import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ROOT_DIR)
import isaacgym
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import torch.nn.functional as F
import statistics
import time
import numpy as np
from collections import deque
from legged_gym.utils.math import trans_matrix
from legged_gym.utils.torch_jit_utils import quaternion2rpy

from legged_gym.envs.base.legged_robot import LeggedRobot

_orig_check_termination = LeggedRobot.check_termination


def _check_termination_with_reasons(self):
    """Run the env's own check, then snapshot each reason before reset_idx wipes the state."""
    _orig_check_termination(self)
    self.tr_contact = torch.any(
        torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
    self.tr_roll = torch.abs(self.roll) > 1.5
    self.tr_pitch = torch.abs(self.pitch) > 1.5
    self.tr_height = self.root_states[:, 2] < -0.25
    self.tr_leave = self.leave_goal_ids.clone()
    self.tr_timeout = self.episode_length_buf > self.max_episode_length
    self.tr_reach = self.cur_goal_idx >= (self.env_goals.size(1) - self.obstacle.last_goal_repeat)
    # use_camera 分支下 check_termination 还会用 reach_last_goal 触发重置(也是成功条件),
    # 不接的话这类终止会全部掉进「其他」桶
    self.tr_reach_last = torch.norm(
        self.root_states[:, :2] - self.env_goals[:, -self.obstacle.last_goal_repeat, :2],
        dim=1) < self.cfg.env.next_goal_threshold


LeggedRobot.check_termination = _check_termination_with_reasons

DIAG_STEPS = 8000


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1024
    env_cfg.env.episode_length_s = 40

    env_cfg.depth.angle = [0, 1]
    env_cfg.depth.depth_noise = 0.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_action = False
    env_cfg.obstacle.randomize_start = True
    env_cfg.obstacle.curriculum = False
    env_cfg.obstacle.randomize_border = False
    env_cfg.env.next_goal_threshold = 0.45

    reach_goal_buffer = deque(maxlen=1000)
    term_counts = {k: 0 for k in
                   ['成功', '偏航', '翻倒', '撞击', '超时', '其他']}
    flag_counts = {k: 0 for k in
                   ['reach_goal', 'reach_last', 'leave_goal', 'roll', 'pitch', 'height', 'contact', 'timeout']}

    if args.use_camera:
        env_cfg.env.num_envs = 256
        env_cfg.env.draw_height_maps = False

    env_cfg.record_video = args.record

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    obs_bbc = env.get_observations_bbc()

    # load policy
    train_cfg.runner.resume = True
    train_cfg.estimator.load_estimator_bbc = False
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(env=env, name=args.task, args=args,
                                                                   train_cfg=train_cfg, return_log_dir=True)

    policy = ppo_runner.get_inference_policy(device=env.device)
    policy_bbc = ppo_runner.get_inference_policy_bbc(device=env.device)
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        ppo_runner.depth_backbone.augment = None
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    # ---------------- video recording ----------------
    video_writer, rec_cam = None, None
    if args.record:
        import cv2
        cam_props = gymapi.CameraProperties()
        cam_props.width, cam_props.height = args.record_width, args.record_height
        cam_props.enable_tensors = False
        rec_cam = env.gym.create_camera_sensor(env.envs[0], cam_props)
        out_dir = os.path.join(ROOT_DIR, "logs", args.proj_name, args.exptid, "videos")
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, "play_{}.mp4".format(time.strftime("%Y%m%d_%H%M%S")))
        fps = int(round(1.0 / env.dt))
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                       (args.record_width, args.record_height))
        print("[record] {} @ {} fps, {} steps".format(video_path, fps, args.record_steps))
        import atexit
        atexit.register(video_writer.release)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None
    infos["delta_yaw_ok"] = torch.ones(env.num_envs, dtype=torch.bool, device=ppo_runner.device)

    n_proprio = env.cfg.env.n_proprio
    n_delta_yaw = env.cfg.env.n_delta_yaw
    n_obst_type = env.cfg.env.n_obst_type
    n_auxiliary = env.cfg.env.n_auxiliary
    n_depth_latent = train_cfg.policy.scan_encoder_dims[-1]

    estimator_cfg = train_cfg.estimator
    n_priv_est = estimator_cfg.priv_states_dim
    n_prop_est = estimator_cfg.num_prop
    n_scan_est = estimator_cfg.num_scan
    delta_yaw_set = []
    delta_yaw_pred_set = []
    abs_vel_set = []
    for i in range(1000 * int(env.max_episode_length)):
        with torch.no_grad():
            priv_states_estimated = estimator(obs[:, :n_prop_est])
            obs[:, n_prop_est + n_auxiliary + n_scan_est:n_prop_est + n_auxiliary + n_scan_est + n_priv_est] = priv_states_estimated
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    obs_student = obs[:, :n_proprio].clone()
                    obs_student[:, n_proprio-n_auxiliary:n_proprio] = 0
                    depth_latent_yaw_type = depth_encoder(infos["depth"], obs_student)
                    depth_latent = depth_latent_yaw_type[:, :n_depth_latent]
                    delta_yaw = depth_latent_yaw_type[:, n_depth_latent:n_depth_latent + n_delta_yaw]
                    obst_type = depth_latent_yaw_type[:, n_depth_latent + n_delta_yaw:]

                obs[:, n_proprio - n_auxiliary:n_proprio - n_auxiliary + n_delta_yaw] = 1.5 * delta_yaw
                obs[:, n_proprio - n_auxiliary + n_delta_yaw:n_proprio] = \
                    F.one_hot(torch.argmax(obst_type.detach(), dim=-1), num_classes=obst_type.shape[-1])
                pred_obst_type = torch.argmax(obst_type.detach(), dim=-1)[env.lookat_id].item()
                pred_obst_type = env.obstacle.obst_types[int(pred_obst_type)]

                delta_yaw_set.append(env.delta_yaw[0].cpu().numpy())
                delta_yaw_pred_set.append(delta_yaw[0].cpu().numpy())
                abs_vel_set.append((env.base_lin_vel[0][0].item()**2 + env.base_lin_vel[0][1].item()**2)**0.5)
            else:
                depth_latent = None

            if hasattr(ppo_runner.alg, "depth_actor"):
                embedding = ppo_runner.alg.depth_actor(obs, hist_encoding=True, scandots_latent=depth_latent)
                prob = ppo_runner.alg.depth_actor.actor_d(embedding)
                mean = ppo_runner.alg.depth_actor.actor_c(embedding)
                actions = torch.cat([torch.argmax(prob, dim=-1).unsqueeze(-1), mean], dim=-1)

            else:
                actions = policy(obs, hist_encoding=True, scandots_latent=depth_latent)

            next_commands = env.set_commands(actions)
            obs_bbc[:, -next_commands.shape[1]:] = next_commands
            actions_bbc = policy_bbc(obs_bbc, hist_encoding=True)

            obs, _, rews, dones, infos, _, _ = env.step(actions_bbc)
            obs_bbc = env.get_observations_bbc()

            if video_writer is not None:
                rp = env.root_states[env.lookat_id, :3].cpu().numpy()
                rq = env.root_states[env.lookat_id, 3:7].cpu().numpy()
                cam_bias = np.array([-1.8, -0.8, 0.8])
                cp = (trans_matrix(rp, [0, 0, quaternion2rpy(rq)[-1]]) @ np.append(cam_bias, 1))[:-1]
                env.gym.set_camera_location(rec_cam, env.envs[0],
                                            gymapi.Vec3(*cp), gymapi.Vec3(*rp))
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                frame = env.gym.get_camera_image(env.sim, env.envs[0], rec_cam, gymapi.IMAGE_COLOR)
                frame = frame.reshape(args.record_height, args.record_width, 4)[:, :, :3]
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if i + 1 >= args.record_steps:
                    video_writer.release()
                    print("[record] saved {} frames -> {}".format(i + 1, video_path))
                    break

            new_ids = (dones > 0).nonzero(as_tuple=False)
            reach_goal_buffer.extend(infos['reach_goal'][new_ids][:, 0].cpu().numpy().tolist())

            if new_ids.numel() > 0:
                ids = new_ids[:, 0]
                reach = env.tr_reach[ids] | env.tr_reach_last[ids]
                leave = env.tr_leave[ids]
                flip = env.tr_roll[ids] | env.tr_pitch[ids] | env.tr_height[ids]
                hit = env.tr_contact[ids]
                tout = env.tr_timeout[ids]

                flag_counts['reach_goal'] += int(env.tr_reach[ids].sum())
                flag_counts['reach_last'] += int(env.tr_reach_last[ids].sum())
                flag_counts['leave_goal'] += int(leave.sum())
                flag_counts['roll'] += int(env.tr_roll[ids].sum())
                flag_counts['pitch'] += int(env.tr_pitch[ids].sum())
                flag_counts['height'] += int(env.tr_height[ids].sum())
                flag_counts['contact'] += int(env.tr_contact[ids].sum())
                flag_counts['timeout'] += int(tout.sum())

                # 互斥归类,按优先级:成功 > 翻倒 > 撞击 > 偏航 > 超时
                taken = torch.zeros_like(reach)
                for name, mask in [('成功', reach), ('翻倒', flip), ('撞击', hit),
                                   ('偏航', leave), ('超时', tout)]:
                    sel = mask & (~taken)
                    term_counts[name] += int(sel.sum())
                    taken |= sel
                term_counts['其他'] += int((~taken).sum())

            if (i + 1) % 1000 == 0:
                tot = sum(term_counts.values())
                sr = statistics.mean(reach_goal_buffer) if reach_goal_buffer else 0.0
                print("[{}步] 终止 {} 次  成功率 {:.4f}".format(i + 1, tot, sr))
                if tot > 0:
                    print("  互斥归类: " + "  ".join(
                        "{} {} ({:.1%})".format(k, v, v / tot) for k, v in term_counts.items() if v))
                    print("  原始标志: " + "  ".join(
                        "{} {}".format(k, v) for k, v in flag_counts.items() if v))
                sys.stdout.flush()

            if i + 1 >= DIAG_STEPS:
                tot = sum(term_counts.values())
                print("\n" + "=" * 62)
                print("诊断完成  {} 步  终止 {} 次  成功率 {:.4f}".format(
                    i + 1, tot, statistics.mean(reach_goal_buffer) if reach_goal_buffer else 0.0))
                print("=" * 62)
                for k, v in term_counts.items():
                    if v:
                        print("  {:<6} {:>6}  {:>7.2%}".format(k, v, v / tot))
                print("-" * 62)
                print("原始标志(可重叠):")
                for k, v in flag_counts.items():
                    print("  {:<12} {:>6}".format(k, v))
                break


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
