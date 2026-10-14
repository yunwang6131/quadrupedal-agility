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

    # ---- 探针 A:终止前一小段时间的真实航向偏差 ----
    # delta_yaw = target_yaw - yaw,是仿真器真值(不是 student 的估计),
    # 每 depth.update_interval 步更新一次,即策略当时实际看到的那个值。
    if hasattr(self, 'delta_yaw'):
        cur_yaw_err = torch.abs(self.delta_yaw)
    else:
        # __init__ 里会先调一次 post_physics_step,此时 compute_observations 还没跑过,
        # delta_yaw 尚不存在。填 NaN,该步自然不参与均值。
        cur_yaw_err = torch.full((self.num_envs,), float('nan'), device=self.device)
    if not hasattr(self, 'yaw_hist'):
        self.yaw_hist = torch.full((YAW_WIN, self.num_envs), float('nan'), device=self.device)
    self.yaw_hist = torch.cat([self.yaw_hist[1:], cur_yaw_err[None, :]], dim=0)
    self.tr_yaw_win = torch.nanmean(self.yaw_hist, dim=0)   # 最近 YAW_WIN 步均值
    self.tr_yaw_now = cur_yaw_err.clone()                   # 终止瞬间值

    # ---- 探针 B:终止时正在处理的障碍类型 ----
    self.tr_obst = (self.cur_obstacle_types.clone()
                    if torch.is_tensor(self.cur_obstacle_types)
                    else torch.zeros(self.num_envs, dtype=torch.long, device=self.device))

    # 重置的环境清空历史,避免上一 episode 的偏差污染下一个
    self.yaw_hist[:, self.reset_buf] = float('nan')


LeggedRobot.check_termination = _check_termination_with_reasons

DIAG_STEPS = 8000
YAW_WIN = 25          # 最近 25 步 ≈ 0.5 s

# ---------------- 消融开关(用环境变量控制,不用改代码) ----------------
# compute_observations 本来就把真值 delta_yaws / obstacle_types 写进了 obs,
# play 脚本再用网络预测覆盖掉。这里控制「覆不覆盖」,即把特权信息还给 student。
#   GT_YAW=1  航向走真值(默认)
#   GT_OBST=1 障碍类型走真值
# 消融阶梯:
#   GT_YAW=0 GT_OBST=0 → 复现 baseline 0.754
#   GT_YAW=1 GT_OBST=0 → 只还航向,剩余差距 = 几何 latent + 障碍分类
#   GT_YAW=1 GT_OBST=1 → 都还,剩余差距 = 纯几何 latent
USE_GT_YAW = os.environ.get('GT_YAW', '1') == '1'
USE_GT_OBST = os.environ.get('GT_OBST', '0') == '1'
MODE_STR = "航向={}  障碍类型={}".format(
    '真值' if USE_GT_YAW else '预测', '真值' if USE_GT_OBST else '预测')


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
    yaw_win_by = {k: [] for k in term_counts}    # 探针 A:各桶的终止前 yaw 偏差
    yaw_now_by = {k: [] for k in term_counts}
    obst_by = {k: {} for k in term_counts}       # 探针 B:各桶按障碍类型的计数
    # 探针 C:按障碍类型统计 yaw 估计误差与障碍分类准确率(懒初始化,需先知道类别数)
    probe_c = {}
    est_err_by = {k: [] for k in term_counts}   # 终止时刻的 yaw 估计误差,按桶分

    if args.use_camera:
        env_cfg.env.num_envs = 256
        env_cfg.env.draw_height_maps = False

    env_cfg.record_video = args.record

    print("\n" + "=" * 62)
    print("消融模式: " + MODE_STR)
    print("(探针 C 仍照常统计网络的估计质量,只是不喂给策略)")
    print("=" * 62 + "\n")
    sys.stdout.flush()

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

                # 不覆盖 = 保留 compute_observations 写入的真值
                if not USE_GT_YAW:
                    obs[:, n_proprio - n_auxiliary:n_proprio - n_auxiliary + n_delta_yaw] = 1.5 * delta_yaw
                if not USE_GT_OBST:
                    obs[:, n_proprio - n_auxiliary + n_delta_yaw:n_proprio] = \
                        F.one_hot(torch.argmax(obst_type.detach(), dim=-1), num_classes=obst_type.shape[-1])
                pred_obst_type = torch.argmax(obst_type.detach(), dim=-1)[env.lookat_id].item()
                pred_obst_type = env.obstacle.obst_types[int(pred_obst_type)]

                # ---------------- 探针 C ----------------
                # 训练里 student 的预测是 1.5 * 网络输出(on_policy_runner.py:355),
                # 真值 env.delta_yaw 以原始弧度写进 obs(compute_observations 未加 obs_scales),
                # 两者同尺度,可直接相减。
                n_obst_cls = obst_type.shape[-1]
                if not probe_c:
                    for key in ['yaw_sum', 'yaw_cnt', 'cls_ok', 'cls_cnt', 'true_sum']:
                        probe_c[key] = torch.zeros(n_obst_cls, device=env.device)
                pred_yaw = (1.5 * delta_yaw[:, 0]).detach().to(env.device)
                est_err = torch.abs(pred_yaw - env.delta_yaw)
                t_true = env.cur_obstacle_types.to(env.device).long()
                ones = torch.ones_like(est_err)
                probe_c['yaw_sum'].index_add_(0, t_true, est_err)
                probe_c['true_sum'].index_add_(0, t_true, torch.abs(env.delta_yaw))
                probe_c['yaw_cnt'].index_add_(0, t_true, ones)
                pred_cls = torch.argmax(obst_type.detach(), dim=-1).to(env.device)
                probe_c['cls_ok'].index_add_(0, t_true, (pred_cls == t_true).float())
                probe_c['cls_cnt'].index_add_(0, t_true, ones)
                env.last_est_err = est_err        # 供终止时按桶取用

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

                def _record(name, sel):
                    sel_ids = ids[sel]
                    if sel_ids.numel() == 0:
                        return
                    v = env.tr_yaw_win[sel_ids]
                    v = v[~torch.isnan(v)]
                    yaw_win_by[name].extend(v.cpu().tolist())
                    yaw_now_by[name].extend(env.tr_yaw_now[sel_ids].cpu().tolist())
                    if hasattr(env, 'last_est_err'):
                        est_err_by[name].extend(env.last_est_err[sel_ids].cpu().tolist())
                    for t in env.tr_obst[sel_ids].cpu().tolist():
                        obst_by[name][int(t)] = obst_by[name].get(int(t), 0) + 1

                for name, mask in [('成功', reach), ('翻倒', flip), ('撞击', hit),
                                   ('偏航', leave), ('超时', tout)]:
                    sel = mask & (~taken)
                    term_counts[name] += int(sel.sum())
                    _record(name, sel)
                    taken |= sel
                rest = ~taken
                term_counts['其他'] += int(rest.sum())
                _record('其他', rest)

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
                print("消融模式: " + MODE_STR)
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

                # ---------------- 探针 A ----------------
                print("\n" + "=" * 62)
                print("探针 A  终止前 |delta_yaw| (弧度,越大说明航向偏得越多)")
                print("=" * 62)
                print("  {:<6} {:>6} {:>14} {:>14}".format(
                    '类别', 'n', '前0.5s均值', '终止瞬间'))
                for k in term_counts:
                    if not yaw_win_by[k]:
                        continue
                    w = yaw_win_by[k]
                    n_now = yaw_now_by[k]
                    print("  {:<6} {:>6} {:>14.4f} {:>14.4f}".format(
                        k, len(w), sum(w) / len(w), sum(n_now) / len(n_now)))
                base = yaw_win_by['成功']
                hitv = yaw_win_by['撞击']
                if base and hitv:
                    r = (sum(hitv) / len(hitv)) / (sum(base) / len(base))
                    print("\n  撞击 / 成功 = {:.2f}x".format(r))
                    if r > 1.5:
                        print("  → 撞击前航向真实偏差远大于成功样本;前0.5s均值与终止瞬间接近,"
                              "说明是先偏后撞而非撞歪。")
                        print("    注意:本探针量的是真实偏差,不是估计误差,"
                              "区分不了「估计不准」还是「控制不到位」——见探针 C。")
                    else:
                        print("  → 撞击前航向偏差与成功样本相当,撞击与航向无关,应查动作原语。")

                # ---------------- 探针 B ----------------
                try:
                    obst_names = list(env.obstacle.obst_types)
                except Exception:
                    obst_names = None

                def _oname(i):
                    if obst_names is not None and 0 <= i < len(obst_names):
                        return str(obst_names[i])
                    return "type_{}".format(i)

                all_t = sorted({t for d in obst_by.values() for t in d})
                print("\n" + "=" * 62)
                print("探针 B  各障碍类型上的终止分布")
                print("=" * 62)
                cols = [k for k in term_counts if term_counts[k]]
                print("  {:<12}".format('障碍') + "".join("{:>8}".format(c) for c in cols)
                      + "{:>9}{:>9}".format('小计', '失败率'))
                for t in all_t:
                    row = [obst_by[c].get(t, 0) for c in cols]
                    sub = sum(row)
                    ok = obst_by['成功'].get(t, 0)
                    fr = 1.0 - ok / sub if sub else 0.0
                    print("  {:<12}".format(_oname(t))
                          + "".join("{:>8}".format(v) for v in row)
                          + "{:>9}{:>8.1%}".format(sub, fr))

                # ---------------- 探针 C ----------------
                if probe_c:
                    print("\n" + "=" * 62)
                    print("探针 C  各障碍上的感知质量(全程逐步统计,非仅终止时刻)")
                    print("=" * 62)
                    print("  {:<12}{:>12}{:>12}{:>12}{:>12}{:>11}".format(
                        '障碍', '估计误差', '真值均值', '误差/真值', '分类准确率', '样本数'))
                    ys = probe_c['yaw_sum'].cpu()
                    yc = probe_c['yaw_cnt'].cpu()
                    co = probe_c['cls_ok'].cpu()
                    cc = probe_c['cls_cnt'].cpu()
                    ts = probe_c['true_sum'].cpu()
                    rows = {}
                    for i in range(len(yc)):
                        if yc[i] < 1:
                            continue
                        e = (ys[i] / yc[i]).item()
                        tv = (ts[i] / yc[i]).item()
                        a = (co[i] / cc[i]).item()
                        ratio = e / tv if tv > 1e-9 else float('nan')
                        rows[i] = dict(name=_oname(i), err=e, true=tv, ratio=ratio, acc=a)
                        print("  {:<12}{:>12.4f}{:>12.4f}{:>12.2f}{:>11.1%}{:>11}".format(
                            _oname(i), e, tv, ratio, a, int(yc[i].item())))
                    print("\n  「误差/真值」≈1 意味着该 yaw 头不比恒定输出 0 更好;"
                          "越接近 0 说明估计越有效。")

                    # 按数据判读:失败率最高的障碍,其归一化估计误差与全场中位数比
                    if rows and all_t:
                        fail_rate = {}
                        for t in all_t:
                            sub = sum(obst_by[c].get(t, 0) for c in cols)
                            if sub:
                                fail_rate[t] = 1.0 - obst_by['成功'].get(t, 0) / sub
                        if fail_rate:
                            wt = max(fail_rate, key=fail_rate.get)
                            vals = sorted(r['ratio'] for r in rows.values())
                            med = vals[len(vals) // 2]
                            wr = rows.get(wt, {}).get('ratio', float('nan'))
                            print("\n  失败率最高的障碍: {} ({:.1%})".format(
                                rows.get(wt, {}).get('name', wt), fail_rate[wt]))
                            print("  其误差/真值 = {:.2f},全场中位数 = {:.2f}".format(wr, med))
                            if wr > 1.3 * med:
                                print("  → 感知问题:该障碍上估计明显更差。"
                                      "查 depth.angle / yaw 头结构,去掉 .detach() 才对症。")
                            elif wr < 0.9 * med:
                                print("  → 控制问题:该障碍上估计反而更好,"
                                      "策略知道往哪走却做不到。改 .detach() 无用,应查动作原语。")
                            else:
                                print("  → 估计质量与全场相当,失败率却最高,"
                                      "倾向控制问题;建议与 teacher 对照确认。")

                    print("\n  终止时刻的 yaw 估计误差(按归类):")
                    for k in term_counts:
                        if est_err_by[k]:
                            v = est_err_by[k]
                            print("    {:<6} n={:<6} {:.4f}".format(k, len(v), sum(v) / len(v)))
                sys.stdout.flush()
                break


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
