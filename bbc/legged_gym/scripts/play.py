import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ROOT_DIR)
import time
import isaacgym
from isaacgym import gymapi
import numpy as np
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.torch_jit_utils import *


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_motor = True
    env_cfg.domain_rand.action_delay = True
    env_cfg.domain_rand.action_curr_step = [1]
    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = 1e10
    env_cfg.env.episode_length_s = 500.
    env_cfg.env.mocap_state_init = False
    env_cfg.env.recovery_init_prob = 0.
    env_cfg.env.root_height_obs = True

    env_cfg.record_video = args.record

    train_cfg.runner.num_preload_transitions = 1
    train_cfg.policy.train_with_estimated_latent = True
    train_cfg.estimator.train_with_estimated_explicit = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    env_cfg.env.play_mode = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    stop_state_log = -1  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_pos_bias = np.array([-2.8, -1.8, 1.8])
    record_cam_bias = np.array([-2.0, -1.2, 0.7])
    img_idx = 0

    # set initial commands (behavior mode: 1.walk 2.pace 3.trot 4.canter 5.jump)
    mode_idx = min(max(args.mode, 1), env.latent_c.shape[-1]) - 1
    env.latent_c[:, :] = 0
    env.commands[:, :] = 0
    env.latent_eps[:, :] = 0
    env.latent_c[:, mode_idx] = 1
    env.commands[:, 0] = args.vel_x
    if mode_idx == env.latent_c.shape[-1] - 1:  # jump
        env.commands[:, 3] = args.jump_height
    else:
        env.commands[:, 4] = 0.30
    print('[play] mode = {} ({}), vel_x = {}, jump_height = {}'.format(
        args.mode, env_cfg.env.mocap_category_all[mode_idx], args.vel_x, env.commands[0, 3].item()))

    # ---------------- video recording ----------------
    video_writer, rec_cam = None, None
    if args.record:
        import cv2
        cam_props = gymapi.CameraProperties()
        cam_props.width, cam_props.height = args.record_width, args.record_height
        cam_props.horizontal_fov = 55.0
        cam_props.enable_tensors = False
        rec_cam = env.gym.create_camera_sensor(env.envs[0], cam_props)
        out_dir = os.path.join(ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'videos')
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, 'play_mode{}_{}.mp4'.format(args.mode, time.strftime('%Y%m%d_%H%M%S')))
        fps = int(round(1.0 / env.dt))
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                       (args.record_width, args.record_height))
        print('[record] {} @ {} fps, {} steps'.format(video_path, fps, args.record_steps))
        base_h = []
        import atexit
        atexit.register(video_writer.release)

    for i in range(1000 * int(env.max_episode_length)):
        time0 = time.time()
        if not env_cfg.env.root_height_obs:
            obs[:, env_cfg.env.num_prop] = 0.0
        actions = policy(obs.detach())
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported',
                                        'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if video_writer is not None:
            rp = env.root_states[0, :3].cpu().numpy()
            rq = env.root_states[0, 3:7].cpu().numpy()
            cp = (trans_matrix_ba(rp, [0, 0, quaternion2rpy(rq)[-1]]) @ np.append(record_cam_bias, 1))[:-1]
            base_h.append(float(rp[2] - env.measured_heights[0].mean().item()))
            env.gym.set_camera_location(rec_cam, env.envs[0], gymapi.Vec3(*cp), gymapi.Vec3(*rp))
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            frame = env.gym.get_camera_image(env.sim, env.envs[0], rec_cam, gymapi.IMAGE_COLOR)
            frame = frame.reshape(args.record_height, args.record_width, 4)[:, :, :3]
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if i + 1 >= args.record_steps:
                video_writer.release()
                print('[record] saved {} frames -> {}'.format(i + 1, video_path))
                print('[record] base height over terrain: min {:.3f} / mean {:.3f} / max {:.3f} m'.format(
                    np.min(base_h), np.mean(base_h), np.max(base_h)))
                break

        if MOVE_CAMERA:
            robot_pos = env.root_states[0, :3].cuda().cpu().numpy()
            camera_position = (trans_matrix_ba(robot_pos, [0, 0, 0]) @ np.append(camera_pos_bias, 1))[:-1]
            env.set_camera(camera_position, robot_pos)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': env.default_dof_pos[0, :].cpu().numpy() + env.cfg.control.action_scale *
                                      env.actions[robot_index, :].cpu().numpy(),
                    'dof_pos': env.dof_pos[robot_index, :].cpu().numpy(),
                    'dof_vel': env.dof_vel[robot_index, :].cpu().numpy(),
                    'dof_torque': env.torques[robot_index, :].cpu().numpy(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
            logger.plot_dof_pos()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)

        if not args.record:
            time1 = time.time()
            dt_target = 4 * 1 / 200
            dt_it = time1 - time0
            if dt_it < dt_target:
                time.sleep(dt_target - dt_it)


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    play(args)
