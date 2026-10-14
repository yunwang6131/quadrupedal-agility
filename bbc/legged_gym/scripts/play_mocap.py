"""Kinematic playback of the retargeted mocap clips (no policy, no physics).

Each robot is driven frame-by-frame straight from the JSON in bbc/mocap_data,
so what you see is exactly the reference motion the discriminator is trained against.

Examples
--------
# one clip, following camera, faithful root motion
python play_mocap.py --motions jump_0 --headless True --record

# five gaits side by side (root xy pinned to each lane, static camera)
python play_mocap.py --motions walk_0,pace_0,trot_0,canter_0,jump_0 --headless True --record
"""
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ROOT_DIR)
import json
import time
import argparse
import isaacgym
from isaacgym import gymapi, gymtorch
import numpy as np
import torch
from legged_gym.envs import *  # noqa: F401,F403  (import order matters: avoids a circular import)
from rsl_rl.datasets.motion_loader import MotionLoader

MOCAP_DIR_LB = os.path.join(ROOT_DIR, 'mocap_data', 'mocap_all_lb')
MOCAP_DIR_ULB = os.path.join(ROOT_DIR, 'mocap_data', 'mocap_all_ulb')
URDF = os.path.join(ROOT_DIR, 'resources', 'robots', 'go2', 'urdf', 'go2.urdf')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--motions', type=str, default='walk_0,pace_0,trot_0,canter_0,jump_0',
                   help='comma separated clip names (without .json)')
    p.add_argument('--dataset', type=str, default='lb', choices=['lb', 'ulb'],
                   help='lb = labeled clips, ulb = unlabeled clips')
    p.add_argument('--headless', type=bool, default=False)
    p.add_argument('--device_id', type=int, default=0)
    p.add_argument('--spacing', type=float, default=1.1, help='lane spacing in y [m]')
    p.add_argument('--speed', type=float, default=1.0, help='playback rate, 1.0 = real time')
    p.add_argument('--pin_xy', type=int, default=-1,
                   help='1 keep each robot in its lane (treadmill), 0 follow the real root path, '
                        '-1 auto (pin when playing more than one clip)')
    p.add_argument('--record', action='store_true', default=False)
    p.add_argument('--record_steps', type=int, default=500)
    p.add_argument('--record_width', type=int, default=1280)
    p.add_argument('--record_height', type=int, default=720)
    return p.parse_args()


def load_motion(path):
    with open(path, 'r') as f:
        motion_json = json.load(f)
    frames = MotionLoader.reorder(np.array(motion_json['Frames']))
    return frames, float(motion_json['FrameDuration'])


def sample_motion(frames, frame_duration, t):
    """Sample a looping clip at time t, keeping the per-cycle displacement (cycle offset)."""
    n = frames.shape[0]
    total = (n - 1) * frame_duration
    cycle, tau = divmod(t, total) if total > 0 else (0, 0.0)
    idx = tau / frame_duration
    i0 = int(np.floor(idx))
    i1 = min(i0 + 1, n - 1)
    b = idx - i0

    f0, f1 = frames[i0], frames[i1]
    out = f0 + (f1 - f0) * b

    # quaternions: nearest-neighbour blend then renormalize
    q0, q1 = f0[3:7], f1[3:7]
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = q0 + (q1 - q0) * b
    out[3:7] = q / (np.linalg.norm(q) + 1e-8)

    out[:2] += cycle * (frames[-1, :2] - frames[0, :2])
    return out


def main():
    args = get_args()
    mocap_dir = MOCAP_DIR_LB if args.dataset == 'lb' else MOCAP_DIR_ULB
    names = [m.strip() for m in args.motions.split(',') if m.strip()]
    motions = [load_motion(os.path.join(mocap_dir, n + '.json')) for n in names]
    num_envs = len(motions)
    pin_xy = (num_envs > 1) if args.pin_xy < 0 else bool(args.pin_xy)
    for n, (f, fd) in zip(names, motions):
        print('[mocap] {:<10} {:>3} frames @ {:.4f} s  ->  {:.2f} s per cycle'.format(
            n, f.shape[0], fd, (f.shape[0] - 1) * fd))

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 50.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0., 0., 0.)  # kinematic replay: states are overwritten every frame
    sim_params.use_gpu_pipeline = False
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.use_gpu = True

    graphics_device = args.device_id  # camera sensors need a graphics device even when headless
    sim = gym.create_sim(args.device_id, graphics_device, gymapi.SIM_PHYSX, sim_params)

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0., 0., 1.)
    plane.static_friction, plane.dynamic_friction, plane.restitution = 1., 1., 0.
    gym.add_ground(sim, plane)

    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    asset_options.disable_gravity = True
    asset_options.collapse_fixed_joints = True
    asset_options.fix_base_link = False
    asset = gym.load_asset(sim, os.path.dirname(URDF), os.path.basename(URDF), asset_options)
    num_dof = gym.get_asset_dof_count(asset)

    # all clips share one env: a camera sensor only renders the env it was created in
    env = gym.create_env(sim, gymapi.Vec3(-20., -20., 0.), gymapi.Vec3(20., 20., 10.), 1)
    origins = []
    for i in range(num_envs):
        origin = np.array([0., i * args.spacing, 0.])
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*origin)
        gym.create_actor(env, asset, pose, names[i], 0, 1)
        origins.append(origin)
    gym.prepare_sim(sim)

    root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(num_envs, num_dof, 2)

    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    video_writer, rec_cam = None, None
    if args.record:
        import cv2
        cam_props = gymapi.CameraProperties()
        cam_props.width, cam_props.height = args.record_width, args.record_height
        cam_props.horizontal_fov = 50.0 if num_envs > 1 else 55.0
        rec_cam = gym.create_camera_sensor(env, cam_props)
        out_dir = os.path.join(ROOT_DIR, 'logs', 'mocap', 'videos')
        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(out_dir, 'mocap_{}_{}.mp4'.format(
            '_'.join(names)[:40], time.strftime('%Y%m%d_%H%M%S')))
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 50,
                                       (args.record_width, args.record_height))
        legend = ' | '.join('{}. {}'.format(i + 1, n) for i, n in enumerate(names))
        print('[record] {} @ 50 fps, {} steps'.format(video_path, args.record_steps))

    dt = sim_params.dt
    t = 0.0
    step = 0
    while True:
        for i, (frames, frame_duration) in enumerate(motions):
            f = sample_motion(frames, frame_duration, t)
            root_states[i, :3] = torch.from_numpy(f[:3].astype(np.float32))
            root_states[i, :3] += torch.from_numpy(origins[i].astype(np.float32))
            if pin_xy:
                root_states[i, :2] = torch.from_numpy(origins[i][:2].astype(np.float32))
            root_states[i, 3:7] = torch.from_numpy(f[3:7].astype(np.float32))
            root_states[i, 7:13] = 0.
            dof_state[i, :, 0] = torch.from_numpy(f[7:19].astype(np.float32))
            dof_state[i, :, 1] = 0.

        # kinematic replay: write the state, render it, never step physics
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state.view(-1, 2)))

        # camera
        if num_envs > 1:
            look = np.array([0.1, (num_envs - 1) * args.spacing / 2., 0.34])
            cam = look + np.array([-5.6, -2.0, 2.0])
        else:
            look = root_states[0, :3].numpy().copy()
            yaw = np.arctan2(2. * (root_states[0, 6] * root_states[0, 5] + root_states[0, 3] * root_states[0, 4]),
                             1. - 2. * (root_states[0, 4] ** 2 + root_states[0, 5] ** 2))
            bias = np.array([-1.9, -1.1, 0.65])
            c, s = np.cos(yaw), np.sin(yaw)
            cam = look + np.array([c * bias[0] - s * bias[1], s * bias[0] + c * bias[1], bias[2]])

        if viewer is not None:
            gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(*cam), gymapi.Vec3(*look))
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
            if gym.query_viewer_has_closed(viewer):
                break

        if video_writer is not None:
            import cv2
            gym.set_camera_location(rec_cam, env, gymapi.Vec3(*cam), gymapi.Vec3(*look))
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            frame = gym.get_camera_image(sim, env, rec_cam, gymapi.IMAGE_COLOR)
            frame = frame.reshape(args.record_height, args.record_width, 4)[:, :, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if num_envs > 1:
                cv2.putText(frame, legend, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
            video_writer.write(frame)
            if step + 1 >= args.record_steps:
                video_writer.release()
                print('[record] saved {} frames -> {}'.format(step + 1, video_path))
                break

        t += dt * args.speed
        step += 1

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
