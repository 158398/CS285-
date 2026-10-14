# -*- coding: utf-8 -*-
"""加载训练好的 agent(checkpoint),跑 N 局并录制 mp4 视频。
用法(在 hw3-q-learning-actor-critic/ 目录下):
  python cs285/scripts/eval_to_gif.py --algo dqn -cfg experiments/dqn/cartpole.yaml \
      --ckpt data/<run_dir>/agent_final.pt --out ../docs/assets/videos/cartpole.mp4 -n 3
"""
import argparse
import os

import numpy as np
import torch
import cv2
import imageio.v2 as imageio  # H.264 写出:浏览器不兼容 cv2 的 mp4v

import cs285.env_configs
from cs285.agents.dqn_agent import DQNAgent
from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import utils
from scripting_utils import make_config


def rollout_render(env, agent, max_length, epsilon=0.02):
    """不依赖 utils.sample_trajectory 的渲染回退:ale-py 0.8 的 render() 不收 mode 参数。"""
    frames, ret = [], 0.0
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    for _ in range(max_length):
        img = env.render()
        if isinstance(img, list):
            img = img[0]
        frames.append(np.asarray(img))
        action = agent.get_action(np.asarray(obs), epsilon)
        out = env.step(action)
        if len(out) == 5:
            obs, r, term, trunc, _ = out
            done = term or trunc
        else:
            obs, r, done, _ = out
        ret += r
        if done:
            break
    return frames, ret


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-cfg", "--config_file", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--algo", choices=["dqn", "sac"], required=True)
    p.add_argument("-n", "--num", type=int, default=3, help="trajectories to record")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max_length", type=int, default=1000)
    args = p.parse_args()

    ptu.init_gpu(use_gpu=False)
    config = make_config(args.config_file)

    env0 = config["make_env"](render=True)
    if args.algo == "dqn":
        agent = DQNAgent(env0.observation_space.shape, env0.action_space.n, **config["agent_kwargs"])
    else:
        agent = SoftActorCritic(env0.observation_space.shape, env0.action_space.shape[0], **config["agent_kwargs"])
    agent.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    agent.eval()
    env0.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = None
    total_frames = 0
    for i in range(args.num):
        env = config["make_env"](render=True)
        try:
            traj = utils.sample_trajectory(env, agent, args.max_length, render=True)
            frames = list(traj["image_obs"])
            ret = traj["episode_statistics"]["r"]
        except TypeError:
            # ale-py 0.8: env.render() 不收 mode 参数,走手写 rollout
            frames, ret = rollout_render(env, agent, args.max_length)
        if writer is None:
            h, w = frames[0].shape[:2]
            # H.264(yuv420p):浏览器通用;cv2 的 mp4v 在浏览器里是黑屏
            writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                        pixelformat="yuv420p", quality=8, macro_block_size=1)
        for f in frames:
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)
            writer.append_data(f)  # imageio 直接吃 RGB,无需 BGR 转换
        total_frames += len(frames)
        print(f"traj {i}: return={ret:.1f} frames={len(frames)}")
    if writer is not None:
        writer.close()
    print(f"saved: {args.out} ({total_frames} frames)")


if __name__ == "__main__":
    main()
