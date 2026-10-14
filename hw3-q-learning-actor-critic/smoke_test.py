# -*- coding: utf-8 -*-
"""HW3 补全代码烟雾测试:用小规模真实环境数据直接驱动 agent 更新,
验证形状契约、梯度流、double_q/backup 各分支、软/硬 target 更新,30 秒出结果。
用法: E:/embodied/isaac/env_cs285/Scripts/python.exe smoke_test.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import gym

import cs285.env_configs
import cs285.infrastructure.pytorch_util as ptu
from cs285.agents.dqn_agent import DQNAgent
from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer

ptu.init_gpu(use_gpu=False)
np.random.seed(0)

# ---------------- DQN ----------------
print("=== DQN smoke (CartPole-v1) ===")
cfg = cs285.env_configs.configs["dqn_basic"](
    "CartPole-v1", total_steps=1000, learning_starts=100, target_update_period=10
)
env = cfg["make_env"]()
agent = DQNAgent(env.observation_space.shape, env.action_space.n, **cfg["agent_kwargs"])

buf = ReplayBuffer(capacity=10000)
obs = env.reset()
for _ in range(2000):
    a = agent.get_action(obs, epsilon=1.0)
    nobs, r, done, info = env.step(a)
    buf.insert(
        observation=obs, action=a, reward=r, next_observation=nobs,
        done=done and not info.get("TimeLimit.truncated", False),
    )
    obs = env.reset() if done else nobs

for step in range(50):
    b = ptu.from_numpy(buf.sample(128))
    out = agent.update(
        obs=b["observations"], action=b["actions"], reward=b["rewards"],
        next_obs=b["next_observations"], done=b["dones"], step=step,
    )
print(f"  single-q   loss={out['critic_loss']:.5f} q={out['q_values']:.3f} "
      f"target={out['target_values']:.3f} grad_norm={out['grad_norm']:.3f}")
assert np.isfinite(out["critic_loss"]) and np.isfinite(out["q_values"])

cfg2 = cs285.env_configs.configs["dqn_basic"]("CartPole-v1", use_double_q=True)
agent2 = DQNAgent(env.observation_space.shape, env.action_space.n, **cfg2["agent_kwargs"])
b = ptu.from_numpy(buf.sample(128))
out2 = agent2.update(
    obs=b["observations"], action=b["actions"], reward=b["rewards"],
    next_obs=b["next_observations"], done=b["dones"], step=0,
)
print(f"  double-q   loss={out2['critic_loss']:.5f} (branch ok)")
assert np.isfinite(out2["critic_loss"])

# ---------------- SAC ----------------
print("=== SAC smoke (Pendulum-v1) ===")
scfg = cs285.env_configs.configs["sac"]("Pendulum-v1", target_update_period=100)
senv = scfg["make_env"]()
sobs = senv.reset()
sbuf = ReplayBuffer(capacity=10000)
for _ in range(1500):
    a = senv.action_space.sample()
    snobs, r, done, info = senv.step(a)
    sbuf.insert(
        observation=sobs, action=a, reward=r, next_observation=snobs,
        done=done and not info.get("TimeLimit.truncated", False),
    )
    sobs = senv.reset() if done else snobs

def make_agent(grad_type, backup, soft):
    kws = dict(scfg["agent_kwargs"])
    kws["actor_gradient_type"] = grad_type
    kws["target_critic_backup_type"] = backup
    kws["num_critic_networks"] = 2 if backup in ("doubleq", "min") else 1
    if soft:
        kws["target_update_period"] = None
        kws["soft_target_update_rate"] = 0.005
    else:
        kws["target_update_period"] = 100
        kws["soft_target_update_rate"] = None
    return SoftActorCritic(
        senv.observation_space.shape, senv.action_space.shape[0], **kws
    )

for grad_type in ["reinforce", "reparametrize"]:
    for backup in ["mean", "doubleq", "min"]:
        for soft in [False, True]:
            sa = make_agent(grad_type, backup, soft)
            for step in range(5):
                b = ptu.from_numpy(sbuf.sample(128))
                out = sa.update(
                    observations=b["observations"], actions=b["actions"],
                    rewards=b["rewards"], next_observations=b["next_observations"],
                    dones=b["dones"], step=step,
                )
            ok = np.isfinite(out["critic_loss"]) and np.isfinite(out["actor_loss"])
            print(f"  {grad_type:14s} backup={backup:8s} "
                  f"{'soft' if soft else 'hard'}-target  "
                  f"critic_loss={out['critic_loss']:.4f} actor_loss={out['actor_loss']:.4f} "
                  f"entropy={out['entropy']:.3f} q={out['q_values']:.1f}")
            assert ok

print("ALL SMOKE TESTS PASSED")
