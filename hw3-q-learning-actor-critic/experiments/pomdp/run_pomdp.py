# -*- coding: utf-8 -*-
"""POMDP 记忆体实验:遮蔽 CartPole 的速度观测 (v, ω),比较三种智能体。

  B 组 盲 MLP   :遮蔽 + 原版 DQN(证明"变瞎")
  C 组 堆帧     :遮蔽 + 最近 4 帧拼接(差分可重建速度,廉价基线)
  D 组 GRU 记忆体:遮蔽 + GRU(主角;序列采样 + burn-in,R2D2-lite)
  finish       :汇总曲线图 + 记忆探针(hidden→v/ω 线性回归 R²)+ 行为视频

用法(在 hw3-q-learning-actor-critic/ 目录下):
  python experiments/pomdp/run_pomdp.py --variant blind|stack|gru [--steps 200000 --seed 1]
  python experiments/pomdp/run_pomdp.py --variant finish
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import gym
import cv2

import cs285.infrastructure.pytorch_util as ptu
from cs285.agents.dqn_agent import DQNAgent
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.env_configs.schedule import PiecewiseSchedule

ROOT = Path(__file__).resolve().parents[2]      # hw3-q-learning-actor-critic/
GC = ROOT.parent                                 # git_commit/
RESULTS = ROOT / "experiments" / "pomdp" / "results"
VIDEOS = GC / "docs" / "assets" / "videos"
FIGS = GC / "docs" / "assets" / "images" / "hw3"
for d in (RESULTS, VIDEOS, FIGS):
    d.mkdir(parents=True, exist_ok=True)

GAMMA, LR, BATCH, TARGET_PERIOD, LEARNING_STARTS = 0.99, 1e-3, 128, 1000, 20000
BURNIN, L_UNROLL, GRU_BATCH = 4, 8, 32


# ---------------- wrappers ----------------
class ObsMaskWrapper(gym.Wrapper):
    """把 CartPole 的 v(idx 1)、ω(idx 3) 置零;保持 4 维形状,各组共享网络结构。"""

    @staticmethod
    def _mask(obs):
        obs = np.asarray(obs, dtype=np.float32).copy()
        obs[1] = 0.0
        obs[3] = 0.0
        return obs

    def reset(self, **kw):
        return self._mask(self.env.reset(**kw))

    def step(self, a):
        o, r, d, i = self.env.step(a)
        return self._mask(o), r, d, i


class FrameStackK(gym.Wrapper):
    """拼接最近 k 帧(不足 k 帧时重复首帧)。"""

    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = None
        self.observation_space = gym.spaces.Box(
            low=np.tile(env.observation_space.low, k),
            high=np.tile(env.observation_space.high, k), dtype=np.float32)

    def _get(self):
        return np.concatenate(self.frames).astype(np.float32)

    def reset(self, **kw):
        o = self.env.reset(**kw)
        self.frames = [o.copy() for _ in range(self.k)]
        return self._get()

    def step(self, a):
        o, r, d, i = self.env.step(a)
        self.frames = self.frames[1:] + [o]
        return self._get(), r, d, i


def make_env(variant, render=False):
    env = gym.make("CartPole-v1", render_mode="rgb_array" if render else None)
    env = ObsMaskWrapper(env)
    if variant == "stack":
        env = FrameStackK(env, k=4)
    return env


# ---------------- 评估与视频 ----------------
def evaluate(get_action, env, n=5, max_len=600):
    rets = []
    for _ in range(n):
        obs = env.reset()
        ret, done, steps = 0.0, False, 0
        while not done and steps < max_len:
            obs, r, done, _ = env.step(get_action(obs))
            ret += r
            steps += 1
        rets.append(ret)
    return float(np.mean(rets))


def record_video(get_action, variant, max_len=600, fps=30):
    env = make_env(variant, render=True)
    obs = env.reset()
    writer, ret, done, steps = None, 0.0, False, 0
    out = VIDEOS / f"pomdp_{variant}.mp4"
    while not done and steps < max_len:
        img = env.render(mode="rgb_array")
        if isinstance(img, list):
            img = img[0]
        img = cv2.resize(img, (250, 250), interpolation=cv2.INTER_CUBIC)
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (250, 250))
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        obs, r, done, _ = env.step(get_action(obs))
        ret += r
        steps += 1
    writer.release()
    env.close()
    print(f"  video {out.name}: return={ret:.0f}, frames={steps}")
    return ret


# ---------------- B/C 组:MLP DQN ----------------
def train_mlp(variant, steps, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env, eval_env = make_env(variant), make_env(variant)

    def make_critic(shape, n):
        return ptu.build_mlp(int(np.prod(shape)), n, 2, 64)

    agent = DQNAgent(
        env.observation_space.shape, env.action_space.n,
        make_critic=make_critic,
        make_optimizer=lambda p: torch.optim.Adam(p, lr=LR),
        make_lr_schedule=lambda o: torch.optim.lr_scheduler.ConstantLR(o, factor=1.0),
        discount=GAMMA, target_update_period=TARGET_PERIOD, use_double_q=True,
    )
    buf = ReplayBuffer(capacity=steps + 1000)
    eps = PiecewiseSchedule([(0, 1.0), (steps * 0.1, 0.02)], outside_value=0.02)
    obs, curve = env.reset(), []
    t0 = time.time()
    for step in range(steps):
        a = agent.get_action(obs, eps.value(step))
        nobs, r, done, info = env.step(a)
        buf.insert(observation=obs, action=a, reward=r, next_observation=nobs,
                   done=done and not info.get("TimeLimit.truncated", False))
        obs = env.reset() if done else nobs
        if step >= LEARNING_STARTS:
            b = ptu.from_numpy(buf.sample(BATCH))
            agent.update(obs=b["observations"], action=b["actions"], reward=b["rewards"],
                         next_obs=b["next_observations"], done=b["dones"], step=step)
        if step % 10000 == 0:
            ev = evaluate(lambda o: agent.get_action(o, 0.02), eval_env)
            curve.append([step, ev])
            print(f"  [{variant}] step {step:>7,} eval={ev:6.1f} ({time.time()-t0:.0f}s)")
    curve.append([steps, evaluate(lambda o: agent.get_action(o, 0.02), eval_env)])
    torch.save(agent.state_dict(), RESULTS / f"{variant}_agent.pt")
    json.dump(curve, open(RESULTS / f"{variant}_curve.json", "w"))
    print(f"[{variant}] done, final eval={curve[-1][1]:.1f}")


# ---------------- D 组:GRU 记忆体 ----------------
class GRUQNet(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2, enc_dim=64, hid_dim=64):
        super().__init__()
        self.enc = ptu.build_mlp(obs_dim, enc_dim, 1, 64)
        self.gru = nn.GRUCell(enc_dim, hid_dim)
        self.q = nn.Linear(hid_dim, n_actions)
        self.hid_dim = hid_dim

    def rollout(self, obs_seq):
        # obs_seq: (S, B, obs_dim) -> qs (S, B, n_actions),h 从零开始
        B = obs_seq.shape[1]
        h = torch.zeros(B, self.hid_dim, device=obs_seq.device)
        qs = []
        for t in range(obs_seq.shape[0]):
            h = self.gru(self.enc(obs_seq[t]), h)
            qs.append(self.q(h))
        return torch.stack(qs)

    def step_once(self, x, h):
        h = self.gru(self.enc(x), h)
        return self.q(h), h


class EpisodeBuffer:
    """按回合存序列,采样 burnin+L+1 的连续小段(供 GRU 展开)。"""

    def __init__(self):
        self.episodes, self.cur = [], None

    def start(self, obs):
        self.cur = {"obs": [np.asarray(obs, dtype=np.float32)], "act": [], "rew": [], "done": []}

    def add(self, a, r, nobs, done_td, episode_end):
        self.cur["act"].append(a)
        self.cur["rew"].append(r)
        self.cur["done"].append(done_td)
        self.cur["obs"].append(np.asarray(nobs, dtype=np.float32))
        if episode_end:
            self.episodes.append({k: np.array(v) for k, v in self.cur.items()})
            self.cur = None

    def __len__(self):
        return len(self.episodes)

    def sample(self, B, burnin, L):
        need = burnin + L + 1
        obs_seq = np.zeros((need, B, 4), dtype=np.float32)
        act = np.zeros((L, B), dtype=np.int64)
        rew = np.zeros((L, B), dtype=np.float32)
        done = np.zeros((L, B), dtype=np.float32)
        for j in range(B):
            while True:
                ep = self.episodes[np.random.randint(len(self.episodes))]
                T = len(ep["act"])
                if T >= burnin + L:
                    break
            t = np.random.randint(0, T - (burnin + L) + 1)
            obs_seq[:, j] = ep["obs"][t:t + need]
            act[:, j] = ep["act"][t + burnin: t + burnin + L]
            rew[:, j] = ep["rew"][t + burnin: t + burnin + L]
            done[:, j] = ep["done"][t + burnin: t + burnin + L]
        return obs_seq, act, rew, done


def train_gru(steps, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env, eval_env = make_env("gru"), make_env("gru")
    online = GRUQNet().to(ptu.device)
    target = GRUQNet().to(ptu.device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=LR)
    buf = EpisodeBuffer()
    eps = PiecewiseSchedule([(0, 1.0), (steps * 0.1, 0.02)], outside_value=0.02)

    def live_get_action_factory():
        state = {"h": torch.zeros(1, online.hid_dim, device=ptu.device)}

        def get_action(obs, epsilon):
            if np.random.random() < epsilon:
                a = np.random.randint(2)
            else:
                with torch.no_grad():
                    q, _ = online.step_once(ptu.from_numpy(np.asarray(obs))[None], state["h"])
                a = int(q.argmax(1).item())
            # hidden 对所有动作候选都同样前进一步(与所执行动作无关,状态推进只依赖观测)
            with torch.no_grad():
                _, state["h"] = online.step_once(ptu.from_numpy(np.asarray(obs))[None], state["h"])
            return a

        def reset_h():
            state["h"] = torch.zeros(1, online.hid_dim, device=ptu.device)

        return get_action, reset_h

    get_action_eps, reset_h = live_get_action_factory()
    obs = env.reset()
    buf.start(obs)
    reset_h()
    curve, t0 = [], time.time()
    for step in range(steps):
        a = get_action_eps(obs, eps.value(step))
        nobs, r, done, info = env.step(a)
        done_td = done and not info.get("TimeLimit.truncated", False)
        buf.add(a, r, nobs, done_td, episode_end=done)
        if done:
            obs = env.reset()
            buf.start(obs)
            reset_h()
        else:
            obs = nobs
        if step >= LEARNING_STARTS and len(buf) > 5:
            obs_seq, act, rew, dn = buf.sample(GRU_BATCH, BURNIN, L_UNROLL)
            obs_seq = ptu.from_numpy(obs_seq)
            act = ptu.from_numpy(act)
            rew = ptu.from_numpy(rew)
            dn = ptu.from_numpy(dn)
            qs_on = online.rollout(obs_seq)                      # (BURNIN+L+1, B, A)
            with torch.no_grad():
                qs_tg = target.rollout(obs_seq)
                a_star = qs_on[BURNIN + 1:BURNIN + L_UNROLL + 1].argmax(-1, keepdim=True)
                q_next = qs_tg[BURNIN + 1:BURNIN + L_UNROLL + 1].gather(-1, a_star).squeeze(-1)
                y = rew + GAMMA * (1 - dn) * q_next              # (L, B)
            q_pred = qs_on[BURNIN:BURNIN + L_UNROLL].gather(-1, act.unsqueeze(-1)).squeeze(-1)
            loss = nn.MSELoss()(q_pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % TARGET_PERIOD == 0:
                target.load_state_dict(online.state_dict())
        if step % 10000 == 0:
            # 评估:独立 live hidden
            state = {"h": torch.zeros(1, online.hid_dim, device=ptu.device)}

            def eval_action(o):
                with torch.no_grad():
                    q, state["h"] = online.step_once(ptu.from_numpy(np.asarray(o))[None], state["h"])
                return int(q.argmax(1).item()) if np.random.random() > 0.02 else np.random.randint(2)

            def eval_reset():
                state["h"] = torch.zeros(1, online.hid_dim, device=ptu.device)

            rets = []
            for _ in range(5):
                o = eval_env.reset()
                eval_reset()
                ret, d, st = 0.0, False, 0
                while not d and st < 600:
                    o, r, d, _ = eval_env.step(eval_action(o))
                    ret += r
                    st += 1
                rets.append(ret)
            curve.append([step, float(np.mean(rets))])
            print(f"  [gru] step {step:>7,} eval={np.mean(rets):6.1f} ({time.time()-t0:.0f}s)")
    torch.save(online.state_dict(), RESULTS / "gru_agent.pt")
    json.dump(curve, open(RESULTS / "gru_curve.json", "w"))
    print(f"[gru] done, final eval={curve[-1][1]:.1f}")


# ---------------- finish:图 + probe + 视频 ----------------
def load_gru():
    agent = GRUQNet().to(ptu.device)
    agent.load_state_dict(torch.load(RESULTS / "gru_agent.pt", map_location=ptu.device))
    agent.eval()
    return agent


def gru_live_action(agent):
    state = {"h": torch.zeros(1, agent.hid_dim, device=ptu.device)}

    def act(obs):
        with torch.no_grad():
            q, state["h"] = agent.step_once(ptu.from_numpy(np.asarray(obs))[None], state["h"])
        return int(q.argmax(1).item())

    def reset():
        state["h"] = torch.zeros(1, agent.hid_dim, device=ptu.device)

    return act, reset


def finish():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # --- A 组(完备观测)曲线:git_commit 内已有的 cartpole run 事件文件 ---
    curves = {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        runs = sorted(ROOT.glob("data/hw3_dqn_dqn_CartPole-v1_*"), key=lambda p: p.stat().st_mtime)
        ea = EventAccumulator(str(runs[-1]), size_guidance={"scalars": 0})
        ea.Reload()
        ev = ea.Scalars("eval_return")
        curves["A 完备观测 DQN"] = ([e.step for e in ev], [e.value for e in ev])
    except Exception as e:
        print("A 组曲线读取失败(跳过):", e)
    for name, label in [("blind", "B 致盲 MLP"), ("stack", "C 遮蔽+堆帧k4"), ("gru", "D 遮蔽+GRU 记忆体")]:
        p = RESULTS / f"{name}_curve.json"
        if p.exists():
            c = json.load(open(p))
            curves[label] = ([x for x, _ in c], [y for _, y in c])

    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = {"A 完备观测 DQN": "#2f5d50", "B 致盲 MLP": "#a63a2b", "C 遮蔽+堆帧k4": "#c2641e", "D 遮蔽+GRU 记忆体": "#7a4d9e"}
    for label, (x, y) in curves.items():
        ax.plot(x, y, lw=2.2, color=colors.get(label), label=label, alpha=0.9)
    ax.axhline(500, color="k", ls="--", lw=1, alpha=0.4)
    ax.set_xlabel("环境步数")
    ax.set_ylabel("eval return")
    ax.set_title("POMDP 记忆体实验:遮蔽速度观测 (v, ω) 后,谁能补回丢失的信息?")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(FIGS / "pomdp_curves.png", dpi=150)
    print("saved fig:", FIGS / "pomdp_curves.png")

    # --- probe:GRU hidden 线性预测真值 v/ω vs 单帧 (x,θ) ---
    agent = load_gru()
    env = make_env("gru")
    # 采集:用与 act 同步推进的 hidden(每步基于历史观测更新),真值取自 env.unwrapped.state
    H, Xs, Vv, Ww = [], [], [], []
    Hraw = {"h": torch.zeros(1, agent.hid_dim, device=ptu.device)}
    for ep in range(10):
        o = env.reset()
        Hraw["h"] = torch.zeros(1, agent.hid_dim, device=ptu.device)
        d, st = False, 0
        while not d and st < 600:
            x = ptu.from_numpy(np.asarray(o))[None]
            with torch.no_grad():
                q, Hraw["h"] = agent.step_once(x, Hraw["h"])
            s = env.unwrapped.state
            H.append(Hraw["h"].squeeze(0).cpu().numpy())
            Xs.append([s[0], s[2]])
            Vv.append(s[1])
            Ww.append(s[3])
            o, r, d, _ = env.step(int(q.argmax(1).item()))
            st += 1
    def r2(X, y, seed=0):
        """岭回归线性探针:标准化 + 随机打乱 70/30 划分(时序划分在非平稳轨迹上不稳定)。"""
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        n = len(X)
        idx = np.random.RandomState(seed).permutation(n)
        X, y = X[idx], y[idx]
        tr, te = slice(0, int(n * 0.7)), slice(int(n * 0.7), n)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xs = (X - mu) / sd
        Xm = np.c_[Xs[tr], np.ones(tr.stop - tr.start)]
        w = np.linalg.solve(Xm.T @ Xm + 1e-2 * np.eye(Xm.shape[1]), Xm.T @ y[tr])
        pred = np.c_[Xs[te], np.ones(n - te.start)] @ w
        ss_res = float(((y[te] - pred) ** 2).sum())
        ss_tot = float(((y[te] - y[te].mean()) ** 2).sum())
        return 1 - ss_res / max(ss_tot, 1e-9)

    probe = {
        "hidden→v": r2(H, Vv),
        "hidden→ω": r2(H, Ww),
        "单帧(x,θ)→v": r2(Xs, Vv),
        "单帧(x,θ)→ω": r2(Xs, Ww),
    }
    json.dump(probe, open(RESULTS / "probe.json", "w"), ensure_ascii=False, indent=2)
    print("probe:", {k: round(v, 3) for k, v in probe.items()})

    fig, ax = plt.subplots(figsize=(8, 4.6))
    labels = list(probe.keys())
    vals = [probe[k] for k in labels]
    bars = ax.bar(labels, vals, color=["#7a4d9e", "#7a4d9e", "#a89d88", "#a89d88"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R²(线性探针,测试集)")
    ax.set_title("记忆里装着什么:GRU hidden 对被遮蔽速度 (v, ω) 的线性可预测性")
    fig.tight_layout()
    fig.savefig(FIGS / "pomdp_probe.png", dpi=150)
    print("saved fig:", FIGS / "pomdp_probe.png")

    # --- 视频:致盲 vs 记忆体 ---
    blind_agent = DQNAgent((4,), 2, make_critic=lambda s, n: ptu.build_mlp(int(np.prod(s)), n, 2, 64),
                           make_optimizer=lambda p: torch.optim.Adam(p, lr=LR),
                           make_lr_schedule=lambda o: torch.optim.lr_scheduler.ConstantLR(o, factor=1.0),
                           discount=GAMMA, target_update_period=TARGET_PERIOD, use_double_q=True)
    blind_agent.load_state_dict(torch.load(RESULTS / "blind_agent.pt", map_location=ptu.device))
    blind_agent.eval()
    record_video(lambda o: blind_agent.get_action(o, 0.0), "blind")

    # GRU:每段视频只跑一局,录前重置 hidden(模拟 episode 开头无历史)
    act3, reseth3 = gru_live_action(agent)
    reseth3()
    record_video(act3, "gru")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["blind", "stack", "gru", "finish"], required=True)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    ptu.init_gpu(use_gpu=False)
    if args.variant in ("blind", "stack"):
        train_mlp(args.variant, args.steps, args.seed)
    elif args.variant == "gru":
        train_gru(args.steps, args.seed)
    else:
        finish()


if __name__ == "__main__":
    main()
