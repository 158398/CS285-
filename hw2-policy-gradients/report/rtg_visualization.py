# -*- coding: utf-8 -*-
"""Reward-to-go 可视化:配对矩阵 + 每步权重 + §9 方差对比 (CS285 HW2 概念图)"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

# ---- Panel 1: T×T (动作, 奖励) 配对矩阵 ----
T = 6
ax = axes[0]
for t in range(T):       # 动作时刻 (x 轴)
    for tp in range(T):  # 奖励时刻 (y 轴)
        kept = tp >= t
        color = "#2ca02c" if kept else "#d62728"
        alpha = 0.85 if kept else 0.30
        ax.add_patch(plt.Rectangle((t, T - 1 - tp), 1, 1, facecolor=color,
                                   alpha=alpha, edgecolor="white"))
ax.set_xlim(0, T)
ax.set_ylim(0, T)
ax.set_xticks(np.arange(T) + 0.5)
ax.set_xticklabels([f"t={i}" for i in range(T)])
ax.set_yticks(np.arange(T) + 0.5)
ax.set_yticklabels([f"t'={T - 1 - i}" for i in range(T)])
ax.set_xlabel("动作时刻 t($\\nabla\\log\\pi(a_t|s_t)$ 来自这一步)")
ax.set_ylabel("奖励时刻 t'")
ax.set_title("① 配对矩阵:rtg 删了什么\n绿=保留 (t'≥t)  红=删除 (t'<t,零均值纯噪声)")

# ---- Panel 2: 每个动作的 score 乘的权重 ----
ax = axes[1]
gamma = 0.9
rewards = np.ones(T)

def reward_to_go(r):
    out, run = [], 0.0
    for x in r[::-1]:
        run = x + gamma * run
        out.append(run)
    return out[::-1]

w_rtg = np.array(reward_to_go(rewards))
w_full = np.full(T, w_rtg[0])
x = np.arange(T)
ax.bar(x, rewards, color="gray", alpha=0.4, label="原始奖励 r_t(=1)")
ax.plot(x, w_full, "o-", color="#d62728", lw=2,
        label=f"full return(恒为 {w_full[0]:.2f})")
ax.plot(x, w_rtg, "s-", color="#1f77b4", lw=2, label="reward-to-go(逐步递减)")
for i, v in enumerate(w_rtg):
    ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f"t={i}" for i in range(T)])
ax.set_title(f"② 动作 t 的 score 乘的权重(γ={gamma}, r≡1)\nfull:所有动作背整条回报;rtg:只背未来")
ax.legend(fontsize=9, loc="lower left")

# ---- Panel 3: §9 两状态 MDP 的梯度估计方差(闭式解) ----
ax = axes[2]
theta = np.linspace(0.02, 0.98, 400)
var_full = (4 * theta**2 + 8 * theta + 1) / (theta * (1 - theta)**4)
var_rtg = (theta**2 + 3 * theta + 1) / (theta * (1 - theta)**4)
ax.semilogy(theta, var_full, color="#d62728", lw=2,
            label="Var(full) = (4θ²+8θ+1) / θ(1−θ)⁴")
ax.semilogy(theta, var_rtg, color="#1f77b4", lw=2,
            label="Var(rtg) = (θ²+3θ+1) / θ(1−θ)⁴")
ax.fill_between(theta, var_rtg, var_full, color="gray", alpha=0.15)
ax.set_xlabel("$\\theta = \\pi(a_1|s_1)$")
ax.set_ylabel("Var(log 尺度)")
ax.set_title("③ §9.3(b):两状态 MDP 的估计方差\nrtg 处处严格更低(灰色=删掉的方差)")
ax.legend(fontsize=9)

fig.tight_layout()
out = "E:/embodied/isaac/cs285-hw/hw2/report/rtg_visualization.png"
fig.savefig(out, dpi=150)
print("saved:", out)
