# HW3 概念小实验:高估偏差 / 死亡三合一 / bootstrap 偏差复利
# 运行: E:/embodied/isaac/env_cs285/Scripts/python.exe E:/embodied/isaac/cs285-hw/hw3/report/concept_demos.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
OUT = "E:/embodied/isaac/cs285-hw/hw3/report/figs/hw3_concepts.png"

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# ---------- Panel (a): max of noisy estimates is biased upward ----------
n_actions, sigma, trials = 10, 1.0, 20000
# 真值 Q*(s,a) = 0 (所有动作一样好), 估计 Q_hat = Q* + N(0, sigma)
noise = rng.normal(0.0, sigma, size=(trials, n_actions))
max_noisy = noise.max(axis=1)
ax = axes[0]
ax.hist(max_noisy, bins=80, density=True, color="steelblue", alpha=0.85)
ax.axvline(0.0, color="k", ls="--", lw=1.5, label="true max $Q^*$ = 0")
ax.axvline(max_noisy.mean(), color="r", lw=1.5,
           label=f"E[max $\\hat{{Q}}$] = {max_noisy.mean():.2f}")
ax.set_title("(a) Overestimation bias\n$\\hat{Q}=Q^*+\\mathcal{N}(0,1)$, 10 actions, $Q^*\\equiv 0$")
ax.set_xlabel("$\\max_a \\hat{Q}(s,a)$")
ax.set_ylabel("density")
ax.legend(fontsize=8)

# ---------- Panel (b): deadly triad, 2-state 1-parameter counterexample ----------
# s1 -> s2 -> s2 (absorbing), r = 0 everywhere, gamma = 0.9  =>  Q* = 0
# tabular: Q1 <- g*Q2, Q2 <- g*Q2  (contraction)
# linear FA: Q(s1)=w, Q(s2)=2w (forced sharing)  ->  fitted Q iteration
gamma, iters = 0.9, 60
Q = np.array([1.0, 2.0]); tab = []
w_uni, w_onp = 1.0, 1.0; fa_uni, fa_onp = [], []
for _ in range(iters):
    tab.append(np.abs(Q).max())
    Q = np.array([gamma * Q[1], gamma * Q[1]])
    fa_uni.append(abs(w_uni))
    y = gamma * 2.0 * w_uni
    w_uni = (1 * y + 2 * y) / (1.0 + 4.0)                      # uniform weights over (s1, s2)
    fa_onp.append(abs(w_onp))
    y2 = gamma * 2.0 * w_onp
    w_onp = (0.01 * 1 * y2 + 1.0 * 2 * y2) / (0.01 + 4.0)      # on-policy-ish weights
ax = axes[1]
ax.semilogy(tab, lw=2, label="tabular: contraction")
ax.semilogy(fa_uni, lw=2, label="linear FA, uniform weight (x1.08/step)")
ax.semilogy(fa_onp, lw=2, label="linear FA, on-policy weight (x0.90/step)")
ax.set_title("(b) Deadly triad toy\n$Q(s_1){=}w,\\ Q(s_2){=}2w,\\ r{=}0,\\ \\gamma{=}0.9$")
ax.set_xlabel("fitted-Q iteration $k$")
ax.set_ylabel("value magnitude (log)")
ax.legend(fontsize=8)

# ---------- Panel (c): bootstrap compounds a small bias (schematic) ----------
# Q_{t+1} = r + gamma * b * Q_t ;  b = 1 + relative bias of the bootstrap target
gamma3, r, T = 0.99, 1.0, 400
ax = axes[2]
for b, lbl in [(1.00, "b=1.00  unbiased"),
               (1.01, "b=1.01  residual bias"),
               (1.05, "b=1.05  max-of-noise bias")]:
    q, traj = 0.0, []
    for _ in range(T):
        q = r + gamma3 * b * q
        traj.append(q)
    ax.semilogy(traj, lw=2, label=lbl)
ax.axhline(r / (1 - gamma3), color="k", ls="--", lw=1, label="true $Q^*$ = 100")
ax.set_title("(c) Bootstrap compounds bias (schematic)\n$Q_{t+1}=r+\\gamma\\,b\\,Q_t$, $\\gamma=0.99$")
ax.set_xlabel("update step")
ax.set_ylabel("Q (log)")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved:", OUT)
print(f"(a) E[max of {n_actions} N(0,1)] = {max_noisy.mean():.3f}  (true max = 0)")
print(f"(b) ratio/step: uniform = {(gamma*2*(1+2))/(1+4):.4f}, on-policy = {(gamma*2*(0.01+2))/(0.01+4):.4f}")
print(f"(c) fixed points: b=1.00 -> {r/(1-gamma3):.0f}, b=1.01 -> {r/(1-gamma3*1.01):.0f}, b=1.05 -> diverges (gamma*b={gamma3*1.05:.4f}>1)")
