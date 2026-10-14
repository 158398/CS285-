# -*- coding: utf-8 -*-
"""解析 4 个 CartPole 小批量实验的 stdout 日志,画对比图 (Eval return vs env steps)"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = "C:/Users/Zaiye/.kimi-code/sessions/wd_workspace_614d949b0747/session_7f7e5229-8a39-4b74-9a39-7cd782538983/agents/main/tasks"
RUNS = {
    "vanilla (无 rtg/na)": f"{BASE}/bash-6y50o0ki/output.log",
    "rtg": f"{BASE}/bash-8fhghqqy/output.log",
    "na": f"{BASE}/bash-h84nmhsl/output.log",
    "rtg+na": f"{BASE}/bash-d2z2tqp8/output.log",
}

def parse(path):
    text = open(path, encoding="utf-8", errors="ignore").read()
    ev = [float(x) for x in re.findall(r"Eval_AverageReturn : ([\d.]+)", text)]
    steps = [float(x) for x in re.findall(r"Train_EnvstepsSoFar : ([\d.]+)", text)]
    loss = [float(x) for x in re.findall(r"Actor Loss : ([\d.eE+-]+)", text)]
    return np.array(steps), np.array(ev), np.array(loss)

fig, ax = plt.subplots(figsize=(10, 5.5))
print(f"{'配置':<22}{'初始':>7}{'末10轮均值':>10}{'末20轮均值':>10}{'峰值':>7}{'首次>=190':>9}")
for name, path in RUNS.items():
    steps, ev, loss = parse(path)
    n = len(ev)
    tail10, tail20 = ev[-10:].mean(), ev[-20:].mean()
    first190 = next((i for i, v in enumerate(ev) if v >= 190), None)
    print(f"{name:<22}{ev[0]:>7.1f}{tail10:>10.1f}{tail20:>10.1f}{ev.max():>7.1f}"
          f"{('itr' + str(first190)) if first190 is not None else '从未':>9}")
    # 平滑一下便于观察趋势 (窗口5), 同时画原始半透明线
    kernel = np.ones(5) / 5
    ev_s = np.convolve(ev, kernel, mode="valid")
    ax.plot(steps, ev, alpha=0.2, lw=0.8)
    ax.plot(steps[2 : 2 + len(ev_s)], ev_s, lw=2, label=f"{name} (末10轮 {tail10:.0f})")

ax.axhline(200, color="k", ls="--", lw=1, alpha=0.5)
ax.text(steps[-1] * 0.02, 203, "满分 200", fontsize=9, alpha=0.7)
ax.set_xlabel("环境步数 Train_EnvstepsSoFar")
ax.set_ylabel("Eval_AverageReturn")
ax.set_title("CartPole-v0 小批量 (b=1000) 四配置对比,seed=1\n粗线=5轮滑动平均, 细线=原始值")
ax.legend(loc="lower right")
ax.set_ylim(0, 230)
fig.tight_layout()
out = "E:/embodied/isaac/cs285-hw/hw2/report/cartpole_sb_compare.png"
fig.savefig(out, dpi=150)
print("saved:", out)
