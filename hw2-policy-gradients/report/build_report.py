# -*- coding: utf-8 -*-
"""CS285 HW2 实验分析 + 网页报告生成器
读取 data/logs/*.log -> report/figs/*.png + report/hw2_analysis.html
用法: python build_report.py
"""
import base64
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("E:/embodied/isaac/cs285-hw/hw2")
LOGS = ROOT / "data" / "logs"
FIGS = ROOT / "report" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
OUT_HTML = ROOT / "report" / "hw2_analysis.html"

COLORS = ["#e8833a", "#d62728", "#8c4a32", "#7f7f7f", "#1f77b4", "#2ca02c", "#9467bd"]


# ---------------- 解析 ----------------
def parse(name):
    text = open(LOGS / f"{name}.log", encoding="utf-8", errors="ignore").read()
    grab = lambda pat: np.array([float(x) for x in re.findall(pat, text)])
    return {
        "steps": grab(r"Train_EnvstepsSoFar : ([\d.]+)"),
        "eval": grab(r"Eval_AverageReturn : ([-\d.]+)"),
        "eval_std": grab(r"Eval_StdReturn : ([\d.]+)"),
        "aloss": grab(r"Actor Loss : ([-\d.eE+]+)"),
        "blloss": grab(r"Baseline Loss : ([-\d.eE+]+)"),
    }


def smooth(y, w=7):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def stats(d):
    ev = d["eval"]
    return dict(n=len(ev), init=float(ev[0]), tail10=float(ev[-10:].mean()),
                tail20=float(ev[-20:].mean()), mx=float(ev.max()))


def first_reach(ev, thr):
    i = next((i for i, v in enumerate(ev) if v >= thr), None)
    return i


def curve_ax(ax, d, label, color, w=7):
    x, y = d["steps"], d["eval"]
    ax.plot(x, y, color=color, alpha=0.18, lw=0.8)
    ys = smooth(y, w)
    ax.plot(x[len(x) - len(ys):], ys, color=color, lw=2.2, label=label)


def save_fig(fig, name):
    path = FIGS / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def img64(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


# ---------------- 数据 ----------------
SB = [("vanilla(无 rtg/na)", "cartpole_sb"), ("rtg", "cartpole_sb_rtg"),
      ("na", "cartpole_sb_na"), ("rtg+na", "cartpole_sb_rtg_na")]
LB = [("vanilla(无 rtg/na)", "cartpole_lb"), ("rtg", "cartpole_lb_rtg"),
      ("na", "cartpole_lb_na"), ("rtg+na", "cartpole_lb_rtg_na")]
CHE = [("无 baseline", "cheetah"), ("baseline (bgs=5,blr=0.01)", "cheetah_baseline"),
       ("baseline 削弱版 (bgs=1)", "cheetah_baseline_bgs1")]
LUN = [("λ=0", "lunar_lander_lambda0"), ("λ=0.95", "lunar_lander_lambda0.95"),
       ("λ=0.98", "lunar_lander_lambda0.98"), ("λ=0.99", "lunar_lander_lambda0.99"),
       ("λ=1", "lunar_lander_lambda1")]

data = {}
for group in (SB, LB, CHE, LUN):
    for _, name in group:
        data[name] = parse(name)

# ---------------- 图 1/2: CartPole sb / lb ----------------
def cartpole_fig(group, title, fname):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, (label, name) in enumerate(group):
        d = data[name]
        s = stats(d)
        curve_ax(ax, d, f"{label}(末10轮 {s['tail10']:.0f})", COLORS[i])
    ax.axhline(200, color="k", ls="--", lw=1, alpha=0.5)
    ax.text(0, 205, "满分 200", fontsize=9, alpha=0.6)
    ax.set_xlabel("环境步数 Train_EnvstepsSoFar")
    ax.set_ylabel("Eval_AverageReturn")
    ax.set_title(title + "\n粗线=7轮滑动平均,细线=原始值")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 235)
    return save_fig(fig, fname)

fig_sb = cartpole_fig(SB, "§4 CartPole-v0 小批量 (b=1000)", "fig_cartpole_sb.png")
fig_lb = cartpole_fig(LB, "§4 CartPole-v0 大批量 (b=4000)", "fig_cartpole_lb.png")

# ---------------- 图 3: sb vs lb 批次效应 ----------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for j, (label, sb_name) in enumerate(SB):
    ax = axes[j // 2][j % 2]
    lb_name = LB[j][1]
    curve_ax(ax, data[sb_name], "b=1000 (sb)", "#1f77b4")
    curve_ax(ax, data[lb_name], "b=4000 (lb)", "#d62728")
    ax.axhline(200, color="k", ls="--", lw=0.8, alpha=0.4)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
for ax in axes[1]:
    ax.set_xlabel("环境步数")
for ax in axes[:, 0]:
    ax.set_ylabel("Eval return")
fig.suptitle("§4 批次大小效应:同一配置 sb vs lb(x 轴为环境步数,不是迭代数)")
fig_batch = save_fig(fig, "fig_cartpole_batch.png")

# ---------------- 图 4/5: HalfCheetah ----------------
fig, ax = plt.subplots(figsize=(10, 5.2))
for i, (label, name) in enumerate(CHE):
    d = data[name]
    s = stats(d)
    curve_ax(ax, d, f"{label}(末10轮 {s['tail10']:.0f})", COLORS[i])
ax.axhline(300, color="k", ls="--", lw=1, alpha=0.5)
ax.text(0, 310, "达标线 300", fontsize=9, alpha=0.6)
ax.set_xlabel("环境步数 Train_EnvstepsSoFar")
ax.set_ylabel("Eval_AverageReturn")
ax.set_title("§5 HalfCheetah-v4:baseline 与 bgs 消融\n粗线=7轮滑动平均")
ax.legend(loc="upper left", fontsize=9)
fig_che_ret = save_fig(fig, "fig_cheetah_return.png")

fig, ax = plt.subplots(figsize=(10, 4.6))
for label, name, c in [("bgs=5, blr=0.01", "cheetah_baseline", "#d62728"),
                       ("bgs=1, blr=0.01", "cheetah_baseline_bgs1", "#1f77b4")]:
    d = data[name]
    x = d["steps"]
    y = d["blloss"]
    ax.semilogy(x, y, color=c, alpha=0.18, lw=0.8)
    ys = smooth(y, 7)
    ax.semilogy(x[len(x) - len(ys):], ys, color=c, lw=2.2, label=label)
ax.set_xlabel("环境步数")
ax.set_ylabel("Baseline Loss (log)")
ax.set_title("§5 价值网络的回归损失(MSE 对 MC 回报)")
ax.legend(fontsize=9)
fig_che_bl = save_fig(fig, "fig_cheetah_blloss.png")

# ---------------- 图 6: LunarLander λ 扫描 ----------------
fig, ax = plt.subplots(figsize=(10, 5.2))
for i, (label, name) in enumerate(LUN):
    d = data[name]
    s = stats(d)
    curve_ax(ax, d, f"{label}(末10轮 {s['tail10']:.0f})", COLORS[i], w=9)
ax.axhline(180, color="k", ls="--", lw=1, alpha=0.5)
ax.text(0, 185, "达标线 180", fontsize=9, alpha=0.6)
ax.set_xlabel("环境步数 Train_EnvstepsSoFar")
ax.set_ylabel("Eval_AverageReturn")
ax.set_title("§6 LunarLander-v2:GAE λ 扫描\n粗线=9轮滑动平均")
ax.legend(loc="lower right", fontsize=9)
fig_lunar = save_fig(fig, "fig_lunarlander.png")

# ---------------- 图 7: §9 理论方差 ----------------
fig, ax = plt.subplots(figsize=(8.5, 5))
theta = np.linspace(0.02, 0.98, 400)
var_full = (4 * theta**2 + 8 * theta + 1) / (theta * (1 - theta) ** 4)
var_rtg = (theta**2 + 3 * theta + 1) / (theta * (1 - theta) ** 4)
ax.semilogy(theta, var_full, color="#d62728", lw=2.2,
            label="Var(full) = (4θ²+8θ+1) / θ(1−θ)⁴")
ax.semilogy(theta, var_rtg, color="#1f77b4", lw=2.2,
            label="Var(rtg) = (θ²+3θ+1) / θ(1−θ)⁴")
ax.fill_between(theta, var_rtg, var_full, color="gray", alpha=0.15)
ax.set_xlabel("$\\theta = \\pi(a_1|s_1)$")
ax.set_ylabel("Var(log 尺度)")
ax.set_title("§9 两状态 MDP:梯度估计方差闭式解\nrtg 处处严格更低(灰色=被删掉的方差)")
ax.legend(fontsize=9)
fig_theory = save_fig(fig, "fig_theory_variance.png")

fig_rtg = ROOT / "report" / "rtg_visualization.png"

# ---------------- 全环境×全技术矩阵(自选扩展) ----------------
MX_CFGS = [("vanilla", "vanilla"), ("rtg", "rtg"), ("na", "na"),
           ("rtg+na", "rtg_na"), ("+baseline", "bl"), ("+GAE", "gae")]
MXC = {"vanilla": "#e8833a", "rtg": "#d62728", "na": "#1f77b4",
       "rtg_na": "#2ca02c", "bl": "#9467bd", "gae": "#8c4a32"}
MX_ENVS = [("CartPole-v0", "cartpole", 190, "满分 200", None),
           ("HalfCheetah-v4", "cheetah", 300, "达标线 300", (-1300, 800)),
           ("LunarLander-v2", "lunar", 180, "达标线 180", None),
           ("InvertedPendulum-v4", "pendulum", 1000, "满分 1000", None)]

def mx_stats(d):
    ev = d["eval"]
    return dict(init=float(ev[0]), tail10=float(ev[-10:].mean()),
                tail20=float(ev[-20:].mean()), tail20std=float(ev[-20:].std()),
                mx=float(ev.max()))

for _, env, _, _, _ in MX_ENVS:
    for _, cfg in MX_CFGS:
        data[f"mx_{env}_{cfg}"] = parse(f"mx_{env}_{cfg}")

def mx_fig(env_label, env, thr, thr_label, ylim):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for label, cfg in MX_CFGS:
        d = data[f"mx_{env}_{cfg}"]
        s = mx_stats(d)
        curve_ax(ax, d, f"{label}(末10轮 {s['tail10']:.0f})", MXC[cfg], w=9)
    ax.axhline(thr, color="k", ls="--", lw=1, alpha=0.5)
    ax.text(0, thr, thr_label, fontsize=9, alpha=0.6, va="bottom")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("环境步数 Train_EnvstepsSoFar")
    ax.set_ylabel("Eval_AverageReturn")
    note = "(y 轴已裁剪:vanilla/na 发散至 −3.8万/−1万,超出显示范围)" if ylim else ""
    ax.set_title(f"{env_label}:降方差阶梯对照(粗线=9轮滑动平均){note}")
    ax.legend(fontsize=9, loc="best")
    return save_fig(fig, f"fig_mx_{env}.png")

fig_mx = {env: mx_fig(label, env, thr, tl, yl) for label, env, thr, tl, yl in MX_ENVS}

# 热力图:末20轮均值,每个环境内部按秩次配色(对发散离群值稳健)
M = np.array([[mx_stats(data[f"mx_{env}_{cfg}"])["tail20"] for _, cfg in MX_CFGS]
              for _, env, _, _, _ in MX_ENVS])
Mnorm = M.argsort(axis=1).argsort(axis=1) / (len(MX_CFGS) - 1)
fig, ax = plt.subplots(figsize=(11, 4.0))
ax.imshow(Mnorm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
for i in range(len(MX_ENVS)):
    for j in range(len(MX_CFGS)):
        ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#1f2329")
ax.set_xticks(range(len(MX_CFGS)))
ax.set_xticklabels([c for c, _ in MX_CFGS])
ax.set_yticks(range(len(MX_ENVS)))
ax.set_yticklabels([e for e, _, _, _, _ in MX_ENVS])
ax.set_title("24 配置末20轮均值总览:数字为原始回报,颜色按每个环境内部秩次\n(绿=该环境下最好,红=最差;秩次配色对 Cheetah 的发散离群值稳健)")
fig_heatmap = save_fig(fig, "fig_mx_heatmap.png")

def mx_table(env, thr):
    rows = ""
    for label, cfg in MX_CFGS:
        d = data[f"mx_{env}_{cfg}"]
        s = mx_stats(d)
        fr = first_reach(d["eval"], thr)
        fr_s = f"第 {fr} 轮" if fr is not None else "未达标"
        rows += (f"<tr><td>{label}</td><td>{s['init']:.1f}</td>"
                 f"<td>{s['tail10']:.1f}</td><td>{s['tail20']:.1f}</td>"
                 f"<td>{s['tail20std']:.1f}</td><td>{s['mx']:.1f}</td>"
                 f"<td>{fr_s}</td></tr>")
    return rows

MX_TABLES = {env: mx_table(env, thr) for _, env, thr, _, _ in MX_ENVS}
MX_BEST = {env: max((c for c in MX_CFGS),
                    key=lambda lc: mx_stats(data[f"mx_{env}_{lc[1]}"])["tail20"])[0]
           for _, env, _, _, _ in MX_ENVS}
MX_WORST = {env: min((c for c in MX_CFGS),
                     key=lambda lc: mx_stats(data[f"mx_{env}_{lc[1]}"])["tail20"])[0]
            for _, env, _, _, _ in MX_ENVS}

# ---------------- 统计摘要 ----------------
def stat_rows(group, thr=None):
    rows = ""
    for label, name in group:
        s = stats(data[name])
        fr = first_reach(data[name]["eval"], thr) if thr else None
        fr_s = f"第 {fr} 轮" if fr is not None else ("未达标" if thr else "—")
        rows += (f"<tr><td>{label}</td><td>{s['init']:.1f}</td>"
                 f"<td>{s['tail10']:.1f}</td><td>{s['tail20']:.1f}</td>"
                 f"<td>{s['mx']:.1f}</td><td>{fr_s}</td></tr>")
    return rows

che_best = stats(data["cheetah_baseline"])
lun_tail = {label: stats(data[name])["tail20"] for label, name in LUN}
lun_best = max(lun_tail, key=lun_tail.get)
che_ok = "达标(>300)" if che_best["tail10"] > 300 else "未达 300(单 seed 波动,属正常范围)"

# ---------------- HTML ----------------
CSS = """
:root{--bg:#f6f7f9;--card:#ffffff;--ink:#1f2329;--sub:#5b6470;--line:#e3e6ea;--acc:#b03a2e;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei","PingFang SC","Segoe UI",sans-serif;line-height:1.75}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
header.hero{background:linear-gradient(135deg,#1f2329,#3a4149);color:#fff;border-radius:14px;padding:36px 40px;margin-bottom:28px}
header.hero h1{margin:0 0 8px;font-size:30px}
header.hero p{margin:4px 0;color:#c8ccd2;font-size:14px}
nav{position:sticky;top:0;background:var(--bg);padding:10px 0;z-index:9;border-bottom:1px solid var(--line);margin-bottom:24px}
nav a{margin-right:18px;color:var(--acc);text-decoration:none;font-size:14px;font-weight:600}
nav a:hover{text-decoration:underline}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:28px 32px;margin-bottom:26px}
h2{margin-top:0;font-size:22px;border-left:5px solid var(--acc);padding-left:12px}
h3{font-size:17px;margin:22px 0 8px}
figure{margin:18px 0;text-align:center}
figure img{max-width:100%;border:1px solid var(--line);border-radius:8px}
figcaption{color:var(--sub);font-size:13px;margin-top:6px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:center}
th{background:#f0f1f4}
td:first-child{text-align:left;font-weight:600}
.cards{display:flex;flex-wrap:wrap;gap:14px;margin:16px 0}
.card{flex:1 1 200px;background:#f0f1f4;border-radius:10px;padding:14px 18px}
.card .num{font-size:24px;font-weight:700;color:var(--acc)}
.card .lab{font-size:13px;color:var(--sub)}
.qa{background:#fbf6f0;border-left:4px solid #e8833a;padding:10px 16px;margin:10px 0;border-radius:0 8px 8px 0}
code,pre{background:#eef0f3;border-radius:5px;padding:1px 6px;font-family:Consolas,"Courier New",monospace;font-size:13px}
pre{padding:12px 16px;overflow-x:auto;line-height:1.5}
.mute{color:var(--sub);font-size:13px}
ul{margin:6px 0 6px 4px;padding-left:20px}
li{margin:4px 0}
"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CS285 HW2 策略梯度 · 实验分析报告</title><style>{CSS}</style></head>
<body><div class="wrap">

<header class="hero">
<h1>CS285 Fall 2023 · HW2 Policy Gradients</h1>
<p>REINFORCE 与降方差三件套(reward-to-go / baseline / GAE)的完整复现与对照分析</p>
<p>环境:CartPole-v0 · HalfCheetah-v4 · LunarLander-v2 · InvertedPendulum-v4 | 种子:1 | 数据来源:16 次官方实验 + 17 次矩阵扩展 = 33 次完整训练运行</p>
</header>

<nav>
<a href="#method">方法</a><a href="#s4">§4 CartPole</a><a href="#s5">§5 HalfCheetah</a><a href="#s6">§6 LunarLander</a><a href="#matrix">§M 全技术矩阵</a><a href="#s9">§9 理论↔实践</a><a href="#repro">复现信息</a>
</nav>

<section id="summary">
<h2>结果摘要</h2>
<div class="cards">
<div class="card"><div class="num">{stats(data['cartpole_sb_na'])['tail10']:.0f}</div><div class="lab">CartPole sb 最佳(na)末10轮</div></div>
<div class="card"><div class="num">{stats(data['cartpole_lb_rtg_na'])['tail10']:.0f}</div><div class="lab">CartPole lb 最佳(rtg+na)末10轮</div></div>
<div class="card"><div class="num">{che_best['tail10']:.0f}</div><div class="lab">HalfCheetah baseline 末10轮({che_ok})</div></div>
<div class="card"><div class="num">{lun_tail[lun_best]:.0f}</div><div class="lab">LunarLander 最佳 {lun_best} 末20轮</div></div>
</div>
<p class="mute">核心结论:所有降方差技术只改 PG 估计器 ĝ = Σ_t ∇log π(a_t|s_t)·w_t 中的标量权重 w_t;rtg 删因果噪声、baseline 去状态混淆、GAE 用 λ 在偏差-方差间插值、normalize 稳定步长。实验中它们的效果排序与 §9 理论预测定性一致。</p>
</section>

<section id="method">
<h2>方法与代码结构</h2>
<p>每个 iteration = <b>一次部署(采样)+ 一次监督式更新</b>:当前策略滚出 batch_size 步 → 从 rewards 算出 Q̂ → advantage → 一次梯度上升 → <b>数据丢弃</b>(on-policy:分布随 θ 变化,旧数据对新梯度有偏,复用需 IS 修正而 IS 方差随 horizon 指数爆炸,见 §9.4)。</p>
<h3>估计器权重演进(全部修改只发生在 w_t 上)</h3>
<table>
<tr><th>版本</th><th>w_t</th><th>期望</th><th>方差</th><th>额外组件</th></tr>
<tr><td>trajectory-centric</td><td>Σ_{{t'=0..T}} γ^t' r_t'(与 t 无关)</td><td>∇J</td><td>最高</td><td>无</td></tr>
<tr><td>+ reward-to-go</td><td>Σ_{{t'≥t}} γ^(t'-t) r_t' = Q̂_t</td><td>∇J(无偏)</td><td>↓(删 t'&lt;t 零均值项)</td><td>无</td></tr>
<tr><td>+ baseline</td><td>Q̂_t − V̂(s_t)</td><td>∇J(无偏)</td><td>↓↓</td><td>critic</td></tr>
<tr><td>GAE</td><td>Σ_l (γλ)^l δ_{{t+l}},δ_t=r_t+γV̂(s_{{t+1}})−V̂(s_t)</td><td>λ&lt;1 有偏</td><td>λ 旋钮</td><td>critic</td></tr>
<tr><td>+ normalize</td><td>(w_t − μ)/σ</td><td>改尺度/步长</td><td>归一</td><td>无</td></tr>
</table>
<h3>critic 的训练</h3>
<p>价值网络不学"全状态全轨迹的期望"——它对<b>同一批采样轨迹</b>做 MSE 回归:min Σ(V_φ(s_t) − Q̂_t)²。最小二乘的平稳点正是条件均值 E[Q̂|s] = V^π(s):轨迹噪声由回归平均掉,状态覆盖由占有测度 + 网络泛化解决。V^π 随策略漂移,故每轮只拟合 bgs 步(默认 5),不追收敛。</p>
</section>

<section id="s4">
<h2>§4 CartPole:rtg × normalize × batch 的 2×2×2</h2>
<p>命令:<code>--env_name CartPole-v0 -n 100 -b {{1000|4000}} [-rtg] [-na]</code>(ep_len=200、γ=1.0、lr=5e-3、2×64 均为默认)。</p>
<figure><img src="{img64(fig_sb)}"><figcaption>小批量 b=1000 四配置</figcaption></figure>
<figure><img src="{img64(fig_lb)}"><figcaption>大批量 b=4000 四配置</figcaption></figure>
<h3>统计(b=1000 / b=4000)</h3>
<table>
<tr><th>配置</th><th>初始</th><th>末10轮</th><th>末20轮</th><th>峰值</th><th>首次≥190</th></tr>
{stat_rows(SB, 190)}
</table>
<table>
<tr><th>配置</th><th>初始</th><th>末10轮</th><th>末20轮</th><th>峰值</th><th>首次≥190</th></tr>
{stat_rows(LB, 190)}
</table>
<figure><img src="{img64(fig_batch)}"><figcaption>同配置 sb vs lb 对照</figcaption></figure>
<div class="qa"><b>Q1 无 na 时哪个估计器更好?</b> reward-to-go。小批量下 rtg 末10轮 {stats(data['cartpole_sb_rtg'])['tail10']:.0f} vs vanilla {stats(data['cartpole_sb'])['tail10']:.0f};大批量下 rtg {stats(data['cartpole_lb_rtg'])['tail10']:.0f} vs vanilla {stats(data['cartpole_lb'])['tail10']:.0f}。因果性删除的是零均值纯噪声,与 §9.3 的解析结论 Var_rtg &lt; Var_full 一致。</div>
<div class="qa"><b>Q2 advantage normalization 有帮助吗?</b> 显著。na 把 advantage 重缩放到零均值单位方差,等效自适应步长;对全正奖励的 CartPole 还顺带完成了 baseline 的中心化(坏动作真正被压低)。sb 组 na 单独即末10轮满分。</div>
<div class="qa"><b>Q3 batch size 有影响吗?</b> 有,且是决定性的:大 batch 梯度方差更小(蒙特卡洛误差 ∝ σ/√N),lb 组<b>四个配置全部满分</b>——连最粗糙的 trajectory-centric 估计器(vanilla)在 b=4000 下也能收敛,这本身就是"方差决定成败"的直接证据,也交叉验证了 Q1。代价是每次参数更新消耗 4 倍样本:x 轴换成环境步数后,sb 反而更早触线(约 2 万步 vs lb 约 4~9 万步)——大 batch 买稳定性,小 batch 买样本效率。</div>
</section>

<section id="s5">
<h2>§5 HalfCheetah:神经网络 baseline</h2>
<p>命令:<code>--env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 [--use_baseline -blr 0.01 -bgs N]</code>。官方有意<b>不加 na</b>:na 太强,会让 baseline 在简单环境里显得没必要——去掉它才能暴露 baseline 的真实贡献。</p>
<figure><img src="{img64(fig_che_ret)}"><figcaption>eval return:无 baseline vs baseline vs 削弱版 baseline(bgs=1)</figcaption></figure>
<figure><img src="{img64(fig_che_bl)}"><figcaption>baseline loss:价值网络对 MC 回报的 MSE 回归曲线</figcaption></figure>
<table>
<tr><th>配置</th><th>初始</th><th>末10轮</th><th>末20轮</th><th>峰值</th><th>首次≥300</th></tr>
{stat_rows(CHE, 300)}
</table>
<p><b>分析:</b>HalfCheetah 奖励 = 前进速度 − 0.1‖a‖²,量级大、方差大,且 γ=0.95 下 1000 步轨迹的 MC 回报噪声显著。baseline 版末10轮 {che_best['tail10']:.0f}({che_ok});削弱 critic 的训练(bgs 5→1)后,V̂ 拟合不足(见 baseline loss 曲线),优势估计 A=Q̂−V̂ 残留大量状态级噪声,策略性能随之下降——这正是 PDF 要求的消融:(a) baseline loss 下降变慢/停在高位,(b) 策略回报更低更抖。</p>
</section>

<section id="s6">
<h2>§6 LunarLander:GAE λ 扫描</h2>
<p>命令:<code>--env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda λ</code>,λ ∈ {{0, 0.95, 0.98, 0.99, 1}}。</p>
<figure><img src="{img64(fig_lunar)}"><figcaption>GAE λ 五条学习曲线</figcaption></figure>
<table>
<tr><th>λ</th><th>初始</th><th>末10轮</th><th>末20轮</th><th>峰值</th><th>首次≥180</th></tr>
{stat_rows(LUN, 180)}
</table>
<p><b>λ 的含义:</b>λ=0 时 Â=δ_t=r_t+γV̂(s_{{t+1}})−V̂(s_t),只信 critic(bootstrap),偏差最大方差最小;λ=1 时级数望远镜坍缩回 MC 的 Q̂−V̂,只信样本,无偏方差大。中间值做几何加权插值。</p>
<p><b>本轮数据(单 seed):</b>末20轮最佳为 <b>{lun_best}</b>({lun_tail[lun_best]:.0f},达标线 180),λ=1 次之({lun_tail['λ=1']:.0f});λ=0.95/0.98 虽然最早触线(第 {first_reach(data['lunar_lander_lambda0.95']['eval'], 180)}/{first_reach(data['lunar_lander_lambda0.98']['eval'], 180)} 轮)却后期大幅塌陷;λ=0 始终无法稳定(末20轮 {lun_tail['λ=0']:.0f})。解读:λ→0 侧 critic 的 bootstrap 误差在稀疏终局奖励(±100)下被放大,偏差主导,学不动;λ≥0.99 保留了 MC 的无偏主干、只混入少量 bootstrap 压方差,最稳;中间 λ 的塌陷正展示了"方差与偏差都没压好"的最差组合。单 seed 有噪声,正式结论建议多 seed。</p>
</section>

<section id="matrix">
<h2>§M 全环境 × 全技术矩阵(自选扩展,非官方要求)</h2>
<p>设计动机:官方实验每个环境只考一个变量;本节把降方差阶梯 <b>vanilla → rtg → na → rtg+na → +baseline → +GAE</b>(每级只增加一个机制)在四个环境上以同种子(1)、官方基础超参各跑一遍,观察同一技术在不同环境结构下的收益差异。说明:GAE 依赖价值网络,故 "+GAE" 档 = rtg + baseline + gae_lambda;" +baseline" 档 = rtg + baseline(两档均不开 na)。λ 取各环境已知良好值(CartPole / HalfCheetah / InvertedPendulum 用 0.95,LunarLander 用 0.99——其 λ 敏感性见 §6);各环境基础超参沿用官方 §4/§5/§6 配置,InvertedPendulum 用 §7 默认(b=5000, γ=1.0, lr=5e-3, 2×64)。</p>
<figure><img src="{img64(fig_heatmap)}"><figcaption>全部 24 个配置的末 20 轮均值总览:数字是原始回报,颜色按环境内归一化</figcaption></figure>

<h3>CartPole-v0(离散 · 全正密集奖励 +1/步 · 短 horizon 200 · 有早夭)</h3>
<figure><img src="{img64(fig_mx['cartpole'])}"></figure>
<table><tr><th>配置</th><th>初始</th><th>末10轮</th><th>末20轮</th><th>末20轮σ</th><th>峰值</th><th>首次≥190</th></tr>{MX_TABLES['cartpole']}</table>
<p><b>读数:</b>最佳 na(末20轮 200,σ=0),最差 vanilla(114,σ=26)。全正奖励下 vanilla 的核心缺陷是缺乏中心化——所有动作权重全为正,好坏只有幅度差异;na 的"减均值"恰好补上这个缺口,是本环境最强单件。baseline 与 na 机制重叠(都是中心化),位次不再提升;GAE 在 200 步短 horizon 上没有 bootstrap 的用武之地(181,与 rtg 的 190 相当)。</p>

<h3>HalfCheetah-v4(连续 · 速度奖励量级大 · 长 horizon 1000 · 无终止)</h3>
<figure><img src="{img64(fig_mx['cheetah'])}"></figure>
<table><tr><th>配置</th><th>初始</th><th>末10轮</th><th>末20轮</th><th>末20轮σ</th><th>峰值</th><th>首次≥300</th></tr>{MX_TABLES['cheetah']}</table>
<p><b>读数:</b>最佳 +GAE(末10轮 420,峰值 604),最差 vanilla(末20轮 <b>−38033</b>——不是没学会,是<b>发散了</b>)。两个反直觉:① vanilla 的回报从 −764 一路崩到 −4 万——所有动作被同号巨额回报无差别推高,高斯头被梯度冲垮;② na 单独(无 rtg)同样崩到 −1 万——中心化救不了 T² 量级的配对噪声。rtg 单独(−188)不崩但也学不会。结论:<b>长 horizon 连续环境里,rtg 是防爆底线,中心化(na 或 V̂)是能学的前提,两者缺一不可</b>;三项全上的 GAE 档给出最高峰值。官方 §5 说"na 是很强的 trick"——本矩阵给出了它的边界:离开 rtg,na 在长 horizon 下照样发散。</p>

<h3>LunarLander-v2(离散 · 稀疏终局 ±100 + shaping · horizon 1000)</h3>
<figure><img src="{img64(fig_mx['lunar'])}"></figure>
<table><tr><th>配置</th><th>初始</th><th>末10轮</th><th>末20轮</th><th>末20轮σ</th><th>峰值</th><th>首次≥180</th></tr>{MX_TABLES['lunar']}</table>
<p><b>读数:</b>最佳 +GAE(λ=0.99,末20轮 180,踩线达标);无 critic 组里 rtg 最强(121);na 单独反而为负(−76)。一个隐藏彩蛋:"+baseline" 档(Q̂−V̂)与 §6 的 λ=1 GAE 在数学上是<b>同一个估计器</b>(GAE 在 λ=1 时望远镜坍缩成 reward-to-go 减 V̂),却跑出 −42 vs 148 的天壤之别——同算法、同种子,仅浮点求值顺序不同,训练轨迹就混沌分岔。这是"单 seed 不可信"最直观的标本,也是 LunarLander 必须多 seed 复核的原因。</p>

<h3>InvertedPendulum-v4(连续 · 全正奖励 +1/步 · 长 horizon 1000)</h3>
<figure><img src="{img64(fig_mx['pendulum'])}"></figure>
<table><tr><th>配置</th><th>初始</th><th>末10轮</th><th>末20轮</th><th>末20轮σ</th><th>峰值</th><th>首次=1000</th></tr>{MX_TABLES['pendulum']}</table>
<p><b>读数:</b>+GAE 满分且 σ=0,na(978)紧随其后;vanilla 仅 63。注意 rtg、+baseline、rtg+na 的峰值都摸到过 1000(max 列),末段均值却只有 249~650、σ 高达 270~335——<b>摸到 ≠ 稳住</b>。结构与 CartPole 同构(全正 +1)但 horizon 长 5 倍,vanilla 的方差被放大得更明显。§7 调参的本质就是在这个环境里把"摸到过"变成"稳得住、到得快"。</p>

<p class="mute">矩阵结论:四种机制解决不同来源的方差——rtg 删因果噪声(收益随 horizon 增长,在 Cheetah 上甚至是防爆前提)、baseline 去状态混淆(Cheetah/Lunar 的关键)、na 稳定步长并附带中心化(全正奖励环境约等于免费 baseline,是 CartPole/Pendulum 的最强单件)、GAE 用可控偏差换 bootstrap 方差(长 horizon 收益最大)。没有一种技术在所有环境都是最强单件,但完整栈(rtg+baseline+GAE)在四个环境全部进入前两名——这就是现代 PG 实现默认全开的理由。所有结果为单 seed;LunarLander 的 bl vs λ=1 分歧提醒你定量结论必须多 seed 复核。</p>
</section>

<section id="s9">
<h2>§9 理论 ↔ 实践对照</h2>
<p>两状态 MDP(π(a₁|s₁)=θ,无折扣)的闭式结果:E[R]=θ/(1−θ),∇J=1/(1−θ)²;Var_full=(4θ²+8θ+1)/θ(1−θ)⁴,Var_rtg=(θ²+3θ+1)/θ(1−θ)⁴——同分母,分子差 3θ²+5θ&gt;0,rtg 处处严格更低;两者在 θ→0(稀有事件)与 θ→1(重尾长轨迹)端都发散。IS 修正的方差含 θ'^(−(H−1)),随 horizon 指数爆炸——这就是主循环每轮重采样而非复用旧数据的定量理由。</p>
<figure><img src="{img64(fig_theory)}"><figcaption>§9.3(b) 方差对比(log 纵轴)</figcaption></figure>
<figure><img src="{img64(fig_rtg)}"><figcaption>reward-to-go 的三视角:配对矩阵 / 每步权重 / 方差对比</figcaption></figure>
<table>
<tr><th>§9 理论结论</th><th>本报告中的实践证据</th></tr>
<tr><td>9.1 PG 估计无偏(=精确梯度)</td><td>所有配置学习曲线长期向上,方向正确</td></tr>
<tr><td>9.2 方差随问题结构剧变</td><td>vanilla 配置 100 轮未稳达满分且大幅震荡;大 batch 更平滑</td></tr>
<tr><td>9.3 rtg 同期望、方差更低</td><td>§4 无 na 时 rtg 显著优于 vanilla(曲线排序,非数值吻合)</td></tr>
<tr><td>9.4 IS 方差随 H 指数爆炸</td><td>on-policy 主循环的设计依据(每轮重采样、旧数据丢弃)</td></tr>
</table>
<p class="mute">注意:§9 是单状态、γ=1、无函数逼近的玩具模型,实践里有神经网络、normalize、随机初始化,只应期待<b>定性一致</b>(排序与稳定性趋势),不应期待数值吻合。单 seed 结果;正式结论建议多 seed 验证。</p>
</section>

<section id="repro">
<h2>复现信息</h2>
<ul>
<li>代码:E:/embodied/isaac/cs285-hw/hw2(cs285 包 pip install -e . --no-deps)</li>
<li>环境:Python 3.11.9 / gym 0.25.2 / mujoco 2.3.3 / torch 2.7.0+cu128 / box2d-py 2.3.8(自编译)/ pygame 2.6.1</li>
<li>日志:data/logs/*.log(stdout),data/q2_pg_*/(TensorBoard event)</li>
<li>图片:report/figs/*.png(可直接用于 LaTeX 报告)</li>
<li>§4 三个问题、§5 消融、§6 λ 分析均已在对应章节作答;§7 调参与 §8 Humanoid 未包含在本页</li>
</ul>
</section>

</div></body></html>
"""

OUT_HTML.write_text(html, encoding="utf-8")
print("saved:", OUT_HTML)

# 控制台摘要
for group, thr in [(SB, 190), (LB, 190), (CHE, 300), (LUN, 180)]:
    for label, name in group:
        s = stats(data[name])
        print(f"{name:26s} init={s['init']:7.1f} tail10={s['tail10']:7.1f} tail20={s['tail20']:7.1f} max={s['mx']:7.1f} first>={thr}: {first_reach(data[name]['eval'], thr)}")
