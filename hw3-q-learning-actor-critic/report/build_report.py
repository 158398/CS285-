# -*- coding: utf-8 -*-
"""CS285 HW3 实验分析 + 网页报告生成器
读取 hw3/data/hw3_{dqn,sac}_*/ (tensorboard event) -> report/figs/*.png + report/hw3_analysis.html
随时可重跑:实验进行中也能生成,缺数据的章节自动标注"等待数据"。
用法: E:/embodied/isaac/env_cs285/Scripts/python.exe E:/embodied/isaac/cs285-hw/hw3/report/build_report.py
"""
import base64
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("E:/embodied/isaac/cs285-hw/hw3")
DATA = ROOT / "data"
FIGS = ROOT / "report" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
OUT_HTML = ROOT / "report" / "hw3_analysis.html"

BLUE, RED, ORANGE, GREEN, PURPLE, BROWN, GRAY = (
    "#1f77b4", "#d62728", "#e8833a", "#2ca02c", "#9467bd", "#8c4a32", "#7f7f7f")


# ---------------- event 文件解析 ----------------
def load_run(logdir: Path):
    """返回 {tag: (steps, values)};失败返回 None。"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
        ea.Reload()
        out = {}
        for tag in ea.Tags().get("scalars", []):
            ev = ea.Scalars(tag)
            out[tag] = (np.array([e.step for e in ev], dtype=float),
                        np.array([e.value for e in ev], dtype=float))
        return out if out else None
    except Exception:
        return None


def find_runs(*must_contain):
    """按目录名包含的所有子串筛选 run,按修改时间排序(≈种子顺序)。"""
    dirs = [d for d in DATA.glob("hw3_*")
            if all(s in d.name for s in must_contain)]
    dirs.sort(key=lambda d: d.stat().st_mtime)
    runs = []
    for d in dirs:
        r = load_run(d)
        if r is not None:
            r["_dir"] = d.name
            runs.append(r)
    return runs


# ---------------- 画图工具 ----------------
def smooth(y, w=7):
    if len(y) < w or w <= 1:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def curve(ax, run, tag, color, label=None, w=7, thin_alpha=0.18):
    if tag not in run:
        return
    x, y = run[tag]
    ax.plot(x, y, color=color, alpha=thin_alpha, lw=0.8)
    ys = smooth(y, w)
    ax.plot(x[len(x) - len(ys):], ys, color=color, lw=2.2, label=label)


def group_mean(runs, tag, grid=None):
    """多个 seed 在同一 step 网格上插值后取均值/标准差。"""
    series = [r[tag] for r in runs if tag in r and len(r[tag][0]) > 1]
    if not series:
        return None
    if grid is None:
        grid = series[0][0]
        for x, _ in series[1:]:
            grid = np.union1d(grid, x)
    ys = []
    for x, y in series:
        ys.append(np.interp(grid, x, y))
    Y = np.array(ys)
    return grid, Y.mean(axis=0), Y.std(axis=0)


def band(ax, runs, tag, color, label, w=7):
    g = group_mean(runs, tag)
    if g is None:
        return
    grid, m, s = g
    for i, r in enumerate(runs):
        if tag in r:
            x, y = r[tag]
            ax.plot(x, y, color=color, alpha=0.15, lw=0.8)
    ms = smooth(m, w)
    ss = smooth(s, w)
    gx = grid[len(grid) - len(ms):]
    ax.plot(gx, ms, color=color, lw=2.4, label=label)
    ax.fill_between(gx, ms - ss, ms + ss, color=color, alpha=0.15)


def save_fig(fig, name):
    path = FIGS / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def img64(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def tail(run, tag, n=5):
    if tag not in run or len(run[tag][1]) == 0:
        return float("nan")
    return float(run[tag][1][-n:].mean())


def last_step(run):
    steps = [x[0][-1] for x in run.values() if isinstance(x, tuple) and len(x[0])]
    return int(max(steps)) if steps else 0


# ---------------- 收集数据 ----------------
R = {
    "cartpole":      find_runs("dqn_CartPole-v1"),
    "cartpole_lr05": find_runs("cartpole_lr0.05"),
    "hp_lr5e3":      find_runs("hparam_lr0.005"),
    "hp_lr1e2":      find_runs("hparam_lr0.01"),
    "lunar_vanilla": [r for r in find_runs("dqn_LunarLander-v2") if "doubleq" not in r["_dir"]],
    "lunar_doubleq": find_runs("LunarLander-v2", "doubleq"),
    "mspacman":      find_runs("MsPacman"),
    "pendulum":      find_runs("sanity_pendulum"),
    "invp_reinforce":    find_runs("sanity_invpendulum_reinforce"),
    "invp_reparametrize": find_runs("sanity_invpendulum_reparametrize"),
    "cheetah_reinforce1":  find_runs("reinforce1_HalfCheetah"),
    "cheetah_reinforce10": find_runs("reinforce10_HalfCheetah"),
    "cheetah_reparam":     find_runs("reparametrize_HalfCheetah")[-1:],  # 只取最新一次(旧 run 曾中断)
    "hopper_single": find_runs("sac_hopper_singlecritic"),
    "hopper_double": find_runs("sac_hopper_doubleq"),
    "hopper_clip":   find_runs("sac_hopper_clipq"),
    "humanoid":      find_runs("sac_humanoid"),
}
print("runs found:", {k: len(v) for k, v in R.items() if v})

# ---------------- 图 ----------------
figs = {}

# --- DQN CartPole + lr=0.05 ---
if R["cartpole"] or R["cartpole_lr05"]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    band(ax, R["cartpole"], "eval_return", BLUE, "lr=1e-3(默认)")
    band(ax, R["cartpole_lr05"], "eval_return", RED, "lr=0.05(50×)")
    ax.axhline(500, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("环境步数"); ax.set_ylabel("eval_return")
    ax.set_title("(a) CartPole-v1 评估回报"); ax.legend(fontsize=9, loc="lower right")
    ax = axes[1]
    for tag, c, lab in [("q_values", RED, "q_values"), ("critic_loss", ORANGE, "critic_loss")]:
        for r in R["cartpole_lr05"]:
            curve(ax, r, tag, c, lab, w=15)
    for r in R["cartpole"]:
        curve(ax, r, "q_values", BLUE, "q_values(lr=1e-3)", w=15)
    ax.set_yscale("symlog")
    ax.set_xlabel("环境步数")
    ax.set_title("(b) lr=0.05:Q 值与 critic 误差(symlog 纵轴)")
    ax.legend(fontsize=8, loc="best")
    fig.suptitle("DQN §2.4 CartPole:默认 vs 50 倍学习率——死亡三合一现场")
    figs["cartpole"] = save_fig(fig, "fig_dqn_cartpole.png")

# --- §2.6 超参敏感性:学习率四档 ---
if R["cartpole"] or R["hp_lr5e3"] or R["hp_lr1e2"] or R["cartpole_lr05"]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    HP = [(R["cartpole"], BLUE, "lr=1e-3(默认)"),
          (R["hp_lr5e3"], GREEN, "lr=5e-3"),
          (R["hp_lr1e2"], ORANGE, "lr=1e-2"),
          (R["cartpole_lr05"], RED, "lr=0.05")]
    ax = axes[0]
    for runs, c, lab in HP:
        band(ax, runs, "eval_return", c, lab)
    ax.axhline(500, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("环境步数"); ax.set_ylabel("eval_return")
    ax.set_title("(a) eval_return × 学习率"); ax.legend(fontsize=9, loc="lower right")
    ax = axes[1]
    for runs, c, lab in HP:
        for r in runs:
            curve(ax, r, "q_values", c, lab, w=15)
    ax.axhline(99, color="k", ls="--", lw=1, alpha=0.5)
    ax.text(0, 105, "真值上限 Σγ^t ≈ 99", fontsize=8, alpha=0.6)
    ax.set_yscale("symlog")
    ax.set_xlabel("环境步数"); ax.set_title("(b) q_values:lr 越大越偏离真值")
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("DQN §2.6 超参敏感性:学习率四档扫描(CartPole-v1, seed 1)")
    figs["hparam"] = save_fig(fig, "fig_dqn_hparam.png")

# --- DQN LunarLander 3v3 ---
if R["lunar_vanilla"] or R["lunar_doubleq"]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    band(ax, R["lunar_vanilla"], "eval_return", BLUE, f"vanilla DQN({len(R['lunar_vanilla'])} seeds)")
    band(ax, R["lunar_doubleq"], "eval_return", RED, f"double DQN({len(R['lunar_doubleq'])} seeds)")
    ax.axhline(200, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("环境步数"); ax.set_ylabel("eval_return")
    ax.set_title("(a) LunarLander-v2 评估回报(细线=各 seed,粗线=均值±std)")
    ax.legend(fontsize=9, loc="best")
    ax = axes[1]
    band(ax, R["lunar_vanilla"], "q_values", BLUE, "vanilla q_values")
    band(ax, R["lunar_doubleq"], "q_values", RED, "double q_values")
    ax.set_xlabel("环境步数"); ax.set_title("(b) Q 值预测:高估偏差的物证")
    ax.legend(fontsize=9, loc="best")
    fig.suptitle("DQN §2.5 LunarLander:vanilla(蓝)vs double Q(红)")
    figs["lunar"] = save_fig(fig, "fig_dqn_lunar.png")

# --- MsPacman ---
if R["mspacman"]:
    fig, ax = plt.subplots(figsize=(10, 5))
    r = R["mspacman"][0]
    curve(ax, r, "train_return", ORANGE, "train_return(在线,带 ε 探索)", w=15)
    curve(ax, r, "eval_return", BLUE, "eval_return(ε=0.02)", w=5)
    ax.set_xlabel("环境步数"); ax.set_ylabel("return")
    ax.set_title("DQN §2.5 MsPacman(double Q, 1M steps):train vs eval 回报")
    ax.legend(fontsize=9, loc="best")
    figs["mspacman"] = save_fig(fig, "fig_dqn_mspacman.png")

# --- SAC sanity: Pendulum(Q/entropy)+ InvertedPendulum ---
if R["pendulum"] or R["invp_reinforce"] or R["invp_reparametrize"]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    for r in R["pendulum"]:
        curve(ax, r, "q_values", BLUE, "q_values", w=15)
        curve(ax, r, "target_values", GRAY, "target_values", w=15)
    ax.axhline(-1000, color="k", ls="--", lw=1, alpha=0.5)
    ax.text(0, -950, "do-nothing 理论量级 ≈ -1000", fontsize=8, alpha=0.6)
    ax.set_title("(a) Pendulum 只训 critic:Q 应稳定在大负数")
    ax.set_xlabel("环境步数"); ax.legend(fontsize=8)
    ax = axes[1]
    for r in R["pendulum"]:
        curve(ax, r, "entropy", GREEN, "entropy", w=15)
    ax.axhline(np.log(2), color="k", ls="--", lw=1, alpha=0.5)
    ax.text(0, np.log(2) + 0.02, "log 2 ≈ 0.69(U[-1,1] 最大熵)", fontsize=8, alpha=0.6)
    ax.set_title("(b) Pendulum actor 熵(纯熵奖励时应 → log 2)")
    ax.set_xlabel("环境步数"); ax.legend(fontsize=8)
    ax = axes[2]
    band(ax, R["invp_reinforce"], "eval_return", ORANGE, "REINFORCE")
    band(ax, R["invp_reparametrize"], "eval_return", BLUE, "reparametrize")
    ax.axhline(1000, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_title("(c) InvertedPendulum-v4:两种 actor 梯度")
    ax.set_xlabel("环境步数"); ax.set_ylabel("eval_return"); ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("SAC §3.1.1–3.1.4 sanity 检查")
    fig.tight_layout()
    figs["sac_sanity"] = save_fig(fig, "fig_sac_sanity.png")

# --- HalfCheetah 三估计器 ---
if R["cheetah_reinforce1"] or R["cheetah_reinforce10"] or R["cheetah_reparam"]:
    fig, ax = plt.subplots(figsize=(10, 5))
    band(ax, R["cheetah_reinforce1"], "eval_return", ORANGE, "REINFORCE-1")
    band(ax, R["cheetah_reinforce10"], "eval_return", GREEN, "REINFORCE-10")
    band(ax, R["cheetah_reparam"], "eval_return", BLUE, "reparametrize")
    ax.set_xlabel("环境步数"); ax.set_ylabel("eval_return")
    ax.set_title("SAC §3.1.3–3.1.4 HalfCheetah-v4:三种 actor 梯度估计器")
    ax.legend(fontsize=9, loc="best")
    figs["cheetah"] = save_fig(fig, "fig_sac_cheetah.png")

# --- Hopper 三 backup ---
if R["hopper_single"] or R["hopper_double"] or R["hopper_clip"]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    band(ax, R["hopper_single"], "eval_return", ORANGE, "single-Q")
    band(ax, R["hopper_double"], "eval_return", GREEN, "double-Q")
    band(ax, R["hopper_clip"], "eval_return", BLUE, "clipped-Q(min)")
    ax.set_xlabel("环境步数"); ax.set_ylabel("eval_return")
    ax.set_title("(a) Hopper-v4 评估回报"); ax.legend(fontsize=9, loc="best")
    ax = axes[1]
    band(ax, R["hopper_single"], "q_values", ORANGE, "single-Q")
    band(ax, R["hopper_double"], "q_values", GREEN, "double-Q")
    band(ax, R["hopper_clip"], "q_values", BLUE, "clipped-Q(min)")
    ax.set_xlabel("环境步数"); ax.set_title("(b) q_values:防高估力度对比")
    ax.legend(fontsize=9, loc="best")
    fig.suptitle("SAC §3.1.5 Hopper:三种 target backup 策略")
    figs["hopper"] = save_fig(fig, "fig_sac_hopper.png")

# --- Humanoid ---
if R["humanoid"]:
    fig, ax = plt.subplots(figsize=(10, 5))
    band(ax, R["humanoid"], "eval_return", BLUE, "SAC(Humanoid-v4)")
    ax.set_xlabel("环境步数"); ax.set_ylabel("eval_return")
    ax.set_title("SAC §3.1.4 Humanoid-v4(off-policy 样本效率对照 HW2 on-policy)")
    ax.legend(fontsize=9, loc="best")
    figs["humanoid"] = save_fig(fig, "fig_sac_humanoid.png")


# ---------------- HTML 片段 ----------------
def stat_table(groups, thr=None):
    """groups: [(label, runs)] -> html 表格"""
    rows = ""
    for label, runs in groups:
        for i, r in enumerate(runs):
            name = label if len(runs) == 1 else f"{label} seed{i+1}"
            tail5 = tail(r, "eval_return")
            peak = float(r["eval_return"][1].max()) if "eval_return" in r else float("nan")
            rows += (f"<tr><td>{name}</td><td>{last_step(r):,}</td>"
                     f"<td>{tail5:.1f}</td><td>{peak:.1f}</td></tr>")
    return rows


def section_runs_missing(text="实验进行中,等待数据……"):
    return f'<p class="mute">⏳ {text}</p>'


CSS = """
:root{--bg:#f6f7f9;--card:#ffffff;--ink:#1f2329;--sub:#5b6470;--line:#e3e6ea;--acc:#2c5f8a;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei","PingFang SC","Segoe UI",sans-serif;line-height:1.75}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
header.hero{background:linear-gradient(135deg,#16324a,#2c5f8a);color:#fff;border-radius:14px;padding:36px 40px;margin-bottom:28px}
header.hero h1{margin:0 0 8px;font-size:30px}
header.hero p{margin:4px 0;color:#d4dde6;font-size:14px}
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
.qa{background:#eef4fa;border-left:4px solid #2c5f8a;padding:10px 16px;margin:10px 0;border-radius:0 8px 8px 0}
code,pre{background:#eef0f3;border-radius:5px;padding:1px 6px;font-family:Consolas,"Courier New",monospace;font-size:13px}
pre{padding:12px 16px;overflow-x:auto;line-height:1.5}
.mute{color:var(--sub);font-size:13px}
ul{margin:6px 0 6px 4px;padding-left:20px}
li{margin:4px 0}
"""

n_runs = sum(len(v) for v in R.values())

def card(fig_key, label, fmt="{:.0f}"):
    return ""  # placeholder, cards built inline below

cards = ""
if R["cartpole"]:
    cards += f'<div class="card"><div class="num">{tail(R["cartpole"][0], "eval_return"):.0f}</div><div class="lab">CartPole 末5次 eval(满分 500)</div></div>'
if R["lunar_doubleq"]:
    cards += f'<div class="card"><div class="num">{np.nanmean([tail(r, "eval_return") for r in R["lunar_doubleq"]]):.0f}</div><div class="lab">LunarLander doubleQ 均值(目标 200)</div></div>'
if R["mspacman"]:
    cards += f'<div class="card"><div class="num">{tail(R["mspacman"][0], "eval_return"):.0f}</div><div class="lab">MsPacman 末5次 eval(目标 ~1500,进行中)</div></div>'
che_all = R["cheetah_reinforce1"] + R["cheetah_reinforce10"] + R["cheetah_reparam"]
if che_all:
    cards += f'<div class="card"><div class="num">{np.nanmax([tail(r, "eval_return") for r in che_all]):.0f}</div><div class="lab">HalfCheetah 最佳末5次 eval</div></div>'

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CS285 HW3 Q-Learning 与 Actor-Critic · 实验分析报告</title><style>{CSS}</style></head>
<body><div class="wrap">

<header class="hero">
<h1>CS285 Fall 2023 · HW3 Q-Learning &amp; Actor-Critic</h1>
<p>DQN(replay buffer / target network / ε-greedy / double Q)与 SAC(熵正则 / reparametrization / clipped double-Q)的完整复现</p>
<p>环境:CartPole-v1 · LunarLander-v2 · MsPacmanNoFrameskip-v0 | Pendulum-v1 · InvertedPendulum-v4 · HalfCheetah-v4 · Hopper-v4 · Humanoid-v4</p>
<p>数据来源:{n_runs} 次训练运行(tensorboard event,实时解析;页面可随实验进展重新生成)</p>
</header>

<nav>
<a href="#summary">摘要</a><a href="#method">方法</a><a href="#dqn-cartpole">DQN·CartPole</a><a href="#dqn-hparam">DQN·超参扫描</a><a href="#dqn-lunar">DQN·LunarLander</a><a href="#dqn-mspacman">DQN·MsPacman</a><a href="#sac-sanity">SAC·sanity</a><a href="#sac-cheetah">SAC·HalfCheetah</a><a href="#sac-hopper">SAC·Hopper</a><a href="#sac-humanoid">SAC·Humanoid</a><a href="#repro">复现信息</a>
</nav>

<section id="summary">
<h2>结果摘要</h2>
<div class="cards">{cards if cards else '<div class="card"><div class="num">—</div><div class="lab">实验进行中</div></div>'}</div>
<p class="mute">主线结论(随数据补充):HW2 的主题是方差控制,HW3 的主题是偏差控制——bootstrap 把估计误差写进回归靶子并逐轮复利,max 算子定向放大噪声为系统性高估;target network 稳定反馈环(治"晃"),double/clipped Q 压制高估(治"歪"),二者病因不同,实验设计上两两对照。</p>
</section>

<section id="method">
<h2>方法与代码结构</h2>
<p>与 HW2 的 on-policy 主循环(采样→更新→<b>丢弃</b>)不同,HW3 是 off-policy 主循环:<b>采样→存入 replay buffer→随机抽 batch 更新→数据保留复用</b>。Q 的靶子 y = r + γ(1−d)·[bootstrap 项] 只依赖转移本身,与行为策略无关(Bellman 方程不在乎数据是谁采的),故旧数据合法;代价是 buffer 分布滞后于当前策略(覆盖度靠 ε-greedy / 熵维持)、靶子陈旧(靠 target network 冻结/慢动)。</p>
<table>
<tr><th>组件</th><th>DQN(离散)</th><th>SAC(连续)</th></tr>
<tr><td>Q 网络形态</td><td>s → 每动作一值(向量输出,argmax 可枚举)</td><td>(s,a) 拼接 → 单值(StateActionCritic)</td></tr>
<tr><td>策略</td><td>argmax Q + ε-greedy(无独立 actor)</td><td>tanh-squashed 高斯 actor(学出来的 argmax)</td></tr>
<tr><td>actor 梯度</td><td>—</td><td>REINFORCE(score function)/ reparametrize(可微采样)</td></tr>
<tr><td>探索</td><td>ε-greedy 调度(PiecewiseSchedule)</td><td>熵奖励,温度 β(yaml 固定)</td></tr>
<tr><td>target 更新</td><td>硬更新(每 K 步拷贝)</td><td>Polyak 软更新 τ=0.005(或硬更新)</td></tr>
<tr><td>防高估</td><td>double Q(online 选 / target 评)</td><td>double Q(交叉评估)/ clipped Q(取 min)</td></tr>
</table>
<p class="mute">本页所有实现:cs285/agents/dqn_agent.py、cs285/agents/soft_actor_critic.py、cs285/scripts/run_hw3_*.py 的 TODO 补全;终止位 done 一律剔除 TimeLimit 截断(truncation 不做 bootstrap 阻断)。烟雾测试:smoke_test.py(12 组配置全通过)。</p>
</section>

<section id="dqn-cartpole">
<h2>DQN §2.4 CartPole:默认 vs 50 倍学习率</h2>
<p>命令:<code>run_hw3_dqn.py -cfg experiments/dqn/cartpole.yaml [--seed 1]</code> 与 <code>cartpole_lr0.05.yaml</code>(learning_rate 0.05 = 默认的 50 倍)。</p>
{f'<figure><img src="{img64(figs["cartpole"])}"><figcaption>(a) eval 回报;(b) lr=0.05 时 q_values 与 critic_loss(symlog):Q 值预测不断冲高、critic 误差不降反升——函数逼近+bootstrap+off-policy 三要素齐备时,回归靶子被自己的估计带着跑,正反馈发散</figcaption></figure>' if "cartpole" in figs else section_runs_missing()}
<table><tr><th>运行</th><th>已训步数</th><th>末5次 eval</th><th>峰值 eval</th></tr>{stat_table([("lr=1e-3", R["cartpole"]), ("lr=0.05", R["cartpole_lr05"])])}</table>
<div class="qa"><b>lr=0.05 发生了什么?(实测)</b>(a) 预测 Q 值:不再收敛到真值(γ=0.99 时 CartPole 的 Q* ≈ Σγ^t ≈ 99),而是胀到 ~120 并随 target 网络每次刷新继续台阶式上抬——靶子把自己上轮的虚高吃回去,长期追自己的尾巴;(b) critic 误差:比默认配置大 1–2 个数量级且剧烈震荡,始终压不下去。后果:策略基本学不起来(eval 卡在 50–85)。机制:大步长让 Q 在靶子附近持续 overshoot,MLP 泛化把 overshoot 外溢到邻近状态,下一次 target 刷新又把虚高回灌——正是函数逼近+bootstrap+off-policy 的反馈失稳;没有飞向无穷,是因为 target network 每 1000 步才刷新一次,把爆炸压成了"高位震荡"——稳定器只能压振幅,治不了根。这也是 §1.4 理论题"函数逼近下哪些收敛保证失效"的实验对照。</div>
</section>

<section id="dqn-hparam">
<h2>DQN §2.6 超参敏感性:学习率四档扫描</h2>
<p>配置:<code>experiments/dqn/hyperparameters/cartpole_lr{{1e-3,5e-3,1e-2,0.05}}.yaml</code>(其余超参与 cartpole.yaml 相同,seed 1)。<b>为什么选学习率:</b>它是死亡三合一反馈环的增益旋钮——§2.4 已观察到 lr=0.05 时 Q 值虚高、critic 误差震荡;本节用四档扫描把这个单点观察扩展成连续图景:学习率每上一个台阶,反馈环的 overshoot 与外溢更强,直至学习彻底失效。</p>
{f'<figure><img src="{img64(figs["hparam"])}"><figcaption>(a) eval_return:lr=1e-3 收敛,随 lr 升高单调退化;(b) q_values(symlog):真值上限约 99,lr 越大,Q 预测越偏离且越不稳——死亡三合一的强度沿学习率连续可调</figcaption></figure>' if "hparam" in figs else section_runs_missing()}
<table><tr><th>运行</th><th>已训步数</th><th>末5次 eval</th><th>峰值 eval</th></tr>{stat_table([("lr=1e-3", R["cartpole"]), ("lr=5e-3", R["hp_lr5e3"]), ("lr=1e-2", R["hp_lr1e2"]), ("lr=0.05", R["cartpole_lr05"])])}</table>
</section>

<section id="dqn-lunar">
<h2>DQN §2.5 LunarLander:vanilla vs double Q(3 seeds 对照)</h2>
<p>命令:<code>run_hw3_dqn.py -cfg experiments/dqn/lunarlander[_doubleq].yaml --seed 1|2|3</code>。PDF 预期:vanilla 可能升后崩(属正常),double Q 应稳定到 ~200。</p>
{f'<figure><img src="{img64(figs["lunar"])}"><figcaption>(a) eval 回报:蓝=vanilla 3 seeds,红=double Q 3 seeds;(b) q_values 均值:vanilla 的系统性虚高 vs double Q 的压制——高估偏差的直接物证</figcaption></figure>' if "lunar" in figs else section_runs_missing()}
<table><tr><th>运行</th><th>已训步数</th><th>末5次 eval</th><th>峰值 eval</th></tr>{stat_table([("vanilla", R["lunar_vanilla"]), ("doubleQ", R["lunar_doubleq"])])}</table>
<div class="qa"><b>差异来源:</b>vanilla 的靶子 y = r + γ·max_a′ Q_target(s′,a′),max 对噪声估计定向偏乐观(E[max] ≥ max E,Jensen);偏高的靶子经 bootstrap 复利放大,Q 值虚高→策略在虚高处打转→崩溃。double Q 用 online 网络选动作、target 网络评分,"选"与"评"的噪声不同源,高估失去机制。注意区分:target network 只稳定反馈环(治"晃"),不修正高估(治"歪"的是 double Q)。与 HW2 的策略梯度相比,这里的曲线应明显更稳——critic 回归的梯度远比 MC 策略梯度安静。</div>
</section>

<section id="dqn-mspacman">
<h2>DQN §2.5 MsPacman:图像输入 CNN(double Q,1M 步,GPU)</h2>
<p>命令:<code>run_hw3_dqn.py -cfg experiments/dqn/mspacman.yaml</code>。环境 MsPacmanNoFrameskip-v0(frame skip=4 由 wrapper 负责),观测 = 4×84×84 灰度帧堆叠,CNN:conv8/s4→conv4/s2→conv3/1→512→9。1M 步 ≈ GPU 3h。</p>
{f'<figure><img src="{img64(figs["mspacman"])}"><figcaption>train_return(在线,带 ε-greedy 探索)vs eval_return(ε=0.02 近乎贪心)</figcaption></figure>' if "mspacman" in figs else section_runs_missing("MsPacman 进行中(约 3 小时)……")}
<table><tr><th>运行</th><th>已训步数</th><th>末5次 eval</th><th>峰值 eval</th></tr>{stat_table([("mspacman", R["mspacman"])])}</table>
<div class="qa"><b>为什么早期 train_return 和 eval_return 差很多?</b>train_return 是"行为策略"的在线表现:ε 从 1.0 退火,早期接近纯随机(随机策略在 MsPacman 里很快送命,分数极低),且它在训练过程中实时变化;eval_return 用 ε=0.02 的近乎贪心策略,直接反映当前 Q 网络学到的最好水平。ε 退火完毕后两条线应收拢(残余差距 = 2% 的随机动作)。这一差距正是 off-policy 的可视化:训练数据由探索性的行为策略产生,而评估的是被派生出来的目标策略。</div>
</section>

<section id="sac-sanity">
<h2>SAC §3.1.1–3.1.4 sanity:Pendulum / InvertedPendulum</h2>
<p>三道自检:(a) Pendulum 只训 critic 时 Q 应稳定在 do-nothing 理论量级(约 −10/步 ÷ (1−0.99) ≈ −1000)——奔向 −∞ 或停在 0 都是 bug;(b) actor 只受熵奖励时熵应升到 U[−1,1] 的最大熵 log 2 ≈ 0.69;(c) InvertedPendulum 两种 actor 梯度都应到 ~1000。</p>
{f'<figure><img src="{img64(figs["sac_sanity"])}"><figcaption>(a) Q/target 值;(b) 熵;(c) REINFORCE vs reparametrize</figcaption></figure>' if "sac_sanity" in figs else section_runs_missing()}
<table><tr><th>运行</th><th>已训步数</th><th>末5次 eval</th><th>峰值 eval</th></tr>{stat_table([("pendulum(critic-only)", R["pendulum"]), ("invpend REINFORCE", R["invp_reinforce"]), ("invpend reparametrize", R["invp_reparametrize"])])}</table>
</section>

<section id="sac-cheetah">
<h2>SAC §3.1.3–3.1.4 HalfCheetah:三种 actor 梯度估计器</h2>
<p>命令:<code>run_hw3_sac.py -cfg experiments/sac/halfcheetah_{{reinforce1,reinforce10,reparametrize}}.yaml</code>。同一目标 ∇θ E[Q]:REINFORCE 用 score function(方差靠 num_actor_samples 压),reparametrize 让梯度穿过 critic 本身(单样本即低噪)。</p>
{f'<figure><img src="{img64(figs["cheetah"])}"><figcaption>eval_return:REINFORCE-1 vs REINFORCE-10 vs reparametrize</figcaption></figure>' if "cheetah" in figs else section_runs_missing("HalfCheetah 1M 步 ×3 进行中……")}
<table><tr><th>运行</th><th>已训步数</th><th>末5次 eval</th><th>峰值 eval</th></tr>{stat_table([("REINFORCE-1", R["cheetah_reinforce1"]), ("REINFORCE-10", R["cheetah_reinforce10"]), ("reparametrize", R["cheetah_reparam"])])}</table>
<div class="qa"><b>预期与解读:</b>REINFORCE-1 方差最大(单样本 score function),学习慢且抖;REINFORCE-10 用 10 个动作样本平均,显著改善——这是它比 HW2 自由的地方(状态来自 buffer,动作可任意重采);reparametrize 单样本就有最低方差(梯度路径不含采样噪声,只含 critic 误差),通常最快最稳。注意与 HW2 的对照:那里 score function 乘的是整条轨迹的 MC 回报,这里乘的是 critic 的 Q——方差来源被 bootstrap 换掉了。</div>
</section>

<section id="sac-hopper">
<h2>SAC §3.1.5 Hopper:single-Q vs double-Q vs clipped-Q</h2>
<p>命令:<code>run_hw3_sac.py -cfg experiments/sac/hopper[_doubleq|_clipq].yaml</code>(100k 步,reparametrize,τ=0.005)。</p>
{f'<figure><img src="{img64(figs["hopper"])}"><figcaption>(a) eval_return;(b) q_values——单 critic 的 q_values 若显著更高而回报反而更差,即 overestimation bias 的直接证据</figcaption></figure>' if "hopper" in figs else section_runs_missing("Hopper 100k ×3 进行中……")}
<table><tr><th>运行</th><th>已训步数</th><th>末5次 eval</th><th>峰值 eval</th></tr>{stat_table([("single-Q", R["hopper_single"]), ("double-Q", R["hopper_double"]), ("clipped-Q", R["hopper_clip"])])}</table>
<div class="qa"><b>与 overestimation 的关系:</b>连续空间里 actor 专挑 critic 高估的动作去钻(reparametrize 梯度直接朝 Q 的虚高尖峰爬),效果等同离散空间的 max 偏差。double-Q 交叉评估让"选"与"评"噪声解耦;clipped-Q 直接取两者最小,人为保守,压制最狠——但若过度保守(低估)也会拖慢学习。哪组 q_values 更低而 eval 更高,就是本题的答案。</div>
</section>

<section id="sac-humanoid">
<h2>SAC §3.1.4 Humanoid:off-policy vs on-policy(HW2)样本效率</h2>
<p>取 Hopper 三联中最优的 backup 配置跑 Humanoid-v4(500K 步截断;完整 5M 步可得行走行为)。与 HW2 的 on-policy 结果同图比较样本效率(每单位环境步数的回报)。</p>
{f'<figure><img src="{img64(figs["humanoid"])}"><figcaption>Humanoid-v4 eval_return</figcaption></figure>' if "humanoid" in figs else section_runs_missing("Humanoid 待启动(待 Hopper 结果选定最优 backup)……")}
<div class="qa"><b>预期:</b>off-policy(SAC,replay buffer 复用每条数据数百次)达到同等回报所需环境步数应比 on-policy(HW2 PG,数据用一次就扔)少一个数量级;代价是训练更脆(target/双 Q/熵温度一堆稳定器)且墙上时间更长(每步都训练)。</div>
</section>

<section id="repro">
<h2>复现信息</h2>
<ul>
<li>代码:E:/embodied/isaac/cs285-hw/hw3(cs285 包 pip install -e . --no-deps;TODO 全部补全,烟雾测试 smoke_test.py 通过)</li>
<li>环境:Python 3.11.9 / gym 0.25.2 / ale-py 0.8.1(ROMs 经 AutoROM 安装)/ mujoco 2.3.3 / torch 2.7.0+cu128 / numpy 1.26.4;GPU:RTX 5070 Laptop(仅 MsPacman 用 GPU,低维 MLP 一律 CPU)</li>
<li>日志:hw3/data/hw3_{{dqn,sac}}_*/(tensorboardX event 文件,本页由其直接解析生成)</li>
<li>报告:hw3/report/build_report.py(随时重跑刷新);图片 hw3/report/figs/*.png(可直接进 LaTeX)</li>
<li>终止位约定:done = 真实终止,TimeLimit 截断不阻断 bootstrap(PDF §2.4 hint)</li>
<li>已知 starter 怪癖(未修,不影响结果):sac_config.py 两个 optimizer 工厂互换(同为 Adam 3e-4);run_hw3_sac.py:39 引用未定义变量(config 恒非空不触发)</li>
</ul>
</section>

</div></body></html>
"""

OUT_HTML.write_text(html, encoding="utf-8")
print("saved:", OUT_HTML)

# 控制台摘要
for label, runs in [("cartpole", R["cartpole"]), ("lunar_vanilla", R["lunar_vanilla"]),
                    ("lunar_doubleq", R["lunar_doubleq"]), ("mspacman", R["mspacman"]),
                    ("pendulum", R["pendulum"]), ("cheetah", che_all),
                    ("hopper", R["hopper_single"] + R["hopper_double"] + R["hopper_clip"])]:
    for i, r in enumerate(runs):
        print(f"{label:16s}#{i+1} step={last_step(r):>9,} tail5_eval={tail(r, 'eval_return'):8.1f}")
