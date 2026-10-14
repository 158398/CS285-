# CS285 Deep RL Foundations

> Reproducing [UC Berkeley CS285 Fall 2023](https://rail.eecs.berkeley.edu/deeprlcourse/) end-to-end —
> from imitation learning to policy gradients to Q-learning & actor-critic —
> as a controlled-experiment study of **why deep RL works, and when it breaks**.
> 配套讲解教程站(与官方讲义互补,不替代):`docs/`(GitHub Pages 托管)。

**中文摘要**:完整复现 Berkeley CS285 三份核心作业(BC/DAgger → REINFORCE 全家桶 → DQN/SAC)。
特色不是"跑通",而是系统的对照实验:24 配置方差矩阵、vanilla DQN 崩溃实录、死亡三合一现场、
三种 actor 梯度估计器对比,外加一个带个人风格的互动教程站(实验室玩具 + 智能体动物园)。

## The storyline: imitation → variance → bias

| HW | Paradigm | Core question | One-line answer (with evidence) |
|----|----------|---------------|--------------------------------|
| 1 | Imitation (BC, DAgger) | Why does BC fail off its training distribution? | Covariate shift: Hopper BC ≈30% of expert → DAgger ≈ expert |
| 2 | On-policy PG (REINFORCE) | The estimator is unbiased — so why does it fail? | **Variance**: baseline −140 → 347; vanilla diverges to −38,033 |
| 3 | Off-policy value learning (DQN, SAC) | The estimator is quiet — so why does it fail? | **Bias**: max-of-noise overestimation, bootstrap compounding, deadly triad |

## Key results

| Environment | Algorithm | Result | Reference |
|---|---|---|---|
| Ant-v4 | BC | **97.9%** of expert | — |
| Hopper-v4 | DAgger | **≈ expert** (BC alone: ≈30%) | — |
| CartPole-v0 | REINFORCE+rtg+na | **200/200**(b=4000 四配置全满分) | 200 |
| HalfCheetah-v4 | REINFORCE+baseline | **347**(no baseline: −140) | 300 |
| LunarLander-v2 | GAE λ=0.99 | **180** | 180 |
| CartPole-v1 | DQN | **452**(Q̂ → 解析值 ≈99) | 500 |
| LunarLander-v2 | Double DQN ×3 seeds | 89/130/59,最优单局 211–277(vanilla:全负) | 200 |
| MsPacman (pixels) | Double DQN, CNN | **1,776**(RTX 5070, 2h05m) | ~1,500 |
| HalfCheetah-v4 | SAC reparametrize | **4,316**(R-10: 1,461;R-1: 97) | — |
| Hopper-v4 | SAC clipped-Q | **544**(double-Q: 440;single-Q: 350) | — |
| Humanoid-v4 | SAC clipped-Q | **1,355** @ 595k steps(truncated) | — |

(数字 = 末 5 次评估回报均值;完整曲线见 `docs/hw{2,3}_analysis.html` 与各 hw 的 README。)

## Repository structure

```
hw1-imitation-learning/      BC + DAgger(README 即第 1 讲教程)
hw2-policy-gradients/        REINFORCE/rtg/baseline/GAE + 24 配置矩阵(README 即第 2 讲教程)
hw3-q-learning-actor-critic/ DQN/Double DQN/SAC(README + 教程站章节)
docs/                        互动教程站:总目录、HW3 完整章节、实验室(死亡三合一滑块/POMDP 企划)、
                             智能体动物园(行为视频)、工程解剖、两份深度分析 HTML
```

## Quickstart

```bash
# 每个 hw 目录是独立的 cs285 包
cd hw3-q-learning-actor-critic
pip install -e . --no-deps
pip install ale-py==0.8.1 "autorom[accept-rom-license]"   # 仅 HW3 Atari 需要

python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 1
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/halfcheetah_reparametrize.yaml

# 训练结束自动保存 agent_final.pt;录制行为视频:
python cs285/scripts/eval_to_gif.py --algo dqn -cfg experiments/dqn/cartpole.yaml \
    --ckpt data/<run_dir>/agent_final.pt --out cartpole.mp4 -n 3
```

环境:Python 3.11 · PyTorch 2.7(cu128)· gym 0.25.2 · mujoco 2.3.3 · ale-py 0.8.1 · box2d-py 2.3.8。
低维 MLP 用 CPU 更快;仅 Atari CNN 需要 GPU(1M 步 ≈ 2–3h)。
图表复现:`hw*/report/build_report.py`(解析 tensorboard 日志,生成分析页与 figs)。

## Roadmap

- [x] HW1 Imitation learning
- [x] HW2 Policy gradients
- [x] HW3 Q-learning & actor-critic
- [ ] 教程站打磨(互动实验室 / 动物园视频墙 / POMDP 记忆体)
- [ ] HW4 Model-based RL

## Attribution

Starter code © UC Berkeley CS285 course staff ([homework_fall2023](https://github.com/berkeleydeeprlcourse/homework_fall2023))。
所有 TODO 实现、实验配置、分析、教程与视频为本人所作(见 NOTICE.md)。Atari ROMs 不随仓库分发(经 AutoROM 安装)。
