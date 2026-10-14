# HW3 · Q-Learning & Actor-Critic:偏差控制(第 3 讲)

> 核心问题:HW2 的估计"对但吵",HW3 的 TD 估计"安静但有方向地错"。什么会坏,什么能修?
> 完整互动章节(官方地图/卡壳点/概念/证据/翻车博物馆/代码解剖/自测)见 `docs/chapters/hw3-q-learning.html`。

## DQN(离散动作)

| 实验 | 结果 | 说明 |
|---|---|---|
| CartPole-v1 | **452**;Q̂ → ≈99 = 解析值 Σγᵗ | 实现正确的解析级证据 |
| CartPole lr=0.05 | Q̂ 胀到 ~120、TD 误差大 1–2 个数量级、eval ~60 | **死亡三合一**现场(FA+bootstrap+off-policy) |
| lr 四档扫描(§2.6) | 仅 1e-3 守住 500;5e-3/1e-2 摸到 500 后塌回 ~100 | "摸到 ≠ 守住";反馈环增益连续可调 |
| LunarLander 3v3 seeds | vanilla 全负;**doubleQ 89/130/59,单局 211–277** | vanilla 的 Q̂ 更高而回报更低——高估活捉 |
| MsPacman(像素,1M 步) | **1,776**(基准 ~1500),GPU 2h05m | CNN 端到端;train/eval 差距=行为策略 vs 目标策略 |

## SAC(连续动作)

| 实验 | 结果 | 说明 |
|---|---|---|
| Pendulum / InvPendulum sanity | Q̂ 稳定在大负数;两种估计器 ~1000(826/920) | 官方三道自检全过 |
| HalfCheetah R-1/R-10/reparam | **97 / 1,461 / 4,316** | 同一目标 ∇θE[Q],三种方差命运 |
| Hopper single/double/clipped | **350 / 440 / 544**,clipped-Q 的 Q̂ 最低 | 保守靶子没有拖累学习,反而成就学习 |
| Humanoid clipped-Q | **1,355** @ 595k 步 | off-policy 样本效率(HW2 on-policy 对照) |

概念小实验:`report/concept_demos.py`(max-of-noise +1.54、两状态死亡三合一反例 ×1.08/步、偏差复利)。

## 实现说明

- TODO 实现:`cs285/agents/{dqn_agent,soft_actor_critic}.py`、`cs285/scripts/run_hw3_*.py`;
  烟雾测试 `smoke_test.py`(2 梯度 × 3 backup × 软/硬 target 共 12 组)。
- **done 一律剔除 TimeLimit 截断**(truncation 不阻断 bootstrap)。
- 训练结束自动保存 `agent_final.pt`;`cs285/scripts/eval_to_gif.py` 录制行为视频(mp4)。
- starter 已知怪癖(未修,不影响结果):`sac_config.py` 两个 optimizer 工厂互换(同为 Adam 3e-4);
  ale-py 不报 `max_episode_steps`(已加 None→27000 兜底)。

## 复现

```bash
pip install -e . --no-deps
pip install ale-py==0.8.1 "autorom[accept-rom-license]"
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 1
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/halfcheetah_reparametrize.yaml
python report/build_report.py   # 图表复现(解析 data/ 下 tensorboard 日志)
```
