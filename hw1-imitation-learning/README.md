# HW1 · Imitation Learning:分布偏移(第 1 讲)

> 在"RL 的四种失败"世界观里,本讲是 **失败 I · 分布偏移**。

> 核心问题:照抄专家(Behavior Cloning)为什么有时会翻车?翻车后,补救的是模型还是数据?

**官方地图**:CS285 Lec 2(Imitation Learning)+ HW1 作业 PDF。先读官方,本页是证据与坑。

## 核心数据

| 策略 | Ant-v4 | Hopper-v4 |
|---|---|---|
| BC(仅专家数据) | **97.9%** of expert | **≈30%** of expert |
| DAgger(10 轮迭代) | ≈ expert | **≈ expert(追平)** |

![BC 数据量扫描](report/figs/figure1_bc_sweep.png)

## 解读:同一个算法,两种命运

- **Ant 为什么行**:四足形态自带稳定,轨迹很难离开专家的状态分布——训练分布≈测试分布,BC 足够。
- **Hopper 为什么崩**:单足平衡,一步踏错就进入专家从未见过的区域;在那里策略只能瞎猜,
  猜错一步就更偏——**复合误差 O(T²ε)**,这不是模型容量问题,是**分布偏移(covariate shift)**。
- **DAgger 修的不是模型,是数据分布**:每轮让学习者自己跑,把"学习者踩出来的状态"交给专家重新打标签,
  再并入训练集。训练分布与策略的占有测度对齐后,复合误差从 O(T²ε) 降回 O(Tε)。

![DAgger Ant](report/figs/figure2_dagger_ant.png)
![DAgger Hopper](report/figs/figure2_dagger_hopper.png)

## 行为视频

`docs/assets/videos/`(教程站动物园同版):`demo_dagger_ant.mp4`、`demo_dagger_hopper.mp4`。

## 概念图(训练分布/占有测度的可视化)

`report/figs/concept*.png`:初始分布、轨迹漂移、占有测度失配的可视化推导。

## 复现

```bash
pip install -e . --no-deps
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl \
  --env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
  --expert_data cs285/expert_data/expert_data_Ant-v4.pkl
# DAgger:--do_dagger --n_iter 10
```

## 自测

1. 为什么"加更多专家数据"救不了 Hopper 的 BC?(数据仍来自专家的占有测度,偏移本身没变)
2. DAgger 的迭代里,谁在产生状态、谁在产生标签?(学习者产生状态,专家给标签——这正是分布对齐的机制)
3. 复合误差从 O(T²ε) 到 O(Tε) 的关键假设变化是什么?(训练分布从"专家的"换成"当前策略诱导的")
