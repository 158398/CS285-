# NOTICE

## 课程代码归属

本仓库包含 UC Berkeley CS285 (Fall 2023) 课程的 starter code,原始来源:

- https://github.com/berkeleydeeprlcourse/homework_fall2023 (hw1 / hw2 / hw3)

课程代码版权归 UC Berkeley CS285 course staff 所有,在此致谢。
本仓库中,starter code 里标注 `TODO(student)` 之处的**实现**、全部实验配置(experiments/)、
分析报告(report/)、教程站(docs/)、视频与图文素材,均为仓库作者本人所作。

## 第三方运行时

- Atari ROMs 不随本仓库分发;请通过 `pip install ale-py==0.8.1 "autorom[accept-rom-license]"` 自行安装(需接受其许可)。
- 环境固定:gym 0.25.2 / mujoco 2.3.3 / numpy 1.26.4 / torch 2.7.0。注意 opencv-python 请装 4.x(5.x 会拉入 numpy 2.x,与 gym 0.25.2 不兼容)。
