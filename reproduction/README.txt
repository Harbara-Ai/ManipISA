Bench2Dex UR5 + Wuji 原始环境复现

验收状态：单任务原始回放通过。
任务：27_ball_box_loading；官方 episode_000000.hdf5。
916 帧，52/52 关节映射，6 路 RGB，各 916 帧；10 路 TacMap。
原始关节和物体位姿数据保持一致，官方成功判据重新计算通过。

运行：在 ManipISA 目录执行 .\reproduction\run_replay.ps1
显示交互窗口：.\reproduction\run_replay.ps1 -Gui
脚本自动映射 I: 到项目目录以避免 Windows 路径长度限制。
每次运行生成独立输出目录，不覆盖官方示范。
运行完成后可执行：
I:\.runtime\env\Scripts\python.exe reproduction\validate_replay.py <输出的 HDF5 路径>

本次输出：reproduction\runs\ball-box-20261003-140245
preview.mp4：胸前＋俯视双视角视频。
validation.json：相机、触觉、轨迹完整性检查。
success-check.log：官方成功判据只读复核。
status.json：环境版本、提交、数据版本、修复和验收范围。
assets-manifest.json：下载资产及上游校验信息。
environment-freeze.txt：实际 Python 包版本，Isaac Lab 使用本地固定提交。

安装适配：Python 3.11.17，Isaac Sim 5.1.0，Isaac Lab v2.3.2，
PyTorch 2.7.0+cu128，NumPy 1.26.4，h5py 3.15.1。
h5py 3.16 在 Windows 上会与 Isaac Sim 的 HDF5 DLL 冲突；
处理依据：https://github.com/isaac-sim/IsaacLab/issues/5076
flatdict 使用 setuptools 80.9.0 构建；wheel 固定为 0.43.0。

已知限制：pip-check.txt 保留官方依赖组合的两项冲突：
Isaac Sim 声明 numpy==1.26.0，但 Bench2Dex README 要求 1.26.4；
Isaac Sim 的 FastAPI 0.115.7 与 Isaac Lab 要求的 Starlette 0.49.1 约束冲突。
本次原始回放实际通过；其他服务和训练流程尚未验收。

数据注意事项：当前上游 collector/data_collector.py 将 collection_mode
固定写成 scripted_hold，demo_eligible 固定为 false。不能仅凭这两个字段
断定采集来源或筛选训练数据。本次保留官方文件不变。

验收范围：一个原始任务的运动学回放、RGB/触觉数据生成与成功标签复核。
尚未复现学习策略的闭环控制成功率，也未运行全部 26 个任务。
Bench2Dex 和 Isaac Lab 上游源码均未改动。
