# 仓库内容与外部依赖

本仓库包含 ManipISA 源码、六条指令的设计契约、Adapter、示例、测试、实验设计文档，以及选定的本地验证证据。代码的已实现范围和限制以 [README](README.md) 与 [验证记录](docs/manipisa-runtime-validation.md) 为准。

## 评测文档与实现组织

总评分标准、Bench2Dex 对照评测协议和主实验设计统一位于 [`evaluation/docs/`](evaluation/docs)，入口见 [Evaluation README](evaluation/README.md)。顶层 `docs/` 保留指令设计、运行时与 Adapter 文档。后续评测代码和冻结配置也放在 `evaluation/`，按职责建立子目录；当前仅有评测文档。

## 只运行单元与契约测试

使用 Python 3.11 或更高版本。以下命令无需启动 Isaac Sim：

```powershell
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -e .
& .venv/Scripts/python.exe -m unittest discover -s tests -v
```

`.venv` 是上述测试环境；现有 Windows 仿真脚本使用的是另行配置的 `.runtime/env`，二者不能直接替代。

## 仿真与原始回放的外部依赖

大型 Python/Isaac Sim 环境、第三方源码、机器人资产与示范数据不随本仓库上传。原工作区验证采用以下版本，详细环境记录见 [reproduction/status.json](reproduction/status.json)：

| 依赖 | 原工作区位置 | 已验证版本 |
| --- | --- | --- |
| [Bench2Dex](https://github.com/Bench2Dex/Bench2Dex) | `Bench2Dex/` | `fd90dcc625b66c2e535a8b25bcf75bc81c7b080d` |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | `IsaacLab/` | v2.3.2，`37ddf626871758333d6ed89cf64ad702aef127d0` |
| Isaac Sim | `.runtime/env/` | 5.1.0 |
| [Bench2Dex Assets](https://huggingface.co/datasets/Bench2Dex/Assets) | `dex2bench_dataset/` | `bf65215f844d1fca30750e4cc70d7266650abf0a` |
| Bench2Dex 官方示范 | `teleopdata/` | `b195787c65046083e6a43776b67bdf1389dfa3eb` |

在项目根目录恢复第三方源码的示例：

```powershell
git clone https://github.com/Bench2Dex/Bench2Dex.git Bench2Dex
git -C Bench2Dex checkout fd90dcc625b66c2e535a8b25bcf75bc81c7b080d
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab
git -C IsaacLab checkout 37ddf626871758333d6ed89cf64ad702aef127d0
```

随后按对应版本的上游说明安装仿真依赖，并准备场景所需资产和数据。`reproduction/` 保留原工作区的下载、检查和回放脚本；其中 `complete_install.ps1` 依赖已有环境、安装日志和本地 wheel，不能作为全新机器的一键安装入口。Windows 环境约束和已知依赖冲突见 [复现记录](reproduction/README.txt)。

原始回放记录保留了官方 RGB/触觉验证结果；ManipISA Runtime 当前不接入触觉图像。两者的验证范围不同。

## 验证证据

`artifacts/` 中保留文档引用的报告、配对比较、测试日志，以及对应最终运行的轨迹和少量 PNG。`reproduction/runs/` 仅保留已通过回放的命令、退出码、验证报告与成功复核日志；原始 HDF5、回放 HDF5、视频和批量图像需要本地生成。

这些是原工作区的历史验证记录，可能包含当时的本地绝对路径，不能据此认定克隆后的机器已经通过仿真验收。新增运行产物由 `.gitignore` 排除，已选入版本控制的历史证据保留。

部分设计文档中的第三方源码链接按原工作区目录保留，需要上述外部 checkout；`../../wuji/` 指向另一个历史参考项目，不是 ManipISA 单元测试所需依赖。
