# ManipISA RoboDojo 打榜专用目录

这是基于 `optimization-20261005`（`60d6338af4ffa435d6f6915a74764d9c555e607b`）的独立评测实现。所有新增内容都在本目录；仓库原有 `manipisa/`、配置和实验代码保持原样。本目录默认导入同一独立 checkout 中的核心代码，不读取另一份日常 ManipISA 项目。

本实现按官方接口限制观测并保留官方评分，属于**待官方审核的提交候选**。本地检查通过不表示已经获得官方 Verified 资格或完成任务评测。配置与待办见 [submission.json](submission.json)，验证结果见 [validation.json](validation.json)。

## 反馈与执行边界

| 内容 | 本提交的处理 |
| --- | --- |
| 三路 RGB | `cam_head`、`cam_left_wrist`、`cam_right_wrist`；使用官方解码后的像素，不反转颜色，不新增视角 |
| 机器人状态 | 双臂各 6 个关节、7 维末端位姿（xyz+wxyz）、1 维夹爪目标；校验数值、形状和四元数 |
| 末端坐标 | `arm_a/tool_a` 为右 X5，`arm_b/tool_b` 为左 X5；官方环境原点坐标，单位米；末端为 **link6**，不是指尖或抓取中心，不额外加 0.145 m 偏移 |
| 夹爪 | 0 关闭、1 打开；反馈为控制目标，不宣称实际开度、接触、抓持或力控成功 |
| 速度 | 基于官方观测周期的有限差分估计，不是高频实测速度或瞬时安全保证 |
| 任务/频率 | 仅官方语言指令、观测频率和单环境编号 0 |
| 不向 Agent 提供 | 物体真值位姿、摩擦/材质参数、接触力/力矩、评分/成功/阶段标签、case/trial 元数据、深度和标定 |
| 自身执行反馈 | ManipISA 指令状态、目标误差、控制句柄、异常和超时；不把指令 `SUCCEEDED` 当作官方任务成功 |

未知字段在入口丢弃，不遍历其嵌套内容；数值和图像复制到独立数组，并剥离 NumPy dtype 的额外 metadata。输入边界在每一次观测交换时重新执行。

当前只支持有公开证据的 `MOVE` 契约，以及不带成功保证的原始夹爪控制。缺少证据的 `SHAPE_HAND`、`MAKE_CONTACT`、`BREAK_CONTACT`、`CONTROL_GRASP` 和 `APPLY_WRENCH` 不会伪装成验证成功。

这份观测白名单是**本提交声明的配置**，不是声称 RoboDojo 所有赛道都禁止深度或标定。XPolicyLab 接口支持可选字段；改变本提交的输入条件需重新记录配置，并在正式提交时确认官方接受。

### 每次工具调用返回什么

- `start_task` 和 `observe` 返回当前公开状态和三路 RGB。
- `execute_python` 只要实际推进了动作，就自动返回最后一帧三路 RGB、当前状态和执行反馈；图像获取不额外推进物理。
- 程序部分执行后报错，返回已执行的状态、图像、错误和已产生的 stdout，避免误以为动作没有发生。
- 自动反馈保留所有活动指令、变化的终态与当前控制句柄；仅省略已经反馈过且不变的终态。显式查询仍完整。
- stdout 最多预览 4000 字符，明确显示省略量；完整 stdout、工具记录和实际反馈图像留在本地 `results/`。
- 编码图像、整理反馈、写日志也计入 Agent 墙钟预算，超时后不再将回复交给模型。

模型只获得四个 MCP 工具：`start_task`、`observe`、`read_api`、`execute_python`。执行环境提供受限数值运算，阻止 NumPy 文件读写和常见私有属性访问路径；它不是对抗恶意 Python 的 OS 安全沙箱。正式评测仍应将策略进程与官方环境隔离。

最终审计区分 `sent_actions`、`confirmed_observation_intervals` 和没有后续观测的动作。最终动作可能已使环境结束；没有新观测时执行情况标为未知，不将策略计数冒充官方步数。

## 固定模型和版本

本提交固定 `gpt-6-astra` / `high`，所有任务使用同一策略流程，不按任务切换模型。调用现有 WSL 原生 Codex 登录，不读取、复制或上传认证文件，也不改全局 Codex 配置。模型服务本身的可复现性和闭源权重公开要求仍需 RoboDojo 官方确认。

固定版本：

- ManipISA 核心：`60d6338af4ffa435d6f6915a74764d9c555e607b`
- RoboDojo：`266130a3ec41ba9b20b0e5648d2f3542f219c06c`
- XPolicyLab：`bb9a0b5f5136a74503b679af830bfd0a3a837d5c`

启动器会核对 30 个核心文件的哈希、官方仓库提交和已跟踪源码是否改动。哈希清单见 [core-files.sha256.json](core-files.sha256.json)。固定旧版本用于复现，不代表官方必然接受该版本；提交前应按官方指定版本重新验证。

## WSL 本地运行

将本分支克隆到新的文件夹，不要在日常实验项目中覆盖文件：

```bash
git clone --branch robodojo-MainpISA https://github.com/Harbara-Ai/ManipISA.git ManipISA-RoboDojo
cd ManipISA-RoboDojo/robodojo_submission
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

按 [RoboDojo 官方说明](https://github.com/RoboDojo-Benchmark/RoboDojo)在本仓库之外准备固定版本的 RoboDojo、XPolicyLab、仿真环境和资产。策略环境不需要导入 Isaac Sim。

在 WSL Bash 中设置实际路径：

```bash
ROBODOJO_ROOT=/path/to/separate/RoboDojo
.venv/bin/python -B run_wsl.py inspect --robodojo "$ROBODOJO_ROOT"
.venv/bin/python -B run_wsl.py protocol --robodojo "$ROBODOJO_ROOT"
.venv/bin/python -B run_wsl.py agent --robodojo "$ROBODOJO_ROOT"
.venv/bin/python -B run_wsl.py server --robodojo "$ROBODOJO_ROOT" --host 127.0.0.2 --port 19000
```

`inspect` 只读配置；`protocol` 检查官方 WebSocket、RGB、动作和重置；`agent` 使用真实 Codex 调用合成位姿回显环境；`server` 等待官方环境连接。前两种检查不调用模型，前三种都不运行物理仿真、不产生成绩。`agent` 会产生模型调用。

输出统一保存在 `robodojo_submission/results/`，默认每次创建新目录；`--output` 只能选择此目录内的路径。`--python-deps` 可指定额外的 Linux 依赖目录，不要给 Linux 解释器加入 Windows 二进制包。

本机默认连接 `ws://127.0.0.2:19000`；内部 MCP 使用 IPv6 回环。`run_wsl.py` 限制本机回环绑定。跨机器部署使用标准 XPolicyLab 服务入口和适当的网络/隧道配置。

## 注册官方评测客户端

使用另外准备的官方 checkout，注册仅新增的策略包：

```bash
.venv/bin/python -B register_policy.py --xpolicylab "$ROBODOJO_ROOT/XPolicyLab"
```

注册器遇到同名不同内容的文件会整体拒绝覆盖。若旧 checkout 已安装旧版 `ManipISA_RoboDojo`，请使用新的独立 checkout。生成的 `submission-root.json` 只在本机部署目录中保存路径，不应提交；更换机器后重新运行注册器。另一台 Linux 环境客户端也应准备此独立 checkout 并注册，以核对版本；不需要登录 Codex 或启动策略模型。

标准包提供 `setup_eval_policy_server.sh`、`setup_eval_env_client.sh` 和 `eval.sh`，沿用官方的参数布局。单机 Linux 配置好两个环境后，可按官方方式运行：

```bash
EVAL_NUM=native bash "$ROBODOJO_ROOT/XPolicyLab/policy/ManipISA_RoboDojo/eval.sh" \
  RoboDojo stack_bowls codex arx_x5 ee 0 0 0 POLICY_CONDA_ENV SIM_CONDA_ENV
```

完整评测使用官方任务清单、种子、布局、次数和汇总器；本目录不实现自己的任务得分。策略服务器的墙钟预算为 600 秒；该值是本提交配置，不是声称 GPT-6 论文或官网统一规定了这个预算。

## Windows 仿真

`run_windows_sim.py` 使用独立官方 RoboDojo checkout 与已准备好的 Windows Python 3.11/Isaac Sim 环境；策略仍运行在 WSL。下列示例在 Windows 终端中执行，路径替换成实际位置：

```powershell
python.exe -B C:\path\ManipISA-RoboDojo\robodojo_submission\run_windows_sim.py `
  --robodojo-root C:\path\RoboDojo --check-only --check-imports
```

实际启动时去掉 `--check-only`，接受 NVIDIA Omniverse EULA 后传 `--accept-eula`。可通过 `--overlay` 提供隔离的仿真依赖、`--protocol-deps` 提供兼容的 WebSocket 依赖；不会向日常项目环境安装包。

默认仅跑 `stack_bowls` 的 1 个诊断回合；`--full` 只恢复**所选任务**原生次数，不表示全任务评测。包含官方 Linux `fcntl` 监控器的任务在 Windows 上明确阻止，不会通过关闭监控器来继续打分。完整评测建议使用官方支持的 Linux 环境。

Windows/WSL 连接检查可运行 `cross_platform_check.py --help`，显式指定 Windows Python、官方 checkout 和两端依赖目录；其合成结果也不算任务成绩。

## 验证与正式提交

从本仓库根目录运行：

```bash
PYTHONPATH=robodojo_submission:. python3.11 -B -m unittest discover -s robodojo_submission/tests -v
python3.11 -B robodojo_submission/verify_submission.py --robodojo "$ROBODOJO_ROOT"
python3.11 -B robodojo_submission/list_tasks.py --robodojo "$ROBODOJO_ROOT" --format json
```

本地日志、图像、模型调用记录、部署路径和认证信息都不纳入 Git 提交。`validation.json` 只记录清理后的验证结论；物理任务和完整得分未完成前，`official_score` 保持 `null`。

[官方协议（检查版本 2026.10.3）](https://robodojo-benchmark.com/leaderboard/protocol)要求官方评测、三个仿真种子、隐藏布局验证、固定系统以及相应的公开材料与研究说明。上传本分支不是向官方提交成绩，也不代表官方已经认可闭源 Codex 的材料提交方式。

上游启动模板的来源和许可保留在 [NOTICE](policy/ManipISA_RoboDojo/NOTICE) 及同目录的许可证副本中。
