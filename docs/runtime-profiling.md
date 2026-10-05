# 执行耗时诊断

计时用于区分接触读取、状态构造、约束检查、控制、物理推进、反馈处理和日志写入的开销。它不改变物理步长、控制频率、成功判据或预算，也不主动增加观测、物理步或 rollout 内的 CUDA 同步。

`examples/bench2dex_episode.py` 的 `--profile` 默认关闭。启用后，普通回合报告的 `performance` 字段保存聚合计时；关闭时该字段为 null。计时在场景、相机和初始观测初始化完成后安装，因此不用于分析场景加载。开启计时会带来少量诊断开销，比较结果应使用相同开关并保留代码版本。

## 运行无模型的短诊断

`scripts/profile_benchmark_runtime.py` 在现有配对初始化上运行固定短动作，不调用模型、不产生新的正式评分回合，也不恢复暂停的批次。它先推进 10 步预热，再让两个手臂各自向上移动 3 cm，按每次 `step(10)` 经过实际 Python 工具、runtime、物理和评价器路径。

以下 PowerShell 示例假定 `I:/` 映射到本项目根目录，且当前目录为需要测试的代码版本。脚本所在目录决定被测 ManipISA 代码，`--asset-root` 提供 Bench2Dex 场景和资产：

```powershell
& 'I:/.runtime/env/Scripts/python.exe' -B scripts/profile_benchmark_runtime.py `
  --asset-root I:/ `
  --snapshot 'I:/artifacts/benchmark-all26-wsl-development-20261005-v1/initializations/27_ball_box_loading.json' `
  --output 'I:/artifacts/runtime-profile-27-v1' `
  --steps 120 --headless
```

`--snapshot` 必须指向已存在且匹配任务、种子与场景的初始化文件。默认任务为 `27_ball_box_loading.yaml`，默认种子为 `20261032`。`--output` 必须是新目录，脚本拒绝覆盖已有目录。`--steps` 必须是 10 到 180 之间的 10 的倍数；它不包含前面的 10 步预热。诊断使用现有回合预算，不修改正式实验配置。

输出包括：

- `profile.json`：实测时间、步数、环境、源文件 SHA256、初始化哈希、动作报告、接触抽样和分段计时。
- `python-profile.pstats` 与 `python-profile.txt`：`cProfile` 的函数级统计及摘要。
- `tool-feedback.jsonl`：每次 Python 工具执行的完整审计记录。
- 发生异常时的 `error.txt`：诊断失败原因；失败的运行不能充当完成的性能样本。

`profile.json.scope` 明确为 `model_free_execution_diagnostic_not_a_benchmark_score`。两条 MOVE 成功只说明这两个短动作完成，不表示球装盒任务成功，也不能用于推算模型 token 消耗或任务成功率。

## 在正式单回合中打开计时

获准运行新的单回合时，在原有命令中追加 `--profile`，其他参数保持一致。例如：

```powershell
& 'I:/.runtime/env/Scripts/python.exe' -B examples/bench2dex_episode.py `
  --task 27_ball_box_loading.yaml --method manipisa --seed 20261032 `
  --snapshot 'I:/artifacts/benchmark-all26-wsl-development-20261005-v1/initializations/27_ball_box_loading.json' `
  --output 'I:/artifacts/episode-27-profile-v1' `
  --profile --headless
```

这个入口会调用测评模型；它与前面的无模型诊断不同。示例省略 `--wall-limit`，沿用 [实验配置](../configs/ur5_wuji_codex.json) 的墙钟上限；模型、预算与相机准备见 [开发运行说明](../evaluation/docs/ur5-wuji-development-benchmark.md)。启用 `--profile` 本身不更改模型配置或实验预算，也不会自动给整批实验开启计时。DIRECT 可使用相同开关，其报告只有实际调用到的标签，不会出现未执行的 ManipISA 路径耗时。

## 如何读取计时

计时器使用 `time.perf_counter`，单位为秒，只累计每个标签的统计量，不保存逐步计时列表：

| 字段 | 含义 |
| --- | --- |
| `count` | 已退出的调用次数，包括以异常退出的调用 |
| `inclusive_seconds` | 包含嵌套计时区间的总时间 |
| `exclusive_seconds` | 从 inclusive 中扣除同线程内已测量的直接子区间 |
| `max_seconds` | 单次调用的最大 inclusive 时间 |
| `failures` | 异常向外传播的调用次数；不等同于指令 REJECTED/FAILED 数量 |
| `active_spans` | 生成快照时尚未退出的区间数，未完成区间尚未计入聚合 |
| `root_seconds` | 没有被其他计时区间包裹的区间总时间；不等于整个回合墙钟时间 |

主要标签如下：

| 标签 | 测量范围 |
| --- | --- |
| `runtime.step` | 一次 runtime 物理推进及其前后检查 |
| `runtime.prepare` | 准备提交与资源、前置条件检查 |
| `runtime.assess` | 对指令的条件、约束与执行状态评估 |
| `adapter.observe` | 构造关节、位姿、谓词和接触观测；其 exclusive 排除了已测量的接触读取 |
| `contacts.read` | 时间戳改变时读取并解析接触数据 |
| `contacts.cache_hit` | 相同时间戳复用既有接触缓存；不是再次读取 PhysX |
| `adapter.command`、`adapter.hold` | 计算控制目标或维持目标 |
| `adapter.step` | 写入目标、物理推进及对象更新的整体路径 |
| `physics.step` | 仿真器 `sim.step` 的 host 调用时间 |
| `robot.write`、`robot.update` | 机器人数据写入与更新；共享机器人不会重复绑定 |
| `evaluator.read_objects`、`evaluator.read_robot` | 读取评价器使用的物体与机器人状态 |
| `evaluator.update` | 官方评价器更新 |
| `tool.program` | 执行 Python 程序，包含程序中调用的物理与状态操作 |
| `feedback.full`、`feedback.compact` | 构造完整反馈及生成自动回复摘要 |
| `feedback.encode` | 回复规范化和 JSON 编码 |
| `feedback.log` | 完整审计记录的 JSON 编码与文件写入 |
| `feedback.visual` | 动作后自动视觉捕获的整体路径，包含其视觉子区间 |
| `visual.render`、`visual.read` | 相机同步与渲染、三路帧读取 |
| `visual.state`、`visual.encode`、`visual.log` | 同步状态构造、PNG 编码、图像与状态落盘 |
| `tool.total` | 短诊断脚本额外包裹的整次工具调用，不是所有正式回合均有的标签 |

不能把 inclusive 列相加；`feedback.visual` 与其 `visual.*` 子区间也不能重复计算。例如 `contacts.read` 已经包含在 `adapter.observe` 中，后者又包含在部分 `runtime.step` 和 `tool.program` 调用中。exclusive 只扣除已插桩的子区间，仍包含未单独计时的代码和计时器本身的少量开销，不能解释为纯算法计算时间。

这些都是 host 时间。既有 `.cpu()`、标量读取或其他同步可能等待此前提交的 GPU 工作，该等待会计入当前 host 调用；不能据此把整个耗时归属为该函数的 GPU kernel 独占时间。rollout 计时不主动调用 `torch.cuda.synchronize()`。模型推理、网络等待、初始化和未插桩代码也不属于这些分段的完整覆盖范围。

短诊断同时开启 `cProfile`，其开销也包含在 `wall_s` 内。当前脚本在预热后重置分段计时，随后提交两条 MOVE，再开始 `wall_s` 和 `cProfile` 的 rollout 区间，因此分段聚合包含提交开销，而 `wall_s` 不包含；两者不是严格守恒的墙钟分解。

可选 `--reference-contacts <旧版 physx_contacts.py>` 用于在最终冻结物理状态上交替读取参考实现和候选实现，核对完整结果相同并记录耗时。该微基准主动同步 CUDA，运行在 rollout 计时之外，结果单独存于 `same_state_contact_comparison`。它验证的仅是该冻结状态，不能替代真实接触、摩擦、异常和缓冲区边界测试。

## 初始基线：2026-10-05

本次开发工作区的 `diagnostics/baseline-v1/profile.json` 记录了反馈精简后、接触读取优化前的基线：任务 27、种子 `20261032`、60 个接触绑定、预热 10 步、测量 120 步。测量区间耗时 **71.2578 秒**，约 **1.684 步/秒**；两条 MOVE 的最终状态均为 `SUCCEEDED`。

| 项目 | 时间 | 相对 rollout 墙钟时间的参考比例 |
| --- | ---: | ---: |
| `contacts.read`，120 次 | 50.8849 秒 | 71.4% |
| `adapter.observe` exclusive | 10.4233 秒 | 14.6% |
| `physics.step`，120 次 | 5.5299 秒 | 7.8% |

这个样本支持优先优化接触读取，而不是先放宽物理预算。`adapter.observe` 总 inclusive 为 61.3099 秒，其中已经包含接触读取，不能再与第一行相加。表中比例仅帮助判断优先级，受前述提交边界和诊断开销影响。

12 个记录样本的 `present_contacts`、`invalid_contacts`、`invalid_wrenches` 均为零。因此，它主要覆盖没有实际接触的运动路径；每 10 步一次的抽样也不能证明中间每一步都无接触。后续空接触优化在该样本上的提速，不能证明真实抓取、持续摩擦或高接触点数量下获得相同比例的收益。

后续对比应保持相同快照、种子、动作、步数、计时开关和环境，保存新的输出目录与源文件哈希。同时核对动作状态及完整接触结果，并另外验证有接触、无效证据和缓冲区边界；不能通过减少约束检查或丢弃异常记录获得表面的加速。

## 本次优化与验证结果

本次先保留 60 个接触 view、每步完整采样、aggregate 一致性检查和摩擦有效性判断，
将六类原始缓冲区改为先取有效片段再传 CPU；有效范围为空时只创建同形状、同 dtype 的 CPU 空数组。
非空范围仍使用独立副本，避免与 PhysX 复用缓冲区共享内存。
同时新增 `Runtime.step(return_snapshot=False)` 供宿主省去未使用的返回副本；默认 API 仍返回隔离快照，全部控制和检查照常执行。

相同初始快照、种子、动作和 120 步诊断中，总耗时从 71.2578 秒变为 44.4464 秒，观察到约 37.6% 的下降。
接触读取从 50.8849 秒变为 30.8828 秒，`.cpu()` 调用次数从 90,600 降到 47,400。
两次独立进程的 GPU/CPU 状态和暖机程度存在波动，不能将全部墙钟降幅解释为代码优化的净收益。

更直接的比较是在同一冻结物理状态下交替执行新旧接触实现 10 组：两者完整接触结果相同，
耗时中位数为 0.318612 秒和 0.281633 秒，新版降低约 11.6%。该冻结状态没有实际接触。
两个 rollout 的 12 个采样状态逐项比较完全一致，两条 MOVE 均成功；这不代表球装盒任务完成。
非空接触、摩擦异常、共享索引缓冲、容量边界和 CPU 缓冲别名由单元测试覆盖，尚未新增真实抓取试验。

验证材料保存在 `artifacts/benchmark-runtime-improvements-20261005-v1/diagnostics/`，
汇总为 `comparison.json`；原始分段数据、函数级 profile 和完整工具反馈均在原工作区保留，本次不额外上传。上面的初始化路径同样属于原工作区示例，使用者需提供自己的匹配快照并调整路径。

## 计时器验证

项目要求 Python 3.11 及以上。无需仿真器即可执行：

```text
python -B -m unittest discover -s tests -p test_profiling.py -v
```

测试覆盖嵌套 inclusive/exclusive、异常原样传播、禁用时不访问时钟或被测对象、方法与函数属性恢复、防止重复绑定、活动区间禁止 reset、统计副本隔离、存储有界，以及接触缓存分类与机器人去重。诊断结束可以调用 `profiler.restore()` 撤销绑定；无活动区间时 `reset()` 清除统计并保留已安装的绑定。
