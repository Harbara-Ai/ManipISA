# 实验工具的精简反馈

本文记录 `execute_python` 的 `compact-visual-v2` 自动回复约定。摘要由宿主生成，模型仍通过原有公开 API 控制仿真。

精简范围是工具自动附加的运行状态和标准输出。物理控制、步数、时间预算、官方评价器与计量规则不因此改变。压缩效果需要用实际回复和后续实验测量，不预设 token 降幅。


## 动作后的视觉反馈

执行一段 Python 后，只要实际物理步数增加且回合尚未终止，宿主自动捕获一次当前 RGB。
回复是一个精简 JSON 文本块加三个真正的 MCP image 块，顺序为 `camera_ids`（俯视、左腕、右腕）。
当前开发配置使用 640×480 的 pinhole 透视相机，腕相机不使用鱼眼投影；配置与外部 Bench2Dex 补丁见 [开发运行说明](../evaluation/docs/ur5-wuji-development-benchmark.md)。
原生 Codex 直接调用实验 MCP 时，图像块随工具结果交付。若通过 `functions.exec` 转接，文本块用 `text(c.text)`，图像块用 `image(c)` 转交模型；
对整个工具结果调用 `text(result)` 会把图片变成 Base64 文本，不能替代图像输入。

`visual_feedback` 提供 `status`、`observation_id`、`physics_step`、`camera_ids` 和同一步的精简 `state`。
`state.robot` 的 `qpos/qvel` 使用初始观测的 `robot.joint_names` 顺序；对象状态保留当前位置、姿态和速度。
每相机记录实际内外参、投影类型、图像路径和 PNG SHA256，路径用于本地审计，不授予模型文件访问权限。
自动反馈不重复附上所有 body、固定关节限位、物体几何或完整历史；显式 `observe` 仍返回完整公开状态。

纯查询、仅提交控制目标而未 step、以及推进前报错，不渲染，返回 `status: unchanged`。
普通程序错误发生在部分移动后仍拍摄现场，保留原异常；相机故障单独记录为 `status: error`，
不给旧图或半套图，回复标记 `isError`。终止时不拍摄新的策略图像、不发送新策略回复。
渲染、编码、落盘和反馈日志都计入墙钟预算，跨过截止时间即丢弃候选回复。

一次工具调用无论含多少次 `step()`，只在返回时附一组图。接近、合拢、试提和搬运应拆为
多个有界调用；本实现不强行截断 `step(n)`，不改变物理步数或指令语义。
成功捕获的初图、显式观察和自动反馈都计入 `queries`；结果的 `rgb_observations` 单独记录自动次数。
未完整捕获不计入成功观测；已捕获但因后续超时被扣留的图像仍有本地记录，以审计的 `reply_withheld` 为准。

## 模型自动收到的内容

每次 `execute_python` 回复包含 `feedback_format: "compact-visual-v2"`、当前 `physics_step` 和标准输出预览。ManipISA 同时附加 `runtime`，其 `schema_version` 为 `manipisa.feedback.summary.v1`；Direct 不附加 runtime 摘要。

- 第一版保留当前 `state` 中除接触压缩外的全部字段，包括 `joints`、`frames`、`predicates`，以及反馈的状态有效性与错误信息。
- 接触状态仅压缩能够确认有效、无接触且力与力矩为零的普通记录。按共同元数据分组保留接触标识；异常、未知、非零、存在接触点或有效性不足的记录完整保留。省略的接触项不能被解释为“没有接触”。
- 保留所有执行中的指令，包括 `RUNNING` 和 `ACTIVE`。终态指令只在新增或完整报告发生变化时自动返回；未变化的旧终态报告不重复发送。
- 保留全部当前控制句柄及其资源、所有者、状态和错误信息。`FAULTED` 或 `EXPIRED` 句柄可能仍占用资源，不能按名称或状态将其删除。
- 状态中的时间与有效性证据仍然保留，未知值不能改写成零或成功。

标准输出预览 `stdout` 最多 4000 个字符，超限时保留头部和尾部，并插入明确的省略标记。`stdout_total_chars` 给出原始长度，`stdout_omitted_chars` 给出未显示的原始字符数。此限制仅针对自动回复中的 stdout，不是整份结构化摘要的长度上限。

## 摘要结构

宿主获取用于审计日志的完整反馈快照，并复用该快照调用纯函数 `summarize_feedback(..., previous_reports=...)`，避免再次读取、序列化同一运行状态。宿主管理本回合已经返回的报告基线，只有通过日志写入和截止时间检查后才推进基线。`Runtime.feedback_summary(previous_reports=...)` 也可供宿主调用，但未新增到模型的控制 namespace。

下表中的字段均位于自动回复的 `runtime` 内。压缩表示只用于自动回复中的 `runtime.state`；公开 `state()` 和 `runtime.feedback()` 返回的原有完整结构保持不变。

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 固定为 `manipisa.feedback.summary.v1` |
| `state.contacts` | 未压缩的完整接触记录 |
| `state.empty_contact_groups` | 无接触记录分组，每组包含 `ids` 和完整相同的 `record` 模板 |
| `state.contact_encoding` | `explicit_plus_identical_empty_groups` |
| `state.contacts_complete` | 为 true 时，接触字典与分组合并后可无损还原全部接触记录 |
| `instructions` | 全部非终态指令，加上新增或变化的终态报告 |
| `instructions_scope` | `active_and_changed_terminal`，不能将缺席的旧终态指令视作不存在 |
| `active_instructions_complete` | 为 true 时，执行中的指令完整 |
| `omitted_unchanged_terminal_count` | 本次省略的未变化终态报告数量 |
| `control_handles` | 全部当前控制句柄及其资源 |
| `control_handles_complete` | 为 true 时，句柄列表完整 |

接触分组必须使用完整相同的记录模板，包括时间戳、有效性和来源；只凭“力为零”分组不能保证无损。返回的接触标识必须在未压缩字典和分组中唯一对应。

## 按需读取完整反馈

公开 API `runtime.feedback()` 和 `runtime.query(call_id)` 继续提供完整当前结果。精简器不能修改运行时保存的报告或这些 API 的返回值。模型可按需检查具体指令、句柄或接触记录，避免每一步打印全部历史。

显式打印完整结果仍经过同一 stdout 预览限制；需要查看大结果时，应选择所需字段或分段打印。自动摘要的终态去重范围限于本回合，不能跨回合复用。

例如，`contact_id` 使用已观察到的接触标识，`call_id` 使用此前提交返回的调用标识：

```python
print(runtime.feedback()["state"]["contacts"][contact_id])
print(runtime.query(call_id).to_dict())
```

## 本地审计记录

每次 Python 执行在回合目录的 `tool-feedback.jsonl` 留存完整反馈快照。记录包含 `feedback_format`、`tool_call`、`physics_step_before`、`physics_step`、原始 `stdout`、`stdout_omitted_chars`、`error`、完整 `runtime`、`reply_withheld` 和候选回复长度 `reply_chars`。Direct 的日志 `runtime` 为 null。ManipISA 每次执行均保存完整 runtime 反馈，即使自动回复只含增量终态记录。

如果日志写入本身跨过截止时间，会额外追加同一 `tool_call` 的 `event: "reply_withheld_after_logging"` 更正记录及终止错误。审计时必须结合此事件解释前一条记录，不能将前一条 `reply_withheld: false` 单独视作已发送回复的证明。

原始日志供离线检查使用，不是模型新增的文件访问通道。第一版不对完整日志做差分优化；未来如改变存储格式，应单独标明版本并保持可还原性。

`visual_feedback` 和 `image_count` 记录该次自动捕获的元数据和图片数量；日志不重复保存图片 Base64。
`reply_chars` 仅统计精简 JSON 文本长度，不包含图像输入的字节数或 token 成本。

## 异常与终止

普通 Python 异常的回复也包含已产生的 stdout、错误类型与信息、当前物理步数，以及可获得的当前运行摘要。执行后报错不代表物理动作被回滚。

`EpisodeStopped` 保持专门的宿主终止路径：保存终止时已有的审计信息，继续向外传播，不向模型发送新的工具回复。摘要或日志处理不能推进物理时钟，也不能把失败状态改成成功。

## 验证边界

离线测试应验证异常/未知接触不丢失、执行中指令与全部占用句柄保留、终态变化可见、完整 API 不受摘要生成影响、stdout 头尾与省略标记、普通异常后的步数准确，以及终止时不发送模型回复。用固定输入比较完整反馈与摘要的字节数只能说明这些输入的输出体积变化，不能替代真实模型成本测量。

现有测试使用标准库 `unittest`；项目要求 Python 3.11 及以上。针对性测试分别为 `test_feedback.py` 与 `test_tool_feedback.py`，例如 `python -B -m unittest discover -s tests -p 'test_tool_feedback.py' -v`。既有执行器回归可用 `python -B -m unittest discover -s tests -p 'test_evaluation.py' -v`。涉及目录清单的测试需要完整项目的 `Bench2Dex` 资产与配置。
