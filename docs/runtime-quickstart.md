# Runtime 调用与显式交接

`submit()` 返回 `ExecutionReport`。`query()`、`cancel()`、`update()` 接收报告的
`call_id` 字符串；`submit(..., replace_handle=...)` 接收终态报告的
`control_handle` 字符串。这两个 ID 用途不同，不能相互替代，也不能直接传整个报告。
如果保存的是 `report.to_dict()`，用 `saved["call_id"]` 或 `saved["control_handle"]`。

| 值 | 用途 |
| --- | --- |
| `report` | 检查 `status`、`reason`、`detail`、`goal` 和 `evidence` |
| `report.call_id` | 查询、显式取消或更新这条指令 |
| `terminal_report.control_handle` | 接替这条指令结束后仍保留的资源 |

以下片段使用预加载的 `runtime`、`step`、`Instruction`、`Opcode`、`PoseGoal`、
`ShapeGoal` 和 `Status`。调用者提供已核验的 `arm_actor`、`hand_actor`、`tool_frame`、
`target_xyz`、`target_quat`、`hand_target`、`next_target_xyz` 和 `step_budget`。
UR5/Wuji 绑定例如 `arm_a`、`hand_a`、`tool_a`；`hand_target` 必须匹配绑定的关节数及物理限位。
位置使用 world 米制坐标，四元数为 wxyz。`step_budget` 是该段允许推进的物理步数。
这些是调用形式示例，坐标和手型需要由当前场景决定。

将接近、合拢、试提拆成独立 `execute_python` 调用。每段推进物理后会自动返回三路新 RGB；
先查看图片及同一步的状态，再决定下一段。一个长循环中的每次 `step()` 不会单独把图片送回模型。

## MOVE

```python
move_report = runtime.submit(Instruction(
    Opcode.MOVE, (arm_actor,), PoseGoal(tool_frame, tuple(target_xyz), tuple(target_quat)),
    max_joint_speed=1.0,
))
print(move_report.status, move_report.reason, move_report.detail)
move_result = move_report
if move_report.status != Status.REJECTED:
    for tick in range(step_budget):
        move_result = runtime.query(move_report.call_id)
        if move_result.status not in (Status.RUNNING, Status.ACTIVE):
            break
        step(1)
    move_result = runtime.query(move_report.call_id)
    print(move_result.status, move_result.reason, move_result.goal, move_result.control_handle)
```

`submit()` 和 `query()` 都不会推进物理时间。步数用尽后指令可能仍是 `RUNNING`；
应先检查状态，再决定继续推进还是显式取消。

## SHAPE_HAND

```python
shape_report = runtime.submit(Instruction(
    Opcode.SHAPE_HAND, (hand_actor,), ShapeGoal(tuple(hand_target)), max_joint_speed=1.0,
))
print(shape_report.status, shape_report.reason, shape_report.detail)
shape_result = shape_report
if shape_report.status != Status.REJECTED:
    for tick in range(step_budget):
        shape_result = runtime.query(shape_report.call_id)
        if shape_result.status not in (Status.RUNNING, Status.ACTIVE):
            break
        step(1)
    shape_result = runtime.query(shape_report.call_id)
    print(shape_result.status, shape_result.reason, shape_result.goal, shape_result.control_handle)
```

`SHAPE_HAND` 使用 REACH 模式；手型到达或位置保持不代表已获得抓取证书。

## 在确认 MOVE 成功后交接同一资源

```python
next_report = None
if move_result.status == Status.SUCCEEDED:
    next_report = runtime.submit(Instruction(
        Opcode.MOVE, (arm_actor,), PoseGoal(tool_frame, tuple(next_target_xyz), tuple(target_quat)),
        max_joint_speed=1.0,
    ), replace_handle=move_result.control_handle)
    print(next_report.status, next_report.reason, next_report.detail)
    if next_report.status != Status.REJECTED:
        for tick in range(step_budget):
            next_result = runtime.query(next_report.call_id)
            if next_result.status not in (Status.RUNNING, Status.ACTIVE):
                break
            step(1)
        next_result = runtime.query(next_report.call_id)
        print(next_result.status, next_result.reason, next_result.control_handle)
```

只有成功接受交接时旧句柄才会消失。失败或取消也可能产生保留句柄；
`EXPIRED`、`FAULTED` 句柄仍可能占用资源，应检查当前 `control_handles`。
交接要求资源集合和耦合域完全一致，不能用手的句柄接替臂，也不能把拒绝指令当成资源所有者。
上述示例不会自动取消任何指令。

若明确决定提前停止，调用 `canceled = runtime.cancel(report.call_id)`，检查返回的状态和
`canceled.control_handle` 后再决定是否接替；取消前检查到的约束违规仍优先记录为失败。
若只修改运行中指令的目标，调用 `runtime.update(report.call_id, revised_instruction)` 并检查
`accepted`、`reason`、`detail`。更新不会重置原超时预算，也不会隐式修改速度上限或其他义务。

## 处理拒绝

- `UNREACHABLE` 且 detail 指向 requested goal：提交目标被 adapter 拒绝，例如目标手型超出物理限位。
- `PRECONDITION_FAILED` 且 `evidence["joint_limits:..."]` 为 `VIOLATED`：当前观测状态已经越界；把目标改回范围内不能自动修复这个前置条件。
- `UNKNOWN` 或 `EVIDENCE_UNAVAILABLE`：查看完整 `truth`、`timestamp`、`source`、`detail`，确认缺失或陈旧证据。未知不等于安全。
- `RESOURCE_CONFLICT`：检查当前所有者及句柄；修正 ID 或显式制定交接，不能依靠自动取消覆盖原控制。

先记录拒绝的 `reason`、`detail`、`goal` 和各项 `evidence`，再决定恢复方式。
这些诊断不会自动放宽限位、速度或接触条件，也不会移动机器人。
文档片段由 `tests/test_runtime_diagnostics.py` 使用确定性 `TestAdapter` 执行，属于 API 契约测试，
不代表真实仿真或物理控制验收。
