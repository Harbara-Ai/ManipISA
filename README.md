# ManipISA Runtime

ManipISA 的可运行执行层：Agent → Instruction → Runtime → Adapter → Controller。结构化反馈与 RGB 回传共用同一套 Runtime；当前不接入 TacMap 或触觉图像，不设置独立 Safety 或 Capability 模块。接触与交互力使用 PhysX 结构化反馈。

六个操作码均已有执行路径；完整语义仍以 [v0.2 指令契约](docs/manipisa-v0.2-core-contracts.md) 为设计目标。支持的参数组合和验证范围见下表及 [接触执行说明](docs/manipisa-interaction-runtime.md)。

| 内容 | 当前范围 |
| --- | --- |
| SHAPE_HAND | 单个手角色的关节坐标构型，REACH |
| MOVE | 单个固定基座臂的 ACTOR_FRAME，世界系绝对位姿，REACH / SUSTAIN |
| 并发 | 空闲空间内不重叠的关节组；重叠资源或已声明的耦合关系拒绝并发 |
| MAKE_CONTACT | 有界末端/构型接近，全部指定物理接触与力阈值成立后完成 |
| BREAK_CONTACT | 有界分离，接触消失且几何间隙成立后完成 |
| CONTROL_GRASP | 已有接触内调整；检查相对运动和有限准静态负载工况，REACH / SUSTAIN，返回带适用范围的 GraspRelation |
| APPLY_WRENCH | 单个臂角色的有界位置导纳，按所选方向闭环控制；六维反馈来自法向/摩擦作用点，SUSTAIN |
| 接触并发 | 单条可声明多个参与者；独立调用按关节、实际接触对和对象刚体互斥；尚无共享对象的并发协调控制 |
| 对象运动 | MOVE(ENTITY_FRAME) 仍明确拒绝；六个操作码可运行不等于全部参数组合完成 |
| 结构化反馈 | 关节/位姿，以及接触对、法向力、完整交互合力/力矩、接触点、几何间隙；时间、有效性、指令状态、误差、证据和控制句柄 |
| 对象观测 | 可注入只读实体刚体状态；当前没有据此执行对象运动 |
| RGB | 可选接入 Bench2Dex CameraRig；时间、相机 ID、标定、有效性和 RGB 数组分开返回 |

首次克隆的环境准备与外部依赖见 [仓库说明](REPOSITORY.md)。

## 运行

在项目根目录使用已有环境运行契约测试，无须启动 Isaac Sim：

```powershell
& .runtime/env/Scripts/python.exe -m unittest discover -s tests -v
```

Windows 仿真验证使用本地 UR5＋Wuji 资产、物理步进和 Isaac Lab DifferentialIKController，不读取示范动作：

```powershell
# 结构化反馈
.\scripts\run_runtime_smoke.ps1

# 同一执行流程，附加 RGB，并验证运行时开启和关闭
.\scripts\run_runtime_smoke.ps1 -Rgb

# 专用双 UR5＋双 Wuji Adapter：非零手掌 TCP 与 12 个原生接触绑定
.\scripts\run_runtime_smoke.ps1 -AdapterChecks

# 四条接触指令的真实物理验收场景（双指夹持与施力探头）
.\scripts\run_runtime_smoke.ps1 -Interactions
```

脚本使用已有的 I: 短路径约定以规避 Windows 路径长度问题；I: 被其他目录占用时明确失败。每次运行生成独立 artifacts 目录，包含 report.json、trace.jsonl，以及启用 RGB 时的两张图像。报告记录源文件哈希。可添加 -Gui 显示窗口；默认 headless。

也可在已安装 Isaac Lab 的 Python 环境中运行 examples/isaaclab_smoke.py --headless，或附加 --rgb。这个示例是无操作物体的空闲空间控制验证，并非 Bench2Dex 正式任务评测。相机数量与分辨率由场景配置决定，示例只使用两路 320×240 RGB，不声称已验证六路相机组合。

## 接入已有仿真循环

场景创建并初始化后，用 Adapter 将通用角色映射到 Articulation、关节和目标帧。双 UR5＋双 Wuji 已有专用的 `DualUr5WujiAdapter`，集中处理 52 个关节、腕部到手掌变换、TCP 配置与指尖/手掌接触绑定，复用 `IsaacLabAdapter` 的执行逻辑和 DLS IK。其他机器人可继续使用 ActorBinding 接入通用 Adapter。导入 manipisa 或专用 Adapter 均不会启动仿真。

```python
from manipisa.adapters import DualUr5WujiAdapter

# 在 sim.reset() 和初始状态应用之后创建。
adapter = DualUr5WujiAdapter(
    sim, robot,
    pre_step_hooks=spawned["pre_step_hooks"],
)
```

默认 arm_a / hand_a 对应右侧，arm_b / hand_b 对应左侧；tool_a / tool_b 保留原来的腕部帧语义，新增 palm_a / palm_b 观测。切换手掌 TCP、改名和接触配置见 [专用 Adapter 接入说明](docs/dual-ur5-wuji-adapter.md)。

```python
from manipisa import Runtime, Instruction, Opcode, PoseGoal

runtime = Runtime(adapter, rgb=optional_rgb_channel)
request = Instruction(
    operation=Opcode.MOVE,
    actors=("arm_a",),
    goal=PoseGoal("tool_a", (0.4, 0.1, 1.2), (1.0, 0.0, 0.0, 0.0)),
)
report = runtime.submit(request)

# 由宿主持续调用，不必等待 Agent 推理完成。
runtime.step()
feedback = runtime.feedback()   # 可 JSON 序列化的状态、指令结果与图像元数据
images = runtime.images()       # 相机 ID → RGBFrame，包含 RGB 数组和采集时间
```

上面的目标只是接口示例，使用时由任务和绑定决定。一个 Runtime 对应一个仿真时钟；宿主调用 runtime.step() 时，不能再调用另一套 sim.step()。本版不是后台服务，也不提供线程安全的多客户端提交；Agent 接入方应让宿主推进循环，并在明确的控制边界串行处理请求。

已支持 submit、query、cancel、目标更新和保留控制句柄的显式接管。update 目前只支持 MOVE / SHAPE_HAND 的目标更新；不重置总时限，ACTIVE 更新后仍需满足目标。四条接触指令通过新调用和显式接管更新，避免重置接近边界或静默更换接触集合。没有队列调度、一般约束更新和任意资源集合之间的接管。

## 结构化反馈与 RGB 开关

RGB 通道使用 Bench2Dex 的相机同步 → render → capture 顺序。建议先配置传感器和渲染环境，再通过 runtime.set_rgb_enabled(True/False) 控制回传。

- 关闭时不调用通道的捕获或渲染，不返回旧图像；宿主自己的可视化渲染由宿主管理。
- 开启时按仿真时间采样，默认间隔 0.10 s；重复读取不触发额外采样。
- 缺帧、格式错误和过期帧显式标为无效，不用黑图伪装有效观测。
- RGB 失败不改变结构化反馈轨道的指令结果。当前图像用于 Agent 观察，没有视觉位姿估计或视觉伺服。
- 像素通过 images() 返回，避免把大数组混入普通 JSON 日志；metadata 和像素中的 timestamp/camera_id 对应。

RGB 关闭与开启都使用相同的实际关节/位姿反馈，当前 state_source 明确标为 simulation_privileged。报告中的 SUCCEEDED 属于指令契约判断，不是 Bench2Dex 的任务成功标签。评价器成功位、阶段分数和未来示范帧没有接入 Agent 观测。

## 执行语义与限制

成功依据实际位姿/构型误差、速度和连续满足时间。命令下发、IK 返回及到达某个中间阶段均不直接算成功。SUSTAIN 先 RUNNING，激活后 ACTIVE，在规定维持时间结束后才能 SUCCEEDED。

当前证据失效策略为立即失败；没有自动用旧观测继续控制的宽限期。失败为终态，后续状态恢复不会改写历史失败。目标获取时限、总时限、具名进入条件和阶段不变量独立处理。

Runtime 检查实际关节速度；Adapter 限制命令增量、关节范围及接触动作相对初始状态的范围。位置驱动可能发生瞬态越限，检测到的违反按失败处理。接触执行还检查力上限、受保护接触、禁止接触和抓持漂移。未绑定的接触区域不会自动被检测，场景必须声明所需接触对；当前没有全场景碰撞规划或闭链协调求解。额外任务条件仍可由可信 predicate_provider 提供。

抓持负载证据采用实际预紧力、校准的保守摩擦系数和接触点，验证明确列出的准静态外载工况。该模型假设接触间可在观测预紧预算内重分配载荷，不能作为动态鲁棒性或任意六维力闭合的证明。施力控制以位置导纳接入现有 PD 执行器；物理验收覆盖法向施力，任意机械臂的独立六轴力矩控制仍需专门验证。

完成、失败、取消后采用显式 HOLD_POSITION，返回仍占用资源的句柄。空闲空间指令锁存实际关节位置；接触指令保留最后的有界位置目标，避免接管时清掉已有预紧。此后不继续更新导纳或认证旧抓持/力目标。调用方通过 submit(..., replace_handle=handle) 请求同一资源集合和耦合域接管；新请求验证失败时旧句柄保留。hold_for_s 到期后采用 FAULT_AND_HOLD：标为 EXPIRED，保留位置输出和资源。物理分离使用显式 BREAK_CONTACT；跨资源集合的通用移交仍未实现。

Isaac Lab 适配器当前仅支持一个环境、固定基座、世界系绝对目标和本地关节坐标构型。TCP 偏置可以绑定；非零偏置的泛化验证与多环境执行仍待后续验证。它不等于完整实现 v0.2 的所有参数组合。

Windows 独立验证入口在同步写完报告和图像后采用 Bench2Dex/replay.py 的直接进程退出方式，避免本地 Kit 关闭阻塞；异常保留非零退出码。此行为只属于示例进程，不在 Runtime 库中。

## 评测与总评分

评分规范、Direct 对照评测协议和论文主实验设计统一见 [evaluation](evaluation/README.md)。总评分采用任务完成程度 30%、执行可靠性 30%、成本效率 40%；成本内部墙钟时间、Native tokens、模型请求数按 5∶4∶1 分配。当前交付为设计文档，正式评测运行器和评分器尚未实现。

## 文件

- [Runtime](manipisa/runtime.py)：接纳、生命周期、并发、监测、取消与接管。
- [共享数据类型](manipisa/types.py)：指令、目标、观测、证据、结果。
- [Isaac Lab Adapter](manipisa/adapters/isaaclab.py)：绑定、IK、目标下发、真实状态读取。
- [双 UR5＋双 Wuji Adapter](manipisa/adapters/dual_ur5_wuji.py)：实体绑定、USD 腕掌变换、工具帧与接触配置。
- [RGB 通道](manipisa/rgb.py)：可选 CameraRig 回传、采样与帧有效性。
- [物理验证入口](examples/isaaclab_smoke.py)：本地 UR5＋Wuji 示例。
- [接触反馈](manipisa/adapters/physx_contacts.py)：PhysX 原始接触与摩擦作用点。
- [接触执行](manipisa/adapters/interaction_control.py)：局部动作、预紧、导纳与持续监测。
- [接触物理验收](examples/isaaclab_interaction_smoke.py)：自由物体夹持与施力场景。
- [契约测试](tests/test_runtime.py)：使用测试替身验证执行规则，与物理验证分开。

验证结果及具体覆盖范围见 [运行验证记录](docs/manipisa-runtime-validation.md)。
