# 双 UR5 与双 Wuji Adapter

`DualUr5WujiAdapter` 集中封装 Bench2Dex 双 UR5＋双 Wuji 资产的角色绑定、腕部到手掌变换、TCP 配置和接触刚体映射。它继承 `IsaacLabAdapter`，直接复用已有 DLS IK、接触控制、状态读取和物理步进。六条 ISA 与 Runtime 的接口保持不变。

实现：[dual_ur5_wuji.py](../manipisa/adapters/dual_ur5_wuji.py)。通用执行逻辑：[isaaclab.py](../manipisa/adapters/isaaclab.py)。

## 默认角色与坐标系

| 物理部分 | 默认角色 | 控制坐标 | 观测帧 |
| --- | --- | --- | --- |
| 右 UR5 | arm_a | 6 个关节，肩到腕固定顺序 | tool_a，默认 wrist_3_link 原点与方向 |
| 左 UR5 | arm_b | 6 个关节，肩到腕固定顺序 | tool_b，默认 L_arm_wrist_3_link 原点与方向 |
| 右 Wuji | hand_a | 20 个关节，finger1～5、各 joint1～4 | palm_a，物理右手掌帧 |
| 左 Wuji | hand_b | 20 个关节，finger1～5、各 joint1～4 | palm_b，物理左手掌帧 |

默认工具帧保留原验证脚本的腕部语义；手掌帧是新增观测。关节数组按绑定顺序返回，不依赖物理引擎内部的关节排列。角色和帧名可用 SideBinding 修改，修改不会改变 ISA 的操作码或语义。

Adapter 从启用的 USD FixedJoint 链组合 `wrist_to_palm`，包括臂法兰和手法兰。左侧、右侧分别解析，不假定镜像对称。不存在固定链、缺少关节、重复名称、非米制或缩放/反射后的机器人帧都会拒绝绑定。

局部变换使用米与 wxyz 四元数，含义为“子坐标系在父坐标系中的位姿”。TCP 可指定以 wrist 或 palm 为父帧。目标仍由通用 Adapter 转成世界系 Jacobian 与关节目标，沿用其非零 TCP 的线速度/Jacobian 偏置修正。

## 接入现有场景

场景负责创建资产、reset 和设置初始关节状态；之后创建 Adapter：

```python
from manipisa import Runtime
from manipisa.adapters import DualUr5WujiAdapter

robot = spawned["interactive_objects"]["global_robot"]
adapter = DualUr5WujiAdapter(
    sim,
    robot,
    pre_step_hooks=spawned["pre_step_hooks"],
)
runtime = Runtime(adapter)
```

Adapter 不自行创建场景、重置状态或改变执行器增益、摩擦与重力补偿。`pre_step_hooks` 原样交给已有执行层，每个物理步调用一次；接入 Bench2Dex 时传入 spawner 返回的 hooks，避免遗漏或重复补偿。导入该类不启动 Isaac Sim，也不提前导入 USD/PhysX。

以物理手掌为 TCP，并叠加 1 cm 局部工具偏置：

```python
from dataclasses import replace
from manipisa.adapters import DualUr5WujiConfig, ToolFrame, RigidTransform

config = DualUr5WujiConfig()
config = replace(
    config,
    right=replace(config.right, tcp=ToolFrame("palm", RigidTransform((0.01, 0.0, 0.0)))),
    left=replace(config.left, tcp=ToolFrame("palm")),
)
adapter = DualUr5WujiAdapter(sim, robot, config=config, pre_step_hooks=spawned["pre_step_hooks"])
```

上述偏置仅演示接口；实际工具偏置由安装标定决定。世界系 MOVE 目标对应配置后的 tool_a / tool_b。

## 接触配置

`WujiContact` 用 side 和 region 选择机器人刚体，调用方不必写 Wuji 的 USD link 路径：

- side：`right` 或 `left`。
- region：`palm`、`thumb_tip`、`index_tip`、`middle_tip`、`ring_tip`、`little_tip`，或各指的 `*_link1`～`*_link4`。
- controller：`hand` 表示由该侧手指关节执行调整；`arm` 表示由该侧机械臂执行接近或施力。手掌自身的位姿运动由臂执行。
- target 与 target_path：场景的观测对象 ID 和实际目标刚体路径。Adapter 检查该路径与对象 PhysX view 一致。

接触报告在 reset 前启用，接触 view 在 reset 后创建：

```python
from manipisa.adapters import WujiContact

contacts = (
    WujiContact(
        name="right_index_object",
        side="right",
        region="index_tip",
        target="object",
        target_path="/World/Object",
        controller="hand",
        separation=gap_provider,
        friction_lower_bound=calibrated_mu,
    ),
)

# 资产已创建，尚未 reset。
DualUr5WujiAdapter.enable_contact_reports(sim.stage, robot.cfg.prim_path, contacts)
sim.reset()
# 按场景约定应用初始状态，再创建适配器。
adapter = DualUr5WujiAdapter(
    sim, robot, contacts=contacts, entities={"object": object_asset},
    pre_step_hooks=spawned["pre_step_hooks"],
)
```

`gap_provider(now)` 返回 `(间隙米数, 观测时刻)`，应由当前碰撞几何计算。`calibrated_mu` 是接触双方材料校准后的保守摩擦下界。两者都可以省略，但缺失间隙时不能证明 BREAK_CONTACT，缺失摩擦校准时不能接纳当前模型的 CONTROL_GRASP。Adapter 不填造这些数据。

可用 `adapter.body_pose(side, region)` 读取实际 link 的位姿、速度和时间，作为几何观测的输入。现有 [几何间隙工具](../manipisa/adapters/physx_contacts.py) 可供场景结合碰撞形状使用。

同一个物理接触对只允许一个 ID 和一个执行角色；不能用两种名称让臂、手独立覆盖同一接口。接触源复用 PhysXContactSource，继续提供法向/摩擦合力与作用点，不接入 TacMap。

## 资源与适用范围

物理关节、接触对、对象耦合锁沿用通用执行层。SideBinding 可通过 `arm_coupling_groups` / `hand_coupling_groups` 声明额外机械或任务耦合。绑定层没有新增双臂闭链求解器，也没有把两个独立 IK 当作共持协调控制。

当前针对一个固定基座、单环境、精确机器人 prim 路径的完整双臂双手 Articulation。自定义缩放、改名资产、拆成多个 Articulation 或实体运动需要相应扩展。抓持/施力的执行范围仍见 [接触执行说明](manipisa-interaction-runtime.md)。

`describe_bindings()` 输出角色、关节、真实路径、腕掌变换和 hook 数量，供配置审查与验证报告使用；这些机器人细节不进入通用 ISA 定义。

## 验证入口

原有 UR5＋Wuji 验证脚本已经改用此 Adapter：

```powershell
.\scripts\run_runtime_smoke.ps1
.\scripts\run_runtime_smoke.ps1 -Rgb
.\scripts\run_runtime_smoke.ps1 -AdapterChecks
```

`-AdapterChecks` 使用左右不同的非零手掌 TCP 偏置与旋转，在运动中比较实际手掌推导出的 TCP 位姿/速度，并接通双手 10 个指尖和两个手掌，共 12 个接触对。接触目标放在远处，该项验证报告绑定和有效观测，不宣称完成 Wuji 抓持任务。验收结果见 [运行记录](manipisa-runtime-validation.md)。
