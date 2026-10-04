# ManipISA Runtime 运行验证记录

更新至 2026-10-04。六条指令均已有可运行组合，结构化接触反馈与可选 RGB 共用 Runtime；双 UR5＋双 Wuji 已封装专用 Adapter。TacMap 未接入。当前实现范围见 [接触执行说明](manipisa-interaction-runtime.md)，不代表 v0.2 所有组合或 Bench2Dex 正式任务成绩。

## 2026 年 10 月 4 日 专用机器人 Adapter

新增 [DualUr5WujiAdapter](../manipisa/adapters/dual_ur5_wuji.py)，集中处理双臂双手的 52 个关节、左右腕部到手掌的固定变换、TCP 与接触区域映射。原验证脚本已改用专用入口，控制、IK 与 Runtime 实现沿用原有代码。接口说明见 [专用 Adapter 文档](dual-ur5-wuji-adapter.md)。

| 验证 | 结果 | 记录 |
| --- | --- | --- |
| 单元与契约测试 | 65 项全部通过 | [日志](../artifacts/dual-adapter-unit-tests.txt) |
| 原生资产与非零手掌 TCP | 13 项检查全部通过，进程退出码 0 | [报告](../artifacts/dual-adapter-checks-final/report.json) |
| 默认配置结构化反馈 | 10 项检查全部通过，进程退出码 0 | [报告](../artifacts/dual-adapter-structured/report.json) |
| 默认配置与 RGB | 13 项检查全部通过，进程退出码 0 | [报告](../artifacts/dual-adapter-rgb/report.json) |
| 默认配置与封装前轨迹比较 | 148 步共同关节/原有末端状态最大差 0 | [兼容记录](../artifacts/dual-adapter-compatibility.json) |
| RGB 开关配对比较 | 148 个共同物理步记录一致，最大差 0，源码哈希匹配 | [比较记录](../artifacts/dual-adapter-rgb-comparison.json) |

新增检查覆盖关节顺序与所有权、角色改名、不同左右侧偏置、变换乘法/逆变换、耦合声明转交、错误配置拒绝，以及导入时不启动 Isaac Sim。Python 编译与 PowerShell 入口语法检查通过。

物理场景在两侧手掌上配置不同的非零平移与旋转 TCP，通过原有 IK 执行 MOVE。在运行的每个检查步，将 USD 固定链推导的 TCP 与物理手掌推导的 TCP 比较，位置误差小于 0.2 mm，姿态误差小于 0.002 rad，线速度误差小于 0.002 m/s。两个手掌和 10 个指尖的原生 PhysX 接触绑定都返回有效观测。

这 12 个接触对的目标位于远处，验收的是接触报告和刚体映射接通，不是 Wuji 抓持成功。默认配置的工具帧仍对应腕部，新增 palm_a / palm_b 观测；与旧轨迹比较仅检查双方共有状态。当前三份物理报告记录的源码哈希均与交付版本一致。dual-adapter-checks-001 属于开发记录，final 为该检查的最终证据。

## 2026 年 10 月 3 日 补齐四条接触指令

新增 MAKE_CONTACT、BREAK_CONTACT、CONTROL_GRASP、APPLY_WRENCH 的类型、准入、控制、成功证据、持续监测和结果。范围由明确的接触绑定与局部动作限定，没有更改上游 Bench2Dex / Isaac Lab 源码或原始机器人资产。

| 验证 | 结果 | 记录 |
| --- | --- | --- |
| 单元与契约测试 | 56 项通过 | [日志](../artifacts/six-ops-unit-tests.txt) |
| Isaac Lab 接触物理验收 | 13 项检查全部通过，进程退出码 0 | [报告](../artifacts/six-ops-interactions/report.json) |
| UR5＋Wuji 结构化反馈回归 | 8 项检查全部通过，进程退出码 0 | [报告](../artifacts/six-ops-ur5-structured/report.json) |
| UR5＋Wuji RGB 回归 | 11 项检查全部通过，进程退出码 0 | [报告](../artifacts/six-ops-ur5-rgb/report.json) |
| RGB 开关配对比较 | 148 个共同物理步记录一致；状态量最大绝对差 0；结果与源码哈希一致 | [比较记录](../artifacts/six-ops-rgb-comparison.json) |

接触物理场景使用有位置驱动的双指夹持器、自由刚体和一个单轴施力探头，运行在相同 Isaac Lab / PhysX 平台上。全部接触、摩擦、位姿和速度来自物理步进；夹持与施力指令没有通过写入物体状态或外力伪造完成。

具体检查包括建立两侧接触、生成有适用范围的抓持关系、在 0.2 N 外部测试载荷下持续抓持、建立探头接触、闭环施力、变换力矩参考点、解除接触并验证至少 3 mm 几何间隙、重新建立接触，以及激活后遭受 20 N 过载的失败处理。外力由验收脚本作为独立扰动施加。

施力目标为世界 X 方向 2 N、容差 0.25 N、激活后维持 0.2 s；记录值为约 2.055 N。过载用例返回 FAILED / GRASP_INVALID；撤去扰动后失败终态保持。力矩参考点转换同时使用实际物理接口数据检查。

契约反例覆盖零力但未分离、无几何间隙、缺失摩擦导致完整 wrench 无效、陈旧证据、接触丢失、保护/禁止接触、力上限、局部动作越界、抓持负载不成立、漂移、资源移交、摩擦请求超过校准值，以及导纳接管时保留预紧。Python 编译与 PowerShell 脚本语法检查通过。

这些结果验证了四条指令的执行路径与具体物理组合。Wuji 原生灵巧抓持、双臂共持、任意独立六轴力矩控制、动态承载和完整 Bench2Dex 任务仍未验收。抓持证据采用有明确假设的准静态有限工况模型；没有将它当作通用动态抓持证明。

开发过程中修正了施力的两处问题：从实测位置限制目标偏置会意外形成力上限；从实测位姿初始化导纳会卸掉已有 PD 预紧。最终路径限制命令轨迹增量、累积力误差，并在接管时保留已有驱动偏置。另修复了结构化反馈中的 NumPy 数值序列化。最终证据以本节的 six-ops-* 目录为准；interactions-001～004 和 interactions-final 是开发诊断记录。

## 首轮记录：MOVE / SHAPE_HAND 与 RGB

以下保留第一次实现的历史结果，文件哈希与覆盖边界均指当时版本；当前结果以上一节为准。

## 实现与环境

实现位于 [manipisa](../manipisa/runtime.py)，使用方式见 [README](../README.md)。主架构为 Agent → Instruction → Runtime → Adapter → Controller；没有独立 Safety 或 Capability 模块。

本地环境为 Windows、Python 3.11、Isaac Sim 5.1.0、Isaac Lab 仓库版本 2.3.2、PyTorch 2.7.0+cu128，执行设备 cuda:0。环境中 isaaclab Python 包元数据版本为 0.54.2，它与仓库 VERSION 的编号口径不同。

仿真使用 Bench2Dex 本地 UR5＋Wuji 资产和原有执行器、重力补偿配置。场景为固定基座机器人与地面，没有待操作物体。默认关节构型在初始化阶段显式写入，之后通过关节目标和物理步进执行，没有用状态回写完成 MOVE 或 SHAPE_HAND。

## 契约测试

[测试日志](../artifacts/runtime-unit-tests.txt)：30 项通过。使用确定性测试替身，不计为物理验证。

覆盖实际观测与命令的区别、速度与连续满足时间、缺测和过期数据、重复观测不能积累满足时间、进入条件重检、冲突与别名资源、显式耦合冲突、SUSTAIN 激活和维持、超时、更新预算不重置、取消、失败终态、控制交接、残余句柄过期、后端异常，以及 RGB 开关、缺帧、采样、过期和渲染失败隔离。

Python 编译检查与 PowerShell 验证脚本语法检查通过。

## 真实物理执行

最终代码运行了两个独立仿真进程，均以退出码 0 完成：

| 运行 | 检查结果 | 证据 |
| --- | --- | --- |
| 结构化反馈 | 8 项检查通过 | [报告](../artifacts/validation-structured-final/report.json)、[轨迹](../artifacts/validation-structured-final/trace.jsonl)、[日志](../artifacts/validation-structured-final.log) |
| 结构化反馈加 RGB 开关 | 相同 8 项及 3 项 RGB 检查通过 | [报告](../artifacts/validation-rgb-final/report.json)、[轨迹](../artifacts/validation-rgb-final/trace.jsonl)、[日志](../artifacts/validation-rgb-final.log) |

共同执行流程：

1. 按默认构型初始化并通过物理步进稳定机器人。
2. 执行单侧 SHAPE_HAND，并检查实际构型误差、速度和连续满足时间。
3. 验证重复资源请求被拒绝，CONTROL_GRASP 尚未实现时明确拒绝。
4. 在自由空间分别提交两侧末端上移 0.02 m 的 MOVE；两侧使用不重叠的关节组并发执行。
5. 通过显式控制句柄接管进入 MOVE/SUSTAIN，验证 ACTIVE 后维持规定时间才完成。
6. 再进入持续模式并取消，验证 CANCELED 与保留控制句柄。

RGB 运行中，在构型调整完成后打开回传。两路 320×240 RGB 共执行 8 次采样；帧有效，具有非空图像内容、采集时间、内参和外参。随后关闭回传，再推进 4 个物理步，采样次数不增加，图像缓存为空。末次图像为 [overview](../artifacts/validation-rgb-final/overview.png) 和 [overhead](../artifacts/validation-rgb-final/overhead.png)，已人工视觉检查两侧机器人可见。相机外参沿用 Bench2Dex.CameraFrame 的坐标约定，不能未经转换当作另一种光学坐标约定使用。

RGB 通过同步挂载相机位姿、渲染、采集生成当前回合观测，没有把示范视频当作在线反馈。未实例化触觉采集链路，未读取官方任务成功位或阶段标签。

## 配对比较

[配对记录](../artifacts/runtime-comparison.json) 比较了两次最终运行的源文件哈希、指令结果和共享物理轨迹。

- 源文件哈希彼此相同，并与该轮保存时的实现和示例代码匹配。
- 共同的 148 个物理步中，仿真相对时间、关节位置/速度、目标帧位置/四元数/线速度/角速度，记录值的最大绝对差均为 0。
- 指令操作码、终态和原因编码相同。
- RGB 运行多出的 4 步专门用于验证关闭开关，不属于共同轨迹比较区间。

比较预设绝对和相对容差均为 1e-5。本次记录实际完全一致，说明这一配对运行中 RGB 回传没有改变控制轨迹；不能外推为所有任务、场景、硬件和渲染负载下都确定性一致。

## 覆盖边界

已验证的是单环境、固定基座、空闲空间的 SHAPE_HAND，以及世界系 ACTOR_FRAME 的 MOVE/REACH 和 MOVE/SUSTAIN。支持按实际资源检查并发，并对声明的耦合域拒绝并发；这不证明双臂共持、闭链或一般碰撞约束可行。

该首轮未实现四条接触指令；本轮已补齐其可运行组合，见本文开头。ENTITY_FRAME 运动、一般约束更新、通用控制移交、多环境、非零 TCP 偏置的泛化、视觉状态估计、触觉图像和完整 Bench2Dex 任务评测仍不在验收范围内。

结构化状态来自模拟器，明确标记 simulation_privileged。当前 RGB 是给 Agent 的附加观察，不替换底层状态来源，也没有用视觉判定任务成功。

## 运行过程中的修正

早期验证发现 Isaac Lab 的 default_joint_pos 需要在初始化时显式应用，示例已修正。最终配对使用修正后的同一版本。

本地 Windows Kit 的普通关闭和关闭时跳过 Replicator 等待均出现阻塞。独立示例在同步保存所有产物后采用 Bench2Dex/replay.py 已有的直接进程退出方式，并保留异常对应的非零退出码；最终两个进程均已退出。该行为不进入 Runtime 库。

上游资产仍有实例化属性及部分视觉引用警告，未修改资产或 Bench2Dex/Isaac Lab 源码来消除这些警告。最后两路图像和本次物理目标通过检查，不据此宣称资产的所有几何与材质均已验收。早期 validation-structured-001、002 与 validation-rgb-001 目录属于开发诊断记录；以上 final 目录是本文的交付证据。
