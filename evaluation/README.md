# ManipISA Evaluation

本目录集中维护 ManipISA 与 Direct 的评分规范、评测协议和主实验设计。评分文档与未来的评测实现共用一个 `evaluation/` 入口，文档放在 `docs/` 子目录，避免维护两套评分规则。

## 阅读入口

| 文档 | 内容 |
| --- | --- |
| [总评分标准](docs/manipisa-overall-scoring.md) | Overall 公式、归一化、成本计量边界、缺失值处理和汇总规则；评分的唯一规范 |
| [Bench2Dex 对照评测协议](docs/bench2dex-manipisa-direct-evaluation.md) | 官方原始指标、Direct 与 ManipISA 公平对照、跨 Agent 计量口径 |
| [主实验设计](docs/cvpr-main-experiment-design.md) | 模型与 Agent 候选、配对实验矩阵、消融和预算裁剪 |

## 评分概览

每个固定模型及 Agent 配置下，Direct 与 ManipISA 分别计算分数，再报告配对差值。

- 任务完成程度：30%。
- 执行可靠性：30%。
- 成本效率：40%；内部墙钟时间、Native tokens、模型推理请求数按 5∶4∶1 分配。

任务判定复用 Bench2Dex，Overall 合成属于 ManipISA。成本项经过任务质量门控；具体定义以[总评分标准](docs/manipisa-overall-scoring.md)为准。通用性不计分，美元费用单独报告。

## 实现状态与目录规划

当前提交的是设计文档，还没有 ManipISA 的正式评测运行器、跨 Agent 计量适配器或可运行评分器，也没有正式对照实验结果。部分进度折扣、质量门控和归一化校准流程仍属于设计稿，三个成本参考值需在开发集校准后冻结。

后续实现放在本目录，按实际需要建立以下子目录：

| 规划子目录 | 职责 |
| --- | --- |
| `metrics/` | Overall 计算、归一化与汇总 |
| `runners/` | 回合边界、配对调度与日志 |
| `adapters/` | 各 Agent 的原生 usage 与推理请求账本 |
| `configs/` | 冻结的任务、模型、运行上限与评分参数 |

这些代码子目录尚未创建。计量机制参考 [DexISA evaluation](https://github.com/Harbara-Ai/DexISA/tree/ad41b25aed8a0914d5936d9331d5b24514bcaf06/evaluation)；旧 MuJoCo 任务和停止规则不作为 ManipISA 评测协议。机器人运行时仍位于 [`manipisa/`](../manipisa)，独立评价器按冻结版本接入 Bench2Dex。
