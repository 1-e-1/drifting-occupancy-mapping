# Drifting V7：保留的一阶时空架构

当前生产配置：`configs/occupancy_drifting_v7_partial_visibility.json`。
核心文件沿用 `model_v3.py` 名字，但运行的是 V7 配置；旧 V1/V2 生成器和 V3–V6 权重已清理。

## 结构与训练

RGB-D/位姿历史 → 时空 Transformer → occupied/free/unknown 三维记忆 →
粗到细 occupied-volume 粒子生成 → 隐式 occupancy/SDF 与外轮廓接口。

输入为 16 帧 RGB-D 和已标定相机位姿，不输入物体数量、类别或真值盒。
图像编码尺寸 128，观测点 8192，输出粒子 2048，隐藏维度 256。
一阶 drifting 训练目标更新 12 次，步长 1/12；推理每个样本 1 NFE，不是 12 次 ODE。
二阶损失函数仅作为同一训练机制的消融工具保留，不存在另一套旧生成器。

训练 / 评估：`train_v3.py`、`evaluate_v3.py`、`benchmark_v3.py`。
从随机初始化分阶段训练可运行 `run_v7_scratch_pipeline.py`，该流程会重新产生阶段权重；
清理只保留了最终选中的 V7 权重，并未保留旧阶段的 optimizer resume 文件。

## 模型

在工程根目录使用 `checkpoints/drifting_v7/best.pt`，epoch 41。
`selection.json` 保留旧训练评估的原始来源记录，绝对历史路径仅用于溯源；
实际当前路径由根目录 `configs/retained_models.json` 指定。
该版本在相同 V7 测试协议的 volume Chamfer / coverage 和 outline IoU 上优于之前 V7 结果，
但不是所有指标都同时更优，也未证明混合新形状的规划安全。

```powershell
$PY = 'D:\\conda_envs\\env_isaaclab\\python.exe'
& $PY -B -m piaoyi_model.evaluate_v3 --config piaoyi_model/configs/occupancy_drifting_v7_partial_visibility.json --checkpoint checkpoints/drifting_v7/best.pt --data_dir data/sequences_v7_partial_visibility_formal --split test --device cpu
& $PY -B -m piaoyi_model.evaluate_outline --particle_splat_mode adaptive --device cpu
```

## 控制与外轮廓

`belief_v4.py`、`online_v4.py` 和相应 PPO 接口仍为本架构服务，保留文件名以避免破坏调用。
最近配套 PPO/World Model 权重位于 `checkpoints/drifting_controller/`，来源是 V6 控制训练。
切换 V7 生成器之后仍需重新训练或评估控制器；本次整理没有做这项训练。

外轮廓为 `outline.py`，视觉引导为 `visual_servo_guidance.py`。
细节见 [V7_PARTIAL_VISIBILITY_DESIGN.md](V7_PARTIAL_VISIBILITY_DESIGN.md) 和
[VISUAL_SERVO_GUIDANCE.md](VISUAL_SERVO_GUIDANCE.md)。
生成的内部 / 遮挡几何不是实际测量，不能直接作为实机自由空间放行依据。
