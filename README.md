# SOP-Vision — 面向电力作业的规程-视觉联合时序行为合规识别

> 项目代号：**SOP-Vision**（Standard Operating Procedure × Vision）

## 项目简介

本项目将电力作业违规形式定义为**漏步**、**乱序**、**非法终止**三个可计算问题，提出**规程知识-视觉感知解耦双阶段架构**。以《电力安全工作规程（发电厂和变电站电气部分）》GB 26860 倒闸操作为状态机蓝本，前端通过骨骼动作识别模型实现逐帧工序状态感知，后端以约束动态时间规整（DTW）算法实现规程知识与感知结果的端到端对齐与合规判断。

**核心创新：** 纯视觉分类器仅能判断"当前动作是什么"，无法检测工序步骤间的逻辑违规。本项目通过将规程规则以 YAML 状态机形式显式注入检测流程，使得漏步、乱序等分类器无法感知的违规类型得以被检测，且结果具备自然语言可解释性。

## 环境配置

- Python 版本：3.8+
- 安装依赖：

```bash
pip install -r requirements.txt
```

> **注意：** PyTorch 的 CUDA 版本请根据 GPU 驱动自行选择。

## 项目结构

```
SOP-Vision/
├── train.py                              # 训练入口（直接改脚本顶部参数）
├── extract.py                            # 骨架提取入口
├── run.py                                # 推理实验入口
├── config/
│   ├── config.yaml                       # 全局配置
│   └── procedure_rules/
│       └── switching_operation.yaml      # 倒闸操作规程
├── data/                                 # 用户自行准备
│   ├── raw_videos/                       # 原始视频
│   ├── skeletons/                        # 骨架序列 .npy
│   └── labels/
│       ├── train.csv                     # 训练标注
│       └── test.csv                      # 测试标注
├── src/
│   ├── config_loader.py
│   ├── data/dataset.py
│   ├── perception/
│   │   ├── st_gcn.py                     # ST-GCN 模型
│   │   └── train.py                      # 训练逻辑
│   ├── knowledge/
│   │   ├── procedure_template.py
│   │   └── state_machine.py
│   └── alignment/
│       ├── utils.py
│       └── compliance_checker.py         # DTW 违规检测
├── evaluation/
│   ├── metrics.py
│   └── baselines.py
├── scripts/
│   ├── 1_extract_skeletons.py
│   ├── 2_train_classifier.py
│   └── 3_run_experiment.py
├── outputs/                              # 自动生成
│   ├── models/best_model.pt              # 训练好的模型
│   ├── inference/test_inference.npy      # 推理缓存
│   ├── results/comparison.csv            # 评估结果
│   ├── dtw/*.png                         # 每样本 DTW 对齐图
│   └── timeline/*.png                    # 每样本违规时间线图
├── requirements.txt
└── README.md
```

## 场景说明

### 五步规程流程

| 步骤 | 动作 | 说明 |
|------|------|------|
| 0 | 接令核对 | 接受调度指令，核对操作票内容及设备编号 |
| 1 | 验电 | 使用验电器确认设备无电压 |
| 2 | 拉闸断电 | 按操作票顺序依次断开断路器及隔离开关 |
| 3 | 挂接地线 | 在检修设备各可能来电侧装设接地线 |
| 4 | 悬挂标示牌 | 在操作手柄处悬挂"有人工作，禁止合闸"标示牌 |

### 违规类型

| 类型 | 定义 | 示例 |
|------|------|------|
| 漏步 | 跳过规程中某个必需步骤 | 验电后直接挂接地线，跳过拉闸断电 |
| 乱序 | 步骤执行顺序与规程不符 | 先拉闸断电后验电 |
| 非法终止 | 未到达规程终态即结束 | 挂接地线后就离开，未悬挂标示牌 |

### 更换场景

前端（动作识别）与后端（规程合规）完全解耦。更换工序仅需：

1. 在 `config/procedure_rules/` 下新建 YAML 规程文件
2. 修改 `config/config.yaml` 中 `compliance.procedure` 字段
3. 前端模型和数据标注保持不变

## 数据准备

### 1. 录制视频

- 按倒闸操作规程录制作业视频
- 覆盖：正常流程 ≥3 次，漏步/乱序/非法终止各 ≥2 次
- 每个视频 1–3 分钟，帧率 ≥15 fps
- 放入 `data/raw_videos/`

### 2. 标注格式

CSV 格式（`data/labels/train.csv` 和 `data/labels/test.csv`）：

```csv
文件名,帧范围标注,工序执行顺序
```

**帧范围标注格式：** `类别:起始帧-结束帧`，多段用 `;` 分隔。

**示例：**

```csv
# 正常流程
worker_A_normal_01.npy,0:0-9;1:10-30;2:31-55;3:56-80;4:81-105;5:106-118;6:119-121,1-2-3-4-5
# 漏步：跳过拉闸断电
worker_A_skip_01.npy,0:0-5;1:6-24;2:25-44;4:45-65;5:66-80;6:81-83,1-2-4-5
# 乱序：先拉闸后验电
worker_B_reorder_01.npy,0:0-5;1:6-24;3:25-45;2:46-66;4:67-87;5:88-102;6:103-105,1-3-2-4-5
# 非法终止：挂完地线就结束
worker_C_early_01.npy,0:0-5;1:6-24;2:25-44;3:45-65;4:66-82;6:83-85,1-2-3-4
```

**7 类动作定义：**

| class_id | 名称 | 说明 |
|----------|------|------|
| 0 | 背景/准备 | 作业前调整、非工序动作 |
| 1 | 接令核对 | 接受调度指令、核对操作票 |
| 2 | 验电 | 使用验电器确认无电压 |
| 3 | 拉闸断电 | 断开断路器及隔离开关 |
| 4 | 挂接地线 | 装设接地线 |
| 5 | 悬挂标示牌 | 悬挂警示标示牌 |
| 6 | 工序间调整 | 步骤间的过渡/穿插动作 |

## 快速开始

```bash
# 1. 提取骨架
python extract.py

# 2. 训练模型
python train.py

# 3. 推理实验
python run.py
```

每个脚本顶部的 `# ==== 参数 ====` 区域可直接修改参数，无需命令行参数。

## 参数说明

### extract.py — 骨架提取

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VIDEO_DIR` | `data/raw_videos` | 原始视频目录 |
| `OUT_DIR` | `data/skeletons` | 骨架 .npy 输出目录 |
| `VIDEO_EXT` | `.mp4` | 视频文件扩展名 |
| `SHOW_PREVIEW` | `False` | 是否显示骨架预览窗口 |

### train.py — 训练

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EPOCHS` | `50` | 训练轮数 |
| `BATCH_SIZE` | `32` | 批次大小 |
| `LR` | `0.001` | 学习率 |
| `WEIGHT_DECAY` | `0.0001` | 权重衰减 |
| `WINDOW_SIZE` | `32` | 输入时间窗口帧数 |

命令行覆盖（可选）：`python train.py --epochs 100 --lr 0.01`

训练完成后模型保存至 `outputs/models/best_model.pt`。

### run.py — 推理实验

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SKIP_INFERENCE` | `False` | `True` = 使用推理缓存，跳过模型推理 |

### config.yaml — 全局配置

`config/config.yaml` 中的主要配置项（命令行参数默认值均从此读取）：

**数据集：**
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `dataset.skeleton_dir` | `data/skeletons/` | 骨架数据目录 |
| `dataset.label_csv_dir` | `data/labels/` | 标注 CSV 目录 |
| `dataset.num_classes` | `7` | 动作类别数 |

**模型：**
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `model.in_channels` | `3` | 输入通道 (x,y,z) |
| `model.num_nodes` | `33` | MediaPipe 关键点数 |
| `model.window_size` | `32` | 时间窗口帧数 |

**训练：**
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `train.batch_size` | `32` | 批次大小 |
| `train.epochs` | `50` | 训练轮数 |
| `train.lr` | `0.001` | 学习率 |
| `train.weight_decay` | `0.0001` | 权重衰减 |
| `train.save_dir` | `outputs/models/` | 模型保存目录 |

**合规检测：**
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `compliance.window_size` | `2` | DTW 搜索窗口 |
| `compliance.transition_cost` | `0.1` | 状态转移代价 |
| `compliance.skip_penalty` | `1.0` | 漏步惩罚 |

**输出：**
| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `output.result_dir` | `outputs/results/` | 结果 CSV 输出 |
| `output.inference_dir` | `outputs/inference/` | 推理缓存 |
| `output.dtw_dir` | `outputs/dtw/` | DTW 对齐图 |
| `output.timeline_dir` | `outputs/timeline/` | 违规时间线图 |

### 命令行覆盖（所有脚本均支持）

除了直接修改脚本顶部的参数外，所有参数都可以在命令行临时覆盖：

```bash
python extract.py --video_dir my_videos --show
python train.py --epochs 100 --batch_size 16 --lr 0.01
python run.py --skip_inference
```

## 输出结果

运行 `python run.py` 后 `outputs/` 目录结构：

```
outputs/
├── models/
│   └── best_model.pt                   # 训练好的 ST-GCN 权重
├── inference/
│   └── test_inference.npy              # 推理缓存（用于 --skip_inference）
├── results/
│   └── comparison.csv                  # 三种方法评测对比
├── dtw/
│   ├── sample_01.png                   # 每样本 DTW 对齐路径图
│   └── ...
└── timeline/
    ├── sample_01.png                   # 每样本违规检测时间线
    └── ...
```

## 引用说明

- ST-GCN 参考 [st-gcn](https://github.com/yysijie/st-gcn)
- DTW 对齐使用 [fastdtw](https://github.com/slaypni/fastdtw)
- 规程知识来源于《电力安全工作规程（发电厂和变电站电气部分）》GB 26860
