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
├── config/
│   ├── config.yaml
│   └── procedure_rules/
│       └── switching_operation.yaml     # 倒闸操作规程：步骤定义+合法转移矩阵
├── data/                                # 用户自行准备
│   ├── raw_videos/                      # 原始视频文件
│   ├── skeletons/                       # MediaPipe 提取的骨架序列 .npy
│   └── labels/
│       ├── train.csv                    # 训练标注
│       └── test.csv                     # 测试标注（含违规流程）
├── src/
│   ├── config_loader.py                 # YAML 配置加载
│   ├── data/
│   │   └── dataset.py                   # PyTorch Dataset + DataLoader
│   ├── perception/
│   │   ├── pose_estimator.py            # MediaPipe 33 关键点封装
│   │   ├── st_gcn.py                    # ST-GCN 模型定义（可替换）
│   │   └── train.py                     # 训练循环
│   ├── knowledge/
│   │   ├── procedure_template.py        # class_id↔步骤映射 + 合规模板
│   │   └── state_machine.py             # 规程状态机（从 YAML 构建）
│   └── alignment/
│       ├── utils.py                     # 序列压缩 / 特征构建
│       └── compliance_checker.py        # DTW 违规检测器
├── evaluation/
│   ├── metrics.py                       # 分类+违规检测评估指标
│   └── baselines.py                     # Baseline1(纯分类)/Baseline2(DBA+DTW)
├── scripts/
│   ├── 1_extract_skeletons.py           # 步骤1：视频→骨架提取
│   ├── 2_train_classifier.py            # 步骤2：训练动作分类器
│   └── 3_run_experiment.py              # 步骤3：端到端对比实验
├── outputs/                             # 自动生成
│   ├── models/                          # 训练好的模型权重 .pt
│   ├── results/                         # 实验结果 CSV
│   └── logs/                            # 训练日志
├── requirements.txt
└── README.md
```

## 场景说明

本项目的默认场景为**电力倒闸操作**，依据《电力安全工作规程》GB 26860。倒闸操作是变电站最常见的作业类型，具有严格的操作序列要求，天然适合工序合规检测任务。

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

本项目的前端（动作识别）与后端（规程合规）完全解耦。更换工序时仅需：

1. 在 `config/procedure_rules/` 下新建 YAML 规程文件
2. 修改 `config/config.yaml` 中 `compliance.procedure` 字段指向新文件
3. 前端模型和数据标注保持不变

## 数据准备

### 1. 录制视频

- 按倒闸操作规程录制作业视频
- 需要覆盖：正常流程 ≥3 次，漏步/乱序/非法终止各 ≥2 次
- 每个视频 1–3 分钟，帧率 ≥15 fps
- 放入 `data/raw_videos/`

### 2. 提取骨架

```bash
python scripts/1_extract_skeletons.py --video_dir data/raw_videos --out_dir data/skeletons
```

输出：每个视频生成一个同名 `.npy` 文件，shape = `(总帧数, 99)`。

### 3. 标注格式

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

## 运行步骤

### 步骤 1：骨架提取

```bash
python scripts/1_extract_skeletons.py --video_dir data/raw_videos --out_dir data/skeletons
```

### 步骤 2：训练动作分类器

```bash
python scripts/2_train_classifier.py \
    --data_dir data/skeletons \
    --train_csv data/labels/train.csv \
    --test_csv data/labels/test.csv \
    --epochs 50 --batch_size 32
```

训练完成后模型保存至 `outputs/models/best_model.pt`。

### 步骤 3：运行对比实验

```bash
python scripts/3_run_experiment.py
```

实验对比三种方法（纯分类器 / DBA模板+DTW / 状态机+DTW），结果输出至 `outputs/results/comparison.csv`。

## 引用说明

- ST-GCN 参考开源实现 [st-gcn](https://github.com/yysijie/st-gcn)
- DTW 对齐使用 [fastdtw](https://github.com/slaypni/fastdtw)
- 规程知识来源于《电力安全工作规程（发电厂和变电站电气部分）》GB 26860
