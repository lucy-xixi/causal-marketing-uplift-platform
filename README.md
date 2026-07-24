# Causal Marketing Uplift Platform

**信贷营销因果 uplift 建模平台**

面向金融科技个人贷款营销场景，提供从数据生成、质量评估、探索性分析、基线建模、因果 uplift 建模到策略评估的**端到端因果推断与 uplift 建模系统**。平台估计营销干预（邮件 / 短信 / Push）对客户贷款申请的**异质性处理效应（CATE）**，支持精准投放与 ROI 优化。

---

## 核心问题

> 哪些客户应该收到营销 offer，才能以最低成本获得**增量**贷款申请？

传统响应预测模型回答的是「谁可能申请」，而 uplift 模型回答的是「谁**因为**营销才会申请」。二者在业务上不等价：

| 方法 | 优化目标 | 局限 |
|------|----------|------|
| 响应预测 | P(申请 \| 客户特征) | 会触达本就会申请的高意向客户 |
| **Uplift 建模** | P(申请 \| 有 offer) − P(申请 \| 无 offer) | 识别真正可被营销 persuadable 的客户 |

---

## 主要能力

- **合成数据生成** — 50,000 客户，含 4 类 uplift 细分（Persuadable / Sure Thing / Lost Cause / Sleeping Dog）
- **数据质量评估** — 缺失值、重复、类型、范围、异常值、处理组平衡
- **探索性分析** — Jupyter Notebook + 可视化图表
- **基线响应模型** — Logistic Regression、XGBoost（ROC-AUC 0.91）
- **因果 Uplift 建模** — EconML：S-Learner、T-Learner、X-Learner、Causal Forest DML
- **Uplift 策略评估** — Uplift 曲线、Qini 曲线、增量收益，对比随机 / 响应模型 / Uplift 模型三种投放策略
- **项目报告** — 面向业务与招聘场景的完整技术报告

---

## 关键结果（测试集 n=10,000）

| 投放策略 | Qini 系数 | Top 30% 增量转化 |
|----------|-----------|------------------|
| 随机投放 | 0.0 | 95.8 |
| 响应模型（XGBoost） | 41.5 | 102.2 |
| **Uplift 模型（EconML 集成）** | **131.9** | **481.8** |

Uplift 模型在 Top 30% 目标客群中，增量转化约为响应模型的 **4.7 倍**。

---

## 技术栈

| 类别 | 工具 |
|------|------|
| 语言 | Python 3.11+ |
| 数据处理 | pandas, numpy, scipy |
| 机器学习 | scikit-learn, XGBoost |
| 因果推断 | **EconML**（Microsoft / PyWhy） |
| 可视化 | matplotlib, seaborn |
| 分析 | Jupyter |

---

## 项目结构

```
causal-marketing-uplift-platform/
├── configs/                    # 实验与流水线配置
│   └── default.yaml
├── data/
│   ├── raw/                    # 原始数据（不可变）
│   │   └── credit_marketing.csv
│   └── processed/              # 建模输出
│       └── customer_uplift_scores.csv
├── notebooks/
│   └── 01_credit_marketing_eda.ipynb
├── reports/                    # 自动生成的报告与图表
├── src/
│   ├── data/                   # 数据生成与质量评估
│   ├── features/               # 特征工程与预处理
│   ├── models/                 # 基线模型 + Uplift 模型
│   ├── evaluation/             # Uplift 评估指标
│   └── simulation/             # 策略仿真（预留）
├── requirements.txt
└── README.md
```

### 模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| `src/data/` | `generate_credit_data.py` | 生成含异质效应的合成信贷营销数据 |
| | `assess_data_quality.py` | 数据质量评估报告 |
| `src/features/` | `preprocessing.py` | 客户特征选择与 sklearn 预处理 |
| `src/models/` | `baseline_response.py` | 基线响应预测（LR / XGBoost） |
| | `uplift_models.py` | EconML uplift 建模与客户评分 |
| `src/evaluation/` | `uplift_evaluation.py` | Uplift / Qini 曲线与策略对比 |
| | `metrics.py` | 分类与 uplift 评估指标 |

---

## 快速开始

### 1. 环境配置

```bash
# 克隆项目后进入目录
cd causal-marketing-uplift-platform

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 安装依赖
pip install -r requirements.txt
```

### 2. 端到端流水线

```bash
# ① 生成合成数据（50,000 客户）
python src/data/generate_credit_data.py

# ② 数据质量评估
python src/data/assess_data_quality.py

# ③ 基线响应模型训练与对比
python -m src.models.baseline_response

# ④ 因果 Uplift 建模（EconML，约 2 分钟）
python -m src.models.uplift_models

# ⑤ Uplift 策略评估
python -m src.evaluation.uplift_evaluation
```

### 3. 探索性分析

```bash
jupyter notebook notebooks/01_credit_marketing_eda.ipynb
```

---

## 输出文件

| 文件 | 说明 |
|------|------|
| `data/raw/credit_marketing.csv` | 原始营销实验数据 |
| `data/processed/customer_uplift_scores.csv` | 客户级 uplift 评分（ITE） |
| `reports/data_quality_report.md` | 数据质量报告 |
| `reports/baseline_model_comparison.md` | 基线模型对比 |
| `reports/uplift_modeling_report.md` | Uplift 建模报告 |
| `reports/uplift_evaluation_report.md` | 策略评估报告 |
| `reports/uplift_evaluation_curves.png` | Uplift / Qini 曲线图 |
| `reports/uplift_incremental_gain.png` | 增量收益柱状图 |
| `reports/causal_uplift_project_report.md` | **完整项目报告（推荐阅读）** |

---

## 因果推断框架

### 符号定义

- **T** = `marketing_offer`（0 = 对照组，1 = 收到营销 offer）
- **Y** = `loan_application`（0 = 未申请，1 = 申请贷款）
- **X** = 客户特征（人口统计、财务、信用、行为）

### 个体处理效应（ITE）

```
ITE(x) = E[Y | T=1, X=x] − E[Y | T=0, X=x]
```

### 客户细分（合成数据基准）

| 细分 | 营销效应 | 投放建议 |
|------|----------|----------|
| **Persuadable** | 强正 uplift（+19.6%） | 优先投放 |
| **Sure Thing** | 高意向，营销增益小（+2.2%） | 抑制投放，节省预算 |
| **Lost Cause** | 低意向，营销无效 | 不投放 |
| **Sleeping Dog** | 负 uplift（−8.7%） | 禁止投放 |

### Uplift 模型

使用 EconML 实现四种估计器，最终 `uplift_score` 为四模型 ITE 的集成平均：

| 模型 | 说明 |
|------|------|
| S-Learner | 单模型，将 treatment 作为输入特征 |
| T-Learner | 对照组与处理组分别建模 |
| X-Learner | 基于倾向性得分的效应插补 |
| Causal Forest DML | 双重机器学习 + 因果森林 |

---

## 数据字段

```
customer_id, customer_segment, age, gender, country,
income, employment_years, account_balance, debt_ratio,
credit_score, previous_loans, payment_history, credit_utilization,
transaction_frequency, app_usage_frequency,
marketing_offer, loan_application
```

Uplift 评分输出见 `customer_uplift_scores.csv`，包含 `uplift_s_learner`、`uplift_t_learner`、`uplift_x_learner`、`uplift_causal_forest` 及集成字段 `uplift_score`。

---

## 业务应用建议

1. **按 `uplift_score` 降序排序**，优先触达 Top 20–30% 客户
2. **抑制** uplift 评分为负的客户（Sleeping Dog），避免营销反效果
3. **不以响应概率作为投放依据**，高 AUC 不等于高 ROI
4. **持续 A/B 实验**，以 Qini 系数监控模型与策略漂移
5. **叠加信贷风控规则**，在 uplift 排序基础上过滤不合规客户

---

## 版本

**v1.0** — 完整流水线：数据生成 → 质量评估 → EDA → 基线建模 → Uplift 建模 → 策略评估 → 项目报告

---

## License

仅供学习与内部研究使用（Proprietary — internal use）。
