# Home Credit 信用风险评分项目设计方案（中文版）

## 1. 项目目标与整体框架

本项目基于 **Home Credit Default Risk** 数据集，目标是构建一套可持续扩展的信用风险分析与评分框架，用于支持以下工作：

- 客户违约风险识别
- 多表数据整合与客户级训练集构建
- 传统信用评分模型与机器学习模型对比
- 风险评分卡、分层与阈值策略设计
- 模型评估、稳定性检查与后续监控

从工程视角看，本项目不应只停留在一次性的竞赛式建模，而应按“**数据理解 → 特征工程 → 建模 → 评分 → 评估 → 监控**”的完整链条来组织。

建议采用双轨目标：

1. **效果导向模型**：如 LightGBM / XGBoost / CatBoost，用于追求更高区分能力。
2. **解释导向模型**：如 WOE + Logistic Regression + PDO Scorecard，用于形成更可解释的传统风险评分体系。

---

## 2. 数据资产总览

项目当前配置文件 `configs/base.yaml` 已定义以下核心数据表：

- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `previous_application.csv`
- `installments_payments.csv`
- `credit_card_balance.csv`
- `POS_CASH_balance.csv`

关键字段：

- **主键**：`SK_ID_CURR`
- **目标列**：`TARGET`

这些表可分为主申请信息、外部征信信息、历史申请信息、还款/账单行为信息四大层次。

---

## 3. 传统数据与另类数据划分

### 3.1 传统数据（Traditional Data）

本项目数据主体仍属于典型的结构化传统金融风控数据，主要包括：

- 客户基本身份信息
- 收入、贷款、年金、商品价格等财务信息
- 他行征信与历史贷款情况
- 历史申请结果与授信情况
- 分期还款、信用卡、POS/Cash loan 行为记录

这些数据是传统信用评分和零售信贷建模中的标准输入。

### 3.2 准另类数据 / 行为型数据（Behavioral / Alternative-like Data）

严格来说，本数据集中的“另类数据”占比不高，但存在一些可视为**行为轨迹型、代理型、准另类**的数据来源：

- 月度账单与还款轨迹（`bureau_balance`, `credit_card_balance`, `POS_CASH_balance`）
- 文件提交与申请行为类字段（如 `FLAG_DOCUMENT_*`, `HOUR_APPR_PROCESS_START`）
- 居住地/工作地偏差、组织类型等代理变量
- 时间窗口聚合出的稳定性、波动性、最近行为特征

因此，更准确的描述是：

> 本项目以**传统结构化金融数据**为主，叠加**行为轨迹衍生特征**，而非典型互联网另类数据项目。

---

## 4. 各数据集说明与特征中文解释

### 4.1 `application_train.csv` / `application_test.csv`

**粒度**：一行对应一个客户当前申请。

**业务作用**：主申请表，是训练集和测试集的骨架。

**关键字段与中文解释（按类别）**：

#### 1）主键与目标
- `SK_ID_CURR`：客户/申请主键
- `TARGET`：是否发生违约/严重逾期（仅训练集有）

#### 2）身份与家庭特征
- `CODE_GENDER`：性别
- `NAME_FAMILY_STATUS`：婚姻/家庭状态
- `CNT_CHILDREN`：子女数量
- `CNT_FAM_MEMBERS`：家庭成员数量
- `NAME_TYPE_SUITE`：陪同申请人类型

#### 3）教育、职业、收入来源
- `NAME_EDUCATION_TYPE`：教育程度
- `NAME_INCOME_TYPE`：收入来源类型
- `OCCUPATION_TYPE`：职业类别
- `ORGANIZATION_TYPE`：工作单位类型

#### 4）贷款与财务特征
- `AMT_INCOME_TOTAL`：总收入
- `AMT_CREDIT`：贷款金额
- `AMT_ANNUITY`：分期/年金金额
- `AMT_GOODS_PRICE`：商品价格

#### 5）资产特征
- `FLAG_OWN_CAR`：是否有车
- `FLAG_OWN_REALTY`：是否有房

#### 6）时间类特征
- `DAYS_BIRTH`：出生距今天数
- `DAYS_EMPLOYED`：就业距今天数
- `DAYS_REGISTRATION`：注册时间距今天数
- `DAYS_ID_PUBLISH`：证件更新距今天数

#### 7）联系方式与地区特征
- `FLAG_MOBIL` / `FLAG_PHONE` / `FLAG_EMAIL`：是否留有联系方式
- `REGION_RATING_CLIENT`：客户所在区域评级
- `REG_CITY_NOT_WORK_CITY`：居住城市与工作城市是否不同

#### 8）外部评分
- `EXT_SOURCE_1` / `EXT_SOURCE_2` / `EXT_SOURCE_3`：外部信用/风险分数

#### 9）居住环境与房产结构类
- `APARTMENTS_*`：公寓相关指标
- `BASEMENTAREA_*`：地下室面积相关
- `YEARS_BUILD_*`：建筑年份相关
- `COMMONAREA_*`：公共区域相关
- `ELEVATORS_*`：电梯相关
- `FLOORSMAX_*` / `FLOORSMIN_*`：楼层相关
- `LIVINGAREA_*`：居住面积相关

#### 10）申请与文件标志类
- `FLAG_DOCUMENT_2` ~ `FLAG_DOCUMENT_21`：各类文件提交标志
- `HOUR_APPR_PROCESS_START`：申请开始小时

---

### 4.2 `bureau.csv`

**粒度**：一行对应客户在其他金融机构的一笔历史贷款/征信记录。

**业务作用**：外部信用历史与负债信息。

**关键字段**：
- `SK_ID_CURR`：客户 ID
- `SK_ID_BUREAU`：外部征信记录 ID
- `CREDIT_ACTIVE`：贷款状态（活跃/已关闭等）
- `CREDIT_CURRENCY`：币种
- `DAYS_CREDIT`：信贷建立距今天数
- `CREDIT_DAY_OVERDUE`：逾期天数
- `DAYS_CREDIT_ENDDATE`：预计结束时间距今天数
- `AMT_CREDIT_SUM`：贷款总额
- `AMT_CREDIT_SUM_DEBT`：未偿债务
- `AMT_CREDIT_SUM_OVERDUE`：逾期金额
- `CNT_CREDIT_PROLONG`：展期次数

---

### 4.3 `bureau_balance.csv`

**粒度**：一行对应某笔 bureau 贷款在某个月份的状态。

**业务作用**：补充外部贷款的月度状态轨迹。

**关键字段**：
- `SK_ID_BUREAU`：关联 `bureau.csv`
- `MONTHS_BALANCE`：月份偏移
- `STATUS`：该月状态

此表适合挖掘：
- 最近状态
- 历史最差状态
- 不良月数
- 连续风险状态长度

---

### 4.4 `previous_application.csv`

**粒度**：一行对应客户过去的一次 Home Credit 历史申请。

**业务作用**：刻画客户过往申请行为与审批结果。

**关键字段**：
- `SK_ID_CURR`：客户 ID
- `SK_ID_PREV`：历史申请 ID
- `NAME_CONTRACT_TYPE`：合同类型
- `AMT_APPLICATION`：申请金额
- `AMT_CREDIT`：批准金额
- `AMT_ANNUITY`：分期金额
- `NAME_CONTRACT_STATUS`：申请状态（批准/拒绝等）
- `DAYS_DECISION`：决策距今天数
- `CNT_PAYMENT`：分期期数

---

### 4.5 `installments_payments.csv`

**粒度**：一行对应一笔历史分期还款记录。

**业务作用**：衡量还款纪律与履约能力。

**关键字段**：
- `SK_ID_CURR`
- `SK_ID_PREV`
- `NUM_INSTALMENT_NUMBER`：第几期
- `DAYS_INSTALMENT`：应还日期
- `DAYS_ENTRY_PAYMENT`：实际付款日期
- `AMT_INSTALMENT`：应还金额
- `AMT_PAYMENT`：实还金额

---

### 4.6 `credit_card_balance.csv`

**粒度**：一行对应某月信用卡余额状态。

**业务作用**：刻画信用卡使用强度、额度利用率与月度风险。

**关键字段**：
- `SK_ID_CURR`
- `SK_ID_PREV`
- `MONTHS_BALANCE`：月份偏移
- `AMT_BALANCE`：余额
- `AMT_CREDIT_LIMIT_ACTUAL`：信用额度
- `AMT_DRAWINGS_CURRENT`：取现/消费金额
- `AMT_PAYMENT_TOTAL_CURRENT`：总还款额
- `SK_DPD` / `SK_DPD_DEF`：逾期天数相关

---

### 4.7 `POS_CASH_balance.csv`

**粒度**：一行对应某月 POS / cash loan 状态。

**业务作用**：反映现金贷/分期消费贷款的月度履约与剩余负担。

**关键字段**：
- `SK_ID_CURR`
- `SK_ID_PREV`
- `MONTHS_BALANCE`
- `CNT_INSTALMENT`：总分期期数
- `CNT_INSTALMENT_FUTURE`：剩余期数
- `SK_DPD` / `SK_DPD_DEF`：逾期情况
- `NAME_CONTRACT_STATUS`：合同状态

---

## 5. 多表整合与聚合设计

### 5.1 总体原则

训练集构建必须以 `application_train` / `application_test` 为主表，确保：

- 一行只代表一个客户（`SK_ID_CURR`）
- 历史表不能直接一对多 merge
- 所有历史明细表需先聚合为客户级特征，再回并到主表

整体流程如下：

1. 主表读取：`application_train/test`
2. 构造各子表客户级聚合特征
3. 所有聚合表按 `SK_ID_CURR` 左连接回主表
4. 形成最终 train / test feature table

---

### 5.2 `bureau.csv` 聚合思路

按 `SK_ID_CURR` 聚合外部贷款信息。

**推荐统计方式**：
- 数量：count, nunique
- 金额：sum, mean, max, min
- 状态：活跃笔数、已关闭笔数、逾期记录数
- 时间：最近贷款、最早贷款、平均剩余期限

**推荐生成特征**：
- 外部贷款总数
- 活跃外部贷款数
- 已关闭贷款数
- 外部债务总额
- 外部逾期金额总额
- 最大逾期天数
- 展期次数总和
- 最近一笔外部信贷距今天数

---

### 5.3 `bureau_balance.csv` 聚合思路

此表需分两步处理：

#### 第一步：按 `SK_ID_BUREAU` 聚合月度状态
生成：
- 最近状态
- 历史最差状态
- 风险状态月数
- 最近 N 月风险状态计数

#### 第二步：merge 到 `bureau.csv`
把 `bureau_balance` 月度状态先并回对应的外部贷款记录。

#### 第三步：再按 `SK_ID_CURR` 聚合
形成客户级特征：
- 风险状态贷款数
- 风险月数总和
- 最差状态最大值
- 最近风险活跃度

---

### 5.4 `previous_application.csv` 聚合思路

按 `SK_ID_CURR` 聚合历史申请行为。

**推荐特征**：
- 历史申请总数
- 批准次数
- 拒绝次数
- 拒绝率 / 批准率
- 历史平均申请金额
- 历史平均批准金额
- 批准金额与申请金额差值
- 最近申请距今天数
- 历史平均分期期数
- 各合同类型计数

---

### 5.5 `installments_payments.csv` 聚合思路

先构造单笔还款行为特征：

- `late_days = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT`
- `pay_diff = AMT_PAYMENT - AMT_INSTALMENT`
- `late_flag = 1(late_days > 0)`
- `underpay_flag = 1(AMT_PAYMENT < AMT_INSTALMENT)`

再按 `SK_ID_CURR` 聚合：

- 分期记录总数
- 平均晚还天数
- 最大晚还天数
- 逾期次数
- 逾期比例
- 少还次数
- 平均支付偏差
- 最近 N 期平均逾期程度
- 还款波动性（std）

---

### 5.6 `credit_card_balance.csv` 聚合思路

先构造中间指标：

- `utilization = AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL`
- `dpd_flag = 1(SK_DPD > 0)`
- `high_util_flag = 1(utilization > 0.8)`

再按 `SK_ID_CURR` 聚合：

- 平均余额
- 最大余额
- 平均额度利用率
- 最大额度利用率
- 平均逾期天数
- 最大逾期天数
- 高利用率月数
- 有逾期月数
- 平均月度支付金额

---

### 5.7 `POS_CASH_balance.csv` 聚合思路

按 `SK_ID_CURR` 聚合：

- 活跃合同数
- 已完成合同数
- 平均剩余分期期数
- 最大 DPD
- 平均 DPD
- 有逾期月数
- 合同状态计数

---

## 6. 特征筛选策略

特征筛选不应只依赖单一技术，而应结合业务、统计与模型三层筛选。

### 6.1 第一层：规则筛选

剔除：
- 缺失率过高特征（如 > 70%）
- 单值率过高特征
- 明显无业务意义特征
- 潜在目标泄漏字段
- 重复、冲突或明显噪声特征

### 6.2 第二层：统计筛选

建议使用：
- 缺失率分析
- 单变量坏账率差异
- 相关性分析
- IV（Information Value）
- KS / 单变量 AUC
- VIF（多重共线性）

### 6.3 第三层：模型筛选

建议使用：
- Logistic 回归系数稳定性
- L1 正则稀疏筛选
- 树模型特征重要性
- SHAP 重要性

---

## 7. PCA、因子分析与特征压缩建议

### 7.1 PCA（主成分分析）

**适合场景**：
- 数值变量很多
- 高维聚合后共线性较强
- 目标是压缩维度或做实验模型

**优点**：
- 降维有效
- 能缓解多重共线性

**缺点**：
- 可解释性弱
- 不适合直接用于传统评分卡解释
- 不利于业务落地与审计表达

### 7.2 因子分析

**适合场景**：
- 想提取“潜在风险维度”
- 用于辅助理解收入压力、偿债能力、历史行为纪律等潜在因子

**优点**：
- 更强调潜在结构解释

**缺点**：
- 工程落地不如监督式特征筛选直接
- 未必提升最终评分模型效果

### 7.3 建议主线

本项目主线不建议一开始依赖 PCA / 因子分析，而建议：

1. 先做业务规则筛选
2. 再做 IV / 相关性 / 模型重要性筛选
3. 需要时将 PCA 作为高维数值聚合特征的实验路线
4. 将因子分析用于研究报告或风险维度解释，而非一开始的生产主模型

---

## 8. 特征工程设计

### 8.1 基础清洗

- 缺失值填补（均值、中位数、众数、特殊值编码）
- 异常值处理（cap / floor / winsorize）
- 类型转换
- 类别变量标准化
- 时间变量转年龄、工龄等可解释指标

### 8.2 比值与结构型特征

建议重点构建：
- `credit_income_ratio = AMT_CREDIT / AMT_INCOME_TOTAL`
- `annuity_income_ratio = AMT_ANNUITY / AMT_INCOME_TOTAL`
- `credit_goods_ratio = AMT_CREDIT / AMT_GOODS_PRICE`
- `employment_age_ratio = DAYS_EMPLOYED / DAYS_BIRTH`
- `children_family_ratio = CNT_CHILDREN / CNT_FAM_MEMBERS`

### 8.3 行为聚合特征

从历史表中提取：
- 历史贷款次数
- 最近贷款活跃度
- 逾期率
- 平均逾期天数
- 最大逾期天数
- 月度风险波动率
- 最近 3/6/12 月风险趋势

### 8.4 分箱与 WOE

适用于评分卡建模，建议对：
- 连续变量做监督式或业务分箱
- 检查坏账率单调性
- 计算 WOE 与 IV
- 保留可解释且稳定的分箱变量

### 8.5 交互特征

建议考虑：
- 收入 × 贷款金额
- 外部评分 × 历史逾期
- 信用卡利用率 × 还款纪律
- 外部债务 × 当前贷款压力

### 8.6 稳定性与波动性特征

行为型数据应重点提取：
- 月度余额波动
- 还款金额波动
- 逾期状态波动
- 最近窗口与长期窗口差异

---

## 9. 建模方案

### 9.1 路线 A：传统评分卡路线

流程：
- 数据清洗
- 特征分箱
- WOE 转换
- Logistic Regression
- 概率校准
- PDO Scorecard 映射

**优点**：
- 可解释性强
- 符合传统信用评分逻辑
- 便于向业务阐释

### 9.2 路线 B：机器学习路线

候选模型：
- LightGBM
- XGBoost
- CatBoost
- Random Forest（可作对照）

**优点**：
- 非线性能力强
- 对缺失值和复杂关系更友好
- 往往能取得更高 AUC / KS

### 9.3 建议：双轨并行

建议同时保留：
- 一个效果更强的树模型作为 performance benchmark
- 一个可解释评分卡模型作为 explainable benchmark

这样既能追求效果，也能保留传统信用风险分析逻辑。

---

## 10. 信用风险评分体系设计

### 10.1 违约概率（PD）输出

模型首先输出客户违约概率，作为风险排序与后续评分的基础。

### 10.2 概率校准

建议使用：
- Platt Scaling
- Isotonic Regression

确保输出概率更接近真实违约率。

### 10.3 评分映射

传统评分卡可采用：
- Base Score
- Base Odds
- PDO（Points to Double the Odds）

将概率映射为直观风险分数。

### 10.4 风险等级与阈值策略

可设计：
- A / B / C / D / E 风险等级
- 自动通过 / 自动拒绝 / 人工审核阈值
- 不同分数段的坏账率与批准率权衡

---

## 11. 模型评估体系

### 11.1 分类效果指标

- ROC-AUC
- KS
- PR-AUC
- Precision / Recall / F1
- Confusion Matrix

### 11.2 风控业务指标

- 分数分箱坏账率
- Lift / Gain
- Top decile risk capture
- cutoff 下的批准率与坏账率平衡

### 11.3 概率质量指标

- Calibration Curve
- Brier Score
- 分箱后预测概率 vs 实际坏账率对比

### 11.4 稳定性指标

- PSI（Population Stability Index）
- 不同时间切片表现
- 训练集 / 验证集 / OOT 稳定性比较

### 11.5 公平性与治理

可对下列分组做基础公平性检查：
- 性别
- 教育水平
- 收入类型
- 区域或组织类型代理变量

建议输出：
- 分组样本量
- 分组坏账率
- 分组通过率
- 分组预测均值

---

## 12. 建议的建模与开发顺序

### Phase 1：最小可用版本

目标：先形成可训练的客户级特征表。

建议优先完成：
- `application_train` 主表清洗
- `bureau` 聚合
- `previous_application` 聚合
- `installments_payments` 聚合
- baseline Logistic Regression / LightGBM

### Phase 2：扩展行为特征

加入：
- `bureau_balance`
- `credit_card_balance`
- `POS_CASH_balance`
- 更丰富的时序窗口特征
- SHAP 与稳定性分析

### Phase 3：评分卡与策略层

完善：
- WOE 分箱
- PDO Scorecard
- 风险等级
- cutoff 策略
- 稳定性监控

---

## 13. Repo 落地建议

建议后续在项目中新增如下模块：

- `src/credit_visable/features/aggregate_bureau.py`
- `src/credit_visable/features/aggregate_previous.py`
- `src/credit_visable/features/aggregate_installments.py`
- `src/credit_visable/features/aggregate_credit_card.py`
- `src/credit_visable/features/aggregate_pos_cash.py`
- `src/credit_visable/features/build_training_table.py`

建议新增 notebook：

- `notebooks/01_eda.ipynb`：主表与目标分布分析
- `notebooks/02_preprocessing.ipynb`：清洗与预处理验证
- `notebooks/03_modeling_baseline.ipynb`：LR / LightGBM baseline
- `notebooks/05_xai_fairness.ipynb`：解释性与公平性检查
- `notebooks/06_scorecard_cutoff.ipynb`：评分卡与阈值分析

---

## 14. 结论

本项目最适合的推进方式不是一开始追求复杂模型，而是先构建**稳定、可复用、可解释的客户级训练集**。在此基础上：

1. 用多表聚合形成高质量训练样本
2. 用规则 + 统计 + 模型的三层筛选选择特征
3. 用双轨模型兼顾效果与解释性
4. 用评分、阈值、稳定性与公平性指标形成完整风险管理链条

这将使项目从“比赛式尝试”升级为“可沉淀、可扩展的信用风险分析工程”。
