# SAXS 数据分析 Agent — 能力报告

> **版本**：2026-11-04（部署优化：会话工作目录 + 窗口建议卡 + model_inbox 收件区，工具面 22）
> **维护纪律**：工具面 / 模型库 / 工作流变化时同步更新本报告
> **配套文档**：用户任务入口 = `prompt_library.md`（16 关键词）；操作 SOP = `saxs_workflow.md`；错题本 = `lessons.md`；设计主线 = `.trae/documents/SAXS数据分析Agent-设计文档.md`；Ruland 规格卡 = `.trae/documents/Ruland取向度分析MCP工具-算法学习与开发规划.md`（v1.2）

## 1. Agent 定位

HEPS 粉光小角站专属 1D SAXS 数据分析专员：

- 所有分析 100% 通过 `heps-saxs` MCP 服务器的 `saxs_*` 工具执行，绝不手写替代分析实现；
- 数据路径、单位等关键信息执行"宁可报错不要猜错"——四要素确认卡（路径 / x 或 q 单位 / 分析窗口=suggest_window 建议卡 / 工作目录）显式确认后才动手；
- 拟合窗口初始默认 = 数据完整范围，`saxs_suggest_window` 按低 q 直通光 / 高 q 噪声证据出建议卡，用户确认后显式传参（261104，不对所有曲线一刀切）；
- 会话工作目录确认后，缺省产物落 `<工作目录>\HEPS_SAXS_analysis\` 专用子目录，不写用户数据根（261104，多用户天然隔离）；未设置兜底 runs\；
- 所有结论先给物理量 + 可靠性分级（稳健 / 条件值 / 归一量），再补技术细节；
- 工具回执仅作参考，关键数值一律读磁盘结果文件核验；产出写入 `runs\` 任务子目录（AI 绝不删除，runs 清空由用户手动执行）；
- 超能力范围请求（2D SAXS / GISAXS / 库外自定义模型）如实告知边界并引导入库流程，绝不现场编造模型。

## 2. 能力地图（五节）

| 能力节 | 内容 | 状态 |
|---|---|---|
| 数据读取与格式适配 | 多编码（UTF-8/GBK/UTF-16）/多分隔符/2 列~N+1 列/文件内嵌元数据利用（`q(A-1)` 列头、EDF 头自动提示，经用户确认生效） | ✅ |
| 模型拟合 | 模型库拟合 + L3 质量门 + 可靠性分级 | ✅ |
| 按需诊断 | 多起点 / 放宽边界 / 参数扫描 / 参数释放 / F 检验 | ✅ |
| 曲线整备 | gap 检测 → 三级裁决 → 填充 → Origin 合并总表 | ✅（M3pre-2） |
| 取向分析 | Ruland 两工具链：B_π/2 触达式峰积分 → B_eq（第二个模型） | ✅（260930） |
| 可视化与报告 | DSL 配置化绘图 + 闭环交付物 + 重放脚本导出（结果可复现、微调权还给用户） | ✅ |

## 3. MCP 工具清单（22 个）

### 3.1 基础设施（5）

| 工具 | 用途 |
|---|---|
| `saxs_selftest` | 服务器自检：模型库契约（manifest 键完整 / 插件契约）+ 拟合引擎 + 整备引擎合成冒烟；改代码重启后必跑 |
| `saxs_set_workdir` | 会话级工作目录（261104）：设后全部缺省产物落 `<工作目录>\HEPS_SAXS_analysis\` 专用子目录；未设置兜底 runs\ |
| `saxs_probe_curve` | 只读格式探测：编码 / 行尾 / 分隔符 / 列数 / 表头结构 / 内嵌单位提示 / 双块与歧义清单，返回格式卡（歧义项由用户裁决） |
| `saxs_suggest_window` | 窗口建议卡（261104）：全范围读入 → 低 q 直通光（平台翼 / 陡于 Porod）与高 q 噪声证据（σ/I / zigzag 震荡）→ 建议窗 + 证据 flag；建议+确认纪律，只建议不采纳 |
| `saxs_load_curve` | 1D 曲线读取：常规 3 列 q I σ 路径（q_unit 必须显式，A⁻¹ 自动×10 转 nm⁻¹）+ fmt 变体路径（用户确认后的格式卡直传）；qmin/qmax=None = 全范围通道 |

### 3.2 拟合分析链（4）

| 工具 | 用途 |
|---|---|
| `saxs_list_models` | 模型库列表：名称 / 版本 / 验证状态 / 可用模式 + 各模型 recommended_workflow（q 窗口 / 单样品 / 多样品参数传递 / 诊断建议） |
| `saxs_fit` | 通用拟合引擎 fit(model, data)：返回 params / perr / chi2_dof / at_bound / derived / quality（L3 质量门：撞界→归一量→稳健分级 + χ² 关注标记）/ nopeg_compare（模式支持对照时） |
| `make_plots` | 纯渲染器：布局全部来自模型插件 plot_layout DSL 声明（panel 原语 fit/residual/profile/curve），未声明即报错——零预设模型 |
| `saxs_report` | 闭环交付物 4 文件：参数表 / 图 / 拟合曲线.csv（图中每条曲线均列出新列）/ 报告 |

### 3.3 按需诊断链（5）

| 工具 | 用途 | 回答的问题 |
|---|---|---|
| `saxs_diag_multistart` | 多起点重拟合 | 全局最优是否唯一？ |
| `saxs_diag_relax_bound` | 放宽某参数边界重拟合 + 相关系数 | 撞界参数敏感性？简并方向？ |
| `saxs_diag_param_scan` | 固定参数网格扫描（warm start） | 宽度-幅度简并定量？ |
| `saxs_diag_release` | 放开跨样品固定参数重拟合 | 固定假设是否漂移？ |
| `saxs_diag_f_test` | 任意嵌套拟合 F 检验 | 加这个参数显著吗？ |

### 3.4 曲线整备链（3，M3pre-2）

| 工具 | 用途 |
|---|---|
| `saxs_gap_detect` | gap/异常点检测：4 码形态学（深谷核心 / 渐变肩部 / 孤立异常点 / 宽缓阴影候选）+ 三级裁决流（Tier A 全自动定案 / Tier B Agent 读图主判 / Tier C 用户裁决）；交付段表 gaptable.json + 标记图 |
| `saxs_gap_fill` | 多段多项式填充：梯级收窗（3×→2× 规则窗 + 不对称收窗重试）/ 连续约束梯级降档 / 包络门 / 相对拟合误差闸；交付 5 件套（_filled.dat / _mask.csv / _标记图 / _整备报告.md / _result.json）；填充值一律按条件值对待并随报告披露；**批量模式**（提速三件套 T-B）：columns + gap_tables 等长对齐一次调用逐列填充（单列路径逐位不变），任一列失败硬报错（P7） |
| `saxs_gap_merge` | 批量合并总表：各列填充产物拼成与源文件同形的 `<原名>_filled_book.dat`（x 与未合并列 token 逐字照抄、合并列 gap 行=填充值），Origin"原始 book + 填充 book"并排拖入对照；行数/列宽/x 不一致报错不合并 |

**适用数据轴**：I-q（A⁻¹ / nm⁻¹ / pixel 探测器半径轴）与 I-phi（deg）；2 列（x I）或 N+1 列（x I1..In）逐列独立处理。

### 3.5 取向分析链（3，Ruland 独立工具链，260930）

| 工具 | 用途 |
|---|---|
| `saxs_ruland_bpi2` | series 级统一选峰（逐列峰检测 → 跨列圆周聚类，φ_ref 确认卡可覆盖）→ 逐列触达式峰积分（f=0.02 主判 + 3σ_local 下限 + 3 点防抖；锚区约束的线性基线；峰区高斯拟合峰高；高斯尾校正 ×1.005）→ B_π/2 表（deg/rad 双单位、依赖填充单元占比逐列披露）+ 逐列 QC 图 + B(q) 汇总图 |
| `saxs_ruland_beq` | Ruland 变换（x=q²，y=q²B²）线性区 OLS → B_eq（deg/rad 双单位）→ **取向度 f_v = 1−2√(ln2/π³)·B_eq_rad（正式产物，弧度制口径，分级继承 B_eq）** + R² / leave-one-out / 残差曲率 F 检验 / 截距诊断（截距 <0 = 系统偏差信号触发复核）；**微孔长度 L 一律不产出**（L=2π/q 原则）；**扫描模式**（提速三件套 T-A）：scan_qmins 逐窗扫描（qmax=数据最大 q），第一个曲率 F 检验 p≥0.05 且非 NaN 的窗自动采纳产出全套交付物（<out>/adopted/，2026-09-30 用户裁决"规则命中即自动产出"），全部窗不通过 → adopt_manual 转人工裁决；人工覆盖 = 显式传 q_window 单窗模式；q_span_narrow 条件值 flag |
| `saxs_ruland_chain` | 取向全流程链式编排（提速三件套 T-C）：三阶段 thin dispatcher——stage='detect' 逐列 gap 检测（停点=逐列目检）/ stage='prepare' 批量填充+merge 总表（gaptable 约定路径自动发现）/ stage='bpi2' filled_book 峰积分（mask 自动接线）；beq 终点走 saxs_ruland_beq scan_qmins；**不提供 detect→fill 一键贯通**（lessons.md 批量执行纪律），全流程 MCP 调用数 32→4 |

**方法学口径**：触达式截断积分 = 只积到"剩余信号≈噪声地板"（与 Origin 默认峰积分同款的标准实践），截断损失由 erf 因子解析补回（f 梯级取证总漏损 ≲1%）；"拖到水平基线"的基线偏置 ∝ 窗宽，YR8 宽窗对照实测 +14.1% 全为基线重分配伪影——详见规格卡 §5.4。算法来源：Thünemann & Ruland, Macromolecules 2000, 33, 1848（记号 s→q 转写）。

### 3.6 结果导出与复现链（1，261007）

| 工具 | 用途 |
|---|---|
| `saxs_export_script` | 从 result.json 生成**自包含重放脚本**（单文件，仅依赖 numpy/scipy/matplotlib）：内联模型源码（sha256 存证）+ 逐位镜像数据读取与拟合引擎，用户改窗口/初值/边界后可独立重跑——"优化权还给用户"。**内置生成时自证**：生成器在本环境实跑脚本与源 result.json 对拍，双判据通过才发 CERTIFIED 证书（tier1 硬判据：χ²/dof、dof、at_bound、n_points 逐位全等；tier2 容差判据：params/perr 逐位计数 + 1e-12 相对 ULP 容差——撞界参数存在服务器进程 vs 子进程末位 ULP 噪声，见 lessons.md C 类 2026-10-07 条）。v1 限制（显式报错不静默降级）：fixed_keys 非空模式、calc 闭包、sasmodels 桥接模型不支持导出；证书声明"改参数不改公式，公式改动走插件重新生成" |

## 4. 模型库

| 模型 | 验证状态 | 模式 | 说明 |
|---|---|---|---|
| `liposome_membrane` | validated（1 号人工审核 PASS） | nopeg / peg | 脂质体双分子层膜剖面模型；peg 模式跨样品固定膜参数 + 无 PEG 对照分析 |
| `alcu_precipitate` | validated（2 号，2026-10-01 终审入库） | main / model1 | Al-Cu 盘状析出相（Tsao 2013）：main=Porod+盘 P(Q)（Model III），model1=盘×PY 硬球 S(Q) |
| `gel_network` | validated（3 号，2026-10-02 入库） | main / porod | 凝胶网络（Wisotzki 2017）：main=关联长度模型 OZ，porod=质量分形幂律 |
| `hardcarbon_ts` | validated（4 号，2026-10-02 入库） | main | 硬碳闭孔 Teubner-Strey 四组分（Adv. Mater. 2025）：幂律+TS 微孔+背景 |
| `unified_power_Rg` | validated（5 号，2026-10-06 入库） | main / full | Beaucage 统一四 level 级联（Chaney 2025 锚）：main=小角区统一拟合，full=全谱+三 Lorentz 峰。**多层级数据主模型** |
| `guinier_porod` | validated（6 号，2026-10-06 入库） | level1 / level2 / level3 | 广义 Guinier-Porod 三参数（Hammouda 2010），tiered 三窗分 level。**unified 的辅助验证与简化替代** |
| sphere / core_shell_sphere / cylinder / ellipsoid / hollow_cylinder / core_shell_cylinder | external（sasmodels 1.1.0 桥接；2026-10-21 扩容 +2） | main | 标准形状开箱即用；参数原生 Å、q 单位内部换算；scale/background→A/B 映射。注意：hollow_cylinder 的 radius=内腔半径、scale=管壁体积分数；core_shell_cylinder 的 scale=总体积（核+壳）分数，两模型 A 口径不同不可跨模型比较 |

**双模型使用策略（2026-10-06 用户确立，两 manifest recommended_workflow.purpose 已声明）**：多层级数据分析以 unified_power_Rg 为主模型（全谱一次拟合、级间自洽）；guinier_porod 作辅助验证——gp 对应窗结果与 unified 基本一致时优先采用 gp（参数更少更简约）；仅 unified 拟合效果好时（层级重叠连续级联，如 Chaney 2025 数据）只用 unified 并披露 gp 验证不适配；gp 独立适用场景 = 结构可分离、窗内仅单一层级（孤立膝）数据集。

新模型入库走开发端流程（manifest 插件制）：文献定义 + 验收数据 + 用户人工审核签字 → candidate → validated；绝不现场编造。
**inbox 收件区（261104）**：待入库模型三件套放入根目录 `model_inbox\<模型名>\` 即被 `saxs_list_models` 展示（zone=inbox，status=inbox，未过验证门）；`saxs_fit` / `saxs_diag_*` 拒收，selftest 仅做契约静态检查不跑冒烟；晋级 validated 仍走验证门 + 人工签字。

## 5. 质量与纪律体系

- **L3 质量门**：撞界参数只作条件值呈现；参数显著性以 F 检验为准；χ²/dof > 3 先诊断后结论；
- **条件值纪律**：整备填充值 = 数学插值而非信号恢复，下游使用必须披露填充占比（写入表头与整备报告）；
- **三级裁决流**：深谷/孤立点全自动（Tier A）、肩部与干净阴影 Agent 目检主判（Tier B）、阴影与真实峰重叠（Tier C）提请用户；
- **runs 单轮生命周期**：AI 生成 → Grep 核验 → 用户 IDE 目检 → 用户手动清空 runs → 下一轮从头生成；签字产物长期归档 runs_archive；
- **错题本机制**：`lessons.md`（A 环境 / B 回显核验 / C 代码 / D 文档 / E 流程目检），新教训当日追加，每次开工必读。

## 6. 用户入口（prompt_library.md，16 关键词）

```text
【完整分析】 E:\data\liposome_A.txt，q_unit=nm^-1，liposome_membrane 的 nopeg 模式
【曲线整备】 E:\data\1.dat，q/pixel，列 1
【批量曲线整备】 E:\data\YR8_ruland.txt，phi/deg，全部 16 列，最后合并总表
【取向分析】 E:\data\YR8_..._filled_book.dat，phi/deg，q_unit=nm^-1
```

## 7. 能力边界（如实声明）

- 不支持 2D SAXS / GISAXS 分析（方位角积分曲线集由站内 pyFAI 链路提供，取向分析吃 1D I(q_c, φ) 文件）；
- 库外自定义物理模型需走入库流程（绝不现场编造）；
- 取向分析 v2 候选（未开放）：σ_B 误差传播（Q8）、Hermans 取向参数 f 换算（Q7，晶体取向量，与孔隙取向度 f_v 为不同物理量）。取向度 f_v（论文公式 3）已于 2026-09-30 固化为 `saxs_ruland_beq` 正式产物。
