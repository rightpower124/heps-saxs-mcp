# SAXS 数据分析工作流（Agent 编排指令）

> 本文件是 heps-saxs MCP 服务器的使用 SOP：任何 Agent 宿主接入本服务器后，按本文件编排工具调用。
> 服务 HEPS 粉光小角站"参考文献方法 + 1D SAXS 数据 → 规范分析结果"。
> 架构与验证门详见设计文档（`.trae/documents/SAXS数据分析Agent-设计文档.md`）。
>
> **文档分层**：本文件全部章节为**跨模型通用纪律**（开工 / 工具 / 流程形态 / 质量门 / 交付物 / 异常）。
> **模型专属流程 = manifest 数据**：各模型的推荐分析序列 / q 窗口 / 诊断建议由其
> `manifest.json` 的 `recommended_workflow` 块声明，Agent 经 `saxs_list_models` 读取编排
> （M2.5-3 能力自发现）——新模型接入**零改动**本文件。

## 0. 开工清单（每次分析会话必做）

0. **读本错题本** `agent_prompts/lessons.md`——历史踩坑与处置纪律（回显核验 / 环境坑 / 措辞规范），新教训当日回写
1. **`saxs_selftest`** —— 服务器健康检查。`ok=true` 才继续；`issues` 逐条原样报告用户，不得带病分析
2. **`saxs_list_models`** —— 模型选型。向用户确认用哪个模型与模式；`validated` 优先，`candidates` 需用户明示同意，`inbox` 收件区禁止用于分析（仅入库流程）
3. **与用户确认四要素**：数据文件路径、`q_unit`（**必须显式指定，不做单位猜测**）、拟合窗口（见第 4 条建议卡流程）、工作目录（用户确认"数据所在文件夹=工作路径"后调 `saxs_set_workdir`，此后缺省产物落 `<工作目录>\HEPS_SAXS_analysis\`；用户未确认不设置，兜底 runs\）
4. **拟合窗口建议卡（2026-11-04 用户裁决）**：初始默认 = 数据**完整范围**，不对所有曲线一刀切用固定窗。拟合前先 `saxs_suggest_window`（全范围读入 → 低 q 直通光 / 高 q 噪声证据 → 建议窗），Agent 呈卡交用户确认，确认后的窗口作为**显式 qmin/qmax** 传入 saxs_fit / saxs_load_curve——**只建议不采纳**（不做静默自动裁剪）

## 1. 工具一览（何时用哪个）

| 工具 | 何时用 |
|---|---|
| `saxs_selftest` | 每次会话开工；改代码重启后必调 |
| `saxs_set_workdir` | 开工四要素确认后设置会话工作目录（261104；此后缺省产物落 `<工作目录>\HEPS_SAXS_analysis\`，不写数据根） |
| `saxs_list_models` | 选模型（拟合前）；inbox 收件区模型仅展示禁拟合 |
| `saxs_suggest_window` | 拟合前出窗口建议卡（261104：全范围数据 → 直通光/噪声证据 → 建议窗，用户确认后显式传参） |
| `saxs_probe_curve` | 读数报错 / 单位存疑 / 格式非常规时：先探测格式卡，ambiguities 与 q_unit 提示交用户裁决，确认后的 fmt 传给 load_curve |
| `saxs_load_curve` | 拟合前数据体检（点数 / 量程 / 丢点 / 非法行）；变体文件传 fmt |
| `saxs_fit` | 主拟合（每模式一次），自动执行 L3 质量门 |
| `saxs_diag_multistart` | 质疑最优解全局性时（多起点是否同收敛） |
| `saxs_diag_relax_bound` | 参数撞界时（放宽边界看敏感性 / 简并） |
| `saxs_diag_param_scan` | 量化参数简并 / 位置依赖（固定网格值逐点重拟合） |
| `saxs_diag_release` | 检验"跨样品固定参数"隐含假设（如膜厚是否漂移） |
| `saxs_diag_f_test` | 任意两个嵌套拟合的显著性（ssr = chi2_dof × dof） |
| `make_plots` | 标准图（每模式一张；布局由插件 `plot_layout` 声明，渲染器通用——M2.5-2 已落地；模式声明 compare 时可传 compare_params 画对照虚线） |
| `saxs_report` | 分析闭环四件套（报告 / 参数表 / 拟合曲线 / result.json） |
| `saxs_export_script` | 用户要微调/复现结果时：从 result.json 生成自包含重放脚本（内置生成时自证对拍，CERTIFIED 证书 = 与源结果对拍通过）；v1 不支持 fixed_keys 非空模式 / calc 闭包 / sasmodels 桥接模型（显式报错） |
| `saxs_doc_extract` | 文献方法学学习前置：PDF/docx/pptx 全文提取为纯文本（落盘 runs/doc_extract/<stem>/，261023 起内化——此前靠临时脚本+用户代跑，Shell 崩溃期完全阻塞；扫描版 PDF 空文本如实报 empty_text，OCR 不在范围） |
| `saxs_gap_detect` | 曲线整备第一步：gap/异常点检测（2 列或 N+1 列逐列）；x_semantic/x_unit 必填不猜，产出段表 + 标记图，按 Tier A/B/C 裁决流定案 |
| `saxs_gap_fill` | 曲线整备第二步：消费定案 gap 表做多段多项式填充（交付 _filled.dat / mask / 标记图 / 整备报告 / result.json）；填充值一律条件值 |
| `saxs_gap_merge` | 曲线整备第三步（打包）：批量逐列填充后把各列拼成与源文件同形的 _filled_book.dat 总表（x/未合并列 token 照抄，合并列 gap 行=填充值），Origin 与原始 book 并排拖入对照；形状/x 不一致报错不合并 |
| `saxs_ruland_bpi2` | 取向分析第二步：series 级统一选峰（自动聚类+确认卡，φ_ref 可覆盖）→ 逐列触达式峰积分（f=0.02 主判+3σ 下限+3 点防抖；线性基线；高斯拟合峰高；高斯尾校正）→ B_π/2 表（deg/rad 双单位 + 依赖填充单元占比披露）+ QC 图；x_unit=deg 必填不猜 |
| `saxs_ruland_beq` | 取向分析第三步：Ruland 变换（x=q²，y=q²B²）线性区 OLS → B_eq（deg/rad 双单位）→ **取向度 f_v = 1−2√(ln2/π³)·B_eq_rad（正式产物，弧度制口径，f_v<0 触发 fv_negative flag）** + R²/LOO/曲率 F 检验/截距诊断（截距 <0 = 系统偏差信号触发复核）；**微孔长度 L 一律不产出**（L=2π/q 原则）；线性区窗口由用户裁决，q_span_narrow 为条件值 flag |

## 2. 单样品流程（通用形态）

```
saxs_load_curve（体检）→ saxs_fit（主拟合）→ make_plots → saxs_report
```

- 拟合窗口 = 建议卡（§0.4）经用户确认后的显式 qmin/qmax（261104 起，不再缺省 [0.3,4.4] 一刀切）
- 各模型的具体编排（步骤取舍 / q 窗口）以其 `recommended_workflow.single_sample` 与
  `q_window_nm` 为准（saxs_list_models 可查）
- 拟合返回的 `quality` 块已自动完成 L3 质量门（残差统计 / 三级分级 / χ² 标记）——**每次必看**
- 返回的 `at_bound` 与 `quality.grade` 直接决定参数表与报告的可靠性分级，无需人工重算

## 3. 多样品 / 模型专属流程（自发现，不在本文档维护）

模型的多样品参数传递与按需诊断序列由其 `recommended_workflow.multi_sample / diagnostics`
声明，Agent 选型后照单编排。通用编排原则：

- **参数传递方向**：先拟合约束强的样品（或无修饰样品）→ 其 params 作为后续样品的 `fixed`
- **对照贯通**：拟合返回的对照参数（若模式支持）传给 make_plots / saxs_report 的
  `compare_params`，图与拟合曲线.csv 同步获得对照曲线
- **诊断按需**：按 manifest `diagnostics` 列表的 when 条件触发，不做默认全跑
- **层级递进缺省（2026-10-09 用户裁决）**：对含层级/级联结构的模型族（gp / unified 族及后续同类模型），数据可能含多层级时缺省从最简起步、逐级递增（1 → 2 → 3 → 4 level；level 越多解释越复杂，奥卡姆剃刀，从单 level 逐步过渡到多 level）；每升一级以嵌套 F 检验 + 可辨识性核查（perr 量级 / multistart 谷结构）裁决是否采纳升阶。各族具体递增序列由其 manifest `recommended_workflow` 声明（unified：lv1→lv2→lv3→main，见其 order_comparison.default_escalation；gp：逐窗单层级对标 level3→level2→level1）
- 示例：liposome_membrane 的 manifest 声明了双样品膜参数传递流程
  （先 nopeg → params 作 peg 的 fixed → nopeg_compare.params 贯通对照）与 4 项按需诊断
- **取向分析链编排（Ruland 独立工具链，非 fit 插件范式）**：曲线整备（盲区/深谷存在时
  gap_detect→fill→merge 出 filled_book）→ `saxs_ruland_bpi2`（φ_ref 确认卡交用户 → 逐列
  QC 图目检）→ `saxs_ruland_beq`（线性区窗口提请用户裁决）；B_eq 结论先给物理量+可靠性
  分级——依赖填充单元的 B 列与 q_span_narrow 均按条件值措辞；触达判据 f/n/m 全系列统一披露
- **提速三件套（2026-09-30 立项）——多列 Ruland 系列推荐用链式编排**（全流程调用数
  32→4，质量门停点不减）：
  1. `saxs_ruland_chain stage='detect'`（逐列检测，内部循环）→ **停点：逐列目检标记图**
     （lessons.md 批量执行纪律：必须逐列 detect→目检→fill，无一键贯通 stage）
  2. `saxs_ruland_chain stage='prepare'`（fill_card_batch 批量填充 + merge 总表；
     gaptable 按约定 runs/<stem>_chain/det/col%02d/ 自动发现）→ **停点：目检整备报告**
  3. `saxs_ruland_chain stage='bpi2'`（filled_book 峰积分，mask 路径自动接线，
     q_unit 必填不猜，q_values 走文件头/显式）→ **停点：目检 B 表+QC 图**
  4. `saxs_ruland_beq scan_qmins=[...]`（线性区扫描内置：qmax=数据最大 q，
     第一个曲率 F 检验 p≥0.05 且非 NaN 的窗自动采纳并直接产出全套交付物于
     `<out>/adopted/`；2026-09-30 用户裁决"规则命中即自动产出"；全部窗不通过或
     F 检验 NaN → 仅出证据链 + adopt_manual 转人工裁决；人工覆盖 = 显式传
     q_window 走单窗模式）
  - 少列/单列或需精细逐列控制时仍可用原单列工具链（saxs_gap_detect/fill/merge
    逐列 + saxs_gap_fill 批量模式 columns+gap_tables 等长对齐）

## 4. 质量门与结论规范（强制，源自已验收 SOP）

- **撞界参数 = 条件值**：数值不得单独进入结论句；结论只依赖稳健参数或模型比较
- **模型显著性以 F 检验为准**（`nopeg_compare.f_test`），不以单参数 sigma 点估计为准——点估计在简并时会误导（实例：drho_peg 点估计 2.7σ，但 tau_peg 与之 corr≈−1，放宽边界后崩到 0.4σ）
- **`chi2_flag=true`（χ²/dof > 3）**：报告用户，先跑诊断工具再下结论
- **归一量**（A / B / 相对电子密度等，插件 `normalized_params` 声明）：跨样品、跨体标不可直接比较
- **定性可靠措辞**：效应量小（如最大相对影响 <5%）且接近检测限时，结论写"定性可靠（log 坐标下肉眼不可辨）"，不做定量断言
- 结论不依赖背景扣除方式时（常数 B 仅吸收高 q 平底），应注明——提升结论稳健性表述

## 5. 交付物与目检纪律

- 每模式一套四件套（`<mode>_分析报告.md / _参数表.csv / _拟合曲线.csv / _result.json`）+ 三联图 PNG
- **分析报告生成规范（2026-10-22 用户确立，适用于未来每一次报告生成）**：
  1. **样品备注**：舍去拟合过程细节、只保留结论，分条展示，每条一行
  2. **模型假设**（refs.assumptions）：同上——舍去细节保留结论，分条每条一行
  3. **公式一律 LaTeX**（`$$...$$` / `$...$`）：分析方法节与算法节禁止纯文本公式（`q^(-dm)` 形态属重大事故）
  4. **"分析方法说明"与"算法与公式"整合为一节**（标题"分析方法说明（模型与算法）"），且该节内部按**五段式架构**组织，禁止公式分散多处展示：
     - **模型段**：本模型完整公式（LaTeX）+ 同族升级/对照版的完整公式也集中在此（所有公式一次展示清楚，不允许两处出现）
     - **各项物理语义段**：语义表逐项列参数与语义；谱系中其他模型的新增参数在本表内单独成行，不另起段
     - **派生量段**：简单物理量的换算公式
     - **拟合协议段**：引擎（最小二乘等）、加权、perr、撞界/perr 爆表判据、诊断工具
     - **备注段**：模型假设指针（§2）、机制对照关系、谱系版本指向等次要内容（每条一行）
  5. 插件 report 钩子（method/algo）按上述规范编写；报告修订时同步修订插件钩子源（下次生成自动合规）
  6. 报告须含**模型谱系说明**：本模型与同族升级版/对照版的关系与 χ² 对比一句话带过（细节指向对应 runs 目录）；**谱系中提到的其他模型必须给出其完整 LaTeX 公式，不允许只有文字描述**（公式展示完全，不能有任何遗漏）
- `拟合曲线.csv` = **绘图全数据单表新增列**：q / I_obs / I_sigma（误差棒）/ I_fit / 各分量与对照列 / **I_background（常数背底独立列，逐点常数 = B；模型声明背景参数 'B' 时由生成器自动增列，2026-10-27 用户裁决）** / rel_residual / z_nm / drho_z_relative（Δρ(z) 剖面，步长 0.02 nm，短列留空）
- **拟合曲线加和强限制（2026-10-27 用户裁决，硬门纪律）**：`I_fit` 必须逐点等于子拟合曲线列的简单加和
  （**Σ分量列 + I_background**，含背底曲线）；由生成器强制执行——插件在模式声明
  `components_sum_to_fit: True` 时，生成器逐点核验相对偏差 >1e-6 即报错**拒绝生成**。
  配套 components 契约细化：分量列为**纯项**（不含常数背景 B，命名 `I_power / I_closedpore / I_sfpeak / ...`），
  背底由 `I_background` 独立列承载；新增/修订插件必须完整分解模型全部项并声明该开关
  （存量插件未声明开关的按旧口径兼容，修订时一并迁移）
- **重放脚本导出（用户要微调/复现时）**：`saxs_export_script(result_json=<mode>_result.json)` 生成
  `replay_<mode>.py` + 实跑产物（replay_*_result.json / _curves.csv / _fit.png）于同目录；
  脚本头部 CERTIFIED 证书 = 生成时已实跑对拍通过（双判据语义见工具描述与 lessons.md C 类
  2026-10-07 条）；导出后 Grep 核验证书块落盘，回执仅作参考
- 产出目录：已设会话工作目录时 `<工作目录>\HEPS_SAXS_analysis\<案例>\`（261104 工作目录制），
  未设置时兜底 `runs\<案例>_<里程碑标签>\`；**产出由用户 IDE 目检后由用户手动清空
  （AI 不执行删除）**——runs 单轮生命周期见设计文档 §9.2
- 重构 / 换引擎后强制回归：与 `baseline\<案例>\regression_baseline.json` **逐位一致**

## 6. 异常处理

- `selftest ok=false` → 停止分析，逐条报告 issues，等用户处理
- 工具报错 → 原样转述错误信息，同一错误重试不超过 2 次，之后升级人工
- 回显可疑（路径 / 数值与上下文矛盾）→ 明确声明"疑似显示层污染"，请用户 IDE 目检核实；**编辑与写入通常正确落盘，被伪造的是回显**
- 撞界 / 简并 / 低显著性 → 按 §4 规范措辞，必要时升级人工把关
- χ² 合理但参数物理意义异常 → 优先怀疑模型适配性，用 diag 工具取证后再向用户报告

## 7. 新模型接入（不在本工作流内展开）

模型 = 插件三件套：`model.py`（物理代码）+ `plugin.py`（适配层：参数表 / calc / profile / compare / report 钩子 + `plot_layout` 布局声明）+ `manifest.json`（元数据：modes / labels / refs / **recommended_workflow**）。
必须过验证门（L1 合成自检 + L2 文献对照 + 人工审核签字）后入 `model_library/validated/`。
manifest 声明 `recommended_workflow` 后，Agent 即可零改动自动编排该模型的流程
（未声明时 saxs_selftest 出 warning 提醒）；流程详见设计文档 §3.4（验证门）与 §10（M2.5）；
新文献模型自动化属 M3 里程碑。

**sasmodels 桥接通道（M2.5-4）**：官方库基础形状（sphere / core_shell_sphere / cylinder / ellipsoid / hollow_cylinder / core_shell_cylinder，2026-10-21 扩容 +2）不在 model_library 建目录，经 `sasmodels_bridge` 直调通道自动出现在 saxs_list_models（zone=`sasmodels_bridge`，status=`external`），fit / make_plots / saxs_report / selftest 全部开箱即用。注意：①模式名固定 `main`；②几何参数保持 sasmodels 原生单位（长度 Å），数据 q 引擎内自动换算；③scale/background 映射为 A/B（归一量）；④external 状态 = 未经本站验证门，报告无模型专属结论，定量结论优先用已过验证门的站内模型；⑤hollow_cylinder 语义：radius=内腔半径、thickness=壁厚、scale=管壁体积分数（core_shell_cylinder 按总体积归一，两模型 A 口径不同不可跨模型比较）。

## 8. 外挂工具边界（Origin MCP，2026-10-25 起生效）

工作区挂载第二台 MCP 服务器 **origin-pro**（leima-max/origin-pro-mcp-skill，23 工具，经 pywin32 COM 驱动本机 Origin 2027 学生版），定位为**纯出图外挂**。与 heps-saxs 的关系纪律如下：

1. **两工具完全解耦、互不依附**：heps-saxs 负责 SAXS 数据解析与拟合分析；Origin MCP 仅负责出图。二者是两个彼此完全不耦合的独立任务，任何一侧不得侵入另一侧的流程。
2. **调用时序门槛（强制）**：仅当数据**完全解析彻底、没有任何疑惑之后**，才可作为独立出图任务调用 Origin MCP；分析未完成、质量门未过、结论仍有疑问时，一律不得触发出图工具。
3. **拟合工具禁用**：Origin MCP 的 `curve_fit` / `list_fitting_functions` 禁止用于任何正式 SAXS 拟合——其结果无 F 检验、无撞界检查、无质量门，不满足本工作流 §4 结论规范；正式拟合 100% 走 `saxs_*` 系列。
4. **典型合法用法**：将 `runs/<task>/` 下已完成分析并经用户目检确认的结果 CSV 导入 Origin，做发表级图形（`create_graph` + `apply_publication_style` + `export_graph`）；`run_labtalk` 可作样式兜底。
5. **环境事实**：服务器代码 `HEPS_SAXS_AGENT\origin-pro-mcp-skill`（与 saxs_mcp 平级）；运行环境 conda env `saxsagent`（入口 `E:\CondaEnvs\saxsagent\Scripts\origin-pro-mcp.exe`）；调用前 Origin 2027 必须已启动且无阻塞弹窗。
