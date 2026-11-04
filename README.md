# heps-saxs-mcp

**HEPS 粉光小角站 1D SAXS 数据分析 MCP 服务器**：把"参考文献方法 + 1D SAXS 数据 →
规范分析结果"固化为可对话调用的 MCP 工具集（22 个 `saxs_*` / `make_plots` 工具），
供支持 Model Context Protocol 的 AI 客户端（如 TRAE、Claude Desktop 等）挂载使用。

设计思想：**分析 100% 走 MCP 工具、关键参数"宁可报错不要猜错"、每个拟合自动过
L3 质量门、结论强制带可靠性分级**——让 AI 编排分析但不替用户做物理裁决。

## 功能总览

| 链路 | 工具 | 说明 |
|---|---|---|
| 基础设施（5） | `saxs_selftest` `saxs_set_workdir` `saxs_probe_curve` `saxs_suggest_window` `saxs_load_curve` | 自检 / 会话工作目录 / 格式探测卡 / 窗口建议卡 / 数据体检 |
| 拟合分析（4） | `saxs_list_models` `saxs_fit` `make_plots` `saxs_report` | 模型库自发现 / 通用拟合引擎（自动 L3 质量门）/ DSL 渲染 / 四件套报告 |
| 按需诊断（5） | `saxs_diag_multistart` `relax_bound` `param_scan` `release` `f_test` | 全局性 / 撞界敏感性 / 简并扫描 / 跨样品固定参数检验 / 嵌套 F 检验 |
| 曲线整备（3） | `saxs_gap_detect` `saxs_gap_fill` `saxs_gap_merge` | 盲区/深谷检测（三级裁决流）→ 多段多项式填充 → Origin 对照总表 |
| 取向分析（3） | `saxs_ruland_bpi2` `saxs_ruland_beq` `saxs_ruland_chain` | Ruland 取向度：B_π/2 峰积分 → B_eq / f_v（链式编排 32→4 次调用） |
| 导出复现（2） | `saxs_export_script` `saxs_doc_extract` | 重放脚本（自证对拍 CERTIFIED）/ 文献全文提取 |

**核心纪律**（内置在工具契约里，不靠自觉）：

- `q_unit` 必须显式（`A^-1` / `nm^-1`），系统不做单位猜测；
- 拟合窗口初始默认 = 数据完整范围，`saxs_suggest_window` 按低 q 直通光 / 高 q 噪声
  证据出建议卡，**用户确认后**才作为显式窗口传入；
- 每次拟合自动执行 L3 质量门：撞界 / perr 爆表 / 归一量三级分级 + χ²/dof 关注标记；
- 确认工作目录后所有产物落 `<工作目录>/HEPS_SAXS_analysis/`，不写用户数据根；
- 数据类文件 >1MB 拒读（防误传 2D 原始帧），`saxs_probe_curve` 诊断通道不设限；
- 模型库三区管理：`validated`（过验证门）/ `candidates`（需用户明示同意）/
  `model_inbox`（待入库收件区，拟合工具拒收）。

## 安装

### 1. conda 环境（一次性）

```bash
conda create -p E:\CondaEnvs\saxsagent python=3.11 -y
E:\CondaEnvs\saxsagent\python.exe -m pip install -r requirements.txt
# 可选：官方基础形状桥接通道（sphere / cylinder / core_shell_* 等 8 个 external 模型）
E:\CondaEnvs\saxsagent\python.exe -m pip install "sasmodels>=1.1"
```

依赖：`mcp>=1.0,<2`（**必须 <2**，2.x API 不兼容）、`numpy`、`scipy`、`matplotlib`。

### 2. 配置 mcp.json（三处占位符）

复制仓库根 `mcp.json` 到你的 MCP 客户端配置（TRAE 用户级
`%APPDATA%\Trae CN\User\mcp.json` 或项目级 `.trae/mcp.json`），替换三处：

| 占位符 | 替换为 | 示例 |
|---|---|---|
| `PYTHON_PATH_PLACEHOLDER` | conda 环境 python.exe 全路径 | `E:/CondaEnvs/saxsagent/python.exe` |
| `REPO_ROOT_PLACEHOLDER` | 本仓库 clone 后的绝对路径 | `E:/HEPS_BB_SAXS/HEPS_SAXS_AGENT` |
| `CONDA_ENV_PLACEHOLDER` | conda 环境根路径（同上，不带 python.exe） | `E:/CondaEnvs/saxsagent` |

> `START_MCP_TIMEOUT_MS=60000` 必须保留——conda 环境启动慢于客户端默认超时，
> 否则会出现"面板绿勾但会话内工具不可见"的静默失败。
> 也可不用 mcp.json，直接运行 `saxs_mcp\run_server.bat`（读取 `SAXSAGENT_ENV`
> 环境变量，缺省回退 `E:\CondaEnvs\saxsagent`）。

### 3. 验证

重启客户端后对话调用 `saxs_selftest`，返回 `ok=true` 即部署成功。

## 快速上手

数据准备：1D SAXS 曲线（3 列 `q I σ` 空白分隔，或先 `saxs_probe_curve` 探测
变体格式）。对话示例：

```text
【完整分析】 E:\data\liposome_A.txt，q_unit=nm^-1，liposome_membrane 的 nopeg 模式
【曲线整备】 E:\data\1.dat，x_semantic=q，x_unit=pixel，列 1
【取向分析】 E:\data\YR8_filled_book.dat，phi/deg，q_unit=nm^-1
【多起点】  上轮 nopeg 拟合的 tau_t，验证全局性
```

Agent 会先回**四要素确认卡**（数据路径 / q_unit / 拟合窗口=建议卡 / 工作目录），
确认后自动编排：体检 → 窗口建议 → 拟合（自动质量门）→ 三联图 → 四件套报告
（`<mode>_分析报告.md / _参数表.csv / _拟合曲线.csv / _result.json`）。

## 模型库（首发 13 个插件模型 + 8 个 sasmodels 桥接）

| 模型 | 状态 | 模式 | 物理内涵（文献锚见各 manifest.json） |
|---|---|---|---|
| liposome_membrane | validated | nopeg / peg | 脂质体双分子层膜剖面（Schilt 2016），双样品膜参数传递 |
| alcu_precipitate | validated | main / model1 | Al-Cu 盘状析出相（Tsao 2013） |
| gel_network | validated | main / porod | 凝胶网络 OZ 关联长度（Wisotzki 2017） |
| hardcarbon_ts | validated | main | 硬碳闭孔 Teubner-Strey 四组分 |
| unified_power_Rg | validated | main / full / lv1–lv3 | Beaucage 统一四 level 级联（多层级数据主模型） |
| guinier_porod | validated | level1–3 | 广义 Guinier-Porod（unified 的辅助验证） |
| fuzzy_sphere_extended | validated | main / sq | 模糊球 + OZ 关联（Henderson） |
| phase_separation_bp / hardcarbon_closedpore | validated | gauss / lorentz 等 | 相分离峰拟合 / 闭孔+表面分形 |
| 4 个 candidates | **未过验证门** | — | 相分离 2seg/ruland/exp、hardcarbon_closedpore_sf——正式分析需用户明示同意 |
| sphere / core_shell_sphere / cylinder / ellipsoid / hollow_cylinder / core_shell_cylinder / broad_peak / teubner_strey | external | main | sasmodels 1.1.0 桥接，开箱即用（装 sasmodels 自动出现） |

新模型走入库流程：三件套放入 `model_inbox/<模型名>/` → 验证门 → validated，
详见 [model_inbox/README.md](model_inbox/README.md)。

## 测试与回归

- `saxs_selftest`：整备/取向/拟合引擎合成冒烟 + 全模型契约检查（21 模型全模式）；
- `test_data/`：脂质体双样品验收数据、7 维格式变体夹具、Ruland L1 套件；
- `baseline/`：回归基准——重构/换引擎后与 `regression_baseline.json` **逐位一致**
  （跨进程单参数末位 ULP 噪声按 1e-12 相对容差裁决，见 baseline 内说明）。

## 项目结构

```
├── saxs_mcp/            # MCP 服务器（14 个模块，无模型知识，插件制加载）
├── model_library/       # validated / candidates 模型插件
├── model_inbox/         # 待入库收件区
├── agent_prompts/       # Agent 编排 SOP / 用户关键词库 / 能力报告 / 错题本
├── test_data/           # 验收与回归测试数据
├── baseline/            # 回归基准
├── mcp.json             # MCP 客户端接入模板（三占位符）
└── requirements.txt
```

## 边界声明

- 仅支持 1D SAXS（2D / GISAXS 不在范围；取向分析吃站内 pyFAI 链路积好的方位角积分曲线）；
- 库外自定义物理模型需走入库流程，工具不现场编造模型；
- 扫描版 PDF 不做 OCR（`saxs_doc_extract` 如实报 empty_text）。

## 致谢

方法学锚定文献：Schilt 2016（脂质体膜）、Tsao 2013（Al-Cu 析出）、Beaucage
（unified）、Hammouda 2010（Guinier-Porod）、Thünemann & Ruland 2000（取向分析）、
Teubner-Strey、sasmodels/SASview 社区——完整文献锚见各模型 manifest.json。

## 许可证

[MIT](LICENSE) © 2026 chenrc
