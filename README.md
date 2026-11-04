English | [简体中文](README.zh-CN.md)

# heps-saxs-mcp

**MCP server for 1D SAXS data analysis at the HEPS pink-beam small-angle scattering beamline.**
It turns "reference-literature methods + 1D SAXS data → standardized analysis results" into a
conversational tool suite of 22 `saxs_*` / `make_plots` MCP tools, mountable by any
Model Context Protocol client (TRAE, Claude Desktop, etc.).

Design principles: **100% of the analysis runs through MCP tools; critical parameters follow
"better an error than a guess"; every fit automatically passes the L3 quality gate; every
conclusion carries a mandatory reliability grade** — the AI orchestrates the analysis but
never makes the physics calls for the user.

## Feature Overview

| Chain | Tools | Description |
|---|---|---|
| Infrastructure (5) | `saxs_selftest` `saxs_set_workdir` `saxs_probe_curve` `saxs_suggest_window` `saxs_load_curve` | Self-test / session work directory / format probe card / window suggestion card / data health check |
| Fitting & analysis (4) | `saxs_list_models` `saxs_fit` `make_plots` `saxs_report` | Self-describing model library / generic fitting engine (auto L3 quality gate) / DSL renderer / 4-file report package |
| On-demand diagnostics (5) | `saxs_diag_multistart` `relax_bound` `param_scan` `release` `f_test` | Global optimum / bound sensitivity / degeneracy scan / cross-sample fixed-parameter test / nested F-test |
| Curve restoration (3) | `saxs_gap_detect` `saxs_gap_fill` `saxs_gap_merge` | Blind-zone/deep-valley detection (3-tier adjudication) → multi-segment polynomial filling → Origin comparison book |
| Orientation analysis (3) | `saxs_ruland_bpi2` `saxs_ruland_beq` `saxs_ruland_chain` | Ruland orientation: B_π/2 peak integration → B_eq / f_v (chained orchestration, 32→4 calls) |
| Export & reproducibility (2) | `saxs_export_script` `saxs_doc_extract` | Replay script (self-certified against source results) / full-text document extraction |

**Core disciplines** (built into tool contracts, not left to good intentions):

- `q_unit` must be explicit (`A^-1` / `nm^-1`) — the system never guesses units;
- The default fit window is the **full data range**; `saxs_suggest_window` produces an
  evidence card (low-q direct-beam / high-q noise flags) and the window is used **only
  after user confirmation**;
- Every fit automatically runs the L3 quality gate: bound-hit / exploded-perr /
  normalized three-tier grading + χ²/dof attention flag;
- Once a work directory is confirmed, all outputs go to
  `<workdir>/HEPS_SAXS_analysis/` — never the user's data root;
- Data files larger than 1 MB are rejected on ingest (protects the shared host from
  accidental 2D frames); the `saxs_probe_curve` diagnostic channel is unlimited;
- Three-zone model management: `validated` (passed verification gate) /
  `candidates` (explicit user consent required) /
  `model_inbox` (submission inbox, rejected by fitting tools).

## Installation

### 1. conda environment (one-time)

```bash
conda create -p E:\CondaEnvs\saxsagent python=3.11 -y
E:\CondaEnvs\saxsagent\python.exe -m pip install -r requirements.txt
# Optional: official shape bridge (sphere / cylinder / core_shell_* — 8 external models)
E:\CondaEnvs\saxsagent\python.exe -m pip install "sasmodels>=1.1"
```

Dependencies: `mcp>=1.0,<2` (**must stay <2**; 2.x renamed FastMCP and is incompatible),
`numpy`, `scipy`, `matplotlib`.

### 2. Configure mcp.json (three placeholders)

Copy the repo-root `mcp.json` into your MCP client configuration (TRAE user-level
`%APPDATA%\Trae CN\User\mcp.json` or project-level `.trae/mcp.json`) and replace:

| Placeholder | Replace with | Example |
|---|---|---|
| `PYTHON_PATH_PLACEHOLDER` | Full path to the conda env python.exe | `E:/CondaEnvs/saxsagent/python.exe` |
| `REPO_ROOT_PLACEHOLDER` | Absolute path of this cloned repo | `E:/HEPS_BB_SAXS/HEPS_SAXS_AGENT` |
| `CONDA_ENV_PLACEHOLDER` | Conda env root (same path, without python.exe) | `E:/CondaEnvs/saxsagent` |

> Keep `START_MCP_TIMEOUT_MS=60000` — conda environments start slower than the client
> default timeout; without it you get the silent "green check in the panel but no tools
> in the session" failure.
> Alternatively skip mcp.json and run `saxs_mcp\run_server.bat` directly (reads the
> `SAXSAGENT_ENV` environment variable, falling back to `E:\CondaEnvs\saxsagent`).

### 3. Verify

Restart the client and invoke `saxs_selftest` in a conversation — `ok=true` means deployed.

## Quick Start

Data preparation: 1D SAXS curve (3 columns `q I σ`, whitespace-separated; use
`saxs_probe_curve` first for variant formats). Conversation examples:

```text
[Full analysis]  E:\data\liposome_A.txt, q_unit=nm^-1, liposome_membrane nopeg mode
[Curve restore]  E:\data\1.dat, x_semantic=q, x_unit=pixel, column 1
[Orientation]    E:\data\YR8_filled_book.dat, phi/deg, q_unit=nm^-1
[Multi-start]    verify global optimum of last nopeg fit, tau_t
```

The agent first replies with a **four-element confirmation card** (data path / q_unit /
fit window = suggestion card / work directory); after confirmation it orchestrates:
health check → window suggestion → fit (auto quality gate) → triple plot → 4-file report
package (`<mode>_分析报告.md / _参数表.csv / _拟合曲线.csv / _result.json`).

## Model Library (13 plugin models + 8 sasmodels bridges at release)

| Model | Status | Modes | Physics (literature anchors in each manifest.json) |
|---|---|---|---|
| liposome_membrane | validated | nopeg / peg | Lipid-bilayer membrane profile (Schilt 2016), cross-sample membrane-parameter transfer |
| alcu_precipitate | validated | main / model1 | Al-Cu disc precipitates (Tsao 2013) |
| gel_network | validated | main / porod | Gel-network OZ correlation length (Wisotzki 2017) |
| hardcarbon_ts | validated | main | Hard-carbon closed-pore Teubner–Strey four-component |
| unified_power_Rg | validated | main / full / lv1–lv3 | Beaucage unified four-level cascade (primary model for multi-level data) |
| guinier_porod | validated | level1–3 | Generalized Guinier–Porod (cross-check of unified) |
| fuzzy_sphere_extended | validated | main / sq | Fuzzy sphere + OZ correlation (Henderson) |
| phase_separation_bp / hardcarbon_closedpore | validated | gauss / lorentz etc. | Phase-separation peak fitting / closed-pore + surface fractal |
| 4 candidates | **gate not passed** | — | phase_separation 2seg/ruland/exp, hardcarbon_closedpore_sf — explicit user consent required for formal analysis |
| sphere / core_shell_sphere / cylinder / ellipsoid / hollow_cylinder / core_shell_cylinder / broad_peak / teubner_strey | external | main | sasmodels 1.1.0 bridge, zero-config (appears automatically when sasmodels is installed) |

New models go through the intake flow: drop the three-piece plugin into
`model_inbox/<model>/` → verification gate → validated; see
[model_inbox/README.md](model_inbox/README.md).

## Tests & Regression

- `saxs_selftest`: synthetic smoke tests for restoration/orientation/fitting engines +
  contract checks for all models (21 models, every mode);
- `test_data/`: liposome two-sample acceptance data, 7-dimension format-variant fixtures,
  Ruland L1 suite;
- `baseline/`: regression baselines — after refactors or engine swaps, results must be
  **bit-identical** to `regression_baseline.json` (cross-process 1-ULP parameter noise is
  adjudicated by the 1e-12 relative tolerance; see notes inside baseline/).

## Project Layout

```
├── saxs_mcp/            # MCP server (14 modules, model-agnostic, plugin-based loading)
├── model_library/       # validated / candidate model plugins
├── model_inbox/         # model submission inbox
├── agent_prompts/       # agent orchestration SOP / user keyword library / capability report / lessons
├── test_data/           # acceptance & regression test data
├── baseline/            # regression baselines
├── mcp.json             # MCP client onboarding template (three placeholders)
└── requirements.txt
```

## Scope & Boundaries

- 1D SAXS only (2D / GISAXS out of scope; orientation analysis consumes azimuthally
  integrated curves I(q_c, φ) produced by the beamline pyFAI pipeline);
- Custom physical models outside the library must go through the intake flow — the
  tools never invent models on the fly;
- No OCR for scanned PDFs (`saxs_doc_extract` reports empty_text honestly).

## Acknowledgments

Methodology anchored to: Schilt 2016 (lipid membrane), Tsao 2013 (Al–Cu precipitates),
Beaucage (unified), Hammouda 2010 (Guinier–Porod), Thünemann & Ruland 2000 (orientation),
Teubner–Strey, and the sasmodels/SASview community — full anchors in each manifest.json.

## License

[MIT](LICENSE) © 2026 chenrc
