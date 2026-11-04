# -*- coding: utf-8 -*-
"""HEPS SAXS 分析 MCP 服务器（M2 通用引擎版）
stdio 传输。注意：stdout 是 MCP 协议通道，全程序禁止 print 到控制台。

注册的工具（定义在各 tools_*.py 的 register() 中）：
  saxs_set_workdir 会话级工作目录：设后缺省产物落 <工作目录>/HEPS_SAXS_analysis/
                   专用子目录（未设兜底 runs/；261104 部署优化）(tools_io + workdir)
  saxs_suggest_window 窗口建议卡：全范围数据→低 q 直通光/高 q 噪声证据→建议窗
                   （建议+确认纪律，只建议不采纳；261104 部署优化）(tools_io + window_suggest)
  saxs_probe_curve 格式探测卡                   (tools_io)
  saxs_load_curve  读曲线 + 体检摘要；qmin/qmax=None = 全范围通道 (tools_io)
  saxs_list_models 列出模型库（validated/candidates/inbox 三区 + 桥接）(tools_fit)
  saxs_fit         通用拟合引擎 fit(model, data)，自动执行 L3 质量门；
                   inbox 区模型拒收 (tools_fit + quality_gate)
  saxs_diag_*      5 个按需诊断工具：multistart / relax_bound / param_scan / release / f_test
                   （tools_diag；inbox 区模型拒收）
  saxs_selftest    服务器自检：引擎/模型库（inbox 仅契约检查）/合成冒烟拟合 (selftest)
  make_plots       三联图（通用）              (tools_report)
  saxs_report      分析闭环四件套（通用；out_dir 可缺省→工作目录）(tools_report)
  saxs_gap_detect  一维曲线 gap 检测标记        (tools_gap, M3pre-2)
  saxs_gap_fill    gap 剔除与多项式填充整备     (tools_gap, M3pre-2)
  saxs_gap_merge   批量填充总表拼合（Origin对照）(tools_gap, M3pre-2)
  saxs_ruland_bpi2 取向分析：选峰+B_π/2 峰积分表  (tools_ruland, 260930)
  saxs_ruland_beq  取向分析：Ruland 变换→B_eq/f_v（含 scan_qmins 扫描自动采纳）(tools_ruland, 260930)
  saxs_ruland_chain 取向全流程链式编排三阶段：detect/prepare/bpi2 (tools_chain, 提速三件套 260930)
  saxs_export_script 从 result.json 生成自包含重放脚本（内置生成时自证对拍）(tools_export, 261007)
模型经 model_loader 从 model_library/{validated,candidates} 与根目录 model_inbox/
（待入库收件区，禁拟合）加载（manifest.json 定位 + plugin.py 适配层）。
"""
import os
import sys
from pathlib import Path

# DLL 污染修复（必须在 numpy/scipy 导入之前执行）：
# 若宿主进程 PATH 里混有其他 conda 环境的 LAPACK/OpenBLAS，scipy 延迟加载会
# 解析到错误版本 DLL 而静默崩溃（0xC06D007F）。把本环境的 DLL 目录前置即可。
# 注意：_ENV 为机器相关配置。GitHub 发布版参数化：环境变量 SAXSAGENT_ENV 优先
# （mcp.json env 块设置），未设置时回退本机缺省——开发端历史行为不变。
_ENV = os.environ.get("SAXSAGENT_ENV", r"E:\CondaEnvs\saxsagent")
os.environ["PATH"] = ";".join([
    os.path.join(_ENV, "Library", "bin"),
    os.path.join(_ENV, "DLLs"),
    _ENV,
    os.environ.get("PATH", ""),
])

# 确保包自身与项目根目录可导入（MCP 宿主以绝对路径启动本文件）
_ROOT = Path(__file__).resolve().parents[1]
for p in (str(Path(__file__).resolve().parent), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---- 服务器运行日志（2026-11-04 用户裁决，部署优化）----
# stdio 传输全程序禁止 print（stdout 是 MCP 协议通道），运行日志一律落
# <工作区根>/logs/heps_saxs_<pid>.log：mcp 库生命周期事件 + 未捕获异常
# 自动进文件；多用户各起各的服务器进程，pid 区分互不覆盖。logs\ 可由用户
# 手动清理（与 runs\ 同属可清空目录）。
import logging
_LOGDIR = _ROOT / 'logs'
_LOGDIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=str(_LOGDIR / ('heps_saxs_%d.log' % os.getpid())),
    encoding='utf-8',
    format='%(asctime)s %(levelname)s %(name)s: %(message)s')

from mcp.server.fastmcp import FastMCP

import tools_io
import tools_fit
import tools_diag
import tools_report
import tools_gap
import tools_ruland
import tools_chain
import tools_export
import tools_docextract
import selftest

mcp = FastMCP('heps-saxs')
tools_io.register(mcp)
tools_fit.register(mcp)
tools_diag.register(mcp)
tools_report.register(mcp)
tools_gap.register(mcp)
tools_ruland.register(mcp)
tools_chain.register(mcp)
tools_export.register(mcp)
tools_docextract.register(mcp)
selftest.register(mcp)

if __name__ == '__main__':
    mcp.run(transport='stdio')
