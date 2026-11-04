# -*- coding: utf-8 -*-
"""会话级工作目录（部署优化，2026-11-04）

背景：默认所有产物落工作区 runs/，多用户共用主机时会互相混淆。
机制：用户在 Agent 层确认"数据所在文件夹 = 工作路径"（四要素确认卡之一）后，
Agent 调 saxs_set_workdir 设置会话级状态（每个用户各起各的 MCP 服务器进程，
内存态即可，随进程生命周期）。此后全部带缺省 out_dir 的工具落盘到
<工作目录>/HEPS_SAXS_analysis/ 专用子目录——绝不写用户数据根目录（防覆盖
用户文件）。未设置时兜底旧行为 <_ROOT>/runs/<...>（逐位一致）。

纪律：显式传 out_dir 的调用路径不受本模块影响；本模块只改"缺省值"分支。
"""
import os

_WORKDIR = None
SUBDIR = 'HEPS_SAXS_analysis'
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_workdir(path):
    """设置会话级工作目录（必须是已存在的文件夹，通常 = 用户数据所在文件夹；
    不自动创建，防路径笔误静默落错位置）。"""
    global _WORKDIR
    p = os.path.abspath(path)
    if not os.path.isdir(p):
        raise ValueError('工作目录不存在（不自动创建，请确认路径）: %s' % p)
    _WORKDIR = p
    return get_workdir()


def get_workdir():
    """当前会话工作目录；未设置返回 None。"""
    return _WORKDIR


def default_out(*parts):
    """缺省产物目录：已设 workdir → <workdir>/HEPS_SAXS_analysis/<parts>；
    未设 → <_ROOT>/runs/<parts>（旧行为逐位一致）。"""
    if _WORKDIR:
        base = os.path.join(_WORKDIR, SUBDIR)
    else:
        base = os.path.join(_ROOT, 'runs')
    return os.path.join(base, *parts)
