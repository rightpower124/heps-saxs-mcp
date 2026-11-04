# -*- coding: utf-8 -*-
"""数据工具（M2.5-1）：saxs_probe_curve 格式探测 + saxs_load_curve 读曲线 + 体检摘要
部署优化（261104）：+ saxs_set_workdir 会话级工作目录 + saxs_suggest_window 窗口建议卡
"""
import os

import workdir
import window_suggest
from curve_io import load_curve, probe_format


def register(mcp):
    @mcp.tool()
    def saxs_set_workdir(path: str) -> dict:
        """设置会话级工作目录（四要素确认卡之一：用户确认"数据所在文件夹=工作
        路径"后由 Agent 调用）。此后所有工具的缺省产物目录重定向到
        <工作目录>/HEPS_SAXS_analysis/ 专用子目录（不写数据根，防覆盖用户文件）；
        未设置时兜底工作区 runs/。目录必须已存在（不自动创建，防路径笔误）。
        每个用户各起各的 MCP 服务器进程，本设置为进程内会话态。"""
        wd = workdir.set_workdir(path)
        return {'ok': True, 'workdir': wd, 'output_subdir': workdir.SUBDIR,
                'note': '此后缺省产物落 <workdir>/HEPS_SAXS_analysis/；显式传 out_dir 不受影响'}

    @mcp.tool()
    def saxs_suggest_window(path: str, q_unit: str = 'A^-1', fmt: dict = None) -> dict:
        """拟合窗口建议卡（建议+确认纪律：只建议不采纳）。
        全范围读入数据（qmin/qmax=None 通道），分析低 q 直通光（平台翼
        direct_beam_plateau / 陡于 Porod 上翘 steeper_than_porod）与高 q 噪声地板
        （sigma_inflated / high_freq_oscillation），返回 suggested_window 与证据
        evidence。无裁剪证据时建议窗 = 完整数据范围（初始默认 = 全范围）。
        用户确认后，将 suggested_window 作为显式 qmin/qmax 传给 saxs_fit /
        saxs_load_curve。fmt 用法同 saxs_load_curve。"""
        if not os.path.isfile(path):
            raise ValueError('文件不存在: %s' % path)
        q, I, sig, s = load_curve(path, q_unit, qmin=None, qmax=None, fmt=fmt)
        card = window_suggest.suggest_window(q, I, sig,
                                             unweighted=bool(s.get('unweighted')))
        card['path'] = path
        card['data_summary'] = {'n_used': s['n_used'], 'n_bad_lines': s['n_bad_lines'],
                                'unweighted': bool(s.get('unweighted', False))}
        return card

    @mcp.tool()
    def saxs_probe_curve(path: str) -> dict:
        """只读探测 1D 数据文件格式，返回格式卡（不改文件、不做猜测性裁决）。
        返回：编码 / 行尾 / 分隔符 / 列数 / 表头结构 / 内嵌单位提示（如 q(A-1) 列头、
        EDF 头）/ 双块与 2θ·方位角疑似标记 / 歧义清单 / suggested_fmt。
        用法：先探测 → ambiguities 与 q_unit 由用户裁决 → 把确认后的 fmt 传给
        saxs_load_curve。q_unit 提示仅为提示，必须经用户确认。
        """
        if not os.path.isfile(path):
            raise ValueError('文件不存在: %s' % path)
        return probe_format(path)

    @mcp.tool()
    def saxs_load_curve(path: str, q_unit: str = 'A^-1', qmin: float = 0.3, qmax: float = 4.4,
                        fmt: dict = None) -> dict:
        """读取 1D SAXS 曲线，返回点数与量程摘要。
        常规路径：三列 q I sigma 空白分隔，q_unit: 'A^-1'（自动×10 转 nm^-1）或
        'nm^-1'，必须显式指定；qmin/qmax: 感兴趣的 q 窗口 (nm^-1)。
        变体路径：先 saxs_probe_curve 探测，格式卡经用户确认后作为 fmt 传入
        （encoding/delimiter/col_q/col_i/col_s/skip_rows/q_unit/block/
        two_theta_wavelength_A）。fmt 中 col_s=None 表示无 σ（摘要标 unweighted，
        perr 被低估）。fmt=None 时行为与常规路径完全一致。
        """
        if not os.path.isfile(path):
            raise ValueError('文件不存在: %s' % path)
        q, I, sig, s = load_curve(path, q_unit, qmin, qmax, fmt)
        s.update({'path': path, 'I_min': float(I.min()), 'I_max': float(I.max()),
                  'sigma_positive': True})
        return s
