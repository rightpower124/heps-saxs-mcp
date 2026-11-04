# -*- coding: utf-8 -*-
"""guinier_porod 适配层（M4-261006 复现锚定）：Hammouda 2010 广义 Guinier–Porod，
tiered 三窗分 level 拟合复现 Chaney 2025 Figure 7a/7b（PM7/氯苯溶液 SAXS 小角区）。
本文件只声明"参数表 + 引用方式 + 报告文本钩子"；物理代码在 model.py。
三模式：level3（大尺度聚集体）/ level2（中尺度域）/ level1（链段/局域尺度）。
"""
import numpy as np

# ---- 布局声明：两联图（拟合曲线 log 图 + 相对残差） ----
_LAYOUT_DOUBLE = {
    'figsize': (10, 8),
    'grid': (2, 1),
    'panels': [
        {'kind': 'fit', 'at': (0, 0), 'yscale': 'log',
         'xlabel': 'q (nm$^{-1}$)', 'ylabel': 'I (a.u.)',
         'title': '{plot_title}, {title}', 'model_label': '{fit_label}'},
        {'kind': 'residual', 'at': (1, 0),
         'xlabel': 'q (nm$^{-1}$)', 'ylabel': '(fit - data)/data',
         'title': '{residual_title}'},
    ],
}

_SHAPE_ANCHOR = ('s 维度锚点（Hammouda 2010）：0=球/等轴三维对象（低 q 平台）、'
                 '1=棒/柱状（中间 q 区 q^-1）、2=片/盘状（中间 q 区 q^-2）')
_POROD_ANCHOR = ('P 形态锚点：≈4 光滑致密界面 Porod 极限；≈3 致密三维对象表面散射'
                 '（Chaney 2025 level3 归因"非球形聚集体"）；<3 质量分形；3–4 表面粗糙度渐增')


def _comp_guinier(q, p, fixed):
    """Guinier 支全程延拓线（显示用：越出 q1 仍延拓，便于目检衔接点位置）"""
    q = np.asarray(q, dtype=float)
    n = max(3.0 - float(p['s']), 1e-12)
    return p['A'] * np.exp(-(q * p['Rg']) ** 2 / n) / q ** float(p['s'])


def _comp_porod(q, p, fixed):
    """Porod 支全程延拓线"""
    q = np.asarray(q, dtype=float)
    s = float(p['s'])
    n = 3.0 - s
    ms = (float(p['P']) - s) / 2.0
    if n <= 0.0 or ms <= 0.0:
        return np.zeros_like(q)
    return p['A'] * np.exp(-ms) * (n * ms / p['Rg'] ** 2) ** ms / q ** float(p['P'])


_COMPONENTS = [
    {'label': 'Guinier branch (cont.)', 'style': 'b:',
     'column': 'I_guinier_branch', 'calc': _comp_guinier},
    {'label': 'Porod branch (cont.)', 'style': 'g--',
     'column': 'I_porod_branch', 'calc': _comp_porod},
]


def _make_method(level_cn, window, paper_anchor):
    def method(fixed):
        return [
            '## 3. 分析方法说明', '',
            '**模型（Hammouda 2010 广义 Guinier–Porod，J. Appl. Cryst. 43, 716）**：',
            '- Guinier 支（q ≤ q₁）：I = A·q^(−s)·exp(−(q·Rg)²/(3−s))',
            '- Porod 支（q > q₁）：I = A·q^(−P)·exp(−ms)·(n·ms/Rg²)^ms',
            '- 其中 n = 3−s，ms = (P−s)/2，衔接点 q₁ = √(n·ms)/Rg（两支值与斜率在 q₁ 双连续）', '',
            '**参数语义**：',
            '- s：' + _SHAPE_ANCHOR,
            '- P：' + _POROD_ANCHOR,
            '- Rg：回转半径（nm；报告同时给出 Å 值直接对标文献）',
            '- A：幅度（a.u. 标度归一量）；B：常数背景', '',
            '**复现锚点（Chaney et al., Chem. Mater. 37 (2025) 7882–7893, Figure 7a/7b）**：'
            '作者对小角区采用 "unified model with four levels" 的 tiered Guinier–Porod '
            '拟合（其 ref 69 = Hammouda 2010），逐级子窗剥离结构层级；'
            '本模式 = ' + level_cn + '，推荐窗 ' + window + ' nm⁻¹，'
            '文献 Figure 7b 锚点：' + paper_anchor + '。',
            'WAXS 区 Lorentz 峰拟合超出本工具模型库能力范围，未纳入复现。', '',
            '**拟合协议**：',
            '- 引擎：scipy `least_squares`（trf，x_scale=jac）；σ 列存在时加权拟合',
            '- 撞界判据：双侧界 |v−bound| < 10⁻³·量程；跨量级界（量程>10³·|v|）改用 '
            '5% 参数尺度相对距离；单侧界用 10⁻³·|v|；撞界参数自动降级为**条件值**', '']
    return method


def _results_md(der, chi2_dof, dof, at, nopeg_compare):
    return [
        '**派生量**（误差为拟合误差数值传播）：',
        '- Rg = **%.2f ± %.2f nm** = **%.1f ± %.1f Å**（Å 值直接对标文献 Figure 7b）'
        % (der['rg_nm'], der['rg_nm_err'], der['rg_A'], der['rg_A_err']),
        '- 衔接点 q₁ = **%.3g ± %.3g nm⁻¹**（q₁·Rg = √(n·ms) = %.4g，无量纲自检量）'
        % (der['q1_nm1'], der['q1_nm1_err'], der['q1_rg']),
        '', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
        % (chi2_dof, dof, '、'.join(at) if at else '无'), '']


def _make_conclusions(level_cn):
    def _conclusions(params, perr, chi2_dof, dof, at, der, nopeg_compare, sample_note):
        _at = at or []
        if 'Rg' in _at:
            c1 = ('1. **结构尺度**：Rg = %.3g Å（χ²/dof=%.3f）为**撞界条件值**，'
                  '不作尺度结论（诊断证据链见样品备注）。' % (der['rg_A'], chi2_dof))
        else:
            c1 = ('1. **结构尺度**：Rg = **%.1f ± %.1f Å**（%.2f ± %.2f nm），衔接点 '
                  'q₁ = %.3g ± %.3g nm⁻¹（χ²/dof=%.3f）。'
                  % (der['rg_A'], der['rg_A_err'], der['rg_nm'], der['rg_nm_err'],
                     der['q1_nm1'], der['q1_nm1_err'], chi2_dof))
        sp_at = [x for x in _at if x in ('s', 'P')]
        if sp_at:
            c2 = ('2. **形态判读**：s = %.3g、P = %.3g——s/P 含撞界参数（%s），'
                  '形态判读降级为**条件值**，仅以拟合曲线+残差佐证。'
                  % (float(params['s']), float(params['P']), '、'.join(sp_at)))
        else:
            c2 = ('2. **形态判读**：s = %.2f ± %.2f（%s）；P = %.2f ± %.2f（%s）。'
                  % (float(params['s']), float(perr['s']), _SHAPE_ANCHOR,
                     float(params['P']), float(perr['P']), _POROD_ANCHOR))
        return [
            c1, c2,
            '3. **可靠性分级**：A/B 为归一量（a.u. 标度）不作跨样品/跨标度直接比较；'
            'Rg/s/P 为标度无关结构参数，物理可引用性以诊断证据链为准；'
            '撞界参数（%s）仅作条件值呈现。' % ('、'.join(_at) if _at else '无'),
            '4. **方法学披露**：tiered 分窗拟合法——作者原文为四 level 统一模型逐级拟合，'
            '本工具以三窗独立 GP 拟合复现其小角区三个层级（本模式 = ' + level_cn + '），'
            '各窗参数独立、单窗 GP 不表达跨层级叠加；数据为作者交付的溶剂/空白扣除后曲线'
            '（sol561−buf555−emp544，a.u. 强度）；WAXS 区 Lorentz 峰未复现'
            '（超出模型库能力范围）。', '']
    return _conclusions


def _make_mode(init, init_qmax, level_cn, window, paper_anchor):
    return {
        'names': None,          # register 内以 M.NAMES 填充（见下）
        'bounds': None,
        'init': init,
        'init_qmax': init_qmax,
        'fixed_keys': [],                 # 无跨样品固定参数
        'normalized_params': ['A', 'B'],  # 幅度量：a.u. 标度内同批可比，跨标度不可直接比较
        'selftest_ab': (1.0, 0.0),        # 冒烟合成数据 A=1、B=0（形状信号主导；B=0 恢复检查自动跳过）
        'calc': None,
        'derived': None,
        'plot_layout': _LAYOUT_DOUBLE,
        'components': _COMPONENTS,
        'report': {
            'method': _make_method(level_cn, window, paper_anchor),
            'results_md': _results_md,
            'conclusions': _make_conclusions(level_cn),
        },
    }


def register(M):
    modes = {
        'level3': _make_mode(M.INIT_LEVEL3, M.INIT_QMAX_LEVEL3,
                             '大尺度聚集体层（tiered level3）', '[0.025, 0.09]',
                             'level3 Rg≈380 Å、P≈3（非球形聚集体）'),
        'level2': _make_mode(M.INIT_LEVEL2, M.INIT_QMAX_LEVEL2,
                             '中尺度域层（tiered level2）', '[0.1, 0.5]',
                             'level2 Rg≈140 Å'),
        'level1': _make_mode(M.INIT_LEVEL1, M.INIT_QMAX_LEVEL1,
                             '链段/局域尺度层（tiered level1）', '[0.4, 2.0]',
                             'level1 Rg≈20 Å'),
    }
    for cfg in modes.values():
        cfg['names'] = M.NAMES
        cfg['bounds'] = M.BOUNDS
        cfg['calc'] = lambda p, q, fixed: M.model_vec(p, q)
        cfg['derived'] = lambda p, perr, fixed: M.derived_main(p, perr)
    return modes
