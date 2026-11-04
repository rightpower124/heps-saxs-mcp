# -*- coding: utf-8 -*-
"""alcu_precipitate 适配层（M3-261001）：把 model.py 物理函数接入通用引擎 saxs_fit。
model.py 物理代码独立成文件；本文件只声明"参数表 + 引用方式 + 报告文本钩子"，
与 manifest.json 的元数据（labels / refs / modes）一起构成完整模型插件。

方法学来源：Tsao & Huang et al., J. Alloys Compd. 579 (2013) 138-146，
Model III = Eq.(5)（含 Porod 杂质项的圆柱盘形式因子，S(Q)≈1）。
"""
import numpy as np

# ---- 布局声明：两联图（拟合曲线 log 图 + 相对残差；盘模型无实空间剖面钩子） ----
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


def _conclusions(params, perr, chi2_dof, dof, at, der, nopeg_compare, sample_note):
    """结论文本钩子：撞界参数不得单独进入结论句（质量门纪律），
    措辞按 at_bound 条件化；可辨识性深度披露由 sample_note 承担。"""
    if 't' in (at or []):
        c1 = ('1. **析出相尺度**：盘状析出相直径 2R = %.1f nm（χ²/dof=%.3f）；'
              '厚度 t = %.3g nm 为**撞界条件值**，不作物理结论'
              '（可辨识性诊断证据链见样品备注）。'
              % (der['diameter_nm'], chi2_dof, der['thickness_nm']))
    else:
        c1 = ('1. **析出相尺度**：盘状析出相直径 2R = %.1f nm、厚度 t = %.2f nm（χ²/dof=%.3f）。'
              % (der['diameter_nm'], der['thickness_nm'], chi2_dof))
    return [
        c1,
        '2. **与文献对照**（Tsao 2013 Table 1/2）：GP 区 R≈0.8–16 nm/t≈0.24 nm，'
        'θ″ 相 R≈8.5–11 nm/t≈1.5–1.9 nm，θ′ 相 R≈22–60 nm/t≈3.7–26 nm；'
        '本结果与各相区间的归属判断，结合样品备注中的可辨识性证据链综合评估。',
        '3. **可靠性分级**：A/s/B 为归一量（a.u. 标度）不作跨样品/跨标度直接比较；'
        'R/t 为标度无关形状参数，其物理可引用性以可辨识性诊断为准（见样品备注）；'
        '撞界参数（%s）仅作条件值呈现。' % ('、'.join(at) if at else '无'),
        '4. **方法学披露**：S(Q)≈1 假设源于数据无结构峰（与文献 A1h 处理一致）；'
        '直径单分布假设沿文献（避免直径多分散度过参数化）；B 为站内背景扩展项。', '']


def _method1(fixed):
    return [
        '## 3. 分析方法说明', '',
        '**模型（Model I，Tsao 2013 Eq.3）**：I(Q) = A·Q⁻⁴ + s·P_disk(Q; R, t)·S(Q; r_hs, η) + B，其中',
        '- P_disk：圆柱盘形式因子取向平均（与 Model III 同款，文献 Eq.1 积分核）',
        '- S(Q)：**Percus-Yevick 硬球结构因子**（Kotlarchyk & Chen 1983 闭合式；公式逐式移植自 '
        'sasmodels 1.1.0 hardsphere.c，含小 X 泰勒分支；r_hs=硬球半径，η=硬球体积分数）',
        '- A·Q⁻⁴：大尺寸杂质析出物的 Porod 幂律项；s 语义与 Model III 相同（= φ·Δρ²·V_disk，P(0)=1），两模式幅度可比',
        '- 适用场景：粒子间空间关联产生结构峰的 GPZ 类数据（文献 Table 1 NA/GPZ 各温度行均用 Model I）', '',
        '**拟合协议**：',
        '- 引擎：scipy `least_squares`（trf，x_scale=jac）；无 σ 列时 unweighted',
        '- 撞界判据：|v−bound| < 10⁻³·量程；撞界参数自动降级为**条件值**',
        '- η 上界 0.50：PY 闭合在 η≳0.5 精度下降（建模选择，披露）；'
        '- **φ 为表观相对体积分数（a.u. 标度）**：本站数据未绝对标定，φ 的绝对值不可与文献 cm⁻¹ 标定值直接比较', '']


def _results1(der, chi2_dof, dof, at, nopeg_compare):
    _at = at or []
    tag = '（%s 撞界 → 同为**条件值**）' % '、'.join([k for k in ('t', 'r_hs', 'eta') if k in _at]) \
        if any(k in _at for k in ('t', 'r_hs', 'eta')) else ''
    return [
        '**派生量**：',
        '- 盘直径 2R = **%.2f ± %.2f nm**，盘厚度 t = **%.2f ± %.2f nm**'
        % (der['diameter_nm'], der['diameter_nm_err'], der['thickness_nm'], der['thickness_nm_err']),
        '- 硬球半径 r_hs = **%.2f ± %.2f nm**，硬球体积分数 η = **%.3f ± %.3f**%s'
        % (der['r_hs_nm'], der['r_hs_nm_err'], der['eta'], der['eta_err'], tag),
        '- 结构因子外推 S(0) = %.3f；PY 结构峰位置（网格）q_peak ≈ %.2f nm⁻¹'
        % (der['S0'], der['q_peak_nm1']),
        '- 表观体积分数 φ = s/(Δρ²·πR²t) = %.3g ± %.3g（**a.u. 标度归一量**，不作跨数据比较）'
        % (der['phi_apparent'] / der['V_disk_nm3'],
           der['phi_apparent_err'] / der['V_disk_nm3'] if der['V_disk_nm3'] else float('nan')),
        '', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
        % (chi2_dof, dof, '、'.join(_at) if _at else '无'), '']


def _conclusions1(params, perr, chi2_dof, dof, at, der, nopeg_compare, sample_note):
    """Model I 结论文本钩子（撞界参数不单独进结论句，质量门纪律）"""
    _at = at or []
    if 't' in _at:
        c1 = ('1. **析出相尺度**：盘直径 2R = %.1f nm（χ²/dof=%.3f）；厚度 t = %.3g nm 为**撞界条件值**，'
              '不作物理结论（可辨识性诊断见样品备注）。'
              % (der['diameter_nm'], chi2_dof, der['thickness_nm']))
    else:
        c1 = ('1. **析出相尺度**：盘直径 2R = %.1f nm、厚度 t = %.2f nm（χ²/dof=%.3f）。'
              % (der['diameter_nm'], der['thickness_nm'], chi2_dof))
    if 'eta' in _at or 'r_hs' in _at:
        c2 = ('2. **粒子间关联**：硬球参数 r_hs/eta 触界（%s）→ 结构因子不可辨识，'
              '结构峰结论以诊断证据链为准（见样品备注）。'
              % ('、'.join([k for k in ('r_hs', 'eta') if k in _at])))
    else:
        c2 = ('2. **粒子间关联**：硬球半径 r_hs = %.2f nm、η = %.3f，PY 结构峰位 q_peak ≈ %.2f nm⁻¹'
              '（其稳健性以诊断证据链为准）。'
              % (der['r_hs_nm'], der['eta'], der['q_peak_nm1']))
    return [
        c1, c2,
        '3. **可靠性分级**：A/s/B 为归一量（a.u. 标度）不作跨样品/跨标度直接比较；'
        'R/t/r_hs/eta 为形状参数，其物理可引用性以可辨识性诊断为准（见样品备注）；'
        '撞界参数（%s）仅作条件值呈现。' % ('、'.join(_at) if _at else '无'),
        '4. **方法学披露**：PY 硬球为单分散闭合近似（η 上界 0.50 建模选择）；'
        '硬球半径 r_hs 未强制 ≥ 盘半径 R（物理自洽性事后核对）；'
        'S(Q) 假设源于文献 Model I（GPZ 相处理）；B 为站内背景扩展项。', '']


def register(M):
    main = {
        'names': M.NAMES,
        'bounds': M.BOUNDS,
        'init': M.INIT,
        'init_qmax': 0.15,                # A0 估计的低 q 窗口上界（nm⁻¹，Porod 主导区）
        'fixed_keys': [],                 # 无跨样品固定参数
        'normalized_params': ['A', 's', 'B'],   # 归一量：幅度/缩放/背景（a.u. 标度，跨数据不可直接比较）
        'calc': lambda p, q, fixed: M.model_vec(p, q),
        'derived': lambda p, perr, fixed: M.derived_main(p, perr),
        'plot_layout': _LAYOUT_DOUBLE,    # 两联图
        'components': [                   # 拟合曲线.csv 的附加分量线
            {'label': 'Porod A*q^-4 + B (no disk)', 'style': 'b:',
             'column': 'I_porod_plus_B',
             'calc': lambda q, p, fixed: p['A'] * q ** (-4.0) + p['B']},
        ],
        'report': {
            'method': lambda fixed: [
                '## 3. 分析方法说明', '',
                '**模型（Model III，Tsao 2013 Eq.5）**：I(Q) = A·Q⁻⁴ + s·P_disk(Q; R, t) + B，其中',
                '- P_disk：圆柱盘（盘状析出相）形式因子的取向平均（文献 Eq.1 积分核：'
                'j₀(x)=sin(x)/x 为球贝塞尔函数，J₁ 为一阶贝塞尔函数，α 为盘法轴与 Q 夹角，∫₀^{π/2}·sinα·dα 加权）',
                '- A·Q⁻⁴：大尺寸杂质析出物的 Porod 幂律项（文献对低 q 尾部的处理）',
                '- s = φ·Δρ²：盘散射缩放；Δρ 统一取 1.05×10⁻⁵ Å⁻²（文献简化口径）',
                '- S(Q)≈1：本数据无空间有序结构峰（与文献 A1h 样品 θ″ 相同处理）', '',
                '**拟合协议**：',
                '- 引擎：scipy `least_squares`（trf，x_scale=jac）；无 σ 列时 unweighted',
                '- 撞界判据：|v−bound| < 10⁻³·量程；撞界参数自动降级为**条件值**',
                '- **φ 为表观相对体积分数（a.u. 标度）**：本站数据未绝对标定，'
                'φ 的绝对值不可与文献 cm⁻¹ 标定值直接比较；R/t 为形状参数（与标度无关），可稳健比较', ''],
            'results_md': lambda der, chi2_dof, dof, at, nopeg_compare: [
                '**派生量**（盘几何）：',
                '- 盘直径 2R = **%.2f ± %.2f nm**，盘厚度 t = **%.2f ± %.2f nm**%s'
                % (der['diameter_nm'], der['diameter_nm_err'], der['thickness_nm'],
                   der['thickness_nm_err'],
                   '（t 撞界 → t/V/R_g 同为**条件值**）' if 't' in (at or []) else ''),
                '- 单盘体积 V = πR²t = %.1f nm³；理论回转半径 R_g = √(R²/2 + t²/12) = %.2f nm'
                % (der['V_disk_nm3'], der['Rg_theory_nm']),
                '- 表观体积分数 φ = s/Δρ² = %.3g ± %.3g（**a.u. 标度归一量**，不作跨数据比较）'
                % (der['phi_apparent'], der['phi_apparent_err']),
                '', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
                % (chi2_dof, dof, '、'.join(at) if at else '无'), ''],
            'conclusions': _conclusions,
        },
    }
    model1 = {
        'names': M.NAMES1,
        'bounds': M.BOUNDS1,
        'init': M.INIT1,
        'init_qmax': 0.15,                # A0 估计的低 q 窗口上界（nm⁻¹，Porod 主导区）
        'fixed_keys': [],                 # 无跨样品固定参数
        'normalized_params': ['A', 's', 'B'],   # 归一量：幅度/缩放/背景（a.u. 标度，跨数据不可直接比较）
        'calc': lambda p, q, fixed: M.model1_vec(p, q),
        'derived': lambda p, perr, fixed: M.derived_1(p, perr),
        'plot_layout': _LAYOUT_DOUBLE,    # 两联图（与 main 同款）
        'components': [                   # 拟合曲线.csv 的附加分量线
            {'label': 'Porod A*q^-4 + B (no disk, no S)', 'style': 'b:',
             'column': 'I_porod_plus_B',
             'calc': lambda q, p, fixed: p['A'] * q ** (-4.0) + p['B']},
            {'label': 'disk x S_PY: s*P_disk*S (no Porod)', 'style': 'g--',
             'column': 'I_disk_S',
             'calc': lambda q, p, fixed: p['s'] * M.P_disk(q, p['R'], p['t'])
                                          * M.S_py(q, p['r_hs'], p['eta'])},
        ],
        'report': {
            'method': _method1,
            'results_md': _results1,
            'conclusions': _conclusions1,
        },
    }
    return {'main': main, 'model1': model1}
