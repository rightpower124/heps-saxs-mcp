# -*- coding: utf-8 -*-
"""liposome_membrane 适配层（M2-T1）：把 model.py 物理函数接入通用引擎 saxs_fit。
model.py 物理代码零改动；本文件只声明"参数表 + 各模式的引用方式 + 报告文本钩子"，
与 manifest.json 的元数据（labels / refs / modes）一起构成完整模型插件。

report 钩子的文本逐字移植自 M1 tools_report._build_md（2026-09-23 用户已审核版本）。
"""
import numpy as np

# ---- M2.5-2 布局声明：三联图（M1 用户已验收形态） ----
# make_plots 是纯渲染器：布局由此 DSL 声明，与模型知识解耦。
# panel 原语 kind ∈ fit / residual / profile / curve；
# 标题 str 模板字段来自 manifest.modes[mode]（如 {plot_title}）+ {title}(调用方附加)；
# kind=profile 未写 title 时回落到该模式的 profile_title 钩子。
_LAYOUT_TRIPLE = {
    'figsize': (10, 7),
    'grid': (2, 2),
    'panels': [
        {'kind': 'fit', 'at': (0, 0), 'colspan': 2,
         'yscale': 'log', 'xlabel': 'q (nm$^{-1}$)', 'ylabel': 'I (a.u.)',
         'title': '{plot_title}, {title}', 'model_label': '{fit_label}'},
        {'kind': 'residual', 'at': (1, 0),
         'xlabel': 'q (nm$^{-1}$)', 'ylabel': '(fit - data)/data',
         'title': '{residual_title}'},
        {'kind': 'profile', 'at': (1, 1),
         'xlabel': 'z (nm)', 'ylabel': r'$\Delta\rho(z)$ (rel.)'},
    ],
}


def register(M):
    # ---- nopeg：3-高斯平膜（9 参数） ----
    nopeg = {
        'names': M.NAMES_NOPEG,
        'bounds': M.BOUNDS_NOPEG,
        'init': M.INIT_NOPEG,
        'init_qmax': 1.5,                 # A0 估计的低 q 窗口上界（与原脚本一致）
        'fixed_keys': [],                 # 无跨样品固定参数
        'normalized_params': ['A', 'B', 'drho_t'],   # 归一量：比例因子/背景/相对电子密度
        'calc': lambda p, q, fixed: M.model_nopeg_vec(p, q),
        'derived': lambda p, perr, fixed: M.derived_nopeg(p, perr),
        'profile': lambda z, p, fixed: M.drho_profile(z, p['zh'], p['tau_h'],
                                                      p['tau_t'], p['drho_t']),
        'z_range': (-3.5, 3.5),
        'profile_n': 500,
        'profile_title': lambda p, der: 'd_hh = %.2f nm' % (2 * p['zh']),
        'plot_layout': _LAYOUT_TRIPLE,    # M2.5-2：布局声明（三联图）
        'components': [                   # 图 1 / 拟合曲线.csv 的附加分量线
            {'label': 'membrane + B (no S2)', 'style': 'b--',
             'column': 'I_mem_plus_B_noS2',
             'calc': lambda q, p, fixed:
                 p['A'] * M.I_membrane(q, p['zh'], p['tau_h'], p['tau_t'], p['drho_t']) + p['B']},
        ],
        'report': {
            'method': lambda fixed: [
                '## 3. 分析方法说明', '',
                '**模型**：I(q) = A · I_mem(q) · S(q) + B，其中',
                '- I_mem：3-高斯对称平膜形状因子（头基×2 + 尾链，高斯电子密度堆叠的解析式）',
                '- S(q) = 1 + f·exp(−q²·sd²/2)·cos(q·d)（双层脂质体层间相关，d 为层间距）',
                '', '**拟合协议**：',
                '- 窗口 q ∈ [0.3, 4.4] nm⁻¹（低端避开直射光/挡板尾部）；权重 1/σ²',
                '- 引擎：scipy `least_squares`（trf，x_scale=jac）；初值由中低 q 归一与高 q 背景自动构造',
                '- 边界 = A 方案（放宽 tau_h 上界至 1.30、drho_t 下界至 −6.00，主拟合采用的方案）',
                '- 撞界判据：|v−bound| < 10⁻³·量程；撞界参数自动降级为**条件值**，不得单独进入结论句', ''],
            'results_md': lambda der, chi2_dof, dof, at, nopeg_compare: [
                '**派生量**：',
                '- 头-头距离 **d_hh = %.3f ± %.3f nm**（= 2·zh）' % (der['d_hh'], der['d_hh_err']),
                '- 双层相关峰 q₁ = 2π/d = %.3f nm⁻¹' % der['q1'],
                '', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
                % (chi2_dof, dof, '、'.join(at) if at else '无'), ''],
            'conclusions': lambda params, perr, chi2_dof, dof, at, der,
                           nopeg_compare, sample_note: [
                '1. **膜结构**：不含 PEG 脂质体的头-头距离 d_hh = %.2f ± %.2f nm，'
                '与文献磷脂双层典型范围（3–4 nm）一致；层间距 d = %.2f nm 表明存在少量双层堆叠相关（f = %.2f）。'
                % (der['d_hh'], der['d_hh_err'], params['d'], params['f']),
                '2. **参数可靠性**：tau_h、drho_t 撞界为条件值，其数值不应单独引用；'
                '但 d_hh 由 zh 决定（稳健值），结论不依赖撞界参数。',
                '3. **定性结论不依赖背景扣除方式**：常数 B 仅吸收高 q 平底；膜振荡峰位不受影响。',
                '4. **后续建议**：膜参数已按本结果固定，用于含 PEG 样品的 5-高斯拟合（见对应报告）；'
                'M2 起可加做多起点、F 检验等诊断。', ''],
        },
    }

    # ---- peg：5-高斯膜 + PEG 层（8 自由参数 + 4 固定膜参数） ----
    peg = {
        'names': M.NAMES_PEG,
        'bounds': M.BOUNDS_PEG,
        'init': M.INIT_PEG,
        # fixed 未提供 f/d/sd 时的回退初值（原脚本 fit_peg 的默认行为）
        'init_defaults': {'f': 0.266, 'd': 6.512, 'sd': 0.644},
        'init_qmax': 1.5,
        'fixed_keys': ['zh', 'tau_h', 'tau_t', 'drho_t'],
        # 放开固定参数做诊断（saxs_diag_release）时的边界：取 A 方案（与 nopeg 一致）
        'fixed_bounds': {k: M.BOUNDS_NOPEG[k] for k in ('zh', 'tau_h', 'tau_t', 'drho_t')},
        'normalized_params': ['A', 'B', 'drho_peg', 'drho_t'],  # 归一量：比例因子/背景/相对电子密度
        'calc': lambda p, q, fixed: M.model_peg_vec(p, q, fixed),
        'derived': lambda p, perr, fixed: M.derived_peg(p, perr, fixed),
        'profile': lambda z, p, fixed: M.drho_profile5(z, p['z_peg'], p['tau_peg'], p['drho_peg'],
                                                       fixed['zh'], fixed['tau_h'],
                                                       fixed['tau_t'], fixed['drho_t']),
        'z_range': (-6, 6),
        'profile_n': 600,
        'profile_title': lambda p, der: ('L_peg = %.2f nm' % der['L_peg'])
                                        if der and 'L_peg' in der
                                        else ('z_peg = %.2f nm' % p['z_peg']),
        'plot_layout': _LAYOUT_TRIPLE,    # M2.5-2：布局声明（三联图）
        'compare': {                      # 无 PEG 对照拟合（drho_peg=0），模型比较用
            'names': ['A', 'f', 'd', 'sd', 'B'],
            'calc': lambda p, q, fixed: M.model_nop_on_peg_vec(p, q, fixed),
            'p0': lambda A0, B0, p0: np.array([A0, p0['f'], p0['d'], p0['sd'], B0]),
            'zero_param': 'drho_peg',     # 计算最大相对影响时置 0 的参数
        },
        'report': {
            'method': lambda fixed: [
                '## 3. 分析方法说明', '',
                '**模型**：I(q) = A · I_5G(q) · S(q) + B，5-高斯膜 = 3-高斯膜（**膜参数固定自不含 PEG 样品**）'
                ' + 对称 PEG 层（±z_peg 镜像同参数）',
                '', '**拟合协议**：',
                '- 膜参数固定：zh=%.4f, tau_h=%.4f, tau_t=%.4f, drho_t=%.4f（不含 PEG 样品 A 方案结果）'
                % (fixed['zh'], fixed['tau_h'], fixed['tau_t'], fixed['drho_t']),
                '- 同一数据附带无 PEG 对照拟合（drho_peg=0）作模型比较',
                '- **PEG 层厚度（10% 判据，与论文一致）**：L_peg = (z_peg + c₁₀·tau_peg) − (zh + c₁₀·tau_h)，'
                'c₁₀ = √(ln10/4ln2) ≈ 0.911（高斯降至峰值 10% 处）',
                '- **构象判据**：L_peg/R_F，R_F ≈ 3.5 nm（PEG2000 Flory 半径）；L_peg < 1.4·R_F → 蘑菇/过渡区', ''],
            'results_md': lambda der, chi2_dof, dof, at, nopeg_compare: [
                '**派生量**：',
                '- 头基 10%% 外缘 %.3f nm，PEG 10%% 外缘 %.3f nm' % (der['edge_head'], der['edge_peg']),
                '- **PEG 层厚度 L_peg = %.3f ± %.3f nm**' % (der['L_peg'], der['L_err']),
                '- L_peg/R_F = %.2f → **%s**' % (der['L_over_RF'], der['conformation']),
                '- drho_peg 显著性 = %.1f σ' % der['drho_peg_sigma'],
                '', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s；%s'
                % (chi2_dof, dof, '、'.join(at) if at else '无',
                   ('无 PEG 模型 χ²/dof = %.3f（dof=%d）'
                    % (nopeg_compare.get('chi2_dof', float('nan')), nopeg_compare.get('dof', 0)))
                   if nopeg_compare else ''), ''],
            'conclusions': lambda params, perr, chi2_dof, dof, at, der,
                           nopeg_compare, sample_note: [
                '1. **PEG 层厚度**：L_peg = %.2f ± %.2f nm（10%% 判据）。' % (der['L_peg'], der['L_err']),
                '2. **构象**：L_peg/R_F = %.2f < 1.4，为**%s**，与 PEG2000 低接枝密度%s 的预期一致。'
                % (der['L_over_RF'], der['conformation'],
                   '（%s）' % sample_note if sample_note else ''),
                '3. **显著性**：drho_peg 点估计 %.1f σ，但 tau_peg 撞界为条件值，'
                '按规范以模型比较为准——含 PEG 模型使 χ² 显著下降，且 PEG 层效应最大 %.1f%%、'
                '接近低接枝体系的检测限，结论为**定性可靠**（log 坐标下该效应肉眼不可辨）。'
                % (der['drho_peg_sigma'], nopeg_compare.get('max_rel_effect_pct', float('nan'))
                   if nopeg_compare else float('nan')),
                '4. **跨样品一致性**：膜参数固定后 d = %.2f nm（不含 PEG 为 6.51 nm），'
                '层间距增大与 PEG 冠层推远双层堆叠的物理图景自洽（z_peg ≈ d/2 = %.2f nm）。'
                % (params['d'], der['edge_peg'] - der['L_peg']), ''],
        },
    }

    return {'nopeg': nopeg, 'peg': peg}
