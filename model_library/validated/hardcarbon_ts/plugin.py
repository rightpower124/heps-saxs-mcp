# -*- coding: utf-8 -*-
"""hardcarbon_ts 适配层（M3-261002）：把 model.py 物理函数接入通用引擎 saxs_fit。
方法学来源：Adv. Mater. 2025, 37, 2420251 SI Methods Eq.1-7（硬碳闭孔 Teubner-Strey 四组分拟合）；
方法学源头 Saurel et al., Energy Storage Mater. 21 (2019) 162；Teubner & Strey 1987。
"""
import numpy as np

# ---- 布局声明：两联图（拟合曲线 log 图 + 相对残差；实空间剖面无对应钩子） ----
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
    """结论文本钩子：撞界参数不得单独进入结论句（质量门纪律）"""
    _at = at or []
    if 'C1' in _at:
        c1 = ('1. **微孔尺度**：孔径 D = %.2f nm、孔间距 d = %.2f nm、相关长度 ξ = %.2f nm'
              '（χ²/dof=%.3f）；C1 撞界 → D 及全部 TS 导出量为**条件值**，不作物理结论。'
              % (der['D_nm'], der['d_nm'], der['xi_nm'], chi2_dof))
    elif 'C2' in _at:
        c1 = ('1. **微孔尺度**：孔径 D = %.2f nm、孔间距 d = %.2f nm、相关长度 ξ = %.2f nm'
              '（χ²/dof=%.3f）；C2 撞界 → d/ξ 为**条件值**（D 不受影响）。'
              % (der['D_nm'], der['d_nm'], der['xi_nm'], chi2_dof))
    else:
        c1 = ('1. **微孔尺度**：孔径 D = %.2f ± %.2f nm、孔间距 d = %.2f ± %.2f nm、'
              '相关长度 ξ = %.2f ± %.2f nm（χ²/dof=%.3f）。'
              % (der['D_nm'], der['D_nm_err'], der['d_nm'], der['d_nm_err'],
                 der['xi_nm'], der['xi_nm_err'], chi2_dof))
    return [
        c1,
        '2. **比表面积**：S_mp = %.1f m²/g（%.1f cm²/g）——**条件值**：依赖 '
        '①数据强度绝对标定假设（abs.units ≈ cm⁻¹）②ρ_s = %.2f g/cm³ 石墨密度上限口径'
        '（用户裁决）；仅作量级参考，不作精确定量结论。'
        % (der['S_mp_m2g'], der['S_mp_cm2g'], der['rho_s_g_cm3']),
        '3. **可靠性分级**：A/I0/B 为幅度量（绝对强度标度内同批可比、跨标度不可直接比较）；'
        'd/ξ/D 为标度无关形状参数，其物理可引用性以诊断证据链为准；'
        '撞界参数（%s）仅作条件值呈现。' % ('、'.join(_at) if _at else '无'),
        '4. **方法学披露**：n 为自由幂律指数（初值 4=Porod 极限）；C₁>0 为孔径公式 D=2√(5C₁) '
        '的适用前提（TS 谱无峰/弱峰形态）；S_mp 由 TS 高 q 渐近（Porod 定律）导出；'
        '孔体积分数 ∅ 不产出（SI Eq.2 反解依赖绝对标定验证，本次未做标定核验）。', '']


def register(M):
    main = {
        'names': M.NAMES,
        'bounds': M.BOUNDS,
        'init': M.INIT,
        'init_qmax': M.INIT_QMAX,         # A0 估计带 [qmin, 0.5] nm⁻¹（低 q 幂律主导区）
        'fixed_keys': [],                 # 无跨样品固定参数
        'normalized_params': ['A', 'I0', 'B'],   # 幅度量：同批绝对标度内可比，跨标度不可直接比较
        'calc': lambda p, q, fixed: M.model_vec(p, q),
        'derived': lambda p, perr, fixed: M.derived_main(p, perr),
        'plot_layout': _LAYOUT_DOUBLE,    # 两联图
        'components': [                   # 拟合曲线.csv 的附加分量线（算术表达式直接消费参数）
            {'label': 'particles+B: A*q^-n+B', 'style': 'b:',
             'column': 'I_particles_plus_B',
             'calc': lambda q, p, fixed: p['A'] * q ** (-p['n']) + p['B']},
            {'label': 'TS pores: I0/(1+C1q2+C2q4)', 'style': 'g--',
             'column': 'I_pores',
             'calc': lambda q, p, fixed: p['I0'] / (1.0 + p['C1'] * q ** 2 + p['C2'] * q ** 4)},
        ],
        'report': {
            'method': lambda fixed: [
                '## 3. 分析方法说明', '',
                '**模型（四组分加和，Adv. Mater. 2025 SI Eq.1 + Figure 2h/S10 分解）**：'
                'I(q) = A·q^(−n) + I₀/(1 + C₁q² + C₂q⁴) + B，其中',
                '- A·q^(−n)：I_particles——颗粒/大结构散射幂律项（n 自由，初值 4=Porod 极限）',
                '- I₀/(1+C₁q²+C₂q⁴)：I_pores——**Teubner-Strey 介观微孔项**（SI Eq.1；'
                '方法学源头 Saurel et al., Energy Storage Mater. 21 (2019) 162；'
                'Teubner & Strey, J. Chem. Phys. 87 (1987) 3195）',
                '- B：I_bckg——常数背景', '',
                '**导出物理量**（SI Eq.3/4/6/7）：',
                '- 孔间距 d = 2π·(½C₂^(−1/2) − C₁/(4C₂))^(−1/2)（相邻微孔平均距离）',
                '- 相关长度 ξ = (½C₂^(−1/2) + C₁/(4C₂))^(−1/2)（有序域延伸尺度）',
                '- 孔径 D = 2√(5C₁)（C₁>0 前提：TS 谱无峰/弱峰形态）',
                '- 比表面积 S_mp = I₀/(2π·C₂·(ΔSLD)²·ρ_s)（TS 高 q 渐近 = Porod 定律：'
                'I₀/C₂ = 2π(ΔSLD)²·S/V；文献 SI Eq.7 字面缺 ρ_s，因文献 I₀ 定义（SI Eq.2）已含 1/ρ_s——'
                '本插件 I₀ 为拟合自由参数，故显式除 ρ_s）', '',
                '**标定口径（用户 2026-10-02 裁决）**：ρ_s = 2.26 g/cm³（石墨密度上限）→'
                ' ΔSLD = 1.916×10⁻⁵ Å⁻² = 1.916×10⁻³ nm⁻²（空孔，ΔSLD=基体 SLD）；'
                '**S_mp 数值有效性条件于数据强度绝对标定假设（abs.units ≈ cm⁻¹），作条件值披露**。', '',
                '**拟合协议**：',
                '- 引擎：scipy `least_squares`（trf，x_scale=jac）；σ 列存在时加权拟合',
                '- 撞界判据：双侧界 |v−bound| < 10⁻³·量程；跨量级界（量程>10³·|v|，如 C1/C2）'
                '改用 5% 参数尺度相对距离；单侧界用 10⁻³·|v|；撞界参数自动降级为**条件值**', ''],
            'results_md': lambda der, chi2_dof, dof, at, nopeg_compare: [
                '**派生量**（微孔结构；误差为拟合误差数值传播）：',
                '- 孔径 D = **%.3f ± %.3f nm**；孔间距 d = **%.2f ± %.2f nm**；'
                '相关长度 ξ = **%.3f ± %.3f nm**'
                % (der['D_nm'], der['D_nm_err'], der['d_nm'], der['d_nm_err'],
                   der['xi_nm'], der['xi_nm_err']),
                '- S/V = %.3g（abs.units，条件于 cm⁻¹ 标定）；'
                '**S_mp = %.1f ± %.1f m²/g**（%.0f ± %.0f cm²/g；ρ_s=%.2f g/cm³ 石墨上限口径）'
                % (der['S_over_V'], der['S_mp_m2g'], der['S_mp_m2g_err'],
                   der['S_mp_cm2g'], der['S_mp_cm2g_err'], der['rho_s_g_cm3']),
                '- S_mp 为**条件值**：依赖 abs.units≈cm⁻¹ 标定假设与 ρ_s 上限口径，仅作量级参考',
                '', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
                % (chi2_dof, dof, '、'.join(at) if at else '无'), ''],
            'conclusions': _conclusions,
        },
    }
    return {'main': main}
