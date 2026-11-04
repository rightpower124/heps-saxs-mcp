# -*- coding: utf-8 -*-
"""phase_separation_bp 适配层（2026-10-22 立项）：幂律底 + 广延峰 + 背景。
方法学与可辨识性披露见 model.py docstring 与 manifest refs；峰形对拍双模式
gauss / lorentz。候选区模型（未过验证门，status=candidate）。"""
import numpy as np

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
    """双模式共用结论钩子：峰参数（amp/q0/峰宽）一律按可辨识性披露"""
    _at = at or []
    is_gauss = 'sigma' in params
    width_name = 'sigma' if is_gauss else 'w'
    if 'q0' in _at:
        c1 = ('1. **特征周期**：峰位 q0 = %.3g nm⁻¹（χ²/dof=%.3f）为**撞界条件值**，'
              'd_period 派生量同不作物理结论。' % (float(params['q0']), chi2_dof))
    else:
        c1 = ('1. **特征周期**：峰位 q0 = %.3f ± %.3f nm⁻¹，**d = 2π/q0 = %.1f ± %.1f nm**，'
              '峰 FWHM = %.3f ± %.3f nm⁻¹（χ²/dof=%.3f）。'
              % (float(params['q0']), float(perr['q0']), der['d_period_nm'],
                 der['d_period_nm_err'], der['fwhm_nm1'], der['fwhm_nm1_err'], chi2_dof))
    if params['amp'] < 0.1 * float(params['A']) * float(params['q0']) ** (-float(params['dm'])):
        c2 = ('2. **峰贡献**：峰幅度 amp = %.3g 相对幂律底在峰位的贡献偏弱（<10%%），'
              '峰参数（amp/%s）沿"底形主导"简并方向可辨识性受限，一律条件值判读。'
              % (float(params['amp']), width_name))
    else:
        c2 = ('2. **峰贡献**：峰幅度 amp = %.3g ± %.3g，%s = %.3f ± %.3f nm⁻¹（峰形参数'
              '可辨识性以 multistart/param_scan 诊断为准）。'
              % (float(params['amp']), float(perr['amp']), width_name,
                 float(params[width_name]), float(perr[width_name])))
    return [
        c1, c2,
        '3. **幂律底**：dm = %.2f ± %.2f（Porod 极限 4=光滑界面；<4 弥散界面/链形态；'
        '>4 提示数据衰减慢于 Porod，界面层分析输入）。' % (float(params['dm']), float(perr['dm'])),
        '4. **可靠性分级**：A/amp/B 为归一量；q0/峰宽/dm 为形状参数，峰浅场景'
        '（首测数据峰/底 log 比 ≈0.11 dex）一律条件值判读并以诊断工具取证；'
        '撞界参数（%s）仅作条件值呈现。' % ('、'.join(_at) if _at else '无'),
        '5. **方法学披露**：本模型 = Hammouda broad_peak 形态（幂律+广延峰）的峰形松绑版'
        '（标准 gauss/lorentz 替换解析峰形）；双连续相分离正源物理 = Teubner & Strey 1987；'
        '峰浅时底与峰尾互相补偿，dm 与峰宽弱耦合；背景 B 吸收未扣背景残余。', '']


def _mk_mode(names, bounds, init, calc, derived, mode_key, smoke_truth):
    return {
        'names': names,
        'bounds': bounds,
        'init': init,
        'init_qmax': 0.5,                 # A0 估计窗覆盖峰区（q0≤0.5 界内）
        'smoke_truth': smoke_truth,       # 冒烟真值覆盖（见 manifest smoke_truth.note）
        'fixed_keys': [],
        'normalized_params': ['A', 'amp', 'B'],
        'calc': calc,
        'derived': derived,
        'plot_layout': _LAYOUT_DOUBLE,
        'components': [
            {'label': 'power-law base A*q^-dm', 'style': 'g--',
             'column': 'I_power_law',
             'calc': lambda q, p, fixed: p['A'] * np.asarray(q, dtype=float) ** (-p['dm'])},
            {'label': 'broad peak (no base/B)', 'style': 'm-.',
             'column': 'I_peak',
             'calc': (lambda q, p, fixed: p['amp'] * np.exp(
                 -((np.asarray(q, dtype=float) - p['q0']) ** 2) / (2.0 * p['sigma'] ** 2))
                 if mode_key == 'gauss' else
                 p['amp'] / (1.0 + ((np.asarray(q, dtype=float) - p['q0']) / p['w']) ** 2))},
            {'label': 'background B (constant)', 'style': 'b:',
             'column': 'I_background',
             'calc': lambda q, p, fixed: np.full_like(np.asarray(q, dtype=float), float(p['B']))},
        ],
        'report': {
            'method': lambda fixed: [
                '## 3. 分析方法说明', '',
                '**模型（phase_separation_bp，幂律底 + 广延峰 + 背景）**：',
                '- mode gauss：I(q) = A·q^(−dm) + amp·exp(−(q−q0)²/2σ²) + B',
                '- mode lorentz：I(q) = A·q^(−dm) + amp/(1+((q−q0)/w)²) + B', '',
                '**物理语义**：',
                '- q0：关联峰位 → **特征周期 d = 2π/q0**（双连续相分离域尺寸，'
                'Teubner & Strey 1987 物理语义）',
                '- dm：幂律底指数（Porod 4=光滑界面；<4 弥散界面；>4 衰减慢于 Porod，'
                '界面层厚度分析输入）',
                '- amp/σ(w)：峰幅度与峰宽（FWHM: gauss=2.355σ / lorentz=2w）', '',
                '**方法学链条**：Hammouda broad_peak（NIST/SASview）峰形松绑版；'
                '正源物理 Teubner & Strey 1987, J. Chem. Phys. 87, 3195；'
                '峰形裁决 = gauss/lorentz 双模式对照 + F 检验。', '',
                '**拟合协议**：scipy least_squares（trf, 1/σ² 加权）；'
                '撞界参数自动降级条件值；峰浅场景峰参数以诊断工具取证。', ''],
            'results_md': lambda der, chi2_dof, dof, at, nopeg_compare: [
                '**派生量**：',
                '- 特征周期 d = 2π/q0 = **%.2f ± %.2f nm**'
                % (der['d_period_nm'], der['d_period_nm_err']),
                '- 峰 FWHM = %.3f ± %.3f nm⁻¹' % (der['fwhm_nm1'], der['fwhm_nm1_err']),
                '', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
                % (chi2_dof, dof, '、'.join(at) if at else '无'), ''],
            'conclusions': _conclusions,
        },
    }


def register(M):
    # 冒烟真值（默认 INIT 峰幅度 0.02 相对 selftest_ab A=1 幂律底为信号淹没形态，
    # lessons 10-06 C 用例同族——冒烟改用峰主导形态验证 amp/q0/峰宽可恢复）
    _smoke_g = {'A': 1.0, 'dm': 2.5, 'amp': 20.0, 'q0': 0.6, 'sigma': 0.5, 'B': 0.001}
    _smoke_l = {'A': 1.0, 'dm': 2.5, 'amp': 20.0, 'q0': 0.6, 'w': 0.5, 'B': 0.001}
    return {
        'gauss': _mk_mode(M.NAMES_G, M.BOUNDS_G, M.INIT_G,
                          lambda p, q, fixed: M.model_gauss_vec(p, q),
                          lambda p, perr, fixed: M.derived_gauss(p, perr),
                          'gauss', _smoke_g),
        'lorentz': _mk_mode(M.NAMES_L, M.BOUNDS_L, M.INIT_L,
                            lambda p, q, fixed: M.model_lorentz_vec(p, q),
                            lambda p, perr, fixed: M.derived_lorentz(p, perr),
                            'lorentz', _smoke_l),
    }
