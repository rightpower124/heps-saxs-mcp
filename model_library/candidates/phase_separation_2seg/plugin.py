# -*- coding: utf-8 -*-
"""phase_separation_2seg 适配层（2026-10-22 末端迭代）：平滑转折双幂律底 + 界面
Porod×Ruland 修正 + 广延峰 + 背景。模型赛马 M4 定形（χ²/dof 5.44→1.23）。
report 钩子按「分析报告生成规范」（2026-10-22 用户确立，workflow §5）编写：
公式一律 LaTeX，方法与算法合并单节，细节舍去结论分条。"""
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
    _at = at or []
    if 'q0' in _at:
        c1 = ('1. **特征周期**：峰位 q0 = %.3g nm⁻¹（χ²/dof=%.3f）为**撞界条件值**。'
              % (float(params['q0']), chi2_dof))
    else:
        c1 = ('1. **特征周期**：峰位 q0 = %.3f ± %.3f nm⁻¹，**d = 2π/q0 = %.1f ± %.1f nm**，'
              '峰 FWHM = %.3f ± %.3f nm⁻¹（χ²/dof=%.3f）。'
              % (float(params['q0']), float(perr['q0']), der['d_period_nm'],
                 der['d_period_nm_err'], der['fwhm_nm1'], der['fwhm_nm1_err'], chi2_dof))
    if 'sigma_int' in _at:
        c2 = ('2. **界面层厚度**：σ_int = %.3g nm 为**撞界条件值**，不作定量物理结论。'
              % float(params['sigma_int']))
    else:
        c2 = ('2. **界面层厚度**：σ_int = %.3f ± %.3f nm（条件值口径以 param_scan/multistart '
              '取证为准；转折位置 q_br = %.2f nm⁻¹）。'
              % (float(params['sigma_int']), float(perr['sigma_int']),
                 float(params['q_br'])))
    c3 = ('3. **双段底形**：低 q 指数 dm1 = %.2f ± %.2f（链/无规两相形态），'
          '高 q 指数 dm2 = %.2f ± %.2f（>Porod 4 = 界面弥散主导段），转折 q_br = %.2f nm⁻¹。'
          % (float(params['dm1']), float(perr['dm1']), float(params['dm2']),
             float(perr['dm2']), float(params['q_br'])))
    return [
        c1, c2, c3,
        '4. **可靠性分级**：A_c/A_int/amp/B 为归一量；dm1/dm2/q_br/σ_int 为结构参数'
        '（dm2 与 A_int/σ_int 共线披露），峰参数条件值判读；撞界参数（%s）仅作条件值呈现。'
        % ('、'.join(_at) if _at else '无'),
        '5. **方法学披露**：本模型 = phase_separation_ruland 的高 q 机制升级版'
        '（模型赛马 5 候选定形，χ²/dof 5.44→1.23）。', '']


def _mk_mode(names, bounds, init, calc, derived, mode_key, smoke_truth):
    return {
        'names': names,
        'bounds': bounds,
        'init': init,
        'init_qmax': 0.5,
        # 幅度初值通用化：主幅度键 A_c（引擎标定）；A_int 比例幅度键
        'init_amp_key': 'A_c',
        'init_amp_scales': ['A_int'],
        'smoke_truth': smoke_truth,
        'fixed_keys': [],
        'normalized_params': ['A_c', 'A_int', 'amp', 'B'],
        'calc': calc,
        'derived': derived,
        'plot_layout': _LAYOUT_DOUBLE,
        'components': [
            {'label': 'base 2-seg power law', 'style': 'g--',
             'column': 'I_two_segment_base',
             'calc': lambda q, p, fixed: (
                 p['A_c'] * np.asarray(q, dtype=float) ** (
                     p['dm1'] * (1.0 - 1.0 / (1.0 + np.exp(
                         -(np.asarray(q, dtype=float) - p['q_br']) / p['w_br'])))
                     + p['dm2'] * (1.0 / (1.0 + np.exp(
                         -(np.asarray(q, dtype=float) - p['q_br']) / p['w_br'])))))},
            {'label': 'interface Porod*Ruland', 'style': 'c-.',
             'column': 'I_interface_porod',
             'calc': lambda q, p, fixed: p['A_int'] * np.asarray(q, dtype=float) ** (-4.0)
             * np.exp(-(p['sigma_int'] * np.asarray(q, dtype=float)) ** 2)},
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
            # 分析报告生成规范（2026-10-22 用户确立，workflow §5）
            'method': lambda fixed: [
                # 五段式架构（workflow §5）：模型段（本模型+同族公式集中）/ 语义表 /
                # 派生量段 / 拟合协议段 / 备注段
                '## 3. 分析方法说明（模型与算法）', '',
                '**模型段（phase_separation_2seg，本报告定版模型）**：', '',
                '$$w(q) = \\frac{1}{1+e^{-(q-q_{br})/w_{br}}}, \\quad'
                ' d_m(q) = d_{m1}(1-w) + d_{m2} w$$',
                '$$I(q) = A_c \\cdot q^{-d_m(q)} + A_{int}\\cdot q^{-4}'
                ' e^{-\\sigma_{int}^2 q^2} + amp\\cdot P(q) + B$$', '',
                '$$P(q) = \\exp\\!\\left(-\\frac{(q-q_0)^2}{2\\sigma^2}\\right)\\ \\text{(gauss)}'
                '\\quad\\text{或}\\quad P(q) = \\frac{1}{1+\\left(\\frac{q-q_0}{w}\\right)^2}'
                '\\ \\text{(lorentz)}$$', '',
                '**同族前代（phase_separation_ruland，单幂律底版本；χ²/dof 5.44 vs 本模型'
                ' 1.23，末端 +40% 上翘由底形两段化消除）**：', '',
                '$$I_{ruland}(q) = A_c\\cdot q^{-d_m} + A_{int}\\cdot q^{-4}'
                ' e^{-\\sigma_{int}^2 q^2} + amp\\cdot P(q) + B$$', '',
                '**各项物理语义段**（本模型参数；ruland 前代差异见备注段）：', '',
                '| 项 | 参数 | 语义 |', '|---|---|---|',
                '| $A_c q^{-d_m(q)}$ | A_c, dm1, dm2, q_br, w_br | 平滑转折双幂律底——'
                'dm1 低 q 链形态；**dm2>4 高 q 界面弥散主导段**（Ruland 修正等效表达）；'
                'q_br 转折位置；w_br logistic 转折宽度 |',
                '| $A_{int} q^{-4} e^{-\\sigma_{int}^2 q^2}$ | A_int, σ_int | 界面 Porod×Ruland 修正；'
                '**σ_int = 界面层厚度**（与 dm2 段互补；数值依赖机制分配假设，与 ruland 版不可直接比较） |',
                '| $amp\\cdot P$ | amp, q0, σ/w | 相分离关联峰（特征周期 d = 2π/q0） |',
                '| $B$ | B | 背景残余 |', '',
                '**派生量段**：$d = 2\\pi/q_0$（误差 $\\sigma_d = 2\\pi\\sigma_{q0}/q_0^2$）；'
                'FWHM $= 2.3548\\sigma$（gauss）/ $2w$（lorentz）；σ_int 即界面层厚度'
                '（拟合参数直接产出）；q_br 与 1/σ_int 的自洽性为机制分配诊断。', '',
                '**拟合协议段**：scipy `least_squares`（trf，1/σ² 加权，'
                '$\\chi^2 = \\sum_i[(I_{obs}-I_{calc})/\\sigma_i]^2$）；'
                'perr = $(J^TJ)^{-1}$ 对角平方根；撞界 / perr 爆表（>100%）自动降级条件值；'
                'multistart / param_scan 取证。', '',
                '**备注段**（次要内容，每条一行）：', '',
                '1. 模型假设见 §2"模型假设"五条（组合模型口径、dm2-A_int-σ_int 共线、'
                'σ_int 高 q 依赖等）。',
                '2. 立项依据 = 模型赛马 5 候选同窗对拍（M0 ruland 5.439 / M1 gen-ruland 5.216 /'
                ' M2 双 Ruland 5.454 / M3 +OZ 尾 3.919 / **M4 本模型 1.226**），'
                '底形两段化是唯一同时改善峰窗与高 q 尾的方案。',
                '3. 证据链与对照交付见 runs/20261022_fullband/（ruland 定版 deliver_ruland_qmax13/）。', ''],
            'results_md': lambda der, chi2_dof, dof, at, nopeg_compare: [
                '**派生量**：',
                '- 特征周期 d = 2π/q0 = **%.2f ± %.2f nm**' % (der['d_period_nm'], der['d_period_nm_err']),
                '- 峰 FWHM = %.3f ± %.3f nm⁻¹' % (der['fwhm_nm1'], der['fwhm_nm1_err']),
                '- **界面层厚度 σ_int = %.3f ± %.3f nm**（条件值口径，见样品备注）'
                % (der['interface_thickness_nm'], der['interface_thickness_nm_err']),
                '', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
                % (chi2_dof, dof, '、'.join(at) if at else '无'), ''],
            'conclusions': _conclusions,
        },
    }


def register(M):
    # 冒烟真值（2026-10-22 取证修订）：amp=20 会使 lorentz 长尾吞掉 98.6-99.96% 总
    # 强度（lessons 10-06 信号淹没型同族——底形转折特征无痕迹，A_c/dm1/dm2/q_br
    # 沿平坦方向漂移假报）；真实物理形态本就是"峰浅底强"，冒烟真值 amp=0.05
    _smoke_g = {'A_c': 0.5, 'dm1': 1.0, 'dm2': 5.0, 'q_br': 0.9, 'w_br': 0.15,
                'A_int': 0.0008, 'sigma_int': 5.15, 'amp': 0.05, 'q0': 0.6, 'sigma': 0.1, 'B': 0.001}
    _smoke_l = {'A_c': 0.5, 'dm1': 1.0, 'dm2': 5.0, 'q_br': 0.9, 'w_br': 0.15,
                'A_int': 0.0008, 'sigma_int': 5.15, 'amp': 0.05, 'q0': 0.6, 'w': 0.1, 'B': 0.001}
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
