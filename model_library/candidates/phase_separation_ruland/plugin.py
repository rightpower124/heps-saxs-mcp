# -*- coding: utf-8 -*-
"""phase_separation_ruland 适配层（2026-10-22 方案 B）：链幂律 + 界面 Porod×Ruland 修正
+ 广延峰 + 背景。σ_int = 界面层厚度直接产出（本任务最终目标物理量）。
方法学与可辨识性披露见 model.py docstring 与 manifest refs。
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
        c1 = ('1. **特征周期**：峰位 q0 = %.3g nm⁻¹（χ²/dof=%.3f）为**撞界条件值**，'
              'd_period 派生量同不作物理结论。' % (float(params['q0']), chi2_dof))
    else:
        c1 = ('1. **特征周期**：峰位 q0 = %.3f ± %.3f nm⁻¹，**d = 2π/q0 = %.1f ± %.1f nm**，'
              '峰 FWHM = %.3f ± %.3f nm⁻¹（χ²/dof=%.3f）。'
              % (float(params['q0']), float(perr['q0']), der['d_period_nm'],
                 der['d_period_nm_err'], der['fwhm_nm1'], der['fwhm_nm1_err'], chi2_dof))
    if 'sigma_int' in _at:
        c2 = ('2. **界面层厚度**：σ_int = %.3g nm 为**撞界条件值**，不作定量物理结论'
              '（可辨识性诊断见样品备注）。' % float(params['sigma_int']))
    else:
        c2 = ('2. **界面层厚度**：σ_int = %.3f ± %.3f nm（Ruland 弥散界面修正直接产出；'
              'Porod 偏离转折位置 q* ≈ 1/σ_int ≈ %.2f nm⁻¹）。'
              % (float(params['sigma_int']), float(perr['sigma_int']),
                 1.0 / float(params['sigma_int']) if float(params['sigma_int']) > 0 else float('nan')))
    c3 = ('3. **两相分解**：链形态幂律 dm = %.2f ± %.2f（幅度 A_c = %.3g）；'
          '界面 Porod 项 A_int = %.3g（∝ 界面比表面积 × Δρ²）。'
          '两项幂律共线风险（窗内 exp→1 极限）以 multistart/param_scan 取证为准。'
          % (float(params['dm']), float(perr['dm']), float(params['A_c']),
             float(params['A_int'])))
    return [
        c1, c2, c3,
        '4. **可靠性分级**：A_c/A_int/amp/B 为归一量；σ_int/dm/q0 为结构参数'
        '（两幂律共线披露见上），峰参数条件值判读；撞界参数（%s）仅作条件值呈现。'
        % ('、'.join(_at) if _at else '无'),
        '5. **方法学披露**：本模型 = Ruland 1971 弥散界面修正 + 链形态幂律 + 广延峰'
        '（方案 B）；与方案 A（stretched-exp 截断）构成高 q 衰减机制对照——'
        '本方案 σ_int 为可解释物理量（界面层厚度，任务最终目标）。', '']


def _mk_mode(names, bounds, init, calc, derived, mode_key, smoke_truth):
    return {
        'names': names,
        'bounds': bounds,
        'init': init,
        'init_qmax': 0.5,
        # 幅度初值通用化声明（2026-10-22，引擎 init_p0 缺口修复配套）：
        # 主幅度键 = A_c（引擎按低 q 窗数据标定 A0 写入该键）；
        # A_int 为比例幅度键（与主幅度等比缩放）
        'init_amp_key': 'A_c',
        'init_amp_scales': ['A_int'],
        'smoke_truth': smoke_truth,
        'fixed_keys': [],
        'normalized_params': ['A_c', 'A_int', 'amp', 'B'],
        'calc': calc,
        'derived': derived,
        'plot_layout': _LAYOUT_DOUBLE,
        'components': [
            {'label': 'chain A_c*q^-dm', 'style': 'g--',
             'column': 'I_chain_power_law',
             'calc': lambda q, p, fixed: p['A_c'] * np.asarray(q, dtype=float) ** (-p['dm'])},
            {'label': 'interface Porod*Ruland A_int*q^-4*exp(-(s*q)^2)', 'style': 'c-.',
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
            # 分析报告生成规范（2026-10-22 用户确立，workflow §5）：五段式架构——
            # 模型段（本模型+同族公式全部集中）/ 语义表（含谱系新增参数行）/
            # 派生量段 / 拟合协议段 / 备注段；公式 LaTeX、不允许分散多处
            'method': lambda fixed: [
                '## 3. 分析方法说明（模型与算法）', '',
                '**模型段（phase_separation_ruland，本报告定版模型）**：', '',
                '$$I(q) = A_c\\cdot q^{-d_m} + A_{int}\\cdot q^{-4}'
                '\\cdot e^{-\\sigma_{int}^2 q^2} + amp\\cdot P(q) + B$$', '',
                '$$P(q) = \\exp\\!\\left(-\\frac{(q-q_0)^2}{2\\sigma^2}\\right)\\ \\text{(gauss)}'
                '\\quad\\text{或}\\quad P(q) = \\frac{1}{1+\\left(\\frac{q-q_0}{w}\\right)^2}'
                '\\ \\text{(lorentz)}$$', '',
                '**同族升级版（phase_separation_2seg，末端底形两段化；'
                'χ²/dof 5.44→1.23，末端上翘消除）**：', '',
                '$$w(q) = \\frac{1}{1+e^{-(q-q_{br})/w_{br}}}, \\quad'
                ' d_m(q) = d_{m1}(1-w) + d_{m2}\\, w$$', '',
                '$$I_{2seg}(q) = A_c\\cdot q^{-d_m(q)} + A_{int}\\cdot q^{-4}'
                '\\cdot e^{-\\sigma_{int}^2 q^2} + amp\\cdot P(q) + B$$', '',
                '**各项物理语义段**（两模型共用参数语义；2seg 新增参数单独列）：', '',
                '| 项 | 参数 | 语义 |', '|---|---|---|',
                '| $A_c q^{-d_m}$ | A_c, dm | 链形态幂律（低 q 无规两相/链贡献；'
                '单幂律底恒定斜率无法表达高 q 段斜率演变） |',
                '| $A_c q^{-d_m(q)}$ | A_c, dm1, dm2, q_br, w_br | （2seg 版）平滑转折双幂律底：'
                'dm1 低 q 链形态；**dm2 > 4 = 界面弥散主导段**（Ruland 修正等效表达）；'
                'q_br = 转折位置；w_br = logistic 转折宽度 |',
                '| $A_{int} q^{-4} e^{-\\sigma_{int}^2 q^2}$ | A_int, σ_int | 界面 Porod 律 × Ruland'
                ' 弥散界面修正；**σ_int = 界面层厚度**（Ruland 1971）；'
                'σ_int 数值依赖机制分配假设，两模型间不可直接比较 |',
                '| $amp\\cdot P$ | amp, q0, σ/w | 相分离关联峰（特征周期 d = 2π/q0） |',
                '| $B$ | B | 背景残余 |', '',
                '**派生量段**：$d = 2\\pi/q_0$（误差 $\\sigma_d = 2\\pi\\sigma_{q0}/q_0^2$）；'
                'FWHM $= 2.3548\\sigma$（gauss）/ $2w$（lorentz）；σ_int 即界面层厚度'
                '（拟合参数直接产出，无派生换算链）；Porod 偏离转折位置'
                ' $q^* \\approx 1/\\sigma_{int}$（$q<q^*$ 纯 Porod，$q>q^*$ 修正显效）。', '',
                '**拟合协议段**：scipy `least_squares`（trf，1/σ² 加权，'
                '$\\chi^2 = \\sum_i[(I_{obs}-I_{calc})/\\sigma_i]^2$）；'
                'perr = $(J^TJ)^{-1}$ 对角平方根；撞界 / perr 爆表（>100%）自动降级条件值；'
                'multistart / param_scan 取证。', '',
                '**备注段**（次要内容，每条一行）：', '',
                '1. 模型假设见 §2"模型假设"五条（Ruland 单参数高斯过渡层近似、'
                '两幂律共线、σ_int 高 q 依赖等）。',
                '2. 与方案 A（stretched-exp 截断）构成高 q 衰减机制对照：'
                'χ² + 残差结构 + σ_int 可辨识性三方裁决。',
                '3. 本报告为 ruland 定版；2seg 为末端迭代升级版，完整交付见 '
                'runs/20261022_fullband/deliver_2seg_gauss/。', ''],
            'results_md': lambda der, chi2_dof, dof, at, nopeg_compare: [
                '**派生量**：',
                '- 特征周期 d = 2π/q0 = **%.2f ± %.2f nm**' % (der['d_period_nm'], der['d_period_nm_err']),
                '- 峰 FWHM = %.3f ± %.3f nm⁻¹' % (der['fwhm_nm1'], der['fwhm_nm1_err']),
                '- **界面层厚度 σ_int = %.3f ± %.3f nm**（拟合参数直接产出）'
                % (der['interface_thickness_nm'], der['interface_thickness_nm_err']),
                '', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
                % (chi2_dof, dof, '、'.join(at) if at else '无'), ''],
            'conclusions': _conclusions,
        },
    }


def register(M):
    # 冒烟真值：链项与界面项等幅（各 0.5）+ 峰主导，验证 A_c/dm/A_int/σ_int 全可恢复；
    # σ_int=0.8 使转折 q*=1.25 在冒烟网格 [0.3,4.4] 内显效
    _smoke_g = {'A_c': 0.5, 'dm': 2.5, 'A_int': 0.5, 'sigma_int': 0.8, 'amp': 20.0,
                'q0': 0.6, 'sigma': 0.5, 'B': 0.001}
    _smoke_l = {'A_c': 0.5, 'dm': 2.5, 'A_int': 0.5, 'sigma_int': 0.8, 'amp': 20.0,
                'q0': 0.6, 'w': 0.5, 'B': 0.001}
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
