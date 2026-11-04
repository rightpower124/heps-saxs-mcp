# -*- coding: utf-8 -*-
"""phase_separation_exp 适配层（2026-10-22 方案 A）：幂律×stretched-exp 底 + 广延峰 + 背景。
方法学与可辨识性披露见 model.py docstring 与 manifest refs；峰形对拍双模式
gauss / lorentz；report 钩子含「算法与公式」节（2026-10-22 用户新需求，方式一固化）。"""
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


def _algo_section(mode_label):
    """「算法与公式」节（2026-10-22 用户新需求，方式一：固化进插件 report 钩子，
    未来任何样品分析自动携带）"""
    return ['### 算法与公式', '',
            '**模型完整数学表达（%s 模式）**：' % mode_label, '',
            '$$I(q) = A\\cdot q^{-d_m}\\cdot \\exp\\!\\left(-\\left(q\\xi\\right)^{2\\beta}\\right)'
            ' + amp\\cdot P(q;q_0,w) + B$$', '',
            '其中峰形 $P(q;q_0,w)$ 按模式取：', '',
            '- gauss: $P = \\exp\\!\\left(-\\dfrac{(q-q_0)^2}{2\\sigma^2}\\right)$'
            '（$\\sigma$ = 标准差，FWHM $= 2\\sqrt{2\\ln 2}\\,\\sigma \\approx 2.3548\\,\\sigma$）',
            '- lorentz: $P = \\dfrac{1}{1+\\left(\\frac{q-q_0}{w}\\right)^2}$'
            '（$w$ = 半高半宽，FWHM $= 2w$）', '',
            '**各项物理语义**：', '',
            '| 项 | 参数 | 语义 |', '|---|---|---|',
            '| $A q^{-d_m}$ | A, dm | 低 q 链形态幂律底（dm=2 无限薄界面/链，→4 光滑界面 Porod 极限） |',
            '| $\\exp(-(q\\xi)^{2\\beta})$ | ξ, β | **stretched-exponential 截断**——高 q 衰减变缓的唯象因子；'
            'ξ=截断尺度（转折位置 q_c=1/ξ），β=拉伸指数（β=1 纯高斯截断，β<1 sub-exp 拉伸，'
            'β>1 压缩指数） |',
            '| $amp\\cdot P$ | amp, q0, σ/w | 相分离关联峰（域周期性），$d=2\\pi/q_0$ 为特征周期 |',
            '| $B$ | B | 常数背景（未扣背景残余+噪声平台） |', '',
            '**派生量计算公式**：', '',
            '- 特征周期：$d = 2\\pi / q_0$（误差 $\\sigma_d = 2\\pi\\,\\sigma_{q0}/q_0^2$）',
            '- 峰宽：FWHM 见上（gauss $2.3548\\sigma$ / lorentz $2w$）',
            '- 截断位置：$q_c = 1/\\xi$（误差 $\\sigma_{q_c} = \\sigma_\\xi/\\xi^2$）', '',
            '**拟合引擎算法**：', '',
            '- scipy `least_squares`（Trust Region Reflective，x_scale=jac）；'
            '目标函数 $\\chi^2 = \\sum_i \\left[\\frac{I_{obs}(q_i)-I_{calc}(q_i)}{\\sigma_i}\\right]^2$'
            '（1/σ² 加权，σ 取数据 Sig 列）',
            '- 参数误差 perr = 协方差矩阵对角线平方根，$C = (J^T J)^{-1}$（J = 加权残差雅可比）',
            '- 撞界判据：$\\|v-bound\\| < 10^{-3}\\times$参数量程；撞界参数自动降级**条件值**',
            '- perr 爆表判据：perr/\\|value\\| > 1.0 → 条件值（quality_gate grade_params 统一实现）', '',
            '**初值策略（init_p0）**：B0 = 数据高 q 末端 100 点均值；'
            'A0 = median[(I−B0)/I_shape]（I_shape = A=1,B=0 纯形状在低 q 窗口内；'
            '估计带 = q ≤ init_qmax），非有限或 ≤0 时兜底（lessons 10-01/10-02/10-09 三变体）', '',
            '**诊断算法**：', '',
            '- multistart：多起点独立重拟合，全局性取证（全部起点同谷 = 全局唯一解）',
            '- param_scan：固定争议参数于网格值、其余 warm-start 重拟合，量化简并宽度',
            '- F 检验（嵌套模型）：'
            '$F = \\frac{(SSR_1-SSR_2)/(k_2-k_1)}{SSR_2/(n-k_2)}$，'
            'SSR = χ²/dof × dof 换算', '',
            '**参考文献锚点**：', '',
            '1. Stretched-exponential scattering cutoff：聚合物/胶体体系通用截断形'
            '（Dingenouts et al., Macromolecules 31 (1998) 或同族；SASview 无内置同名模型，站内自实现）',
            '2. 广延峰：Hammouda B.F. (NIST), broad peak model（SASview/sasmodels broad_peak 同源）',
            '3. 双连续相分离正源物理：Teubner M., Strey R., J. Chem. Phys. 87 (1987) 3195', '']


def _conclusions(params, perr, chi2_dof, dof, at, der, nopeg_compare, sample_note):
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
    c2 = ('2. **截断行为（高 q 衰减变缓）**：截断尺度 ξ = %.3g ± %.3g nm（转折位置 '
          'q_c = 1/ξ = %.3f ± %.3f nm⁻¹），拉伸指数 β = %.3g ± %.3g——高 q 区数据衰减'
          '相对纯幂律变缓的唯象表达（弥散界面/有限域关联）。'
          % (float(params['xi']), float(perr['xi']), der['cutoff_q_nm1'],
             der['cutoff_q_nm1_err'], float(params['beta']), float(perr['beta'])))
    c3 = ('3. **幂律底**：dm = %.2f ± %.2f（低 q 链形态等效指数；与截断因子 ξ/β 在'
          ' q << 1/ξ 极限共线，A/dm/ξ/β 联合判读）。' % (float(params['dm']), float(perr['dm'])))
    return [
        c1, c2, c3,
        '4. **可靠性分级**：A/amp/B 为归一量；ξ/β/dm 为形状参数（低 q 共线披露见上），'
        '峰参数条件值判读并以 multistart/param_scan 取证；撞界参数（%s）仅作条件值呈现。'
        % ('、'.join(_at) if _at else '无'),
        '5. **方法学披露**：本模型 = Hammouda broad_peak 峰形松绑 + stretched-exp 截断底形'
        '（方案 A）；与方案 B（Ruland 界面修正）构成高 q 衰减机制的两种唯象对照；'
        '完整「算法与公式」见报告末节。', '']


def _mk_mode(names, bounds, init, calc, derived, mode_key, smoke_truth):
    mode_label = 'gauss' if mode_key == 'gauss' else 'lorentz'
    return {
        'names': names,
        'bounds': bounds,
        'init': init,
        'init_qmax': 0.5,
        'smoke_truth': smoke_truth,
        'fixed_keys': [],
        'normalized_params': ['A', 'amp', 'B'],
        'calc': calc,
        'derived': derived,
        'plot_layout': _LAYOUT_DOUBLE,
        'components': [
            {'label': 'base A*q^-dm*exp(-(q*xi)^(2b))', 'style': 'g--',
             'column': 'I_power_law_cutoff',
             'calc': lambda q, p, fixed: p['A'] * np.asarray(q, dtype=float) ** (-p['dm'])
             * np.exp(-((np.asarray(q, dtype=float) * p['xi']) ** p['beta']) ** 2)},
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
                '**模型（phase_separation_exp，方案 A：幂律×stretched-exp 底 + 广延峰 + 背景）**：',
                '- mode gauss：I(q) = A·q^(−dm)·exp(−(qξ)^(2β)) + amp·exp(−(q−q0)²/2σ²) + B',
                '- mode lorentz：I(q) = A·q^(−dm)·exp(−(qξ)^(2β)) + amp/(1+((q−q0)/w)²) + B', '',
                '**物理语义**：',
                '- 底形 = 低 q 链形态幂律 × stretched-exponential 截断（ξ=截断尺度，'
                'β=拉伸指数；高 q 衰减变缓的唯象表达）',
                '- 峰 = 相分离关联峰（q0 → 特征周期 d=2π/q0）',
                '- B = 背景残余', '',
                '**方法学锚点**：Hammouda broad_peak 峰形松绑；Teubner & Strey 1987 物理语义；'
                'stretched-exp 截断层站内自实现（SASview 无内置同名模型）。', '',
                '**拟合协议**：scipy least_squares（trf, 1/σ² 加权）；撞界/perr 爆表自动降级条件值；'
                '峰浅场景峰参数以 multistart/param_scan 取证。', ''],
            'results_md': lambda der, chi2_dof, dof, at, nopeg_compare: [
                '**派生量**：',
                '- 特征周期 d = 2π/q0 = **%.2f ± %.2f nm**' % (der['d_period_nm'], der['d_period_nm_err']),
                '- 峰 FWHM = %.3f ± %.3f nm⁻¹' % (der['fwhm_nm1'], der['fwhm_nm1_err']),
                '- 截断位置 q_c = 1/ξ = %.3f ± %.3f nm⁻¹' % (der['cutoff_q_nm1'], der['cutoff_q_nm1_err']),
                '', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
                % (chi2_dof, dof, '、'.join(at) if at else '无'), ''],
            'conclusions': _conclusions,
            # 「算法与公式」节（2026-10-22 用户新需求，方式一固化）
            'algo': lambda: _algo_section(mode_label),
        },
    }


def register(M):
    _smoke_g = {'A': 1.0, 'dm': 2.5, 'xi': 2.0, 'beta': 1.0, 'amp': 20.0,
                'q0': 0.6, 'sigma': 0.5, 'B': 0.001}
    _smoke_l = {'A': 1.0, 'dm': 2.5, 'xi': 2.0, 'beta': 1.0, 'amp': 20.0,
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
