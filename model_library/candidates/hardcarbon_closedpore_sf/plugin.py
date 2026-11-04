# -*- coding: utf-8 -*-
"""hardcarbon_closedpore_sf 适配层（候选区，2026-10-27 立项）：
把 model.py 物理函数接入通用引擎 saxs_fit。
结构 = hardcarbon_closedpore main（闭孔谱族，物理关切）+ 加性 Lorentzian 峰项
（结构因子峰，nuisance：峰为用户确认的真实结构，但其结构参数不作物理结论，
仅作低 q 信号修正）。
嵌套关系：Apk=0 → hardcarbon_closedpore main（跨模式嵌套 F 检验，
ssr=chi2_dof×dof 换算纪律）。
"""
import numpy as np

# ---- 布局声明：两联图（拟合曲线 log 图 + 相对残差），与父模型同构 ----
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


def _f(x):
    x2 = x * x
    return x2 * x2 / (1.0 + x2) ** 2


def _fmt_at(names_at):
    return '、'.join(names_at) if names_at else '无'


def _conclusions(params, perr, chi2_dof, dof, at, der, nopeg_compare, sample_note):
    """结论文本钩子：撞界参数不得单独进入结论句（质量门纪律）"""
    _at = at or []
    if 'a1' in _at:
        c1 = ('1. **闭孔尺度**：D_pore = %.2f nm、Rg = %.2f nm、q_knee = %.2f nm⁻¹'
              '（χ²/dof=%.3f）；a1 撞界 → 全部闭孔尺度导出量为**条件值**，不作物理结论。'
              % (der['D_pore_nm'], der['Rg_nm'], der['q_knee_nm'], chi2_dof))
    else:
        c1 = ('1. **闭孔尺度**：等效孔径 D_pore = %.2f ± %.2f nm（直径口径，= 2·a1·√10）、'
              '回转半径 Rg = %.2f ± %.2f nm、膝位 q_knee = %.2f ± %.2f nm⁻¹'
              '（χ²/dof=%.3f）。'
              % (der['D_pore_nm'], der['D_pore_nm_err'], der['Rg_nm'], der['Rg_nm_err'],
                 der['q_knee_nm'], der['q_knee_nm_err'], chi2_dof))
    return [
        c1,
        '2. **膝位落窗核查（必读）**：闭孔尺度量的物理可引用性条件于 q_knee 落在拟合窗内'
        '（q_knee ∈ [qmin×1.5, qmax/1.5]）；膝在窗外时 a1 沿幅度组合简并谷漂移，'
        '仅幂律指数 a 稳健——以诊断工具（multistart/param_scan）证据链为准。',
        '3. **峰项 nuisance 披露**：Lorentzian 峰项为用户确认的真实结构信号'
        '（2026-10-27 裁决），但其结构参数（q0/FWHM/d）**不作物理结论**，仅作低 q '
        '信号修正；闭孔四件套为本模型唯一物理关切量。',
        '4. **可靠性分级**：A/Bp/Apk/B 为幅度量（归一量：同批数据强度标度内可比、'
        '跨标度不可直接比较）；a1 及全部闭孔导出尺度量为形状参数，可引用性以证据链'
        '为准；撞界参数（%s）仅作条件值呈现。' % _fmt_at(_at),
        '5. **方法学披露**：幂律指数 a 上界预置 7.0（2026-10-27 relax_bound 取证：'
        '样品2 full 模式 a=5.114±0.118 稳健）；a 为自由指数时 A 与 Bp 在 a→4 处存在'
        '共线风险，以 F 检验与诊断工具裁决。', '']


def _method(fixed):
    """main 模式方法学（五段式；全部公式集中于模型段 LaTeX 完整展示，逐条一行分条展示）"""
    return [
        '## 3. 分析方法说明（模型与算法）', '',
        '**模型段**（本模型全部公式集中于此，LaTeX 完整展示）：', '',
        'main 模式模型式（8 参数；幂律项 + Debye-Bueche 闭孔项 + Lorentzian 峰项 + 常数背景）：',
        '$$I(q) = A\\,q^{-a} + B_p\\,q^{-4}\\,F(a_1 q) + A_{\\mathrm{pk}}\\,L(q;q_0,w) + B$$',
        '',
        '闭孔项 Debye-Bueche 形状函数（Porod 口径：高 q→1，低 q→x⁴ 平台）：',
        '$$F(x) = \\frac{x^4}{\\left(1+x^2\\right)^2}$$',
        '',
        '峰项 Lorentzian 形状函数（Hammouda/NIST 口径，加性结构因子峰）：',
        '$$L(q;q_0,w) = \\frac{1}{1+\\left[(q-q_0)/w\\right]^2}$$',
        '',
        '嵌套关系：$A_{\\mathrm{pk}}=0$ 退化为 hardcarbon_closedpore main 模式（5 参数），可用嵌套 F 检验裁决峰项必要性。',
        '',
        '与文献字面式的参数映射（Porod 口径改写，逐项恒等）：',
        '$$B_p \\equiv B_{\\mathrm{SI}}\\;(\\mathrm{AEM\\ SI}),\\qquad B_p = \\Delta\\rho\\,B\\,a_1^2\\;(\\mathrm{Stevens\\ Eq.1})$$',
        '',
        '派生量公式（闭孔尺度为物理关切，误差为 $a_1$ 误差解析传播；峰量仅描述性）：',
        '$$R_g = a_1\\sqrt{6}$$',
        '',
        '$$R_{\\mathrm{sphere}} = a_1\\sqrt{10}$$',
        '',
        '$$D_{\\mathrm{pore}} = 2 a_1 \\sqrt{10}$$',
        '',
        '$$q_{\\mathrm{knee}} = 1/a_1$$',
        '',
        '$$\\mathrm{FWHM} = 2w$$',
        '',
        '$$d = 2\\pi/q_0$$',
        '',
        '幂律指数说明：Stevens 2000 原文钉值 3.3 为狭缝准直 Smearing（无孔石墨标样定标）；本站针孔准直无 Smearing，$a$ 自由拟合。',
        '',
        '**各项物理语义段**：',
        '',
        '| 参数 | 语义 | 单位 |',
        '|---|---|---|',
        '| $A$ | 幂律项幅度（颗粒/大结构散射） | a.u. |',
        '| $a$ | 幂律指数（自由；上界 7.0，relax_bound 取证） | — |',
        '| $B_p$ | 闭孔项 Porod 口径幅度（高 q 渐近 $B_p q^{-4}$） | a.u. |',
        '| $a_1$ | 闭孔 Debye 特征长度（电子密度涨落关联长度） | nm |',
        '| $A_{\\mathrm{pk}}$ | 峰幅度（**nuisance**） | a.u. |',
        '| $q_0$ | 峰位（**nuisance**，样品2 实测 ≈0.70 nm⁻¹） | nm⁻¹ |',
        '| $w$ | 峰半高半宽 HWHM（**nuisance**） | nm⁻¹ |',
        '| $B$ | 常数背景 | a.u. |',
        '',
        '**派生量段**（物理语义与口径说明；公式见模型段）：',
        '- $R_g$：闭孔 Debye 关联长度的回转半径口径（Porod 定义，Stevens 2000 引 Glatter & Kratky 1982）。',
        '- $R_{\\mathrm{sphere}}$：等效球形孔半径（由 $R_g = R\\sqrt{0.6}$ 反解）。',
        '- $D_{\\mathrm{pore}}$：等效球形孔**直径**；AEM SI 公式 "$R = 2 a_1 \\sqrt{10}$" 的记号 "R" 实为直径，跨文献比对注意 2 倍口径差。',
        '- $q_{\\mathrm{knee}}$：Debye 膝特征波数；落窗核查（可辨识性前提）见结论。',
        '- $\\mathrm{FWHM} = 2w$、$d = 2\\pi/q_0$：峰描述量，**nuisance，不作物理结论**。',
        '',
        '**拟合协议段**：',
        '- 引擎：scipy `least_squares`（trf，x_scale=jac）；σ 列存在时加权拟合。',
        '- 初值：A0 由低 q 估计带自动标定（init_qmax=0.5 nm⁻¹，避开峰区，防 A0 被峰抬高把峰项压入关闭谷）；B0 取高 q 末端均值。',
        '- 闭孔三参数初值取父模型收敛邻域（a=2.9 / Bp=3.0 / a1=0.50 nm）；峰参数按样品2 目测锚（q0=0.70 / w=0.15 / Apk=0.05）。',
        '- 撞界判据：双侧界 |v−bound| < 10⁻³·量程；跨量级界改用 5% 参数尺度相对距离；撞界参数自动降级为**条件值**。',
        '- 诊断工具：multistart（峰项+闭孔幅度简并排查）、param_scan（a1/q0 网格）、嵌套 F 检验（vs closedpore main，Apk=0）。',
        '',
        '**备注段**：',
        '- 方法学源头：Stevens & Dahn, J. Electrochem. Soc. 147 (12) 4428-4431 (2000)；Debye, Anderson & Brumberger, J. Appl. Phys. 28 (1957) 679。',
        '- 应用文献：Zheng et al., Adv. Energy Mater. 2024, 14, 2303064 SI Fig.S10；峰形：Hammouda B.F. (NIST) Lorentzian 峰模型口径。',
        '- 峰项立项依据（2026-10-27）：样品2 q≈0.65-0.70 nm⁻¹ 真实局部极大（SNR≈520，用户确认非本底扣除伪影）。',
        '- 模型选型诊断链：closedpore main χ²/dof=924（结构性失配）；full 双 Debye χ²/dof=377.6（Debye 形状函数无法产生局部极大）；hardcarbon_ts 试探 χ²/dof=474（C1>0 设计性禁峰）。',
        '- 空孔对比度固定（无 SLD 自由参数），无 SLD 族简并。',
        '- 模型假设见报告 §2（refs.assumptions）。', '']


def _results_md(der, chi2_dof, dof, at, nopeg_compare):
    return [
        '**派生量**（闭孔尺度为物理关切；误差为 a1 误差解析传播）：',
        '- D_pore = **%.3f ± %.3f nm**（直径口径）；R_sphere = %.3f ± %.3f nm；'
        'Rg = %.3f ± %.3f nm'
        % (der['D_pore_nm'], der['D_pore_nm_err'],
           der['R_sphere_nm'], der['R_sphere_nm_err'],
           der['Rg_nm'], der['Rg_nm_err']),
        '- q_knee = **%.3f ± %.3f nm⁻¹**（膝位；落窗核查见结论）'
        % (der['q_knee_nm'], der['q_knee_nm_err']),
        '- 峰项（nuisance，仅描述性）：q0 = %.3f nm⁻¹、FWHM = %.3f nm⁻¹、d = %.2f nm'
        % (der['peak_q0_inv_nm'], der['peak_FWHM_inv_nm'], der['peak_d_nm']),
        '', '**拟合质量**：χ²/dof = **%.3f**（dof=%d）；撞界参数：%s'
        % (chi2_dof, dof, _fmt_at(at or [])), '']


def register(M):
    main = {
        'names': M.NAMES,
        'bounds': M.BOUNDS,
        'init': M.INIT,
        'init_qmax': M.INIT_QMAX,
        'fixed_keys': [],
        'normalized_params': ['A', 'Bp', 'Apk', 'B'],
        'calc': lambda p, q, fixed: M.model_vec(p, q),
        'derived': lambda p, perr, fixed: M.derived_main(p, perr),
        'plot_layout': _LAYOUT_DOUBLE,
        # components 纯项契约（2026-10-27 用户裁决）：子曲线不含常数背景 B，
        # B 由拟合曲线.csv 的 I_background 独立列承载；
        # components_sum_to_fit=True 接受生成器加和硬门（I_fit ≡ Σ分量 + B）
        'components_sum_to_fit': True,
        'components': [
            {'label': 'power law: A*q^-a', 'style': 'b:',
             'column': 'I_power',
             'calc': lambda q, p, fixed: p['A'] * q ** (-p['a'])},
            {'label': 'closed pores: Bp*q^-4*F(a1*q)', 'style': 'g--',
             'column': 'I_closedpore',
             'calc': lambda q, p, fixed: p['Bp'] * q ** (-4.0) * _f(p['a1'] * q)},
            {'label': 'SF peak (nuisance): Apk*L(q0,w)', 'style': 'm-.',
             'column': 'I_sfpeak',
             'calc': lambda q, p, fixed: p['Apk']
                     / (1.0 + ((q - p['q0']) / p['w']) ** 2)},
        ],
        'report': {
            'method': _method,
            'results_md': _results_md,
            'conclusions': _conclusions,
        },
    }
    return {'main': main}
