# -*- coding: utf-8 -*-
"""guinier_porod 物理模型（Hammouda 2010 广义 Guinier–Porod）
公式与 sasmodels 内置 guinier_porod 参考实现逐式一致（L1b DirectModel 对拍验证）：
  n  = 3 - s
  ms = (P - s) / 2
  q1 = sqrt(n * ms) / Rg              # Guinier 支 → Porod 支衔接点（值与斜率双连续）
  q <= q1:  I = A * q^-s * exp(-(q·Rg)^2 / n)
  q >  q1:  I = A * q^-P * exp(-ms) * (n·ms / Rg^2)^ms
退化守卫（与 sasmodels 行为一致）：Rg<=0 或 P<=s → I = A·q^-s；n<=0 → I = 0。
q 单位 nm^-1，Rg 单位 nm（引擎 q_unit='A^-1' 时自动 ×10 换算，插件内部统一 nm）。
幅度量纲约定（L1b 对拍 261006 确认）：GP 幅度 G 带量纲，无量纲组合是 G·Rg^s
（Guinier/Porod 两支同携 Rg^s 因子）→ 同一对象在 Å 约定与 nm 约定下 A 差 10^s 倍；
A 只在本插件 nm 约定内自洽使用（归一量，禁跨单位/跨标度直接比较）。

复现锚点：Chaney et al., Chem. Mater. 37 (2025) 7882-7893 Figure 7a/7b
（tiered Guinier-Porod；论文 ref 69 = Hammouda, J. Appl. Cryst. 43 (2010) 716）。
"""
import numpy as np

NAMES = ['A', 'Rg', 's', 'P', 'B']

BOUNDS = {
    'A':  (0.0, np.inf),        # 幅度（a.u. 标度归一量）
    'Rg': (0.01, 500.0),        # 回转半径 nm
    's':  (0.0, 3.0),           # 维度参数 0=球 / 1=棒 / 2=片（Hammouda 口径）
    'P':  (0.5, 6.0),           # 高 q Porod 指数
    'B':  (0.0, np.inf),        # 常数背景
}

# 三模式初值（对标 Figure 7b 锚点：level3 Rg≈380 Å / level2 ≈140 Å / level1 ≈20 Å）
INIT_LEVEL3 = {'Rg': 38.0, 's': 0.0, 'P': 3.0}   # 窗 [0.025, 0.09] nm^-1，膝 0.056 在窗内
INIT_LEVEL2 = {'Rg': 14.0, 's': 0.0, 'P': 4.0}   # 窗 [0.1, 0.5] nm^-1，膝 0.175 在窗内
INIT_LEVEL1 = {'Rg': 2.0, 's': 0.0, 'P': 4.0}    # 窗 [0.4, 2.0] nm^-1，膝 1.22 在窗内
INIT_QMAX_LEVEL3 = 0.09    # A0 估计窗上界（=拟合窗全宽，数据 ~37 点）
INIT_QMAX_LEVEL2 = 0.25    # 半窗上界；selftest 冒烟网格低 q 补点 [0.125,0.3] 覆盖膝区
INIT_QMAX_LEVEL1 = 2.0     # A0 估计窗上界（=拟合窗全宽）


def _gp_core(q, rg, s, p_exp):
    """广义 Guinier-Porod 谱形（不含 A/B 标度）；q 为 ndarray"""
    n = 3.0 - s
    ms = (p_exp - s) / 2.0
    if rg <= 0.0 or p_exp <= s:
        return np.power(q, -s)
    if n <= 0.0:
        return np.zeros_like(q)
    q1 = np.sqrt(n * ms) / rg
    gu = np.exp(-(q * rg) ** 2 / n) / np.power(q, s)
    po = np.exp(-ms) * (n * ms / rg ** 2) ** ms / np.power(q, p_exp)
    return np.where(q <= q1, gu, po)


def model_vec(p, qq):
    """引擎接口：p 按 NAMES 顺序 [A, Rg, s, P, B]（向量或列表）"""
    A, rg, s, p_exp, B = p
    q = np.asarray(qq, dtype=float)
    return A * _gp_core(q, float(rg), float(s), float(p_exp)) + float(B)


def _q1_of(rg, s, p_exp):
    """衔接点 q1 = sqrt((3-s)(P-s)/2)/Rg；退化情形返回 NaN"""
    n = 3.0 - s
    ms = (p_exp - s) / 2.0
    if rg <= 0.0 or p_exp <= s or n <= 0.0:
        return float('nan')
    return float(np.sqrt(n * ms) / rg)


def derived_main(p, perr):
    """导出量：rg_A（Å，直接对标文献 Figure 7b）、q1_nm1（衔接点，数值差分误差传播）"""
    rg, s, p_exp = float(p['Rg']), float(p['s']), float(p['P'])
    rg_e, s_e, p_exp_e = float(perr['Rg']), float(perr['s']), float(perr['P'])
    q1 = _q1_of(rg, s, p_exp)
    # 中心差分传播（步长取参数误差与 1e-6 的大者，防 perr=0 时步长归零）
    h_r, h_s, h_p = max(abs(rg_e), 1e-6), max(abs(s_e), 1e-6), max(abs(p_exp_e), 1e-6)
    dq_dr = (_q1_of(rg + h_r, s, p_exp) - _q1_of(rg - h_r, s, p_exp)) / (2.0 * h_r)
    dq_ds = (_q1_of(rg, s + h_s, p_exp) - _q1_of(rg, s - h_s, p_exp)) / (2.0 * h_s)
    dq_dp = (_q1_of(rg, s, p_exp + h_p) - _q1_of(rg, s, p_exp - h_p)) / (2.0 * h_p)
    q1_err = float(np.sqrt((dq_dr * rg_e) ** 2 + (dq_ds * s_e) ** 2 + (dq_dp * p_exp_e) ** 2))
    return {
        'rg_nm': rg, 'rg_nm_err': rg_e,
        'rg_A': rg * 10.0, 'rg_A_err': rg_e * 10.0,   # ×10 直接对标文献 Å 值
        'q1_nm1': q1, 'q1_nm1_err': q1_err,
        'q1_rg': q1 * rg,    # = sqrt(n·ms) 无量纲自检（与 s、P 有关，与 Rg 无关）
    }
