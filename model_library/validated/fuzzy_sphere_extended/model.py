# -*- coding: utf-8 -*-
"""
fuzzy_sphere_extended 物理模型（Rathgeber blob 形式；模型库 5 号，Neubert 2026 桥接复现）

物理模型（main 纯形式因子）：
  I(q) = A·[ amp²(q) + rel_amp·P_blob(q) ] + B

  amp(q)   : 模糊球振幅 = [3j₁(qR)/(qR)]·exp(−(q·σ)²/2)
             R = 半密度半径（SLD 降为一半处），σ = 界面模糊宽度（高斯阻尼）
  P_blob(q): Rathgeber blob 散射（Ornstein-Zernike 型涨落项，Rathgeber et al.,
             J. Chem. Phys. 117 (2002) 4047-4062）：
             P_blob_raw(q) = sin(μ·arctan(q*)) / [ μ·q*·(1+q*²)^(μ/2) ]
             μ = 1/ν − 1（ν = Flory 指数，质量分形维数 D_f = 1/ν）
             q* = q·ξ / [erf(q·Rg/√6)]³（OZ 关联长度 ξ 开启因子，与 Beaucage
             统一模型的 q* 结构同族——Rg 大时低 q 截断）
  A        : 幅度（吸收 SASview 管线的 scale·Δρ²·V·1e-4 全链，归一量）
  B        : 常数背景

实现来源（L1b 对拍基准）：
  作者亲笔 SASview 插件 fuzzy_sphere_extended.py/.c（Neubert 2026 数据包
  SASviewModels/ 归档，Macromol. Chem. Phys. 2026, 227, e00493 应用锚）。
  .c 公式逐式转写：i_sphere = prefactor·amp²、i_blob = prefactor·rel_amp·P_blob_raw、
  V-scaling（Rathgeber 形式，非 Stieger fuzzy_sphere_fluct 的 V² + 1/cm i_fluc）。
  幅度链披露：SASview 管线 I = scale·(Δρ²·V·[amp²+rel_amp·P_blob_raw])·1e-4 + bkg
  中 scale·Δρ²·V·1e-4 对 a.u./任意标度数据不可辨（SI 表 scale=123.94 与数据
  幅度链条存在未闭合口径，见 manifest refs），站内以自由参数 A 整体吸收——
  结构参数（R/σ/rel_amp/Rg/ν/ξ）全部标度无关，不受幅度链口径影响。

sq 模式：I(q) = A·[amp² + rel_amp·P_blob_raw]·S_PY(q; r_hs, η) + B
  Percus-Yevick 硬球结构因子（Kotlarchyk & Chen 1983 闭合式，公式逐式移植自
  sasmodels 1.1.0 hardsphere.c，与 alcu_precipitate model1 同源同版）。
  适用：20 °C 低 q 粒子间关联（作者 SI Chapter 6：硬球 S(Q) 改善 20 °C 拟合，
  Hayter MSA 不适用——作者原文结论，见 manifest refs）。

与作者实现的已知差异（L2 披露项）：
  ① polydispersity 未纳入：作者 radius.width=0.4 nm（PD≈2.2%）高斯平均，
     站内第一版单分散；
  ② q→0 单点：作者 .c 在 erf<1e-10 时 P_blob_raw=0，站内取数学极限
     P_blob_raw(0)=1（连续，q=0 不在拟合域内，无实际影响）；
  ③ flory_nu 下界 1/3（作者 .py 为 0.25）：μ=3 时高 q 尾 sin(3π/2)<0 产生
     负强度，ν≥1/3 保证 μ≤2 → sin(μ·arctan(q*))>0 恒正。

单位约定：引擎内 q 为 nm⁻¹，几何参数 R/σ/Rg/ξ/r_hs 以 nm 计；
rel_amp/ν/η 无量纲。
"""
import numpy as np
from scipy import special

# ---------- 参数表（main：纯形式因子） ----------
NAMES = ['A', 'R', 'sigma_fuzzy', 'rel_amp', 'rg', 'flory_nu', 'xi', 'B']

BOUNDS = {
    'A': (0.0, np.inf),
    'R': (0.5, 200.0),          # nm（SI 锚 17.81；微凝胶~ dendrimer 尺度带）
    'sigma_fuzzy': (0.0, 100.0),  # nm（SI 锚 10.311）
    'rel_amp': (1e-6, 10.0),    # 无量纲（SI 锚 0.12621）
    'rg': (0.5, 100.0),         # nm（SI 锚 21）
    'flory_nu': (1.0 / 3.0, 0.7),  # 作者 .py 上界 0.7；下界 0.25→1/3 见文件头差异③
    'xi': (0.1, 100.0),         # nm（SI 锚 9.3354）
    'B': (0.0, np.inf),
}
# INIT 锚 SI Chapter 6 "Pure microgel in water" 20 °C 行（L2 复现起点；A/B 由引擎自估）
INIT = {'R': 17.81, 'sigma_fuzzy': 10.311, 'rel_amp': 0.126,
        'rg': 21.0, 'flory_nu': 0.4, 'xi': 9.335}


# ---------- 模糊球振幅 ----------
def sph_amp(qq, R, sigma):
    """amp(q) = [3j₁(qR)/(qR)]·exp(−(qσ)²/2)；qR→0 极限 = 1。"""
    qr = np.asarray(qq, dtype=float) * float(R)
    small = qr < 1e-12
    qrs = np.where(small, 1.0, qr)
    b = np.where(small, 1.0, 3.0 * special.j1(qrs) / qrs)
    return b * np.exp(-0.5 * (np.asarray(qq, dtype=float) * float(sigma)) ** 2)


# ---------- Rathgeber blob 散射（rel_amp 已乘） ----------
def pb_term(qq, rel_amp, rg, nu, xi):
    """rel_amp·sin(μ·arctan(q*)) / [μ·q*·(1+q*²)^(μ/2)]，q*→0 极限 = rel_amp。"""
    mu = 1.0 / float(nu) - 1.0
    q = np.asarray(qq, dtype=float)
    erf_val = special.erf(q * float(rg) / np.sqrt(6.0))
    denom = erf_val ** 3
    small = denom < 1e-12                     # q→0 截断（q=0 不在拟合域）
    qd = np.where(small, 1.0, denom)
    qs = q * float(xi) / qd
    num = np.sin(mu * np.arctan(qs))
    den = mu * qs * np.power(1.0 + qs * qs, mu / 2.0)
    raw = np.where(small, 1.0, num / den)     # q*→0 极限 = 1（文件头差异②）
    return float(rel_amp) * raw


# ---------- 向量形式模型（least_squares 用；参数顺序见 NAMES） ----------
def model_vec(p, qq):
    """8 参数：A, R, sigma_fuzzy, rel_amp, rg, flory_nu, xi, B（顺序按 NAMES）"""
    A, R, sig, ra, rg, nu, xi, B = p
    amp = sph_amp(qq, R, sig)
    return A * (amp * amp + pb_term(qq, ra, rg, nu, xi)) + float(B)


# ---------- 派生量（main） ----------
def derived_main(p, perr):
    """模糊球几何（R_SANS 外口径）、blob 分形维数、球形式因子第一极小。"""
    R, sig = float(p['R']), float(p['sigma_fuzzy'])
    nu = float(p['flory_nu'])
    r_sans = R + 2.0 * sig
    return {
        'R_nm': R,
        'R_nm_err': float(perr['R']),
        'two_sigma_nm': 2.0 * sig,
        'two_sigma_nm_err': 2.0 * float(perr['sigma_fuzzy']),
        'R_SANS_nm': r_sans,
        'R_SANS_nm_err': float(np.sqrt(float(perr['R']) ** 2
                                       + (2.0 * float(perr['sigma_fuzzy'])) ** 2)),
        'V_R_SANS_nm3': float(4.0 / 3.0 * np.pi * r_sans ** 3),
        'mu': 1.0 / nu - 1.0,
        'D_f': 1.0 / nu,
        'D_f_err': float(perr['flory_nu']) / nu ** 2,
        'q_first_min_nm1': 4.4934 / R,
        'q_first_min_nm1_err': 4.4934 * float(perr['R']) / R ** 2,
        'rg_nm': float(p['rg']),
        'rg_nm_err': float(perr['rg']),
        'xi_nm': float(p['xi']),
        'xi_nm_err': float(perr['xi']),
        'rel_amp': float(p['rel_amp']),
        'rel_amp_err': float(perr['rel_amp']),
        'flory_nu': nu,
        'flory_nu_err': float(perr['flory_nu']),
    }


# =====================================================================
# sq 模式：I(q) = A·[amp² + rel_amp·P_blob_raw]·S_PY(q; r_hs, η) + B
#   作者 SI Chapter 6（20 °C）：硬球 S(Q) 改善低 q 拟合；
#   radius.effective=32.81 nm、volfrac=0.1（SI 锚）；Hayter MSA 不适用（作者结论）。
# =====================================================================
NAMES_SQ = ['A', 'R', 'sigma_fuzzy', 'rel_amp', 'rg', 'flory_nu', 'xi',
            'r_hs', 'eta', 'B']

BOUNDS_SQ = {
    'A': (0.0, np.inf),
    'R': (0.5, 200.0),
    'sigma_fuzzy': (0.0, 100.0),
    'rel_amp': (1e-6, 10.0),
    'rg': (0.5, 100.0),
    'flory_nu': (1.0 / 3.0, 0.7),
    'xi': (0.1, 100.0),
    'r_hs': (0.5, 200.0),       # nm（SI 锚 radius.effective=32.81）
    'eta': (1e-4, 0.50),        # PY 闭合在 η≳0.5 精度下降（alcu 同款披露）
    'B': (0.0, np.inf),
}
INIT_SQ = {'R': 17.81, 'sigma_fuzzy': 10.311, 'rel_amp': 0.126,
           'rg': 21.0, 'flory_nu': 0.4, 'xi': 9.335,
           'r_hs': 32.81, 'eta': 0.1}


def S_py(qq, r_hs, eta):
    """Percus-Yevick 硬球结构因子（单分散，Kotlarchyk & Chen 1983 闭合式）。
    公式移植自 sasmodels 1.1.0 models/hardsphere.c（与 alcu_precipitate
    model1 同源同版，含小 X 泰勒分支）；自检：S(0)=(1−η)⁴/(1+2η)²，
    η=0.2、R=50 Å、q=0.001 Å⁻¹ → 0.209128 与 sasmodels tests 一致。
    r_hs: 硬球半径 nm；eta: 硬球体积分数；X = 2·q·r_hs = q·直径。
    """
    X0 = 1.0 / (1.0 - eta)
    aa = ((1.0 + 2.0 * eta) * X0) ** 2            # α
    bb = -6.0 * eta * ((1.0 + 0.5 * eta) * X0) ** 2   # β
    gg = 0.5 * eta * aa                            # γ
    X = 2.0 * np.asarray(qq, dtype=float) * float(r_hs)
    small = X < 0.05                               # sasmodels 双精度 CUTOFFHS
    Xs = np.where(small, 1.0, X)                   # 防 0 除（小 X 支取 1，另行泰勒）
    X2 = Xs * Xs
    X4 = X2 * X2
    sn, cs = np.sin(Xs), np.cos(Xs)
    ff_main = (aa * (sn - Xs * cs) / Xs
               + bb * (2.0 * Xs * sn - (X2 - 2.0) * cs - 2.0) / X2
               + gg * ((4.0 * X2 - 24.0) * Xs * sn - (X4 - 12.0 * X2 + 24.0) * cs + 24.0) / X4)
    S_main = 1.0 / (1.0 + 24.0 * eta * ff_main / X2)
    # 小 X 泰勒（X² 阶，sasmodels hardsphere.c L51）
    ff0 = 8.0 * aa + 6.0 * bb + 4.0 * gg
    ff_small = ff0 + (-0.8 * aa - bb / 1.5 - 0.5 * gg
                      + (aa / 35.0 + 0.0125 * bb + 0.02 * gg) * X * X) * X * X
    S_small = 1.0 / (1.0 + eta * ff_small)
    return np.where(small, S_small, S_main)


def model_sq_vec(p, qq):
    """10 参数：A, R, sigma_fuzzy, rel_amp, rg, flory_nu, xi, r_hs, eta, B"""
    A, R, sig, ra, rg, nu, xi, r_hs, eta, B = p
    amp = sph_amp(qq, R, sig)
    return (A * (amp * amp + pb_term(qq, ra, rg, nu, xi))
            * S_py(qq, r_hs, eta)) + float(B)


def derived_sq(p, perr):
    """sq 派生量：main 全部 + 硬球参数 + S(0) + PY 结构峰位置（网格取 max）。"""
    der = derived_main(p, perr)
    eta, r_hs = float(p['eta']), float(p['r_hs'])
    der['r_hs_nm'] = r_hs
    der['r_hs_nm_err'] = float(perr['r_hs'])
    der['eta'] = eta
    der['eta_err'] = float(perr['eta'])
    der['S0'] = float((1.0 - eta) ** 4 / (1.0 + 2.0 * eta) ** 2)
    qg = np.linspace(0.01, 30.0, 600)
    Sq = S_py(qg, r_hs, eta)
    der['q_peak_nm1'] = float(qg[int(np.argmax(Sq))])
    return der
