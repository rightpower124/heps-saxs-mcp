# -*- coding: utf-8 -*-
"""
Al-Cu 合金盘状析出相 SAXS 模型插件（模型库 2 号模型，M3 首个文献自动化模型）
物理模型（Model III，Tsao & Huang et al., J. Alloys Compd. 579 (2013) 138-146, Eq.1+Eq.5）：

  I(Q) = A·Q⁻⁴ + s·P_disk(Q; R, t) + B

  A·Q⁻⁴   : 大尺寸杂质析出物的 Porod 幂律项（文献 Eq.5 首项，处理低 q 尾部）
  P_disk  : 圆柱盘（plate-like precipitate）形式因子的取向平均（文献 Eq.1 积分核）：
            P(Q) = ∫₀^{π/2} [2·j₀((Qt/2)·cosα)·J₁(QR·sinα)/(QR·sinα)]²·sinα dα
            j₀(x)=sin(x)/x；J₁(x)/x 在 x→0 取 0.5；P(0)=1（Guinier 平台归一）
  s       : 盘散射缩放 = φ·Δρ²（φ=体积分数，Δρ=散射长度密度差）
  B       : 常数背景（站内扩展：文献数据已完成绝对标定与背景扣除，
            本站数据为 a.u. 强度，以常数 B 吸收噪声平台；属使用披露项）

单位约定：引擎内 q 为 nm⁻¹，几何参数 R/t 以 nm 计；
Δρ 取文献统一口径 1.05×10⁻⁵ Å⁻² = 1.05×10⁻³ nm⁻²（θ′ 值，文献对全部相统一使用以简化相对体积分数比较）。
适用相：GP 区 / θ″（GP II）/ θ′ 前驱体等盘状析出相（文献 Model III 对应 θ″ 相；
数据无空间有序结构峰时 S(Q)≈1，本插件即该假设——与文献 A1h 样品处理一致）。
"""
import numpy as np
from scipy import special

# 文献统一 Δρ 口径（θ′ 值）：1.05e-5 Å⁻² = 1.05e-3 nm⁻²（1 Å⁻² = 100 nm⁻²）
DRHO = 1.05e-3          # nm⁻²
DRHO2 = DRHO * DRHO     # nm⁻⁴

# ---------- 参数表 ----------
NAMES = ['A', 's', 'R', 't', 'B']

# 边界：R/t 覆盖文献 Table 1/2 全部析出相尺度
#   GPZ  R≈0.8–16 nm, t≈0.24 nm；θ″ R≈8.5–11 nm, t≈1.5–1.9 nm；θ′ R≈22–60 nm, t≈3.7–26 nm
BOUNDS = {
    'A': (0.0, np.inf),
    's': (0.0, np.inf),
    'R': (0.5, 80.0),
    't': (0.05, 25.0),
    'B': (0.0, np.inf),
}
# 初值：θ″ 尺度附近起步（文献 Table 2 RT：R=8.54 nm, t=1.46 nm）
INIT = {'s': 100.0, 'R': 5.0, 't': 1.5}

# ---------- 圆柱盘形式因子取向平均（Eq.1 积分核） ----------
_ALPHA_N = 121          # 取向积分节点数（收敛性经 L1 验证）


def P_disk(qq, R, t, n_alpha=_ALPHA_N):
    """圆柱盘形式因子取向平均，P(0)=1 归一。
    qq: nm⁻¹ 数组；R: 盘半径 nm；t: 盘厚度 nm。
    被积函数端点奇异已按极限处理：J₁(x)/x→0.5（x→0），j₀(0)=1。
    """
    alpha = np.linspace(0.0, np.pi / 2.0, int(n_alpha))
    sa, ca = np.sin(alpha), np.cos(alpha)
    Qt2 = np.outer(qq, (t / 2.0) * ca)            # (n_q, n_a)  j₀ 自变量
    QRs = np.outer(qq, R * sa)                    # (n_q, n_a)  J₁ 自变量
    j0 = np.sinc(Qt2 / np.pi)                     # np.sinc(x)=sin(πx)/(πx) → sin(x)/x = np.sinc(x/π)
    b1 = np.full_like(QRs, 0.5)                   # x→0 极限 J₁(x)/x = 0.5
    nz = QRs > 1e-12
    b1[nz] = special.j1(QRs[nz]) / QRs[nz]
    core = 2.0 * j0 * b1
    # numpy 2.x：np.trapz 已移除，用 np.trapezoid（lessons.md 2026-09-30）
    return np.trapezoid(core * core * sa[None, :], alpha[None, :], axis=1)


# ---------- 向量形式模型（least_squares 用；参数顺序见 NAMES） ----------
def model_vec(p, qq):
    """5 参数：A, s, R, t, B（顺序按 NAMES）"""
    A, s, R, t, B = p
    return A * qq ** (-4.0) + s * P_disk(qq, R, t) + B


# ---------- 派生量 ----------
def derived_main(p, perr):
    """表观体积分数（a.u. 标度，依赖强度绝对标定——归一量披露）、盘几何与理论回转半径"""
    R, t = float(p['R']), float(p['t'])
    return {
        'phi_apparent': float(p['s']) / DRHO2,
        'phi_apparent_err': float(perr['s']) / DRHO2,
        'diameter_nm': 2.0 * R,
        'diameter_nm_err': 2.0 * float(perr['R']),
        'thickness_nm': t,
        'thickness_nm_err': float(perr['t']),
        'V_disk_nm3': float(np.pi * R * R * t),
        'Rg_theory_nm': float(np.sqrt(R * R / 2.0 + t * t / 12.0)),
    }


# =====================================================================
# Model I（文献 Eq.3）：I(Q) = A·Q⁻⁴ + s·P_disk(Q;R,t)·S_PY(Q; r_hs, η) + B
#   与 Model III 的差别：盘分量乘以 Percus-Yevick 硬球结构因子 S(Q)，
#   用于存在粒子间关联（结构峰）的 GPZ 类数据（文献 Table 1 NA/GPZ 各温度行）。
#   s 语义与 Model III 相同（= φ·Δρ²·V_disk，P(0)=1），保证两模式幅度可比。
# =====================================================================
NAMES1 = ['A', 's', 'R', 't', 'r_hs', 'eta', 'B']

BOUNDS1 = {
    'A': (0.0, np.inf),
    's': (0.0, np.inf),
    'R': (0.5, 80.0),
    't': (0.05, 25.0),
    'r_hs': (0.5, 120.0),      # 硬球半径 nm（峰位 ~X/(2·r_hs)，X 峰≈5–7）
    'eta': (1e-4, 0.50),       # 硬球体积分数：PY 闭合在 η≳0.5 精度下降，上界取 0.50（披露）
    'B': (0.0, np.inf),
}

# 初值（2026-10-01 multistart 双样品实证修订）：原起点 (R=2,t=0.5,r_hs=5,eta=0.05) 使
# T4-2days 落入坏局部盆（chi2 0.204，r_hs 撞下界）；本起点 (R=8,t=1.5,r_hs=8,eta=0.02)
# 使 2days/48days 单起点分别收敛到各自全局最优（chi2 0.0631 / 0.1157），48days 无回归。
INIT1 = {'s': 100.0, 'R': 8.0, 't': 1.5, 'r_hs': 8.0, 'eta': 0.02}


def S_py(qq, r_hs, eta):
    """Percus-Yevick 硬球结构因子（单分散，Kotlarchyk & Chen 1983 闭合式）。
    公式移植自 sasmodels 1.1.0 models/hardsphere.c（逐式核对，含小 X 泰勒分支）；
    自检：S(0)=(1−η)⁴/(1+2η)²，η=0.2、R=50 Å、q=0.001 Å⁻¹ → 0.209128 与 sasmodels tests 一致。
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


def model1_vec(p, qq):
    """7 参数：A, s, R, t, r_hs, eta, B（顺序按 NAMES1）"""
    A, s, R, t, r_hs, eta, B = p
    return A * qq ** (-4.0) + s * P_disk(qq, R, t) * S_py(qq, r_hs, eta) + B


def derived_1(p, perr):
    """Model I 派生量：盘几何（同 Model III）+ 硬球参数 + S(0) + 结构峰位置（网格取 max）"""
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
