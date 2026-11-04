# -*- coding: utf-8 -*-
"""
相分离全带宽广延峰模型（方案 A：phase_separation_exp，候选区开发中）
物理形态（phenomenological，锚定双连续相分离 + 高 q 弥散衰减）：

  mode gauss / lorentz 与 phase_separation_bp 同源，底形扩展为
  **幂律 × 拉伸指数截断**（stretched exponential cutoff）：

      I(q) = A·q^(−dm)·exp(−(q·ξ)^(2β)) + amp·Peak(q; q0, 宽) + B

  A   : 幂律幅度（a.u. 归一量）
  dm  : 低 q 幂律指数（链形态；首测窗内等效 2.34）
  ξ   : 截断尺度（nm）——高 q 衰减由 q^dm 变缓的转折位置
  β   : 拉伸指数（0<β≤1 sub-exp 拉伸 / β=1 纯高斯截断 exp(−(qξ)²)）
  amp : 峰幅度（a.u. 归一量）
  q0  : 峰位（nm⁻¹）
  σ/w : 峰宽（gauss 标准差 / lorentz 半高半宽，nm⁻¹）
  B   : 常数背景

立项依据（2026-10-22 相分离 1.dat 实测）：
- 峰窗 [0.075, 0.9] 内 phase_separation_bp gauss χ²/dof=2.31（dm=2.34），
  但窗外 q∈[0.9, 1.456] 衰减仅 ≈q^-2.6——单幂律底跨两段斜率行为结构性不足；
- stretched-exp 截断 = 弥散界面/有限域关联的通用唯象表达（高 q 衰减变缓
  = 界面电子密度渐变 + 域尺寸有限性），与方案 B 的 Ruland 修正互为对照。

方法学锚点：
- Stretched exponential scattering cutoff：Dingenouts et al. / Bender & Ehrlich
  等聚合物胶体体系通用截断形（SASview 无内置同名模型，站内自实现）；
- 广延峰：Hammouda broad_peak（NIST/SASview）峰形松绑版（phase_separation_bp 同款）；
- 双连续相分离正源物理：Teubner & Strey 1987, J. Chem. Phys. 87, 3195。

可辨识性注意（报告措辞依据）：
- ξ 与 dm 低 q 共线（q << 1/ξ 时 exp→1，I≈A·q^-dm 仅乘积可辨——首测窗内
  dm=2.34 即此极限的等效指数），ξ/β 主要由高 q 段（q > 1/ξ）约束；
- 峰浅场景（首测峰/底比 1.3）峰参数条件值判读，multistart 取证必备。
"""
import numpy as np

# ---------- 参数表（gauss 模式：幂律×stretched-exp 底 + 高斯峰 + 背景） ----------
NAMES_G = ['A', 'dm', 'xi', 'beta', 'amp', 'q0', 'sigma', 'B']

BOUNDS_G = {
    'A': (0.0, np.inf),
    'dm': (0.5, 6.0),
    'xi': (0.3, 50.0),
    'beta': (0.3, 1.5),
    'amp': (0.0, np.inf),
    'q0': (0.1, 0.95),
    'sigma': (0.02, 0.6),
    'B': (0.0, np.inf),
}
# 初值：dm/amp/q0/sigma 沿 phase_separation_bp 首测谷；xi/beta 由高 q 段形态估
# （q≈1 nm⁻¹ 处衰减开始变缓 → 1/ξ≈0.5 → ξ≈2 nm；β=1 纯高斯截断起步）
INIT_G = {'dm': 2.3, 'xi': 2.0, 'beta': 1.0, 'amp': 0.02, 'q0': 0.4, 'sigma': 0.15}


def _base_vec(p, qq):
    """幂律 × stretched-exp 截断底形（amp 峰与背景之前的公共部分）"""
    A, dm, xi, beta = p['A'], p['dm'], p['xi'], p['beta']
    return A * qq ** (-dm) * np.exp(-((qq * xi) ** beta) ** 2)


def model_gauss_vec(p, qq):
    """8 参数：A, dm, xi, beta, amp, q0, sigma, B（顺序按 NAMES_G）。"""
    p = p if isinstance(p, dict) else dict(zip(NAMES_G, p))
    return (_base_vec(p, qq)
            + p['amp'] * np.exp(-((qq - p['q0']) ** 2) / (2.0 * p['sigma'] ** 2))
            + p['B'])


def derived_gauss(p, perr):
    """特征周期 d=2π/q0、峰 FWHM=2.3548σ、截断尺度 1/ξ（高 q 衰减变缓位置）"""
    q0 = float(p['q0']); q0_e = float(perr['q0'])
    sg = float(p['sigma']); sg_e = float(perr['sigma'])
    xi = float(p['xi']); xi_e = float(perr['xi'])
    return {
        'd_period_nm': 2.0 * np.pi / q0,
        'd_period_nm_err': 2.0 * np.pi * q0_e / q0 ** 2,
        'fwhm_nm1': 2.3548200450309493 * sg,
        'fwhm_nm1_err': 2.3548200450309493 * sg_e,
        'cutoff_q_nm1': 1.0 / xi,
        'cutoff_q_nm1_err': xi_e / xi ** 2,
    }


# ---------- 参数表（lorentz 模式） ----------
NAMES_L = ['A', 'dm', 'xi', 'beta', 'amp', 'q0', 'w', 'B']

BOUNDS_L = {
    'A': (0.0, np.inf),
    'dm': (0.5, 6.0),
    'xi': (0.3, 50.0),
    'beta': (0.3, 1.5),
    'amp': (0.0, np.inf),
    'q0': (0.1, 0.95),
    'w': (0.02, 0.6),
    'B': (0.0, np.inf),
}
INIT_L = {'dm': 2.3, 'xi': 2.0, 'beta': 1.0, 'amp': 0.02, 'q0': 0.4, 'w': 0.15}


def model_lorentz_vec(p, qq):
    """8 参数：A, dm, xi, beta, amp, q0, w, B（顺序按 NAMES_L）。"""
    p = p if isinstance(p, dict) else dict(zip(NAMES_L, p))
    return (_base_vec(p, qq)
            + p['amp'] / (1.0 + ((qq - p['q0']) / p['w']) ** 2)
            + p['B'])


def derived_lorentz(p, perr):
    q0 = float(p['q0']); q0_e = float(perr['q0'])
    w = float(p['w']); w_e = float(perr['w'])
    xi = float(p['xi']); xi_e = float(perr['xi'])
    return {
        'd_period_nm': 2.0 * np.pi / q0,
        'd_period_nm_err': 2.0 * np.pi * q0_e / q0 ** 2,
        'fwhm_nm1': 2.0 * w,
        'fwhm_nm1_err': 2.0 * w_e,
        'cutoff_q_nm1': 1.0 / xi,
        'cutoff_q_nm1_err': xi_e / xi ** 2,
    }
