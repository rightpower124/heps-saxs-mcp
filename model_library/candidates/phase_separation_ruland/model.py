# -*- coding: utf-8 -*-
"""
相分离全带宽模型（方案 B：phase_separation_ruland，候选区开发中）
物理形态（三项叠加，锚定 Ruland 1971 弥散界面修正）：

  mode gauss / lorentz 峰形与 phase_separation_bp 同源，底形替换为
  **链形态幂律 + 界面 Porod×Ruland 修正**两项：

      I(q) = A_c·q^(−dm) + A_int·q^(−4)·exp(−σ_int²·q²) + amp·Peak(q) + B

  A_c   : 链形态幂律幅度（a.u. 归一量）
  dm    : 链形态幂律指数（首测窗内等效 2.34）
  A_int : 界面 Porod 项幅度（a.u. 归一量；∝ 界面比表面积 × Δρ²）
  σ_int : **界面层厚度（nm）——Ruland 弥散界面修正因子，本模型最终目标物理量**
  amp   : 峰幅度（a.u. 归一量）
  q0    : 峰位（nm⁻¹）
  σ/w   : 峰宽（gauss 标准差 / lorentz 半高半宽，nm⁻¹）
  B     : 常数背景

物理依据（Ruland 1971, J. Appl. Cryst. 4, 70）：
- 两相体系 Porod 律 I ∝ q^-4 在高 q 的偏离 = 界面非无限锐利；
- Ruland 修正：界面电子密度过渡层（厚度 σ_int）使 Porod 项乘
  exp(−σ_int²·q²)——q 增大时界面"看起来"变弥散，散射衰减快于 q^-4；
- σ_int 直接产出**界面层厚度**（本文件夹主题"界面层厚度"的目标物理量）；
- 首测数据高 q 衰减 ≈q^-2.6 "变缓"的解读：q^-4·exp(−σ²q²) 相对窗内
  等效幂律（dm≈2.3）在高 q 反而衰减更慢（exp 因子截掉的正是 q^-4 的
  过陡尾部，综合等效指数介于 2~4 之间随 q 滑动）——两项底形可表达该行为。

与方案 A（phase_separation_exp stretched-exp 截断）的对照：
- A：高 q 衰减变缓 = 通用唯象截断（无界面物理语义）；
- B：高 q 行为 = 物理机制分解（链形态贡献 + 界面 Porod×弥散修正），
  σ_int 为可解释物理量；
- 两者以 χ² + 残差结构 + 参数可辨识性对拍裁决。

可辨识性注意（报告措辞依据）：
- A_c·q^-dm 与 A_int·q^-4·exp(−σ²q²) 两幂律项共线风险：窗内 q<1/σ_int 时
  exp→1，两项仅靠指数差（dm vs 4）区分——dm 与 4 差距不足时简并，
  multistart + param_scan(σ_int) 取证必备；
- 首测窗内 dm=2.34 离 4 有距离（两机制可分），但 σ_int 需高 q 数据（q>1/σ_int）
  强约束——拟合窗必须延伸到数据最大 q；
- 峰浅场景峰参数条件值判读（phase_separation_bp 同款）。
"""
import numpy as np

# ---------- 参数表（gauss 模式：链幂律 + 界面 Porod×Ruland + 高斯峰 + 背景） ----------
NAMES_G = ['A_c', 'dm', 'A_int', 'sigma_int', 'amp', 'q0', 'sigma', 'B']

BOUNDS_G = {
    'A_c': (0.0, np.inf),
    'dm': (0.5, 6.0),
    'A_int': (0.0, np.inf),
    'sigma_int': (0.05, 10.0),
    'amp': (0.0, np.inf),
    'q0': (0.1, 0.95),
    'sigma': (0.02, 0.6),
    'B': (0.0, np.inf),
}
# 初值：dm/amp/q0/sigma 沿 phase_separation_bp 首测谷；A_int/sigma_int 由高 q 段形态估
# （q=1 nm⁻¹ 处 I≈0.01，若以界面项为主 A_int≈I·q⁴≈0.01；σ_int 初值 0.5 nm 待高 q 段修正）
INIT_G = {'dm': 2.3, 'A_int': 0.01, 'sigma_int': 0.5, 'amp': 0.02, 'q0': 0.4, 'sigma': 0.15}


def _base_vec(p, qq):
    """链形态幂律 + 界面 Porod×Ruland 修正（公共底形）"""
    A_c, dm = p['A_c'], p['dm']
    A_int, s_int = p['A_int'], p['sigma_int']
    return A_c * qq ** (-dm) + A_int * qq ** (-4.0) * np.exp(-(s_int * qq) ** 2)


def model_gauss_vec(p, qq):
    """8 参数：A_c, dm, A_int, sigma_int, amp, q0, sigma, B（顺序按 NAMES_G）。"""
    p = p if isinstance(p, dict) else dict(zip(NAMES_G, p))
    return (_base_vec(p, qq)
            + p['amp'] * np.exp(-((qq - p['q0']) ** 2) / (2.0 * p['sigma'] ** 2))
            + p['B'])


def derived_gauss(p, perr):
    """特征周期 d=2π/q0、峰 FWHM、界面层厚度即 sigma_int 本身（显式透传）"""
    q0 = float(p['q0']); q0_e = float(perr['q0'])
    sg = float(p['sigma']); sg_e = float(perr['sigma'])
    s = float(p['sigma_int']); s_e = float(perr['sigma_int'])
    return {
        'd_period_nm': 2.0 * np.pi / q0,
        'd_period_nm_err': 2.0 * np.pi * q0_e / q0 ** 2,
        'fwhm_nm1': 2.3548200450309493 * sg,
        'fwhm_nm1_err': 2.3548200450309493 * sg_e,
        'interface_thickness_nm': s,
        'interface_thickness_nm_err': s_e,
    }


# ---------- 参数表（lorentz 模式） ----------
NAMES_L = ['A_c', 'dm', 'A_int', 'sigma_int', 'amp', 'q0', 'w', 'B']

BOUNDS_L = {
    'A_c': (0.0, np.inf),
    'dm': (0.5, 6.0),
    'A_int': (0.0, np.inf),
    'sigma_int': (0.05, 10.0),
    'amp': (0.0, np.inf),
    'q0': (0.1, 0.95),
    'w': (0.02, 0.6),
    'B': (0.0, np.inf),
}
INIT_L = {'dm': 2.3, 'A_int': 0.01, 'sigma_int': 0.5, 'amp': 0.02, 'q0': 0.4, 'w': 0.15}


def model_lorentz_vec(p, qq):
    """8 参数：A_c, dm, A_int, sigma_int, amp, q0, w, B（顺序按 NAMES_L）。"""
    p = p if isinstance(p, dict) else dict(zip(NAMES_L, p))
    return (_base_vec(p, qq)
            + p['amp'] / (1.0 + ((qq - p['q0']) / p['w']) ** 2)
            + p['B'])


def derived_lorentz(p, perr):
    q0 = float(p['q0']); q0_e = float(perr['q0'])
    w = float(p['w']); w_e = float(perr['w'])
    s = float(p['sigma_int']); s_e = float(perr['sigma_int'])
    return {
        'd_period_nm': 2.0 * np.pi / q0,
        'd_period_nm_err': 2.0 * np.pi * q0_e / q0 ** 2,
        'fwhm_nm1': 2.0 * w,
        'fwhm_nm1_err': 2.0 * w_e,
        'interface_thickness_nm': s,
        'interface_thickness_nm_err': s_e,
    }
