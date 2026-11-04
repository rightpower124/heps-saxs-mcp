# -*- coding: utf-8 -*-
"""
相分离/界面层广延峰模型插件（phase_separation_bp，候选区开发中）
物理形态（ phenomenological，锚定双连续相分离散射特征）：

  mode gauss = 幂律底 + 高斯广延峰 + 常数背景：
      I(q) = A·q^(−dm) + amp·exp(−(q−q0)²/(2σ²)) + B
  mode lorentz = 幂律底 + Lorentz 广延峰 + 常数背景：
      I(q) = A·q^(−dm) + amp / (1 + ((q−q0)/w)²) + B

  A   : 幂律幅度（a.u. 归一量）
  dm  : 幂律指数（底形；Porod 4=光滑界面，<4 弥散界面/链形态）
  amp : 峰幅度（a.u.，与 A 同量纲归一量）
  q0  : 峰位（nm⁻¹）——关联峰顶，d_period = 2π/q0 为特征周期
  σ/w : 峰宽（gauss: 标准差 nm⁻¹ / lorentz: 半高半宽 nm⁻¹）
  B   : 常数背景

方法学定位：
- 幂律+峰叠加形态 = Hammouda broad_peak（NIST/SASview）的"峰形松绑"版——
  broad_peak 的解析峰形（width_exp/shape_exp 强约束）对非对称宽浅峰失配
  （2026-10-22 相分离 1.dat 实证 χ²/dof=38.9-49.3 残差准周期摆动），
  本插件以标准 gauss/lorentz 峰形替换，峰位/峰宽全自由；
- 双连续相分离正源物理 = Teubner & Strey 1987（J. Chem. Phys. 87, 3195）：
  TS 关联函数在 c1<0 时产出周期性峰，d_period=2π/q0 与 ξ 的物理语义同 TS；
- gauss/lorentz 双模式构成峰形对拍对（嵌套不可比但形态可对照），
  以 F 检验/残差结构裁决峰形。

可辨识性注意（报告措辞依据）：
- 峰浅（amp ≪ 底）时 amp/q0/σ 沿"底形主导"简并谷滑动——首测数据峰/底比
  ≈1.3（log 尺度 0.11 dex），峰参数一律条件值判读，multistart 取证必备；
- 幂律底与峰在峰尾区（q 离 q0 >3σ）互相补偿，dm 与 σ 弱耦合。
"""
import numpy as np

# ---------- 参数表（gauss 模式） ----------
NAMES_G = ['A', 'dm', 'amp', 'q0', 'sigma', 'B']

BOUNDS_G = {
    'A': (0.0, np.inf),
    'dm': (0.5, 6.0),
    'amp': (0.0, np.inf),
    'q0': (0.1, 0.95),
    'sigma': (0.02, 0.6),
    'B': (0.0, np.inf),
}
# 初值（首测数据形态：q0=0.4 nm⁻¹ 峰顶、峰超量 ~0.02 a.u.、dm≈2.5）
INIT_G = {'dm': 2.5, 'amp': 0.02, 'q0': 0.4, 'sigma': 0.15}


def model_gauss_vec(p, qq):
    """6 参数：A, dm, amp, q0, sigma, B（顺序按 NAMES_G）。"""
    A, dm, amp, q0, sigma, B = p
    return A * qq ** (-dm) + amp * np.exp(-((qq - q0) ** 2) / (2.0 * sigma ** 2)) + B


def derived_gauss(p, perr):
    """特征周期 d=2π/q0 与峰 FWHM=2.3548σ"""
    q0 = float(p['q0']); q0_e = float(perr['q0'])
    sg = float(p['sigma']); sg_e = float(perr['sigma'])
    return {
        'd_period_nm': 2.0 * np.pi / q0,
        'd_period_nm_err': 2.0 * np.pi * q0_e / q0 ** 2,
        'fwhm_nm1': 2.3548200450309493 * sg,
        'fwhm_nm1_err': 2.3548200450309493 * sg_e,
    }


# ---------- 参数表（lorentz 模式） ----------
NAMES_L = ['A', 'dm', 'amp', 'q0', 'w', 'B']

BOUNDS_L = {
    'A': (0.0, np.inf),
    'dm': (0.5, 6.0),
    'amp': (0.0, np.inf),
    'q0': (0.1, 0.95),
    'w': (0.02, 0.6),
    'B': (0.0, np.inf),
}
INIT_L = {'dm': 2.5, 'amp': 0.02, 'q0': 0.4, 'w': 0.15}


def model_lorentz_vec(p, qq):
    """6 参数：A, dm, amp, q0, w, B（顺序按 NAMES_L）。"""
    A, dm, amp, q0, w, B = p
    return A * qq ** (-dm) + amp / (1.0 + ((qq - q0) / w) ** 2) + B


def derived_lorentz(p, perr):
    """特征周期 d=2π/q0 与峰 FWHM=2w"""
    q0 = float(p['q0']); q0_e = float(perr['q0'])
    w = float(p['w']); w_e = float(perr['w'])
    return {
        'd_period_nm': 2.0 * np.pi / q0,
        'd_period_nm_err': 2.0 * np.pi * q0_e / q0 ** 2,
        'fwhm_nm1': 2.0 * w,
        'fwhm_nm1_err': 2.0 * w_e,
    }
