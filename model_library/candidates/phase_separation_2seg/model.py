# -*- coding: utf-8 -*-
"""
相分离全带宽模型（M4 定形：phase_separation_2seg，候选区开发中）
物理形态（三项叠加：平滑转折双幂律底 + 界面 Porod×Ruland 修正 + 广延峰 + 背景）：

  底形 = 两段幂律的 logistic 平滑转折：
      dm_eff(q) = dm1·(1−w) + dm2·w,  w = 1/(1+exp(−(q−q_br)/w_br))
      I_base(q) = A_c·q^(−dm_eff(q))

  mode gauss:   I(q) = I_base(q) + A_int·q^(−4)·exp(−σ_int²q²) + amp·exp(−(q−q0)²/2σ²) + B
  mode lorentz: I(q) = I_base(q) + A_int·q^(−4)·exp(−σ_int²q²) + amp/(1+((q−q0)/w)²) + B

  A_c      : 底形幅度（a.u. 归一量）
  dm1      : 低 q 幂律指数（链形态；首测等效 1.05）
  dm2      : 高 q 幂律指数（**可超越 Porod 极限 4**——dm2>4 段即 Ruland 弥散界面
             修正的等效表达：界面电子密度渐变使高 q 衰减陡于 q^-4 又慢于
             纯指数截断；首测 5.10）
  q_br     : 转折位置（nm⁻¹）——链形态主导到界面弥散主导的 crossover
  w_br     : 转折宽度（nm⁻¹）——logistic 平滑尺度（越小转折越锐）
  A_int    : 界面 Porod 项幅度（与 dm2>4 段互补描述界面散射，归一量）
  σ_int    : 界面层厚度（nm，Ruland 1971 修正因子；注意双幂律底承接部分界面
             行为后 σ_int 数值会与单幂律底版本不同——机制分配变化）
  amp/q0/σ(w) : 相分离关联峰（d=2π/q0 特征周期）
  B        : 常数背景

立项依据（2026-10-22 模型赛马，5 候选同窗对拍，1/σ² 加权 multistart）：
  M0 定版 ruland χ²/dof=5.439 / M1 gen-ruland(p_int 自由) 5.216 /
  M2 双 Ruland 5.454 / M3 +OZ 尾 3.919 / **M4 两段幂律 1.226**——
  末端 +40% 上翘取证（局部斜率表：数据维持 -2.5~-3.7 而单幂律/Ruland 组合
  无法产生持续中间斜率）指向"底形需要两段斜率"，M4 以 logistic 平滑转折
  双幂律直接表达，峰窗与高 q 尾同步改善（rms 0.022→0.013 / 0.188→0.052）。

物理语义定调：
- dm1（≈1）= 低 q 链/无规两相形态；dm2（≈5，>Porod 4）= 高 q 界面弥散主导段；
- 转折 q_br ≈ 0.94 nm⁻¹（首测）≈ 1/σ_int 量级——与 Ruland 修正转折位置自洽，
  两个机制（底形两段化 vs 界面指数修正）在数学上部分互补，A_int/σ_int 与
  dm2 存在共线，multistart + param_scan 取证必备；
- 本模型是 M0（phase_separation_ruland）的高 q 机制升级版：σ_int 仍直接产出
  界面层厚度，但底形不再承担"单幂律硬拟合全带宽"的结构性负担。

可辨识性注意（报告措辞依据）：
- dm2 与 A_int/σ_int 共线（高 q 段三者共同决定衰减形状），param_scan(σ_int)
  必备；dm2>4 的物理解读=界面弥散（Ruland 等效），不是超 surfaces 分形；
- logistic 转折 w_br 很小时 dm_eff 接近硬转折（数值稳定，exp 无奇异）。
"""
import numpy as np

# ---------- 参数表（gauss 模式） ----------
NAMES_G = ['A_c', 'dm1', 'dm2', 'q_br', 'w_br', 'A_int', 'sigma_int', 'amp', 'q0', 'sigma', 'B']

BOUNDS_G = {
    'A_c': (0.0, np.inf),
    'dm1': (0.5, 6.0),
    'dm2': (0.5, 8.0),
    'q_br': (0.3, 3.0),
    'w_br': (0.02, 1.0),
    'A_int': (0.0, np.inf),
    'sigma_int': (0.05, 10.0),
    'amp': (0.0, np.inf),
    'q0': (0.1, 0.95),
    'sigma': (0.02, 0.6),
    'B': (0.0, np.inf),
}
# 初值（模型赛马 M4 谷）：A_c 由引擎 init_amp_key 标定，其余沿赛马谷
INIT_G = {'dm1': 1.05, 'dm2': 5.1, 'q_br': 0.94, 'w_br': 0.13,
          'A_int': 0.0008, 'sigma_int': 5.15, 'amp': 0.02, 'q0': 0.4, 'sigma': 0.15}


def _base_vec(p, qq):
    """平滑转折双幂律底形"""
    A_c = p['A_c']
    w = 1.0 / (1.0 + np.exp(-(qq - p['q_br']) / p['w_br']))
    dm_eff = p['dm1'] * (1.0 - w) + p['dm2'] * w
    return A_c * qq ** (-dm_eff)


def model_gauss_vec(p, qq):
    """11 参数（顺序按 NAMES_G）。"""
    p = p if isinstance(p, dict) else dict(zip(NAMES_G, p))
    return (_base_vec(p, qq)
            + p['A_int'] * qq ** (-4.0) * np.exp(-(p['sigma_int'] * qq) ** 2)
            + p['amp'] * np.exp(-((qq - p['q0']) ** 2) / (2.0 * p['sigma'] ** 2))
            + p['B'])


def derived_gauss(p, perr):
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
NAMES_L = ['A_c', 'dm1', 'dm2', 'q_br', 'w_br', 'A_int', 'sigma_int', 'amp', 'q0', 'w', 'B']

BOUNDS_L = dict(BOUNDS_G)
BOUNDS_L['w'] = (0.02, 0.6)
INIT_L = {'dm1': 1.05, 'dm2': 5.1, 'q_br': 0.94, 'w_br': 0.13,
          'A_int': 0.0008, 'sigma_int': 5.15, 'amp': 0.02, 'q0': 0.4, 'w': 0.15}


def model_lorentz_vec(p, qq):
    """11 参数（顺序按 NAMES_L）。"""
    p = p if isinstance(p, dict) else dict(zip(NAMES_L, p))
    return (_base_vec(p, qq)
            + p['A_int'] * qq ** (-4.0) * np.exp(-(p['sigma_int'] * qq) ** 2)
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
