# -*- coding: utf-8 -*-
"""
硬碳闭孔 SAXS 模型插件（模型库 4 号模型候选，M3 任务）
物理模型（四组分加和，Adv. Mater. 2025, 37, 2420251 SI Methods Eq.1-7；
方法学源头：Saurel et al., Energy Storage Mater. 21 (2019) 162；Teubner & Strey, J. Chem. Phys. 87 (1987) 3195）：

  I(q) = A·q^(−n) + I₀/(1 + C₁q² + C₂q⁴) + B

  A·q^(−n)      : I_particles——颗粒/大结构散射项（双对数直线幂律，文献 Figure 2h/S10
                  黑色点划虚线；n 自由拟合，Porod 极限 n=4）
  I₀/(1+C₁q²+C₂q⁴) : I_pores——Teubner-Strey 介观微孔项（文献 SI Eq.1；
                  低 q 饱和平台 → 中 q 弯折 → 高 q Porod q⁻⁴ 渐近）
  B             : I_bckg——常数背景（文献 Figure 2h/S10 橙色虚线）

导出物理量（文献 SI Eq.3/4/6/7）：
  孔间距   d = 2π·(½C₂^(−1/2) − C₁/(4C₂))^(−1/2)              （相邻微孔平均距离）
  相关长度 ξ = (½C₂^(−1/2) + C₁/(4C₂))^(−1/2)                  （有序域延伸尺度）
  孔径     D = 2√(5C₁)                                          （要求 C₁>0）
  比表面积 S_mp = I₀/(2π·C₂·(ΔSLD)²·ρ_s)                        （TS 高 q 渐近=Porod 定律
                  I₀/C₂ = 2π(ΔSLD)²·S/V 与 S_mp=(S/V)/ρ_s 合成；
                  文献 SI Eq.7 字面缺 ρ_s，因其 I₀ 定义（SI Eq.2）已含 1/ρ_s——
                  本插件 I₀ 为拟合自由参数（数据强度单位），故显式除 ρ_s）

单位约定：引擎内 q 为 nm⁻¹，C₁ 以 nm²、C₂ 以 nm⁴ 计，d/ξ/D 以 nm 计。
ΔSLD 口径（用户 2026-10-02 裁决：石墨密度上限）：ρ_s = 2.26 g/cm³，
  ΔSLD = SLD_carbon = ρ_s·N_A·Z·r_e/M = 1.916×10⁻⁵ Å⁻² = 1.916×10⁻³ nm⁻²（空孔：ΔSLD=基体 SLD）。
  S_mp 数值有效性条件于数据强度绝对标定（abs.units ≈ cm⁻¹）——条件值披露。
"""
import numpy as np

# ---------- 物理常数（X 射线 SLD，石墨密度上限口径） ----------
RHO_S = 2.26                     # g/cm³（石墨结构密度上限，用户裁决）
DSLD = 1.916e-3                  # nm⁻²（ΔSLD = 基体电子密度，由 RHO_S 计算：1.916e-5 Å⁻²）
RHO_GRA_REF = 2.26               # g/cm³（文献 SI Eq.5 的 ρ_gra 参考值）

# ---------- 参数表 ----------
NAMES = ['A', 'n', 'I0', 'C1', 'C2', 'B']

BOUNDS = {
    'A':  (0.0, np.inf),         # 幂律幅度（引擎按低 q 形状自动估计初值）
    'n':  (1.0, 6.0),            # 幂律指数：Porod 极限 4 为初值；上限 6 覆盖粗糙表面/过渡形态
    'I0': (0.0, np.inf),         # TS 微孔项幅度
    'C1': (1e-6, 50.0),          # nm²：D=2√(5C₁) → D ∈ [0.0045, 31.6] nm；C₁>0 为 D 公式要求
    'C2': (1e-8, 1e4),           # nm⁴：四阶系数
    'B':  (0.0, np.inf),         # 常数背景（引擎按高 q 末端自动估计初值）
}

# 初值：中 q 肩位 q_sh ≈ 4 nm⁻¹（数据形态识别）→ d ≈ 2π/q_sh ≈ 1.57 nm；
# 文献锚 D ≈ 0.6 nm（STHC-MS）→ C₁ = D²/20 ≈ 0.018 nm²；
# ξ ≈ 0.3 nm → C₂^(−1/2) = 4π²/d² + 1/ξ² ≈ 26.5 → C₂ ≈ 1.4×10⁻³ nm⁴
INIT = {'n': 4.0, 'I0': 0.05, 'C1': 0.018, 'C2': 1.4e-3}

# A0 初值估计带的上界（nm⁻¹）：数据 qmin=0.175 nm⁻¹；1.2 nm⁻¹ 仍为幂律主导区
# （0.5→1.2 修订 2026-10-02：支持中高 q 窗口依赖诊断，qmin>0.5 的子窗原报
# "init_qmax 与数据 q 范围不相交"硬错；幂律区内 median 估计对窗宽不敏感）
INIT_QMAX = 1.2


# ---------- 向量形式模型（least_squares 用；参数顺序见 NAMES） ----------
def model_vec(p, qq):
    """6 参数：A, n, I0, C1, C2, B（顺序按 NAMES）"""
    A, n, I0, C1, C2, B = p
    return A * qq ** (-n) + I0 / (1.0 + C1 * qq ** 2 + C2 * qq ** 4) + B


# ---------- 导出量（SI Eq.3/4/6/7） ----------
def _ts_d_xi(C1, C2):
    """由 C1/C2 计算 d、ξ（nm）；根号内非正（C₁ ≥ 2√C₂）时返回 NaN 并标记"""
    half = 0.5 * C2 ** (-0.5)
    v = C1 / (4.0 * C2)
    dm = half - v          # d 的根号内
    xp = half + v          # ξ 的根号内
    d = 2.0 * np.pi * dm ** (-0.5) if dm > 0 else float('nan')
    xi = xp ** (-0.5) if xp > 0 else float('nan')
    return d, xi


def derived_main(p, perr):
    """导出量 + 数值差分误差传播（d/ξ/D 对 C1,C2；S_mp 对 I0,C2）。
    S_mp 单位：abs.units/g/cm³ → 若 abs.units=cm⁻¹ 则 cm²/g（条件值，报告披露）；
    另给 m²/g 换算列。"""
    I0, C1, C2 = float(p['I0']), float(p['C1']), float(p['C2'])
    sI0, sC1, sC2 = float(perr['I0']), float(perr['C1']), float(perr['C2'])
    d, xi = _ts_d_xi(C1, C2)
    D = 2.0 * np.sqrt(5.0 * C1)
    sv = I0 / (2.0 * np.pi * C2 * DSLD ** 2)      # S/V，abs.units（条件于 cm⁻¹ 标定）
    smp = sv / RHO_S                              # cm²/g（同上条件）
    smp_m2g = smp * 1e-4                          # m²/g

    # 数值中心差分误差传播（步长 = 1e-4×参数量级）
    def _num_grad(fn, vals, errs):
        gs = []
        for i, (v, e) in enumerate(zip(vals, errs)):
            h = max(1e-4 * abs(v), 1e-12)
            vp = list(vals); vm = list(vp)
            vp[i] = v + h; vm[i] = v - h
            try:
                g = (fn(*vp) - fn(*vm)) / (2.0 * h)
            except (ValueError, ZeroDivisionError):
                g = float('nan')
            gs.append(abs(g) * e if np.isfinite(g) else float('nan'))
        s2 = 0.0; any_nan = False
        for g in gs:
            if np.isfinite(g):
                s2 += g * g
            else:
                any_nan = True
        return (float(np.sqrt(s2)) if not any_nan else float('nan'))

    d_err = _num_grad(lambda c1, c2: _ts_d_xi(c1, c2)[0], [C1, C2], [sC1, sC2])
    xi_err = _num_grad(lambda c1, c2: _ts_d_xi(c1, c2)[1], [C1, C2], [sC1, sC2])
    D_err = _num_grad(lambda c1: 2.0 * np.sqrt(5.0 * c1), [C1], [sC1])
    smp_err = _num_grad(lambda i0, c2: i0 / (2.0 * np.pi * c2 * DSLD ** 2) / RHO_S,
                        [I0, C2], [sI0, sC2])

    return {
        'd_nm': float(d), 'd_nm_err': d_err,
        'xi_nm': float(xi), 'xi_nm_err': xi_err,
        'D_nm': float(D), 'D_nm_err': D_err,
        'S_over_V': float(sv), 'S_over_V_err': smp_err * RHO_S,
        'S_mp_cm2g': float(smp), 'S_mp_cm2g_err': smp_err,
        'S_mp_m2g': float(smp_m2g), 'S_mp_m2g_err': smp_err * 1e-4,
        'rho_s_g_cm3': RHO_S, 'dSLD_nm2': DSLD,
    }
