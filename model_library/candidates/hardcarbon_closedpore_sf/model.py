# -*- coding: utf-8 -*-
"""
硬碳闭孔 Debye 关联 + 结构因子峰（nuisance）SAXS 模型插件（候选区，2026-10-27 立项）

立项背景（2026-10-27 马文杰-闭孔-真密度 样品2）：数据 q≈0.65-0.70 nm⁻¹ 存在真实
关联峰（局部极大，SNR≈520，用户确认非本底扣除伪影）。hardcarbon_closedpore
main 模式 chi2/dof=924（多起点确认结构性失配）；full 双 Debye 模式 chi2/dof=377.6
（Debye-Bueche 形状函数只能形成肩部，数学上无法产生局部极大）；hardcarbon_ts
试探失败（其 C1>0 设计性禁峰，chi2/dof=474）。用户裁决（2026-10-27）：峰为真实
结构但**结构参数不关心**，闭孔部分才是物理关切——峰项按 nuisance 处理，仅作
信号修正。

物理模型（hardcarbon_closedpore main 全保留 + 加性 Lorentzian 峰项；加性口径，
对应 Hammouda (NIST) Lorentzian 峰 / SasView 峰类模型惯例）：

  I(q) = A·q^(−a) + Bp·q^(−4)·F(a1·q) + Apk·L(q; q0, w) + B

  F(x) = x⁴/(1+x²)²,    L(q) = 1/[1+((q−q0)/w)²]

  前两项 + B 语义与 hardcarbon_closedpore main 完全一致（Stevens-Dahn 闭孔谱族，
  Porod 口径改写逐项恒等）；L 峰项：q0=峰位（nm⁻¹）、w=半高半宽 HWHM（nm⁻¹）、
  Apk=峰幅度（nuisance）；高 q 渐近 Apk·w²·q⁻²（q⁻² 尾，与闭孔 q⁻⁴ 尾简并风险低）。

导出物理量：
  闭孔（物理关切，与 closedpore main 同口径）：Rg=a1√6 / R_sphere=a1√10 /
  D_pore=2a1√10 / q_knee=1/a1
  峰项（nuisance，仅描述性披露，不作物理结论）：q0、FWHM=2w、d=2π/q0

边界裁量（a 上界 7.0，与父模型 4.5 不同）：2026-10-27 样品2 relax_bound 取证
（full 模式 a 上界 4.5→7 重拟合，a=5.114±0.118 稳健、无撞界、无简并失控），
故本模型 a 上界预置 7.0 并按撞界/条件值纪律披露。

单位约定：引擎内 q 为 nm⁻¹，a1 以 nm 计，q0/w 以 nm⁻¹ 计，
A/Bp/Apk/B 为强度量（数据强度单位 a.u.，归一量）。
"""
import numpy as np

# ---------- 参数表（main 模式，8 参数） ----------
NAMES = ['A', 'a', 'Bp', 'a1', 'Apk', 'q0', 'w', 'B']

BOUNDS = {
    'A':   (0.0, np.inf),     # 幂律幅度（引擎 init_amp_key='A' 自动低 q 标定 A0）
    'a':   (1.0, 7.0),        # 幂律指数：上界 7.0（2026-10-27 relax_bound 取证，见文件头）
    'Bp':  (0.0, np.inf),     # 闭孔 Porod 口径幅度（命名避让引擎 'B' 背景语义）
    'a1':  (0.05, 5.0),       # Debye 特征长度 nm：膝位 1/a1 ∈ [0.2, 20] nm⁻¹
    'Apk': (0.0, np.inf),     # 峰幅度（nuisance）
    'q0':  (0.05, 5.0),       # 峰位 nm⁻¹（样品2 预期 ≈0.70）
    'w':   (0.005, 3.0),      # 半高半宽 nm⁻¹（下界防 δ 函数塌缩）
    'B':   (0.0, np.inf),     # 常数背景（引擎 init_p0 按名 'B' 自动末端均值估计）
}

# 初值：闭孔三参数取 hardcarbon_closedpore full 对样品2 的收敛邻域（a1≈0.47）；
# 峰参数按样品2 目测（q0≈0.70、HWHM≈0.15-0.2、幅度≈峰上超出量的 5%）
INIT = {'a': 2.9, 'Bp': 3.0, 'a1': 0.50, 'Apk': 0.05, 'q0': 0.70, 'w': 0.15}

# A0 初值估计带上界（nm⁻¹）：取 0.5（q<0.5 为幂律主导区，避开 0.55-1.2 峰区，
# 防止 A0 被峰抬高把峰项压入关闭谷）。限制：诊断窄窗 qmin 须 <0.5，否则
# 估计带倒空（父模型 hardcarbon_closedpore 10-09 教训同类）
INIT_QMAX = 0.5


# ---------- 形状函数与向量形式模型 ----------
def _f(x):
    """Debye-Bueche 形状函数 x⁴/(1+x²)²（Porod 口径：高 q→1，低 q→x⁴ 平台）"""
    x2 = x * x
    return x2 * x2 / (1.0 + x2) ** 2


def model_vec(p, qq):
    """main 模式 8 参数：A, a, Bp, a1, Apk, q0, w, B（顺序按 NAMES）"""
    A, a, Bp, a1, Apk, q0, w, B = p
    lorentz = 1.0 / (1.0 + ((qq - q0) / w) ** 2)
    return A * qq ** (-a) + Bp * qq ** (-4.0) * _f(a1 * qq) + Apk * lorentz + B


# ---------- 导出量（闭孔 = a1 线性/反比函数解析传播；峰 = nuisance 描述量） ----------
def derived_main(p, perr):
    """main 模式导出量：闭孔四件套（物理关切）+ 峰描述量（nuisance）"""
    a1 = float(p['a1'])
    sa1 = float(perr['a1'])
    r6 = float(np.sqrt(6.0))
    r10 = float(np.sqrt(10.0))
    q0 = float(p['q0'])
    sq0 = float(perr['q0'])
    w = float(p['w'])
    sw = float(perr['w'])
    out = {
        'Rg_nm': a1 * r6, 'Rg_nm_err': sa1 * r6,
        'R_sphere_nm': a1 * r10, 'R_sphere_nm_err': sa1 * r10,
        'D_pore_nm': 2.0 * a1 * r10, 'D_pore_nm_err': 2.0 * sa1 * r10,
        'q_knee_nm': 1.0 / a1 if a1 > 0 else float('nan'),
        'q_knee_nm_err': (sa1 / (a1 * a1)) if a1 > 0 else float('nan'),
        'a1_nm': a1, 'a1_nm_err': sa1,
        # 峰项描述量（nuisance，仅披露不作物理结论）
        'peak_q0_inv_nm': q0, 'peak_q0_inv_nm_err': sq0,
        'peak_FWHM_inv_nm': 2.0 * w, 'peak_FWHM_inv_nm_err': 2.0 * sw,
        'peak_d_nm': (2.0 * np.pi / q0) if q0 > 0 else float('nan'),
        'peak_d_nm_err': (2.0 * np.pi * sq0 / (q0 * q0)) if q0 > 0 else float('nan'),
    }
    return out
