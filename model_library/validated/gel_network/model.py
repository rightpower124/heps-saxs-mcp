# -*- coding: utf-8 -*-
"""
凝胶/聚合物网络 SAXS 模型插件（模型库 3 号模型，M3 第二个文献自动化模型）
物理模型（Wisotzki, Tempesti, Fratini & Mayr, Phys. Chem. Chem. Phys. 19 (2017) 12064-12074）：

  mode main = 文献 Eq.(1) correlation length model（全 q 区）：
      I(q) = I(0) / (1 + (ξ·q)^m) + B
      I(0) : q→0 外推散射强度（本插件记作参数 A，与引擎幅度约定同名）
      ξ    : 相关长度 correlation length（nm）——网络中散射中心平均间距
      m    : Porod 指数——局部链形态：1.67 良溶剂溶胀线团 / 2 theta 溶剂（OZ 极限）/
             2~3 支化（化学凝胶）/ →1 半柔性棒状结构；随 q 增大 I~q^-m
      B    : 常数背景（站内扩展：吸收噪声平台，文献数据已扣背景故原文无此项）
      派生量：qc = 1/ξ（双幂律 crossover / 肩部位置），mesh = 2π/qc = 2πξ（网络网目尺寸）
      —— mesh=2πξ 口径经文献数值链验证：ξ=3.0 nm → mesh 18.8 nm ≈ 文献 17.5±0.5；
         ξ=4.0 → 25.1 ≈ 26.0±0.5 ✓

  mode porod = 文献 Eq.(3) Porod plot / 质量分形维数（中段局部幂律，文献窗 0.3–2 nm⁻¹）：
      I(q) = A · q^(-dm)
      dm   : 质量分形维数——1 线性棒状 / 1.67 良溶剂线团 / 2~3 支化化学凝胶

  文献 Eq.(2)（OZ–DB 叠加 Picot–Benoit 低 q 上翘项）不入库：文献自证对辐照凝胶
  ξ 与 ε 同量级、无物理意义（"results did not appear to have physical meaning"），
  且 6 参数过拟合风险高（其弃用理由成立）。

单位约定：引擎内 q 为 nm⁻¹，ξ 以 nm 计；A/B 为 a.u. 标度（归一量）。
可辨识性注意（报告措辞依据）：数据低 q 无平台（未及 q << 1/ξ）时 A 与 ξ^m 强共线
（I ≈ (A/ξ^m)·q^-m 仅乘积可辨）——A 单独值在该场景降级条件值，形态结论以拟合曲线+残差为准。
"""
import numpy as np

# ---------- 参数表（main 模式：correlation length model，文献 Eq.1） ----------
NAMES = ['A', 'xi', 'm', 'B']

# 边界：ξ 覆盖文献凝胶 ξ≈3–6 nm 与更宽网络尺度；m 覆盖 Porod 指数物理域
#   （0.5≈极端粗糙/软界面 → 4=Porod 上限；文献域 1.69–2.55）
BOUNDS = {
    'A': (0.0, np.inf),
    'xi': (0.05, 100.0),
    'm': (0.5, 4.0),
    'B': (0.0, np.inf),
}
# 初值：文献物理凝胶典型值（ξ=3.0 nm、m=1.8）；A/B 由引擎自动估计
# （init_p0：A0=低 q 窗中值估计、B0=高 q 末端 100 点均值）
INIT = {'xi': 3.0, 'm': 1.8}


def model_vec(p, qq):
    """4 参数：A, xi, m, B（顺序按 NAMES）。文献 Eq.(1)。"""
    A, xi, m, B = p
    return A / (1.0 + (xi * qq) ** m) + B


# ---------- 派生量（main） ----------
def derived_main(p, perr):
    """qc = 1/ξ（crossover/肩部位置）与 mesh = 2π/qc = 2πξ（网目尺寸，文献 Mesh size 口径）"""
    xi = float(p['xi'])
    xi_e = float(perr['xi'])
    qc = 1.0 / xi
    qc_e = xi_e / xi ** 2          # |d(1/ξ)/dξ|·σξ
    mesh = 2.0 * np.pi * xi
    return {
        'qc_nm1': qc,
        'qc_nm1_err': qc_e,
        'mesh_nm': mesh,
        'mesh_nm_err': 2.0 * np.pi * xi_e,
    }


# ---------- 参数表（porod 模式：质量分形幂律，文献 Eq.3） ----------
NAMES_POROD = ['A', 'dm']

# 边界：dm 质量分形维数物理域 1–3（1 线性棒 / 1.67 良溶剂线团 / 2–3 支化），留余量
BOUNDS_POROD = {
    'A': (0.0, np.inf),
    'dm': (0.5, 3.5),
}
# 初值：文献域中庸值；A 由引擎自动估计（I_shape = q^-dm，A0=低 q 窗中值）
INIT_POROD = {'dm': 1.8}


def model_porod_vec(p, qq):
    """2 参数：A, dm（顺序按 NAMES_POROD）。文献 Eq.(3)：I = A·q^(-dm)。"""
    A, dm = p
    return A * qq ** (-dm)


def derived_porod(p, perr):
    """porod 模式无额外派生量（dm 即产物）；保持接口一致返回空 dict"""
    return {}
