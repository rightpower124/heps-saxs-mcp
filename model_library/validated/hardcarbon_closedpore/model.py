# -*- coding: utf-8 -*-
"""
硬碳闭孔 Debye 关联（Stevens-Dahn）SAXS 模型插件（模型库 5 号候选，M3-261025 立项，
261025-GXL 复现轮：用户批准，四样品独立，拟合窗 [0.15, 2.26] nm⁻¹）

物理模型（幂律 + Debye-Bueche 闭孔项 + 常数背景）：
方法学源头：Stevens & Dahn, J. Electrochem. Soc. 147 (12) 4428-4431 (2000) Eq.1/2
（Debye, Anderson & Brumberger, J. Appl. Phys. 28 (1957) 679 关联函数）；
应用文献：Zheng et al., Adv. Energy Mater. 2024, 14, 2303064 SI Fig.S10。

  main（AEM SI Fig.S10 所用简化式，Porod 口径改写——高 q 渐近显式为 q^-4）：

    I(q) = A·q^(−a) + Bp·q^(−4)·F(a1·q) + B,    F(x) = x⁴/(1+x²)²

    与 SI 字面式 I = A/q^a + B_SI·a1⁴/(1+a1²q²)² + D **逐项恒等**
    （Bp·q^-4·(a1·q)⁴ = Bp·a1⁴，故 Bp ≡ B_SI）。改写仅改变幅度参数的量纲语义：
    Bp = "闭孔项的 Porod 外推幅度"（q→大时闭孔项 → Bp·q^-4），与幂律幅度 A 同口径、
    跨项/跨样品可比；SI 字面 B_SI 的量纲则与 a1 耦合（不可比）。

  full（Stevens 2000 Eq.2 完整式：双 Debye 项 + 显式中孔组分）：

    I(q) = A·q^(−a) + Bp·q^(−4)·F(a1·q) + Bmp·q^(−4)·F(a2·q) + B

    与 Stevens 字面式 I = A/q^3.3 + Δρ·B·a1⁶/(1+a1²q²)² + C·a2⁴/(1+a2²q²)² + D 的映射：
    Bp = Δρ·B·a1²（Porod 口径吸收 Δρ 与 a1²；原式 B 正比纳米孔总数）、Bmp = C·a2²。
    原文幂律指数 3.3 为**狭缝准直仪器 Smearing** 的实测钉值（原文用无孔石墨标样定标）；
    本站针孔准直无 Smearing，指数 a 自由拟合（AEM SI 同为自由 a）。
    Stevens 的 a2（中孔）> a1（纳米孔），中孔膝位 1/a2 在更低 q。
    **适配性边界（261025 立项裁决）**：本批 GXL 数据 q_max=2.26 nm⁻¹ 且
    a1≈0.6-0.77 / a2≈1 nm 两膝位（1.0–1.7 nm⁻¹）间距 <2.5 倍——full 模式对本批
    数据预期结构性简并（lessons 10-06 多层级重叠纪律），本批仅用 main；
    full 服务未来两膝分离良好的数据集（如 Stevens 原位对比度实验）。

导出物理量（Porod 回转半径定义，Stevens 2000 原文 §Debye length）：
  Rg = a1·√6           （"mean square distance from the center of gravity"）
  R_sphere = a1·√10    （等效球形孔半径：Rg_sphere = R·√0.6 → R = a1·√10）
  D_pore = 2·a1·√10    （等效孔**直径**——即 AEM SI 公式 "R = 2·a1·√10" 的数值口径；
                         SI 记号 "R" 实为直径，跨文献比对须注意 2 倍口径差）
  q_knee = 1/a1        （Debye 膝特征波数）

单位约定：引擎内 q 为 nm⁻¹，a1/a2 以 nm 计，Rg/R_sphere/D_pore 以 nm 计；
A/Bp/Bmp/B 为强度量（数据强度单位 a.u.，归一量）。

可靠性语义（质量门，入 manifest assumptions 同步披露）：
- D_pore/q_knee/Rg 的可辨识性条件于膝位落窗（q_knee ∈ [qmin×1.5, qmax/1.5]）；
  膝在窗外时 a1 沿 A·Rg^a 类幅度组合简并谷漂移（lessons 10-06 GP 同族指纹：
  χ² 不变 + a1/Rg 大幅漂移 + 指数 a 铁稳），届时仅 a 为稳健值；
- 空孔对比度固定（无 SLD 自由参数），本模型无 SLD 族精确简并，冒烟无需真值覆盖；
- 背景参数命名 'B'（引擎 init_p0 按名自动末端估计），闭孔幅度避开 'B' 命名
  （'Bp'）防止引擎背景初值语义误挂——命名契约见 plugin.register。
"""
import numpy as np

# ---------- 参数表（main 模式） ----------
NAMES = ['A', 'a', 'Bp', 'a1', 'B']

BOUNDS = {
    'A':  (0.0, np.inf),      # 幂律幅度（引擎 init_amp_key='A' 自动低 q 标定 A0）
    'a':  (1.0, 4.5),         # 幂律指数：GXL 数据实测斜率 2.0–2.6；Porod 极限 4+余量封顶
                              # （10-06 关尾逃逸纪律：上界=物理极限+余量）
    'Bp': (0.0, np.inf),      # 闭孔 Porod 口径幅度（命名避让引擎 'B' 背景语义）
    'a1': (0.05, 5.0),        # Debye 特征长度 nm：膝位 1/a1 ∈ [0.2, 20] nm⁻¹
    'B':  (0.0, np.inf),      # 常数背景（引擎 init_p0 按名 'B' 自动末端均值估计）
}

# 初值：a 取数据两段斜率折中；a1 文献锚（HCMP 3.82 nm → 0.604 / HCMP-CO2 4.86 nm →
# 0.769 nm 直径换算），折中 0.70；Bp 按数据量级手估（q≈2 处闭孔贡献与幂律可比）
INIT = {'a': 2.3, 'Bp': 10.0, 'a1': 0.70}

# A0 初值估计带上界（nm⁻¹）：修订史——
# 0.5（261025 入库，按 GXL 主窗 qmin=0.15 定制：q≤0.5 幂律主导区）→
# 1.2（261025 窄窗诊断修订：窗口依赖诊断需 qmin=1.0 的窄窗对照，0.5 倒空报错；
#      10-09 教训：估计带常量不是物理参数，换窗即失配。修订后主窗 [0.45,2.26]
#      回归对拍 χ²=0.9759450/a1=0.7214807 逐位一致——A0 仅初值不影响谷位置）
INIT_QMAX = 1.2

# ---------- 参数表（full 模式） ----------
FULL_NAMES = ['A', 'a', 'Bp', 'a1', 'Bmp', 'a2', 'B']

FULL_BOUNDS = {
    'A':   (0.0, np.inf),
    'a':   (1.0, 4.5),
    'Bp':  (0.0, np.inf),
    'a1':  (0.05, 5.0),
    'Bmp': (0.0, np.inf),
    'a2':  (0.1, 10.0),       # 中孔特征长度 nm（Stevens 钉值 10 Å=1 nm 为初值）
    'B':   (0.0, np.inf),
}

FULL_INIT = {'a': 2.3, 'Bp': 8.0, 'a1': 0.70, 'Bmp': 3.0, 'a2': 1.0}


# ---------- 形状函数与向量形式模型 ----------
def _f(x):
    """Debye-Bueche 形状函数 x⁴/(1+x²)²（Porod 口径：高 q→1，低 q→x⁴ 平台）"""
    x2 = x * x
    return x2 * x2 / (1.0 + x2) ** 2


def model_vec(p, qq):
    """main 模式 5 参数：A, a, Bp, a1, B（顺序按 NAMES）"""
    A, a, Bp, a1, B = p
    return A * qq ** (-a) + Bp * qq ** (-4.0) * _f(a1 * qq) + B


def model_vec_full(p, qq):
    """full 模式 7 参数：A, a, Bp, a1, Bmp, a2, B（顺序按 FULL_NAMES）"""
    A, a, Bp, a1, Bmp, a2, B = p
    return (A * qq ** (-a)
            + Bp * qq ** (-4.0) * _f(a1 * qq)
            + Bmp * qq ** (-4.0) * _f(a2 * qq) + B)


# ---------- L1b 逐式转写对拍（文献字面式独立实现，仅验证脚本消费） ----------
def model_vec_stevens_raw(p, qq):
    """Stevens 2000 Eq.1 字面式（狭缝准直指数 3.3 固定）：参数 A, B_raw, a1, D。
    与 model_vec 在 a=3.3 时的恒等关系：B_raw ≡ Bp。"""
    A, B_raw, a1, D = p
    return A * qq ** (-3.3) + B_raw * a1 ** 4 / (1.0 + (a1 * qq) ** 2) ** 2 + D


def model_vec_full_raw(p, qq):
    """Stevens 2000 Eq.2 字面式（Δρ·B·a1⁶ 显式 + C·a2⁴ 中孔）：
    参数 A, dr_Bn（=Δρ·B 乘积）, a1, C, a2, D。
    与 model_vec_full 在 a=3.3 时的映射：dr_Bn·a1² ≡ Bp（即 dr_Bn = Bp/a1²）、C ≡ Bmp。"""
    A, dr_Bn, a1, C, a2, D = p
    return (A * qq ** (-3.3)
            + dr_Bn * a1 ** 6 / (1.0 + (a1 * qq) ** 2) ** 2
            + C * a2 ** 4 / (1.0 + (a2 * qq) ** 2) ** 2 + D)


# ---------- 导出量（a1 的线性/反比函数，误差解析传播） ----------
def derived_main(p, perr):
    """main 模式导出量：Rg / R_sphere / D_pore / q_knee（nm / nm⁻¹）"""
    a1 = float(p['a1'])
    sa1 = float(perr['a1'])
    r6 = float(np.sqrt(6.0))
    r10 = float(np.sqrt(10.0))
    return {
        'Rg_nm': a1 * r6, 'Rg_nm_err': sa1 * r6,
        'R_sphere_nm': a1 * r10, 'R_sphere_nm_err': sa1 * r10,
        'D_pore_nm': 2.0 * a1 * r10, 'D_pore_nm_err': 2.0 * sa1 * r10,
        'q_knee_nm': 1.0 / a1 if a1 > 0 else float('nan'),
        'q_knee_nm_err': (sa1 / (a1 * a1)) if a1 > 0 else float('nan'),
        'a1_nm': a1, 'a1_nm_err': sa1,
    }


def derived_full(p, perr):
    """full 模式导出量：纳米孔（a1）+ 中孔（a2）两组同构量"""
    out = {}
    for key, len_key in (('nano', 'a1'), ('meso', 'a2')):
        al = float(p[len_key])
        sal = float(perr[len_key])
        r6 = float(np.sqrt(6.0))
        r10 = float(np.sqrt(10.0))
        out.update({
            'Rg_%s_nm' % key: al * r6, 'Rg_%s_nm_err' % key: sal * r6,
            'R_sphere_%s_nm' % key: al * r10, 'R_sphere_%s_nm_err' % key: sal * r10,
            'D_pore_%s_nm' % key: 2.0 * al * r10, 'D_pore_%s_nm_err' % key: 2.0 * sal * r10,
            'q_knee_%s_nm' % key: 1.0 / al if al > 0 else float('nan'),
            'q_knee_%s_nm_err' % key: (sal / (al * al)) if al > 0 else float('nan'),
        })
    out['a1_nm'] = float(p['a1']); out['a1_nm_err'] = float(perr['a1'])
    out['a2_nm'] = float(p['a2']); out['a2_nm_err'] = float(perr['a2'])
    return out
