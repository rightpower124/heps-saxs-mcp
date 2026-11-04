# -*- coding: utf-8 -*-
"""unified_power_Rg 物理模型（Beaucage 1995 统一四 level，级联形式）
公式与 sasmodels 内置 unified_power_Rg 逐式一致（L1b DirectModel 对拍验证）：
  I(q) = B + SUM_i [ G_i·exp(-(q·Rg_i)^2/3)
                     + K_i·exp(-(q·Rg_{i-1})^2/3)·(q_i*)^-P_i ]     (i = 1..4)
  q_i* = q / erf(q·Rg_i/sqrt(6))^3
  K_i = G_i·P_i / Rg_i^P_i · Gamma(P_i/2)   ← 固定关系，非自由参数
级联方向（与论文 Level 编号一致，L1 曲线形态锁定）：
  Level 4 = 最大尺度（最低 q 平台）→ Level 1 = 最小尺度（最高 q 幂律尾）；
  K_i 的压制因子用"下一级（更小尺度，编号 i-1）"的 Guinier 因子——
  K4 尾被 exp(-(q·Rg3)^2/3) 压制、K3 被 Rg2、K2 被 Rg1、K1 无压制（最高 q
  主导项）。对应 sasmodels 内核循环 rg[0]=最大尺度（L1b 对拍映射：
  sasmodels rg1←Rg4、rg2←Rg3、rg3←Rg2、rg4←Rg1，B[i] 同序传公式值）。
K 固定关系的依据（L1 实证 + 方法学）：
  ① Beaucage "unified" 本意 = 每级 Guinier 幅度与 Porod 尾由同一对象的
     尺寸/维度统一锁定（sasmodels 文档 Debye 例子：P=2 时 B=2G/Rg²，
     与 random() 自洽公式 K=G·P/Rg^P·Gamma(P/2) 同款）；
  ② 2026-10-06 L1 实测：K 作自由参数时 17 参数版存在结构性简并谷——
     相邻级被压制幂律尾在有限窗内 K·q^-P 与 K'·q^-P' 互相补偿
     （A 用例 chi2=0.931 完美但 P3 撞上界 6、K3 漂 3 个量级、多起点不一致）；
     固定关系后 13 参数（G/Rg/P×4 + B）简并解除。
参数命名：K_i ≡ sasmodels 的 B[i]（每级 Porod 尾幅度；内部派生，不进参数表）；
scale 等效固定 1（幅度由 G_i 直接承载，消除 A-G 简并）。

q 单位 nm^-1，Rg 单位 nm（引擎 q_unit='A^-1' 时自动 ×10 换算，插件内部统一 nm；
L1b 对拍时 sasmodels 侧 Rg×10 转 Å）。

复现锚点：Chaney et al., Chem. Mater. 37 (2025) 7882-7893 Figure 7a/7b
（"A unified model with four levels"；论文 ref 65 = Beaucage, J. Appl. Cryst.
28 (1995) 717；ref 69 = Hammouda 2010 广义 GP 概念）。
"""
import numpy as np
from numpy import errstate, exp, sqrt
from scipy.special import erf, gamma

NAMES = ['G1', 'Rg1', 'P1',
         'G2', 'Rg2', 'P2',
         'G3', 'Rg3', 'P3',
         'G4', 'Rg4', 'P4',
         'B']

BOUNDS = {
    'G1': (0.0, np.inf), 'G2': (0.0, np.inf),
    'G3': (0.0, np.inf), 'G4': (0.0, np.inf),   # 每级 Guinier 幅度（a.u. 归一量）
    'Rg1': (0.01, 500.0), 'Rg2': (0.01, 500.0),
    'Rg3': (0.01, 500.0), 'Rg4': (0.01, 500.0),  # 回转半径 nm
    'P1': (0.5, 4.5), 'P2': (0.5, 4.5),
    'P3': (0.5, 4.5), 'P4': (0.5, 4.5),          # 每级 Porod 指数（上界 4.5：Porod 极限
                                                 # P=4（光滑表面）+ 余量；论文披露 P∈[1,3.8]。
                                                 # 2026-10-06 L2 实证：上界 6 时 fitter 用
                                                 # P→6 的"关尾逃逸"（K∝Rg^-6→尾部消失，纯
                                                 # Guinier 阶梯拟合）致 P 全撞界+双谷简并）
    'B': (0.0, np.inf),                          # 常数背景（数据已扣背景 → 预期 ~0）
}


def k_of(g, rg, p_exp):
    """每级 Porod 尾幅度（固定关系，非自由参数）：
    K = G·P/Rg^P·Gamma(P/2)（Guinier-Porod crossover 连续；sasmodels random 同款）"""
    return float(g * p_exp / rg ** p_exp * gamma(p_exp / 2.0))


# 论文 Figure 7b 锚定初值（Rg/P = 论文披露值 ÷10Å→nm；G 为数值起步点——论文不披露
# 幅度，按 Chaney 数据真实形态"低 q 平台由最大级主导、幅度跨 4 个量级"分配
# （2026-10-06 L2 实证：均匀 ΣG≈3 分配使 INIT 直跑落入"Level 3 死"次优谷，
#  multistart 6 起点取证谷 X 为全局最优，Rg/P 论文锚起点 + 数据形态 G 分配
#  可直收敛谷 X）
INIT = {
    'G1': 0.02, 'Rg1': 2.0, 'P1': 1.0,
    'G2': 0.5, 'Rg2': 13.0, 'P2': 3.1,
    'G3': 8.0, 'Rg3': 36.0, 'P3': 3.8,
    'G4': 100.0, 'Rg4': 150.0, 'P4': 2.9,
}
# Level 4 Rg 论文披露为 ">1000 Å"（下限式锚标），初值取 150 nm=1500 Å 落在约束区

# A0 估计窗上界（A 不在参数表，此窗仅保证 init_p0 低 q 估计不空窗）。
# 2026-10-06 修订：0.02 → 0.05——Chaney 数据实际 q_min=0.02496 nm^-1，原值触发
# "init_qmax 与数据 q 范围不相交"显式报错（宁可报错不要猜错契约，行为正确）；
# 0.05 覆盖数据最前段（Level 4 近平台区）且不空窗。
# 2026-10-08 修订：0.05 → 0.09——Henderson Figure-6 数据 q_min=0.07642 nm^-1
# 超过 0.05，估计带空窗同款报错；0.09 覆盖该数据最前段（3 点）。A0 不进
# unified 参数表（NAMES 无 'A'），估计带变化对 x0 零影响；Chaney main 回归
# 锚点 χ²/dof=0.07126431714626671 改后对拍逐位一致（当轮验证）。
# selftest 冒烟网格低 q 补点 geomspace(0.025, 0.3, 20)，覆盖 Level4/3 膝区边缘
# 2026-10-27 修订：0.09 → 0.4——闭孔硬碳数据（马文杰 A1/A2_azimAvg_sub.dat）
# q_min=0.3018 nm^-1 超过 0.09，估计带空窗报错阻断拟合；A0 不进本模型参数表
#（NAMES 无 'A'），估计带变化对 x0 零影响；main 模式同步声明 smoke_q 专属
# 冒烟网格（与 lv/full 同款），使 selftest 合成网格不随本值漂移。
INIT_QMAX = 0.4


def _level(q, g, rg, p_exp, rg_next):
    """单级两项：Guinier 支 + 被下一级 Guinier 因子压制的 Porod 尾。
    rg_next=None 表示最小尺度级（Level 1），尾无压制。q 为 ndarray。"""
    gu = g * np.exp(-(q * rg) ** 2 / 3.0)
    pow_factor = (erf(q * rg / sqrt(6.0)) ** 3 / q) ** p_exp    # = (q_i*)^-P
    kk = k_of(g, rg, p_exp)
    if rg_next is None:
        return gu + kk * pow_factor
    return gu + kk * np.exp(-(q * rg_next) ** 2 / 3.0) * pow_factor


def _pd(p):
    if isinstance(p, dict):
        return p
    return dict(zip(NAMES, p))


def model_vec(p, qq):
    """引擎接口：p 按 NAMES 顺序（向量/列表）或参数 dict"""
    d = _pd(p)
    q = np.asarray(qq, dtype=float)
    with errstate(divide='ignore', invalid='ignore'):
        total = (_level(q, d['G4'], d['Rg4'], d['P4'], d['Rg3'])
                 + _level(q, d['G3'], d['Rg3'], d['P3'], d['Rg2'])
                 + _level(q, d['G2'], d['Rg2'], d['P2'], d['Rg1'])
                 + _level(q, d['G1'], d['Rg1'], d['P1'], None))
    return total + d['B']


def level_curve(q, d, i):
    """第 i 级（论文编号 1-4）完整贡献线（components 显示/CSV 分量列共用）。
    d 为参数 dict。"""
    q = np.asarray(q, dtype=float)
    rg_next = d['Rg%d' % (i - 1)] if i > 1 else None
    with errstate(divide='ignore', invalid='ignore'):
        return _level(q, d['G%d' % i], d['Rg%d' % i], d['P%d' % i], rg_next)


def derived_main(p, perr):
    """导出量：四级 rg_A（Å，直接对标文献 Figure 7b）+ q_knee（Guinier 支
    特征弯折点 sqrt(3)/Rg；Beaucage 无硬 crossover，此为参考位置）
    + K_i（固定关系派生幅度，披露用）"""
    out = {}
    for i in range(1, 5):
        rg = float(p['Rg%d' % i])
        rge = float(perr['Rg%d' % i])
        out['rg%d_nm' % i] = rg
        out['rg%d_nm_err' % i] = rge
        out['rg%d_A' % i] = rg * 10.0          # ×10 直接对标文献 Å 值
        out['rg%d_A_err' % i] = rge * 10.0
        out['q_knee%d_nm1' % i] = float(np.sqrt(3.0)) / rg
        out['k%d' % i] = k_of(float(p['G%d' % i]), rg, float(p['P%d' % i]))
    return out


# ---------------------------------------------------------------------------
# full 模式（2026-10-06 增）：小角统一四 level + WAXS 区三个 Lorentz 峰
# 忠实复现 Chaney 2025 Figure 7a 全谱拟合："A unified model with four levels
# ... was used to fit the small-angle scattering region. Lorentz peaks were
# used to fit features in the wide-angle scattering region."
# 论文峰锚（Figure 7b 表，读图级精度）：
#   (100) broad  中心 0.27 Å⁻¹ = 2.7 nm⁻¹，FWHM 0.08 Å⁻¹ = 0.8 nm⁻¹
#   (100) narrow 中心 0.29 Å⁻¹ = 2.9 nm⁻¹，FWHM 0.02 Å⁻¹ = 0.2 nm⁻¹
#   (012)        中心 0.66 Å⁻¹ = 6.6 nm⁻¹，FWHM 0.04 Å⁻¹ = 0.4 nm⁻¹
# ---------------------------------------------------------------------------

NAMES_PEAK = ['A_b', 'c_b', 'w_b',      # (100) broad：幅度/中心/FWHM
              'A_n', 'c_n', 'w_n',      # (100) narrow
              'A_0', 'c_0', 'w_0']      # (012)

NAMES_FULL = NAMES + NAMES_PEAK         # 22 自由参数（13 统一 + 9 峰）

# 2026-10-29 用户授权放宽：ZWH-Sam-D08-285-A（USAXS 硬碳）q≈0.1 nm^-1 宽鼓包
# 需 Lorentz 结构因子落位，原 Chaney 锚定窗 c_b/c_n∈[2.0,3.5]、c_0∈[5.0,8.0]
# 覆盖不到 0.1 → 峰中心下界统一放宽至 0.02 nm^-1（上界不动）；
# 换数据集时峰中心边界应按数据形态重新裁决并在报告披露
BOUNDS_PEAK = {
    'A_b': (0.0, np.inf), 'c_b': (0.02, 3.5), 'w_b': (0.05, 2.0),
    'A_n': (0.0, np.inf), 'c_n': (0.02, 3.5), 'w_n': (0.05, 2.0),
    'A_0': (0.0, np.inf), 'c_0': (0.02, 8.0), 'w_0': (0.05, 2.0),
}

# 峰幅度 INIT 按 L2 体检实估值起步：WAXS 区峰顶净高（数据 I_max − unified 尾
# 外推）——(100) 顶 ~3.4e-4（narrow 主导）、(012) 顶 ~4e-5；σ/I≈26% 下峰信噪
# 2–3σ，INIT 量级正确即可，fitter 自调
INIT_PEAK = {
    'A_b': 1.0e-4, 'c_b': 2.7, 'w_b': 0.8,
    'A_n': 2.0e-4, 'c_n': 2.9, 'w_n': 0.2,
    'A_0': 4.0e-5, 'c_0': 6.6, 'w_0': 0.4,
}


def lorentz(q, a, c, w):
    """Lorentz 峰：A / (1 + 4(q−q0)²/FWHM²)，w = FWHM（直接对标论文锚口径）"""
    q = np.asarray(q, dtype=float)
    return a / (1.0 + 4.0 * (q - c) ** 2 / w ** 2)


def peak_curve(q, d, key):
    """第 key 个峰（'b'/'n'/'0'）完整贡献线（components/CSV 分量列共用）"""
    return lorentz(q, d['A_' + key], d['c_' + key], d['w_' + key])


def model_full(p, qq):
    """full 模式引擎接口：统一四 level + 三 Lorentz 峰（共享常数背景 B）。
    向量按 NAMES_FULL 解包（22 键；_pd 只认 13 键 NAMES，不能复用）"""
    d = p if isinstance(p, dict) else dict(zip(NAMES_FULL, p))
    return (model_vec(d, qq)
            + lorentz(qq, d['A_b'], d['c_b'], d['w_b'])
            + lorentz(qq, d['A_n'], d['c_n'], d['w_n'])
            + lorentz(qq, d['A_0'], d['c_0'], d['w_0']))


def derived_full(p, perr):
    """full 模式导出量：统一部分（Rg/q_knee/K）+ 峰中心/FWHM（nm⁻¹，
    直接对标论文 Figure 7b 峰锚）"""
    out = derived_main(p, perr)
    for k, key in (('b', '100b'), ('n', '100n'), ('0', '012')):
        out['peak%s_center' % key] = float(p['c_' + k])
        out['peak%s_center_err' % key] = float(perr['c_' + k])
        out['peak%s_fwhm' % key] = float(p['w_' + k])
        out['peak%s_fwhm_err' % key] = float(perr['w_' + k])
    return out


# ---------------------------------------------------------------------------
# 降阶 unified 族 lv1/lv2/lv3（2026-10-09 增）：level 数 1-3 的级联截断，
# 用于模型阶数比较（奥卡姆剃刀：lv1→lv2→lv3→main 构成嵌套模型族，阶数取舍
# 以嵌套 F 检验 + 参数可辨识性为准，χ²/dof 单一指标不作降阶依据）。
# 与 sasmodels unified_power_Rg level=k 同式；编号语义与 main 一致
# （G1=最小尺度级尾无压制，Gk=最大尺度级）。
# ---------------------------------------------------------------------------
NAMES_N = {
    1: ['G1', 'Rg1', 'P1', 'B'],
    2: ['G1', 'Rg1', 'P1', 'G2', 'Rg2', 'P2', 'B'],
    3: ['G1', 'Rg1', 'P1', 'G2', 'Rg2', 'P2', 'G3', 'Rg3', 'P3', 'B'],
}

BOUNDS_LV = {k: {n: BOUNDS[n] for n in NAMES_N[k]} for k in (1, 2, 3)}

# INIT 按 Henderson Figure-6 数据形态分配（2026-10-09）：低 q 斜率 ≈−1.6 由
# 最大级 Porod 尾承载（其膝在窗外/窗缘），高 q 渐近斜率 ≈−2.8 由更小尺度级
# Guinier 交越承载；幅度对齐该数据 I_max≈1e4 量级。降阶模式无独立文献锚，
# 换数据集应按目标形态重配 INIT（fitter 自调 + multistart 取证）。
INIT_LV = {
    # 2026-10-28 重配（闭孔硬碳测试数据 A1/A2_test 曲线形态，multistart 取证的
    # 全局谷折中值 G1=500/Rg1=20/P1=4：平台→膝（q≈0.06-0.09 nm⁻¹）→q⁻⁴ 尾；
    # 原 Henderson 形态（20000/30/2）在本数据落 6475 坏谷（丢平台纯幂律解）。
    1: {'G1': 500.0, 'Rg1': 20.0, 'P1': 4.0},
    # 2026-10-28 重配（闭孔硬碳 A1/A2 数据形态，对照 lv3 全局谷的两级合并预期）：
    # 最大级 G2/Rg2≈900/22 承载低 q 幂律（lv3 的 G3/Rg3 谷位），最小级 G1/Rg1≈0.03/1
    # 承载高 q 交越；原 Henderson 形态（60000/50/1.6）在本数据落局部极小。
    2: {'G1': 0.03, 'Rg1': 1.0, 'P1': 2.9, 'G2': 900.0, 'Rg2': 22.0, 'P2': 3.7},
    # 2026-10-27 重配（闭孔硬碳 A1/A2 数据形态，multistart 暖起点取证的全局谷；
    # 原 Henderson 形态 INIT 在本数据上落局部极小：A1 定背底 χ²/dof 34.7 vs 1.30、
    # A2 6.14 vs 1.16）：幂律尾主导形态——最大级 Rg3≈22 nm 承载低 q 幂律、
    # 中间级 Rg2≈4.3 nm、最小级 Rg1≈1 nm；幅度按本数据 a.u. 标度。
    # G3/Rg3 取 A1（1124/22.8）与 A2（340/21.3）两全局谷的折中值，全程窗
    #（0.149–6.59 nm⁻¹）双样品标准拟合均直收敛各自全局谷（当轮 multistart 取证）。
    3: {'G1': 0.02, 'Rg1': 1.0, 'P1': 3.0, 'G2': 0.6, 'Rg2': 4.3, 'P2': 2.7,
        'G3': 600.0, 'Rg3': 22.0, 'P3': 3.7},
}


def model_n(p, qq, k):
    """k 级 unified 级联引擎接口（k=4 时与 model_vec 逐位一致——同序求和）。
    p 按 NAMES_N[k] 顺序向量或参数 dict"""
    d = p if isinstance(p, dict) else dict(zip(NAMES_N[k], p))
    q = np.asarray(qq, dtype=float)
    with errstate(divide='ignore', invalid='ignore'):
        if k == 1:      # 单级：尾无压制（特殊分支防引用不存在的 Rg0）
            total = _level(q, d['G1'], d['Rg1'], d['P1'], None)
        else:
            total = _level(q, d['G%d' % k], d['Rg%d' % k], d['P%d' % k],
                           d['Rg%d' % (k - 1)])
            for i in range(k - 1, 1, -1):
                total = total + _level(q, d['G%d' % i], d['Rg%d' % i],
                                       d['P%d' % i], d['Rg%d' % (i - 1)])
            total = total + _level(q, d['G1'], d['Rg1'], d['P1'], None)
    return total + d['B']


def derived_n(p, perr, k):
    """降阶 lvk 模式导出量：k 级 rg_nm/_A/q_knee/K（与 derived_main 同口径）"""
    out = {}
    for i in range(1, k + 1):
        rg = float(p['Rg%d' % i])
        rge = float(perr['Rg%d' % i])
        out['rg%d_nm' % i] = rg
        out['rg%d_nm_err' % i] = rge
        out['rg%d_A' % i] = rg * 10.0          # ×10 对标文献 Å 口径
        out['rg%d_A_err' % i] = rge * 10.0
        out['q_knee%d_nm1' % i] = float(np.sqrt(3.0)) / rg
        out['k%d' % i] = k_of(float(p['G%d' % i]), rg, float(p['P%d' % i]))
    return out
