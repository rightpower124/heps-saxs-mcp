# -*- coding: utf-8 -*-
"""
脂质体平膜模型插件（模型库 1 号模型）
物理模型：A · I_mem(q) · [1 + f·exp(-q²·sd²/2)·cos(q·d)] + B
  I_mem : N-高斯对称平膜（Schilt 2016 Eq.1 简化形式），头基 Δρ 归一为 1，尾区为相对值
  [·]   : 双层脂质体的两膜层间相关（Z=2 结构因子简化式，d=层间距）
  B     : 常数背景（代替未测量的缓冲液扣除）

数学逐行移植自 baseline/liposome/fit_nopeg_membrane.py 与 fit_peg_membrane.py，
未作任何改动；仅把原脚本的全局固定膜参数（ZH/TAU_H/TAU_T/DRHO_T）改为显式传参，
以适配 MCP 工具的无状态调用方式。
"""
import numpy as np

LN2 = np.log(2.0)

# 高斯降到峰值 10% 处的半宽系数：z0 ± c10·tau
C10 = np.sqrt(np.log(10) / (4 * LN2))

# PEG2000 Flory 半径（构象判据用）
R_F = 3.5

# ---------- 参数表（dict 进出；数值语义与原脚本一致） ----------
NAMES_NOPEG = ['A', 'zh', 'tau_h', 'tau_t', 'drho_t', 'f', 'd', 'sd', 'B']
NAMES_PEG = ['A', 'f', 'd', 'sd', 'B', 'z_peg', 'tau_peg', 'drho_peg']

# 边界 = 原脚本 A 方案（放宽 tau_h 上界 0.90->1.30、drho_t 下界 -4.00->-6.00）
BOUNDS_NOPEG = {
    'A': (0.0, np.inf), 'zh': (0.90, 2.20), 'tau_h': (0.25, 1.30),
    'tau_t': (0.50, 1.80), 'drho_t': (-6.00, -0.10), 'f': (0.0, 1.0),
    'd': (5.50, 16.0), 'sd': (0.30, 6.00), 'B': (0.0, np.inf),
}
# 初值（cryo-TEM 膜厚 2.9 nm -> d_hh~2.9；0.7~0.9 nm-1 肩峰 -> d ~ 2π/0.8）
INIT_NOPEG = {'zh': 1.45, 'tau_h': 0.50, 'tau_t': 1.00, 'drho_t': -1.00,
              'f': 0.60, 'd': 8.50, 'sd': 1.50}

BOUNDS_PEG = {
    'A': (0.0, np.inf), 'f': (0.0, 1.0), 'd': (4.00, 16.0), 'sd': (0.30, 6.00),
    'B': (0.0, np.inf), 'z_peg': (2.00, 5.00), 'tau_peg': (0.80, 5.00),
    'drho_peg': (0.00, 0.50),
}
# PEG 初值：蘑菇构象、低对比度；f/d/sd 取不含 PEG 拟合值
INIT_PEG = {'z_peg': 2.8, 'tau_peg': 2.5, 'drho_peg': 0.06}

# ---------- 3-高斯膜（不含 PEG） ----------
def drho_profile(z, zh, tau_h, tau_t, drho_t):
    return (np.exp(-4*LN2*(z + zh)**2 / tau_h**2)
            + np.exp(-4*LN2*(z - zh)**2 / tau_h**2)
            + drho_t * np.exp(-4*LN2 * z**2 / tau_t**2))

def I_membrane(qq, zh, tau_h, tau_t, drho_t):
    z = np.array([-zh, 0.0, zh])
    tau = np.array([tau_h, tau_t, tau_h])
    dr = np.array([1.0, drho_t, 1.0])
    out = np.zeros_like(qq)
    for i in range(3):
        for j in range(3):
            out += (dr[i]*dr[j]*tau[i]*tau[j] / (16*LN2)
                    * np.exp(-qq**2*(tau[i]**2 + tau[j]**2)/(16*LN2))
                    * np.cos(qq*(z[i] - z[j])))
    return 2*np.pi**3 / qq**2 * out

# ---------- 5-高斯膜（含 PEG；膜参数显式传入） ----------
def drho_profile5(z, z_peg, tau_peg, drho_peg, zh, tau_h, tau_t, drho_t):
    return (np.exp(-4*LN2*(z + zh)**2/tau_h**2)
            + np.exp(-4*LN2*(z - zh)**2/tau_h**2)
            + drho_t*np.exp(-4*LN2 * z**2/tau_t**2)
            + drho_peg*(np.exp(-4*LN2*(z - z_peg)**2/tau_peg**2)
                        + np.exp(-4*LN2*(z + z_peg)**2/tau_peg**2)))

def I_membrane5(qq, z_peg, tau_peg, drho_peg, zh, tau_h, tau_t, drho_t):
    z = np.array([-z_peg, -zh, 0.0, zh, z_peg])
    tau = np.array([tau_peg, tau_h, tau_t, tau_h, tau_peg])
    dr = np.array([drho_peg, 1.0, drho_t, 1.0, drho_peg])
    out = np.zeros_like(qq)
    for i in range(5):
        for j in range(5):
            out += (dr[i]*dr[j]*tau[i]*tau[j]/(16*LN2)
                    * np.exp(-qq**2*(tau[i]**2 + tau[j]**2)/(16*LN2))
                    * np.cos(qq*(z[i] - z[j])))
    return 2*np.pi**3 / qq**2 * out

def Sq_of(qq, f, d, sd):
    return 1.0 + f*np.exp(-qq**2*sd**2/2.0)*np.cos(qq*d)

# ---------- 向量形式模型（least_squares 用；参数顺序见 NAMES_*） ----------
def model_nopeg_vec(p, qq):
    """9 参数：A, zh, tau_h, tau_t, drho_t, f, d, sd, B"""
    A, zh, tau_h, tau_t, drho_t, f, d, sd, B = p
    Sq = Sq_of(qq, f, d, sd)
    return A*I_membrane(qq, zh, tau_h, tau_t, drho_t)*Sq + B

def model_peg_vec(p, qq, fixed):
    """8 参数：A, f, d, sd, B, z_peg, tau_peg, drho_peg；fixed = 膜参数 dict"""
    A, f, d, sd, B, z_peg, tau_peg, drho_peg = p
    Sq = Sq_of(qq, f, d, sd)
    return (A*I_membrane5(qq, z_peg, tau_peg, drho_peg,
                          fixed['zh'], fixed['tau_h'], fixed['tau_t'], fixed['drho_t'])*Sq + B)

def model_nop_on_peg_vec(p, qq, fixed):
    """5 参数无 PEG 对照模型（drho_peg=0，PEG 层形状参数取占位值不生效）"""
    A, f, d, sd, B = p
    Sq = Sq_of(qq, f, d, sd)
    return (A*I_membrane5(qq, 2.8, 2.5, 0.0,
                          fixed['zh'], fixed['tau_h'], fixed['tau_t'], fixed['drho_t'])*Sq + B)

# ---------- 派生量 ----------
def derived_nopeg(p, perr):
    """头-头距离 d_hh 与双层相关峰位 q1=2π/d"""
    return {
        'd_hh': 2*float(p['zh']), 'd_hh_err': 2*float(perr['zh']),
        'q1': 2*np.pi/float(p['d']),
    }

def derived_peg(p, perr, fixed):
    """PEG 层厚度（10% 判据，与论文一致）、构象判据、显著性、最大相对影响"""
    edge_head = fixed['zh'] + C10*fixed['tau_h']        # 固定膜的头基外缘
    edge_peg = float(p['z_peg']) + C10*float(p['tau_peg'])
    L_peg = edge_peg - edge_head
    L_err = float(np.sqrt(perr['z_peg']**2 + (C10*perr['tau_peg'])**2))
    conf = '蘑菇/过渡区构象' if L_peg < 1.4*R_F else '刷状构象'
    return {
        'edge_head': float(edge_head), 'edge_peg': float(edge_peg),
        'L_peg': float(L_peg), 'L_err': L_err,
        'L_over_RF': float(L_peg/R_F), 'conformation': conf,
        'drho_peg_sigma': float(p['drho_peg']/perr['drho_peg']),
    }
