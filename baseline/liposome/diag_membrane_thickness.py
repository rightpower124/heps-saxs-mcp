# -*- coding: utf-8 -*-
"""
诊断: 膜厚度参数映射检验
当前方案把膜参数(zh, tau_h, tau_t, drho_t)固定自不含PEG样品,
隐含假设 = 两样品膜厚度相同。
本脚本在含PEG数据上放开膜参数重拟合, 检验 d_hh = 2*zh 是否漂移:
  方案0: 膜全固定 (原方案, 对照)
  方案1: 放开 zh, tau_h (tau_t, drho_t 仍固定)
  方案2: 膜 4 参数全放开 (12 参数)
"""
import numpy as np
from scipy.optimize import least_squares

FILE = r"D:\博士后研究\B自编程序\2026-09-17-脂质体\测试数据\20260603-含PEG脂质体.dat"
QMIN, QMAX = 0.3, 4.4
LN2 = np.log(2.0)

# 固定膜参数（A方案, 来自不含PEG样品）
ZH0, TAU_H0, TAU_T0, DRHO_T0 = 1.6428, 1.3000, 0.6566, -6.0000

q_all, I_all, s_all = [], [], []
for line in open(FILE, encoding='utf-8', errors='ignore'):
    p = line.split()
    if len(p) == 3:
        try:
            q_all.append(float(p[0])*10.0)
            I_all.append(float(p[1]))
            s_all.append(float(p[2]))
        except ValueError:
            pass
q_all, I_all, s_all = map(np.asarray, (q_all, I_all, s_all))
m = (q_all >= QMIN) & (q_all <= QMAX) & (s_all > 0)
q, I, sig = q_all[m], I_all[m], s_all[m]
print('含PEG数据: %d 点, q = %.2f ~ %.2f nm-1' % (len(q), q[0], q[-1]))

def I_membrane(qq, zh, tau_h, tau_t, drho_t, z_peg, tau_peg, drho_peg):
    z = np.array([-z_peg, -zh, 0.0, zh, z_peg])
    tau = np.array([tau_peg, tau_h, tau_t, tau_h, tau_peg])
    dr = np.array([drho_peg, 1.0, drho_t, 1.0, drho_peg])
    out = np.zeros_like(qq)
    for i in range(5):
        for j in range(5):
            out += (dr[i]*dr[j]*tau[i]*tau[j]/(16*LN2)
                    * np.exp(-qq**2*(tau[i]**2 + tau[j]**2)/(16*LN2))
                    * np.cos(qq*(z[i] - z[j])))
    return 2*np.pi**3/qq**2*out

def Sq(qq, f, d, sd):
    return 1.0 + f*np.exp(-qq**2*sd**2/2.0)*np.cos(qq*d)

c10 = np.sqrt(np.log(10)/(4*LN2))

def report(tag, x, chi, idx):
    """idx: 各量在参数向量中的位置, None 表示固定"""
    zh = x[idx['zh']] if idx['zh'] is not None else ZH0
    th = x[idx['tau_h']] if idx['tau_h'] is not None else TAU_H0
    zpeg = x[idx['z_peg']]
    tpeg = x[idx['tau_peg']]
    d = x[idx['d']]
    edge_head = zh + c10*th
    print('\n[%s]  chi2/dof = %.4f' % (tag, chi))
    print('  zh = %.4f nm,  tau_h = %.4f nm' % (zh, th))
    print('  d_hh = 2*zh      = %.3f nm' % (2*zh))
    print('  疏水核心(粗估)   = %.2f nm  (2*zh - tau_h)' % (2*zh - th))
    print('  头基10%%外缘      = %.3f nm,  全膜厚 = %.3f nm' % (edge_head, 2*edge_head))
    print('  水层 d-全膜厚    = %.2f nm' % (d - 2*edge_head))
    print('  PEG外缘/总外壳   = %.2f nm  (2*(z_peg+0.91*tau_peg))' % (2*(zpeg + c10*tpeg)))

def run(resid, p0, lo, hi, npar, tag, idx):
    r = least_squares(resid, p0, bounds=(lo, hi), method='trf', x_scale='jac',
                      max_nfev=40000)
    chi = (r.fun**2).sum()/(len(q)-npar)
    pcov = np.linalg.pinv(r.jac.T @ r.jac)*chi
    perr = np.sqrt(np.diag(pcov))
    report(tag, r.x, chi, idx)
    if idx['zh'] is not None:
        k = idx['zh']
        print('  (zh 误差 %.4f -> d_hh = %.3f +- %.3f nm)'
              % (perr[k], 2*r.x[k], 2*perr[k]))
    return r.x, chi

# ---- 方案0: 膜全固定 (8参数, 原方案) ----
def r0(p):
    A, f, d, sd, B, z_peg, tau_peg, drho_peg = p
    return (A*I_membrane(q, ZH0, TAU_H0, TAU_T0, DRHO_T0, z_peg, tau_peg, drho_peg)
            *Sq(q, f, d, sd) + B - I)/sig
run(r0, [1e-4, 0.335, 9.05, 1.82, 0.0061, 4.5, 0.79, 0.05],
    [0, 0, 4, 0.3, 0, 2.0, 0.3, 0],
    [np.inf, 1, 16, 6, np.inf, 5, 5, 0.5], 8, '方案0 膜全固定(原方案)',
    dict(zh=None, tau_h=None, z_peg=5, tau_peg=6, d=2))

# ---- 方案1: 放开 zh, tau_h (10参数) ----
def r1(p):
    A, zh, tau_h, f, d, sd, B, z_peg, tau_peg, drho_peg = p
    return (A*I_membrane(q, zh, tau_h, TAU_T0, DRHO_T0, z_peg, tau_peg, drho_peg)
            *Sq(q, f, d, sd) + B - I)/sig
run(r1, [1e-4, 1.6428, 1.30, 0.335, 9.05, 1.82, 0.0061, 4.5, 0.79, 0.05],
    [0, 0.90, 0.25, 0, 4, 0.3, 0, 2.0, 0.3, 0],
    [np.inf, 2.20, 2.00, 1, 16, 6, np.inf, 5, 5, 0.5], 10, '方案1 放开zh+tau_h',
    dict(zh=1, tau_h=2, z_peg=7, tau_peg=8, d=4))

# ---- 方案2: 膜全放开 (12参数) ----
def r2(p):
    A, zh, tau_h, tau_t, drho_t, f, d, sd, B, z_peg, tau_peg, drho_peg = p
    return (A*I_membrane(q, zh, tau_h, tau_t, drho_t, z_peg, tau_peg, drho_peg)
            *Sq(q, f, d, sd) + B - I)/sig
run(r2, [1e-4, 1.6428, 1.30, 0.6566, -6.0, 0.335, 9.05, 1.82, 0.0061, 4.5, 0.79, 0.05],
    [0, 0.90, 0.25, 0.25, -6.00, 0, 4, 0.3, 0, 2.0, 0.3, 0],
    [np.inf, 2.20, 2.00, 1.80, -0.10, 1, 16, 6, np.inf, 5, 5, 0.5], 12, '方案2 膜全放开',
    dict(zh=1, tau_h=2, z_peg=9, tau_peg=10, d=6))
