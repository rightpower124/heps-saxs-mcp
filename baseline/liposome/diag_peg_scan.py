# -*- coding: utf-8 -*-
"""
诊断脚本2：PEG参数简并度量化
诊断4: tau_peg 扫描（固定tau拟合其余7参数）-> L_peg 对 tau 的依赖（宽度-幅度简并）
诊断5: z_peg 扫描 -> 检验PEG高斯是否真的偏好水层中部(z~d/2)，
       还是头基附近也能拟合（后者意味着只是膜模型误差补偿，不是PEG）
"""
import numpy as np
from scipy.optimize import least_squares

FILE = r"D:\博士后研究\B自编程序\2026-09-17-脂质体\测试数据\20260603-含PEG脂质体.dat"
QMIN, QMAX = 0.3, 4.4
ZH, TAU_H, TAU_T, DRHO_T = 1.6428, 1.3000, 0.6566, -6.0000
LN2 = np.log(2.0)

q_all, I_all, s_all = [], [], []
for line in open(FILE, encoding='utf-8', errors='ignore'):
    p = line.split()
    if len(p) == 3:
        try:
            q_all.append(float(p[0]) * 10.0)
            I_all.append(float(p[1]))
            s_all.append(float(p[2]))
        except ValueError:
            pass
q_all, I_all, s_all = map(np.asarray, (q_all, I_all, s_all))
m = (q_all >= QMIN) & (q_all <= QMAX) & (s_all > 0)
q, I, sig = q_all[m], I_all[m], s_all[m]

def I_membrane5(qq, z_peg, tau_peg, drho_peg):
    z = np.array([-z_peg, -ZH, 0.0, ZH, z_peg])
    tau = np.array([tau_peg, TAU_H, TAU_T, TAU_H, tau_peg])
    dr = np.array([drho_peg, 1.0, DRHO_T, 1.0, drho_peg])
    out = np.zeros_like(qq)
    for i in range(5):
        for j in range(5):
            out += (dr[i]*dr[j]*tau[i]*tau[j]/(16*LN2)
                    * np.exp(-qq**2*(tau[i]**2 + tau[j]**2)/(16*LN2))
                    * np.cos(qq*(z[i] - z[j])))
    return 2*np.pi**3 / qq**2 * out

def Sq_of(qq, f, d, sd):
    return 1.0 + f*np.exp(-qq**2*sd**2/2.0)*np.cos(qq*d)

c10 = np.sqrt(np.log(10)/(4*LN2))
edge_head = ZH + c10*TAU_H

def fit7(zfix=None, tfix=None, p0=None):
    """固定 z_peg 或 tau_peg 之一，拟合其余 7 参数"""
    if p0 is None:
        p0 = [1.0e-4, 0.335, 9.05, 1.82, 0.0061, 4.5, 0.06]
    if tfix is not None:
        def model_p(p, qq):
            A, f, d, sd, B, z_peg, drho = p
            return A*I_membrane5(qq, z_peg, tfix, drho)*Sq_of(qq, f, d, sd) + B
        lo = [0.0, 0.0, 4.00, 0.30, 0.0, 2.00, 0.00]
        hi = [np.inf, 1.0, 16.0, 6.00, np.inf, 5.00, 0.50]
    else:
        def model_p(p, qq):
            A, f, d, sd, B, tau_peg, drho = p
            return A*I_membrane5(qq, zfix, tau_peg, drho)*Sq_of(qq, f, d, sd) + B
        lo = [0.0, 0.0, 4.00, 0.30, 0.0, 0.30, 0.00]
        hi = [np.inf, 1.0, 16.0, 6.00, np.inf, 5.00, 0.50]
    resid = lambda p: (model_p(p, q) - I)/sig
    r = least_squares(resid, p0, bounds=(lo, hi), method='trf',
                      x_scale='jac', max_nfev=20000)
    chi = (r.fun**2).sum()/(len(q)-7)
    return r.x, chi

# ============================================================
# 诊断4: tau_peg 扫描
# ============================================================
print('========== 诊断4: tau_peg 扫描（固定tau, 拟合7参数） ==========')
print('%8s %10s %10s %10s %10s' % ('tau_peg', 'chi2/dof', 'z_peg', 'drho_peg', 'L_peg'))
warm = None
for tfix in [0.3, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0]:
    p0 = warm if warm is not None else [1.0e-4, 0.335, 9.05, 1.82, 0.0061, 4.5, 0.06]
    x, chi = fit7(tfix=tfix, p0=list(p0))
    warm = list(x)
    z_peg, drho = x[5], x[6]
    L = z_peg + c10*tfix - edge_head
    print('%8.2f %10.4f %10.3f %10.4f %10.2f' % (tfix, chi, z_peg, drho, L))

# ============================================================
# 诊断5: z_peg 扫描（tau_peg 自由, 下界0.30）
# ============================================================
print('\n========== 诊断5: z_peg 扫描（固定z, tau自由） ==========')
print('%8s %10s %10s %10s' % ('z_peg', 'chi2/dof', 'tau_peg', 'drho_peg'))
warm = None
for zfix in [2.5, 3.0, 3.5, 4.0, 4.5, 4.53, 5.0]:
    p0 = warm if warm is not None else [1.0e-4, 0.335, 9.05, 1.82, 0.0061, 0.8, 0.06]
    x, chi = fit7(zfix=zfix, p0=list(p0))
    warm = list(x)
    tau_peg, drho = x[5], x[6]
    print('%8.2f %10.4f %10.3f %10.4f' % (zfix, chi, tau_peg, drho))

# 对照: 完全自由拟合中 drho_peg 与 tau_peg 的相关系数（诊断2的协方差重新算一遍）
def model_peg8(p, qq):
    A, f, d, sd, B, z_peg, tau_peg, drho_peg = p
    return A*I_membrane5(qq, z_peg, tau_peg, drho_peg)*Sq_of(qq, f, d, sd) + B

p0 = [1.0e-4, 0.335, 9.05, 1.82, 0.0061, 4.53, 0.30, 0.12]
lo_p = [0.0, 0.0, 4.00, 0.30, 0.0, 2.00, 0.30, 0.00]
hi_p = [np.inf, 1.0, 16.0, 6.00, np.inf, 5.00, 5.00, 0.50]
r = least_squares(lambda p: (model_peg8(p, q) - I)/sig, p0,
                  bounds=(lo_p, hi_p), method='trf', x_scale='jac', max_nfev=20000)
chi = (r.fun**2).sum()/(len(q)-8)
pcov = np.linalg.pinv(r.jac.T @ r.jac)*chi
perr = np.sqrt(np.diag(pcov))
corr = pcov/np.outer(perr, perr)
print('corr(drho_peg, tau_peg) = %.3f' % corr[7, 6])
print('corr(drho_peg, A)       = %.3f' % corr[7, 0])
print('自由拟合: tau_peg=%.2f (下界0.30), drho_peg=%.3f +- %.3f, chi2/dof=%.4f'
      % (r.x[6], r.x[7], perr[7], chi))
