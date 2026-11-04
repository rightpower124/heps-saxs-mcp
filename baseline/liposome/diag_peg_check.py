# -*- coding: utf-8 -*-
"""
诊断脚本：验证含PEG拟合结果的两个关键问题
诊断1: 无PEG模型从含PEG拟合的共享参数(d=9.05)出发重新拟合
       -> 确认 chi0=1.606 是全局最优而非局部极小（关系到PEG检出显著性的判读）
诊断2: 放宽 tau_peg 下界 0.80 -> 0.30，看PEG宽度想去哪、L_peg 如何变化
       -> 原拟合中 tau_peg=0.8000 恰好撞下界
"""
import numpy as np
from scipy.optimize import least_squares

FILE = r"D:\博士后研究\B自编程序\2026-09-17-脂质体\测试数据\20260603-含PEG脂质体.dat"
QMIN, QMAX = 0.3, 4.4

# ---- 膜参数固定（与 fit_peg_membrane.py 完全一致）----
ZH, TAU_H, TAU_T, DRHO_T = 1.6428, 1.3000, 0.6566, -6.0000
LN2 = np.log(2.0)

# ---------- 1. 读数据（与原脚本一致） ----------
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
print('数据点: %d,  q = %.2f ~ %.2f nm-1' % (len(q), q[0], q[-1]))

# ---------- 2. 模型（与原脚本一致） ----------
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

def model_peg(p, qq):
    A, f, d, sd, B, z_peg, tau_peg, drho_peg = p
    return A*I_membrane5(qq, z_peg, tau_peg, drho_peg)*Sq_of(qq, f, d, sd) + B

def model_nop(p, qq):
    A, f, d, sd, B = p
    return A*I_membrane5(qq, 2.8, 2.5, 0.0)*Sq_of(qq, f, d, sd) + B

def resid_peg(p):
    return (model_peg(p, q) - I)/sig

def resid_nop(p):
    return (model_nop(p, q) - I)/sig

c10 = np.sqrt(np.log(10)/(4*LN2))
edge_head = ZH + c10*TAU_H

# ============================================================
# 诊断1: 无PEG模型换起点重拟合（原脚本从 d=6.512 出发）
# ============================================================
print('\n========== 诊断1: 无PEG模型全局最优验证 ==========')
lo_0 = [0.0, 0.0, 4.00, 0.30, 0.0]
hi_0 = [np.inf, 1.0, 16.0, 6.00, np.inf]

# 起点A: 从含PEG拟合收敛的共享参数出发
p_start_a = np.array([1.0e-4, 0.335, 9.05, 1.82, 0.0061])
r1a = least_squares(resid_nop, p_start_a, bounds=(lo_0, hi_0), method='trf',
                    x_scale='jac', max_nfev=20000)
chi_a = (r1a.fun**2).sum()/(len(q)-5)
print('起点(d=9.05,sd=1.82): chi2/dof = %.4f' % chi_a)
print('  A=%.4e  f=%.4f  d=%.4f  sd=%.4f  B=%.5f' % tuple(r1a.x))

# 起点B: 原脚本初值（d=6.512）
p_start_b = np.array([1.43e-4, 0.266, 6.512, 0.644, 0.006086])
r1b = least_squares(resid_nop, p_start_b, bounds=(lo_0, hi_0), method='trf',
                    x_scale='jac', max_nfev=20000)
chi_b = (r1b.fun**2).sum()/(len(q)-5)
print('起点(d=6.512, 原脚本): chi2/dof = %.4f' % chi_b)
print('  A=%.4e  f=%.4f  d=%.4f  sd=%.4f  B=%.5f' % tuple(r1b.x))

chi0_best = min(chi_a, chi_b)
print('-> 无PEG模型最优 chi2/dof = %.4f (798-5=793 dof, chi2=%.1f)'
      % (chi0_best, chi0_best*(len(q)-5)))

# ============================================================
# 诊断2: 放宽 tau_peg 下界 0.80 -> 0.30
# ============================================================
print('\n========== 诊断2: tau_peg 下界放宽到 0.30 ==========')
p_peg0 = np.array([1.0e-4, 0.335, 9.05, 1.82, 0.0061, 4.50, 0.79, 0.0494])
lo_p = [0.0, 0.0, 4.00, 0.30, 0.0, 2.00, 0.30, 0.00]
hi_p = [np.inf, 1.0, 16.0, 6.00, np.inf, 5.00, 5.00, 0.50]
r2 = least_squares(resid_peg, p_peg0, bounds=(lo_p, hi_p), method='trf',
                   x_scale='jac', max_nfev=20000)
chi_2 = (r2.fun**2).sum()/(len(q)-8)
pcov = np.linalg.pinv(r2.jac.T @ r2.jac)*chi_2
perr = np.sqrt(np.diag(pcov))
names = ['A', 'f', 'd', 'sd', 'B', 'z_peg', 'tau_peg', 'drho_peg']
for n, v, e in zip(names, r2.x, perr):
    print('  %-8s %10.4f  +- %.4f' % (n, v, e))
print('chi2/dof = %.4f (dof=%d, chi2=%.1f)' % (chi_2, len(q)-8, chi_2*(len(q)-8)))

edge_peg2 = r2.x[5] + c10*r2.x[6]
L2 = edge_peg2 - edge_head
print('头基10%%外缘 = %.2f nm,  PEG 10%%外缘 = %.2f nm' % (edge_head, edge_peg2))
print('L_peg = %.2f nm (原结果 2.40)' % L2)

eff = (model_peg(r2.x, q) - model_peg(np.concatenate([r2.x[:7], [0.0]]), q))/model_peg(r2.x, q)
print('PEG层最大相对影响 = %.1f%% (原结果 3.9%%)' % (100*np.max(np.abs(eff))))

# 参数相关矩阵中 drho_peg 与 A 的相关
corr = pcov/np.outer(perr, perr)
print('corr(drho_peg, A) = %.3f' % corr[7, 0])
print('corr(drho_peg, z_peg) = %.3f' % corr[7, 5])
print('corr(drho_peg, B) = %.3f' % corr[7, 4])

# ============================================================
# 诊断3: 检出显著性（似然比 / F检验）
# ============================================================
print('\n========== 诊断3: PEG项检出显著性 ==========')
chi0 = chi0_best*(len(q)-5)
chi1 = chi_2*(len(q)-8)
dchi = chi0 - chi1
F = (dchi/3.0)/(chi1/(len(q)-8))
print('无PEG最优  chi2 = %.1f (5参数)' % chi0)
print('含PEG最优  chi2 = %.1f (8参数)' % chi1)
print('Delta chi2 = %.1f,  F(3,%d) = %.1f  -> p << 1e-30' % (dchi, len(q)-8, F))
print('若视为单一幅度参数(PEG贡献 ~ A*drho_peg): 显著性 ~ %.1f sigma' % np.sqrt(dchi-1))
print('边际显著性(drho_peg/perr) = %.1f sigma' % (r2.x[7]/perr[7]))
