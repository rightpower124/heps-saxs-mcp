# -*- coding: utf-8 -*-
"""
含PEG脂质体 SAXS 拟合（第一轮：不扣背景，膜参数固定）
模型 = A · I_5G(q) · [1 + f·exp(-q^2·sd^2/2)·cos(q·d)] + B
  I_5G : 5-高斯膜 = 3-高斯膜（固定，来自不含PEG样品A方案拟合）
         + 对称PEG层（外层 +z_peg / 内层 -z_peg，镜像同参数）
  附带模型比较：同一数据再用 drho_peg=0（无PEG）拟合一次，
  两个 chi2/dof 之差 = PEG 层的检测显著性
数据  : 20260603-含PEG脂质体.dat，三列 q[A^-1]  I  sigma
输出  : PEG层参数/厚度/构象判据 + 拟合图
"""
import numpy as np
from scipy.optimize import least_squares

FILE = r"D:\博士后研究\B自编程序\2026-09-17-脂质体\测试数据\20260603-含PEG脂质体.dat"
QMIN, QMAX = 0.3, 4.4      # nm^-1，与不含PEG样品保持一致

# ---- 膜参数固定（来自不含PEG样品A方案拟合, chi2/dof=2.09）----
ZH, TAU_H, TAU_T, DRHO_T = 1.6428, 1.3000, 0.6566, -6.0000

# ---------- 1. 读数据 ----------
q_all, I_all, s_all = [], [], []
for line in open(FILE, encoding='utf-8', errors='ignore'):
    p = line.split()
    if len(p) == 3:
        try:
            q_all.append(float(p[0]) * 10.0)     # A^-1 -> nm^-1
            I_all.append(float(p[1]))
            s_all.append(float(p[2]))
        except ValueError:
            pass
q_all, I_all, s_all = map(np.asarray, (q_all, I_all, s_all))
m = (q_all >= QMIN) & (q_all <= QMAX) & (s_all > 0)
q, I, sig = q_all[m], I_all[m], s_all[m]
print('数据点: %d,  q = %.2f ~ %.2f nm-1' % (len(q), q[0], q[-1]))

LN2 = np.log(2.0)

# ---------- 2. 模型 ----------
def drho_profile5(z, z_peg, tau_peg, drho_peg):
    return (np.exp(-4*LN2*(z + ZH)**2/TAU_H**2)
            + np.exp(-4*LN2*(z - ZH)**2/TAU_H**2)
            + DRHO_T*np.exp(-4*LN2 * z**2/TAU_T**2)
            + drho_peg*(np.exp(-4*LN2*(z - z_peg)**2/tau_peg**2)
                        + np.exp(-4*LN2*(z + z_peg)**2/tau_peg**2)))

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

def model_nopeg(p, qq):
    A, f, d, sd, B = p
    return A*I_membrane5(qq, 2.8, 2.5, 0.0)*Sq_of(qq, f, d, sd) + B

# ---------- 3. 初值 ----------
f0, d0, sd0 = 0.266, 6.512, 0.644          # 取不含PEG拟合值
zpeg0, taupeg0, dpeg0 = 2.8, 2.5, 0.06     # PEG: 蘑菇构象, 低对比度
B0 = float(I[-100:].mean())
w = q <= 1.5
Iraw0 = I_membrane5(q, zpeg0, taupeg0, dpeg0)*Sq_of(q, f0, d0, sd0)
A0 = float(np.median((I[w] - B0)/Iraw0[w]))
p_peg0 = np.array([A0, f0, d0, sd0, B0, zpeg0, taupeg0, dpeg0])
p_nop0 = np.array([A0, f0, d0, sd0, B0])
print('初值: A0 = %.3e,  B0 = %.3e' % (A0, B0))

# ---------- 4. 拟合（含PEG + 无PEG模型比较）----------
def resid_peg(p):
    return (model_peg(p, q) - I)/sig

def resid_nop(p):
    return (model_nopeg(p, q) - I)/sig

lo_p = [0.0, 0.0, 4.00, 0.30, 0.0, 2.00, 0.80, 0.00]
hi_p = [np.inf, 1.0, 16.0, 6.00, np.inf, 5.00, 5.00, 0.50]
res = least_squares(resid_peg, p_peg0, bounds=(lo_p, hi_p), method='trf', x_scale='jac', max_nfev=20000)

lo_0 = [0.0, 0.0, 4.00, 0.30, 0.0]
hi_0 = [np.inf, 1.0, 16.0, 6.00, np.inf]
res0 = least_squares(resid_nop, p_nop0, bounds=(lo_0, hi_0), method='trf', x_scale='jac', max_nfev=20000)

dof1, dof0 = len(q) - 8, len(q) - 5
chi1 = (res.fun**2).sum()/dof1
chi0 = (res0.fun**2).sum()/dof0
pcov = np.linalg.pinv(res.jac.T @ res.jac)*chi1
perr = np.sqrt(np.diag(pcov))

names = ['A 比例因子', 'f 双层强度分数', 'd 层间距 (nm)', 'sd 层间距涨落 (nm)', 'B 常数背景',
         'z_peg PEG中心 (nm)', 'tau_peg PEG宽度 (nm)', 'drho_peg/drho_h']
print('\n========== 含PEG拟合结果（膜参数固定） ==========')
for n, v, e in zip(names, res.x, perr):
    print('%-22s %10.4f  +- %.4f' % (n, v, e))
print('-' * 58)
print('无PEG模型 (5参数)   chi2/dof = %.3f' % chi0)
print('含PEG模型 (8参数)   chi2/dof = %.3f' % chi1)
print('drho_peg 显著性     = %.1f sigma' % (res.x[7]/perr[7]))

# PEG 层厚度：10%判据（与论文一致）
c10 = np.sqrt(np.log(10)/(4*LN2))            # 高斯降到峰值10%处 = z0 +- c10*tau
edge_head = ZH + c10*TAU_H                   # 固定膜的头基外缘
edge_peg = res.x[5] + c10*res.x[6]
L_peg = edge_peg - edge_head
L_err = np.sqrt(perr[5]**2 + (c10*perr[6])**2)
print('头基10%%外缘 = %.2f nm,  PEG 10%%外缘 = %.2f nm' % (edge_head, edge_peg))
print('PEG层厚度 L_peg      = %.2f +- %.2f nm' % (L_peg, L_err))

R_F = 3.5                                     # PEG2000 Flory半径
print('Flory半径 R_F ~ 3.5 nm,  L_peg/R_F = %.2f' % (L_peg/R_F))
if L_peg < 1.4*R_F:
    print('-> 蘑菇/过渡区构象（与Zeta结果相符的预期）')
else:
    print('-> 刷状构象')

p_nopeg_cur = res.x.copy(); p_nopeg_cur[7] = 0.0
eff = (model_peg(res.x, q) - model_peg(p_nopeg_cur, q))/model_peg(res.x, q)
print('PEG层对曲线的最大相对影响 = %.1f%%' % (100*np.max(np.abs(eff))))

# ---------- 5. 作图 ----------
import matplotlib.pyplot as plt
qq = np.linspace(q[0], q[-1], 800)

fig = plt.figure(figsize=(10, 7))
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax3 = plt.subplot2grid((2, 2), (1, 1))

ZZ0 = model_peg(res.x, qq)

ax1.errorbar(q, I, sig, fmt='.', ms=2, lw=0.4, alpha=0.4, color='k', label='data')
ax1.plot(qq, model_peg(res.x, qq), 'r-', lw=1.5, label='fit: 5G membrane (PEG free)')
ax1.plot(qq, model_nopeg(res0.x, qq), color='0.5', ls=':', lw=1.5, label='best no-PEG model (drho_peg=0)')
ax1.set_yscale('log')
ax1.set_xlabel('q (nm$^{-1}$)')
ax1.set_ylabel('I (a.u.)')
ax1.set_title('PEG liposome, chi2/dof: noPEG=%.2f, PEG=%.2f' % (chi0, chi1))
ax1.legend(fontsize=8)

ax2.plot(q, (model_peg(res.x, q) - I)/I, '.', ms=2)
ax2.axhline(0, color='r', lw=0.8)
ax2.set_xlabel('q (nm$^{-1}$)')
ax2.set_ylabel('(fit - data)/data')
ax2.set_title('relative residual (PEG model)')

zz = np.linspace(-6, 6, 600)
ax3.plot(zz, drho_profile5(zz, *res.x[5:8]), 'g-')
ax3.axhline(0, color='gray', lw=0.6)
ax3.set_xlabel('z (nm)')
ax3.set_ylabel(r'$\Delta\rho(z)$ (rel.)')
ax3.set_title('L_peg = %.2f +- %.2f nm,  %.1f sigma' % (L_peg, L_err, res.x[7]/perr[7]))

plt.tight_layout()
plt.savefig(r"D:\博士后研究\B自编程序\2026-09-17-脂质体\fit_peg_result.png", dpi=200)
plt.show()
