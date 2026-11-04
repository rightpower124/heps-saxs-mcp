# -*- coding: utf-8 -*-
"""
不含PEG脂质体 SAXS 膜拟合（第一轮：不扣背景）
模型 = A · I_mem(q) · [1 + f·exp(-q^2·sd^2/2)·cos(q·d)] + B
  I_mem : 3-高斯对称平膜（Schilt 2016 Eq.1），头基Δρ归一为1，尾区为相对值
  [·]   : 双层脂质体的两膜层间相关（Z=2 结构因子简化式，d=层间距）
  B     : 常数背景（代替未测量的缓冲液扣除）
数据  : R0-26080301-不含PEG脂质体.dat，三列 q[A^-1]  I  sigma
输出  : 参数表 + 拟合图（曲线/残差/Δρ(z)膜轮廓）
"""
import numpy as np
from scipy.optimize import least_squares

FILE = r"D:\博士后研究\B自编程序\2026-09-17-脂质体\测试数据\R0-26080301-不含PEG脂质体.dat"
QMIN, QMAX = 0.3, 4.4      # nm^-1，拟合区间（低端避开直射光/挡板尾部）

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

def model(p, qq):
    A, zh, tau_h, tau_t, drho_t, f, d, sd, B = p
    Sq = 1.0 + f*np.exp(-qq**2*sd**2/2.0)*np.cos(qq*d)
    return A*I_membrane(qq, zh, tau_h, tau_t, drho_t)*Sq + B

# ---------- 3. 初值 ----------
zh0, tau_h0, tau_t0, drt0 = 1.45, 0.50, 1.00, -1.00   # cryo-TEM 膜厚 2.9 nm -> d_hh~2.9
f0, d0, sd0 = 0.60, 8.50, 1.50                        # 0.7~0.9 nm-1 肩峰 -> d ~ 2π/0.8
B0 = float(I[-100:].mean())
w = q <= 1.5                                          # 只用中低q归一尺度，避免高q信噪比低
Iraw0 = I_membrane(q, zh0, tau_h0, tau_t0, drt0) * (1.0 + f0*np.exp(-q**2*sd0**2/2)*np.cos(q*d0))
A0 = float(np.median((I[w] - B0) / Iraw0[w]))
p0 = np.array([A0, zh0, tau_h0, tau_t0, drt0, f0, d0, sd0, B0])
print('初值: A0 = %.3e,  B0 = %.3e' % (A0, B0))

# ---------- 4. 拟合 ----------
def resid(p):
    return (model(p, q) - I) / sig

# A方案：放宽 tau_h 上界 0.90->1.30、drho_t 下界 -4.00->-6.00，看参数停在哪
# lo = [0.0, 0.90, 0.25, 0.50, -4.00, 0.0, 5.50, 0.30, 0.0]
# hi = [np.inf, 2.20, 0.90, 1.80, -0.10, 1.0, 16.0, 6.00, np.inf]
lo = [0.0, 0.90, 0.25, 0.50, -6.00, 0.0, 5.50, 0.30, 0.0]
hi = [np.inf, 2.20, 1.30, 1.80, -0.10, 1.0, 16.0, 6.00, np.inf]
res = least_squares(resid, p0, bounds=(lo, hi), method='trf', x_scale='jac', max_nfev=20000)

npar = 9
dof = len(q) - npar
chi2 = (res.fun**2).sum() / dof
pcov = np.linalg.pinv(res.jac.T @ res.jac) * (res.fun**2).sum() / dof
perr = np.sqrt(np.diag(pcov))

names = ['A 比例因子', 'zh 头基半距 (nm)', 'tau_h 头基FWHM (nm)', 'tau_t 尾链FWHM (nm)',
         'drho_t/drho_h', 'f 双层强度分数', 'd 层间距 (nm)', 'sd 层间距涨落 (nm)', 'B 常数背景']
print('\n========== 拟合结果 ==========')
for n, v, e in zip(names, res.x, perr):
    print('%-22s %10.4f  +- %.4f' % (n, v, e))
print('-' * 50)
print('头-头距离 d_hh   = %.3f +- %.3f nm' % (2*res.x[1], 2*perr[1]))
print('双层相关峰 q1   = 2pi/d = %.3f nm-1' % (2*np.pi/res.x[6]))
print('chi2/dof        = %.2f' % chi2)

# ---------- 5. 作图 ----------
import matplotlib.pyplot as plt
pf = res.x
qq = np.linspace(q[0], q[-1], 800)

fig = plt.figure(figsize=(10, 7))
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax3 = plt.subplot2grid((2, 2), (1, 1))

ZZ0 = model(pf, qq)

ax1.errorbar(q, I, sig, fmt='.', ms=2, lw=0.4, alpha=0.4, color='k', label='data')
ax1.plot(qq, model(pf, qq), 'r-', lw=1.5, label='fit: membrane*S2 + B')
ax1.plot(qq, pf[0]*I_membrane(qq, *pf[1:5]) + pf[8], 'b--', lw=1.0, label='membrane + B (no S2)')
ax1.set_yscale('log')
ax1.set_xlabel('q (nm$^{-1}$)')
ax1.set_ylabel('I (a.u.)')
ax1.set_title('no-PEG liposome, chi2/dof = %.2f' % chi2)
ax1.legend(fontsize=8)

ax2.plot(q, (model(pf, q) - I)/I, '.', ms=2)
ax2.axhline(0, color='r', lw=0.8)
ax2.set_xlabel('q (nm$^{-1}$)')
ax2.set_ylabel('(fit - data)/data')
ax2.set_title('relative residual')

zz = np.linspace(-3.5, 3.5, 500)
ax3.plot(zz, drho_profile(zz, *pf[1:5]), 'g-')
ax3.axhline(0, color='gray', lw=0.6)
ax3.set_xlabel('z (nm)')
ax3.set_ylabel(r'$\Delta\rho(z)$ (rel.)')
ax3.set_title('d_hh = %.2f +- %.2f nm' % (2*pf[1], 2*perr[1]))

plt.tight_layout()
# plt.savefig(r"D:\博士后研究\B自编程序\2026-09-17-脂质体\fit_nopeg_result.png", dpi=200)
plt.savefig(r"D:\博士后研究\B自编程序\2026-09-17-脂质体\fit_nopeg_result_A.png", dpi=200)
plt.show()
