# -*- coding: utf-8 -*-
"""Ruland 取向度分析工具薄壳（orientation）：saxs_ruland_bpi2 /
saxs_ruland_beq 两工具链（规格卡 §5.1，Q2=A）。引擎与写盘渲染在 ruland.py
card 层；本模块只做参数透传与校验。"""
import os

from ruland import bpi2_card, beq_card


def register(mcp):
    @mcp.tool()
    def saxs_ruland_bpi2(path: str, q_unit: str, q_values: list = None,
                         columns: list = None, phi_ref: float = None,
                         params: dict = None, mask_files: list = None,
                         out_dir: str = None) -> dict:
        """Ruland 取向分析第二步：series 统一选峰 + 逐列触达式峰积分 → B_π/2 表。
        输入 N+1 列方位角积分文件（第 1 列 φ/deg，第 2..N+1 列各 q_c 下的 I；
        典型为 gap 整备后的 filled_book——gap 整备前置，未整备数据基线会异常）。
        q_unit: q_c 清单单位 'nm^-1'/'A^-1'，必填不猜；
        q_values: 各列 q_c 清单（nm^-1 或 A^-1 按 q_unit），须与选用列数等长，
        清单须已经用户确认；缺省 None 时从文件头注释行解析兜底（Q9：候选行须
        唯一——含 q 字样且浮点数个数==因变量列数，歧义报错列出全部头行）；
        columns: 1-based 因变量列号列表（缺省全部列）；
        phi_ref: 统一选峰位置 deg（None=自动跨列聚类，确认卡披露 φ_ref 与对称侧
        候选；可指定覆盖，360° 圆周口径，不硬编码 90°）；
        params: 触达判据覆盖（f_touch 缺省 0.02 / n_noise 3.0 / m_hold 3 /
        peak_tol_deg 5 / prominence_k 5 / tail_correction True 等）；
        mask_files: gap_fill 各列 <stem>_col<k>_mask.csv 显式路径列表（与
        columns 等长对齐；缺省 None=填充未统计并披露）；
        out_dir: 交付物目录，缺省 runs/<文件名>_ruland_bpi2/。
        产出：B_π/2 表 CSV（q_nm/B_deg/B_rad/B_rad_corr/n_filled/filled_frac/
        flags，Origin 可拖）+ 逐列 QC 网格图 PNG + B(q) 汇总图 PNG + result.json。
        纪律：某列 φ_ref 容差窗内无有效峰 → 硬报错（Q13）并列对称侧候选；
        填充依赖列 = 条件值（gap 插值非信号恢复）；
        B_rad_corr 含高斯尾校正（f 触达截断）。"""
        if not os.path.isfile(path):
            raise ValueError('文件不存在: %s' % path)
        return bpi2_card(path, q_unit, q_values=q_values, columns=columns,
                         phi_ref=phi_ref, params=params, mask_files=mask_files,
                         out_dir=out_dir)

    @mcp.tool()
    def saxs_ruland_beq(bpi2_csv: str, q_window: list = None,
                        b_column: str = 'B_rad_corr', params: dict = None,
                        out_dir: str = None, scan_qmins: list = None) -> dict:
        """Ruland 取向分析第三步：Ruland 变换 q² vs q²·B² + 线性区域 OLS → B_eq。
        bpi2_csv: 上一步 saxs_ruland_bpi2 产出的 B 表 CSV 路径（或同形手写表，
        必需列 q_nm/B_rad/B_rad_corr）；
        q_window: 线性区域 [qmin, qmax] nm⁻¹（单窗模式，人工裁决线性区）；
        scan_qmins: 线性区扫描 qmin 清单 nm⁻¹（提速三件套 T-A，与 q_window
        互斥）——逐窗扫描（qmax=数据最大 q）后按最大无曲率窗规则自动采纳
        （2026-09-30 用户裁决：规则命中即自动产出；第一个曲率 F 检验 p≥0.05
        且非 NaN 的窗），采纳窗全套交付物写入 <out_dir>/adopted/；全部窗
        不通过 → 仅出证据链 CSV+报告 + adopt_manual 转人工裁决。人工覆盖：
        显式传 q_window 重跑单窗模式；
        b_column: 'B_rad_corr'（缺省，含高斯尾校正）或 'B_rad'；
        params: 拟合诊断参数覆盖（r2_min 0.95 / loo_warn_rel 0.10 /
        q_span_min 3.0）；
        out_dir: 缺省 runs/<B表stem>_ruland_beq/（扫描模式 …_ruland_beq_scan/）。
        产出：单窗模式 = Ruland 图 PNG（标题含 f_v）+ 分析报告 md（B_eq +
        取向度 f_v + 诊断 + 条件值口径）+ result.json；扫描模式另加逐窗证据链
        CSV（Origin 可拖）+ 扫描报告 md（规则披露+证据表）。
        纪律：B_eq = √斜率为取向参数（稳健/条件值分级由诊断 flags 定）；
        取向度 f_v = 1 − 2√(ln2/π³)·B_eq_rad 为正式产物（弧度制口径，
        分级继承 B_eq，f_v<0 触发 fv_negative flag）；
        微孔长度 L 不产出（L=2π/q 原则），截距仅作质量诊断量——负截距 =
        系统偏差信号（背景残留/线性区不当/B 口径不一致），触发线性区复核；
        残差曲率 F 检验显著 → 提示复核 q 窗口（双 L 群体/背景残留）。"""
        if not os.path.isfile(bpi2_csv):
            raise ValueError('B 表不存在: %s' % bpi2_csv)
        return beq_card(bpi2_csv, q_window=q_window, b_column=b_column,
                        params=params, out_dir=out_dir, scan_qmins=scan_qmins)
