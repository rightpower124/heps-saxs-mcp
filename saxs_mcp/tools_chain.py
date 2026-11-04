# -*- coding: utf-8 -*-
"""Ruland 全流程链式编排工具薄壳（提速三件套 T-C）：saxs_ruland_chain。
三阶段 thin dispatcher（detect / prepare / bpi2），质量门停点保留；
beq 终点走 saxs_ruland_beq（scan_qmins 内置扫描）。引擎与写盘在 chain.py
card 层，本模块只做参数透传与校验。"""
import os

from chain import chain_bpi2, chain_detect, chain_prepare


def register(mcp):
    @mcp.tool()
    def saxs_ruland_chain(path: str, stage: str, x_semantic: str,
                          x_unit: str, q_unit: str = None,
                          q_values: list = None, columns: list = None,
                          phi_ref: float = None, params: dict = None,
                          fit_log: bool = None, out_dir: str = None) -> dict:
        """Ruland 全流程链式编排（提速三件套 T-C）：把 Ruland 取向分析的机械
        接线收拢为三阶段调用，全流程 MCP 调用数 32 → 4；质量门停点全部保留。
        path: N+1 列方位角积分文件（第 1 列 φ/deg）或 prepare 后的 filled_book；
        stage: 'detect'（逐列 gap 检测，停点=逐列目检标记图）| 'prepare'
        （批量填充 + 合并总表，输入原始文件；停点=目检整备报告）| 'bpi2'
        （filled_book 峰积分，mask 自动接线；停点=目检 B 表+QC 图）。
        其余 stage 一律报错——不提供 detect→fill 一键贯通（lessons.md 批量
        执行纪律：必须逐列 detect→目检→fill）；beq 终点直接调
        saxs_ruland_beq（传 scan_qmins 扫描自动采纳或 q_window 人工单窗）。
        x_semantic: 'phi'，必填不猜；x_unit: 'deg' 等，必填不猜；
        q_unit: 仅 stage='bpi2' 需要（'nm^-1'/'A^-1'，必填不猜）；
        q_values: 各列 q_c 清单（bpi2 阶段；缺省从文件头解析，Q9 唯一候选
        才用，歧义报错）；
        columns: 1-based 因变量列号列表（缺省全部列，三阶段通用）；
        phi_ref: 统一选峰位置 deg（仅 bpi2；None=自动跨列聚类）；
        params: 各阶段引擎参数覆盖（detect/fill 共用 _DEFAULTS 口径、
        bpi2 触达判据口径）；
        fit_log: 仅 prepare（覆盖拟合域）；
        out_dir: 链产物根目录，缺省 runs/<stem>_chain/（det/ gapfix/ bpi2/
        子目录；stem 对 filled_book 输入自动剥 _filled_book 后缀对齐同链）。
        推荐调用序列（000064 实测 4 次调用收官）：
        ①chain stage=detect → ②目检 16 列标记图 → ③chain stage=prepare →
        ④目检整备报告 → ⑤chain stage=bpi2(path=filled_book, q_unit=…) →
        ⑥目检 B 表 → ⑦saxs_ruland_beq(bpi2_csv, scan_qmins=…)。
        纪律：填充值=条件值；gaptable/mask 路径按确定性约定发现（col%02d 目录、
        <stem>_gaptable_col<k>.json、<stem>_col<k>_mask.csv），缺失硬报错
        （P7 报错不猜测）；采纳窗自动规则见 saxs_ruland_beq 的 scan_qmins。"""
        if stage == 'detect':
            return chain_detect(path, x_semantic, x_unit, columns=columns,
                                params=params, out_dir=out_dir)
        if stage == 'prepare':
            return chain_prepare(path, x_semantic, x_unit, columns=columns,
                                 params=params, fit_log=fit_log,
                                 out_dir=out_dir)
        if stage == 'bpi2':
            if not q_unit:
                raise ValueError("stage='bpi2' 需要 q_unit（'nm^-1'/'A^-1'，"
                                 '必填不猜）')
            return chain_bpi2(path, q_unit, q_values=q_values, columns=columns,
                              phi_ref=phi_ref, params=params, out_dir=out_dir)
        raise ValueError("stage 须为 'detect'/'prepare'/'bpi2'（实得 %r）；"
                         'beq 终点走 saxs_ruland_beq，不支持 detect→fill '
                         '一键贯通（lessons.md 批量执行纪律）' % stage)