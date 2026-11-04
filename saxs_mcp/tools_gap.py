# -*- coding: utf-8 -*-
"""曲线整备工具（M3pre-2 curve_conditioning）：saxs_gap_detect / saxs_gap_fill /
saxs_gap_merge。两步式设计（规格卡 R4）：detect 产候选段表与标记图，经裁决流
定案后由 fill 消费（gap_table JSON 可经用户/Agent 手改后传入，显式可复现）；
批量逐列填充后由 merge 拼合与源文件同形的总表（Origin 拖入对照用）。"""
import os

from gap_fill import (detect_card, fill_card, fill_card_batch,
                      merge_filled_book)


def register(mcp):
    @mcp.tool()
    def saxs_gap_detect(path: str, x_semantic: str, x_unit: str,
                        column: int = 1, detect_params: dict = None,
                        out_dir: str = None) -> dict:
        """一维曲线 gap/异常点检测（曲线整备第一步）。
        输入 2 列（x I）或 N+1 列（x I1..In，逐列独立检测）数据文件。
        x_semantic: 'q'（径向积分）或 'phi'（方位角切割），必填不猜；
        x_unit: x_semantic='q' 时 'A^-1'/'nm^-1'/'pixel'（探测器像素半径轴），
        'phi' 时 'deg'，必填不猜；
        column: 因变量列号（1-based，默认 1）；detect_params: 检测参数覆盖
        （D_deep/s_tol/D_broad/D_ext/W_min/w_frac/d_max 等，缺省用引擎内置）；
        out_dir: 交付物目录，缺省 runs/<文件名>_gapdetect/。
        返回候选段表（start_idx/end_idx/x_start/x_end/code/label/confidence/note
        + segment_counts + files 交付物路径）并在磁盘写出标记图 PNG 与 gap_table.json。
        code 形态学：1 深谷核心 / 2 渐变肩部 / 3 孤立异常点 / 4 宽缓阴影候选。
        裁决流：深谷/孤立点（Tier A）全自动定案；肩部与"干净"阴影（Tier B）须
        Agent 读图目检主判留痕；阴影与真实峰结构重叠或低置信（Tier C）提请用户
        裁决。段表定案后再进入填充环节。"""
        if not os.path.isfile(path):
            raise ValueError('文件不存在: %s' % path)
        return detect_card(path, x_semantic, x_unit, col=column,
                           params=detect_params, out_dir=out_dir)

    @mcp.tool()
    def saxs_gap_fill(path: str, x_semantic: str, x_unit: str, gap_table: str,
                      column: int = 1, params: dict = None, fit_log: bool = None,
                      out_dir: str = None, columns: list = None,
                      gap_tables: list = None) -> dict:
        """一维曲线 gap/异常点剔除与多项式填充（曲线整备第二步）。
        消费 saxs_gap_detect 产出的 gap 表（裁决定案后）：gap_table 为
        gaptable JSON 文件路径（含 segments 键，可经用户手改后传入），
        也可由用户手写段表直接给出。
        单列模式（缺省）：column= 因变量列号（1-based，默认 1），
        gap_table= 该列 gaptable JSON 路径。
        批量模式（提速三件套 T-B）：columns= 1-based 列号列表 + gap_tables=
        等长对齐的各列 gaptable JSON 路径列表（此时 column/gap_table 参数
        忽略）——内部循环逐列独立填充（单列路径逐位不变），任一列失败硬报错
        （P7），返回逐列摘要。
        x_semantic: 'q' 或 'phi'，必填不猜；
        x_unit: 'A^-1'/'nm^-1'/'pixel'（q 语义）/'deg'（phi 语义），必填不猜；
        params: 填充参数覆盖（w_frac/d_max/p_f/margin 等，批/单列共用）；
        fit_log: 覆盖拟合域（缺省 x_semantic=q → log I，phi → 按动态范围
        自动选并披露）；
        out_dir: 缺省 runs/<文件名>_gapfix/。
        返回 files 交付物清单（主输出 <原名>_filled.dat：x 照抄 + 填充后 y，
        带 # 表头信息块；QC 对照图 PNG；mask 表；gap_table 副本；result.json）
        + summary（各段阶数/RMS/包络违例/填充占比）。
        纪律：填充值一律按"条件值"对待——被遮挡处无光子记录，填充是数学插值
        而非信号恢复；包络违例段 clip + 强制披露；填充占比写入下游使用披露。"""
        if not os.path.isfile(path):
            raise ValueError('文件不存在: %s' % path)
        if columns is not None:
            if gap_tables is None or len(gap_tables) != len(columns):
                raise ValueError('批量模式须给出与 columns 等长对齐的 gap_tables')
            return fill_card_batch(path, x_semantic, x_unit, columns,
                                   gap_tables, params=params, fit_log=fit_log,
                                   out_dir=out_dir)
        return fill_card(path, x_semantic, x_unit, col=column, gap_table=gap_table,
                         params=params, fit_log=fit_log, out_dir=out_dir)

    @mcp.tool()
    def saxs_gap_merge(path: str, out_dir: str, columns: list = None,
                       filled_paths: list = None) -> dict:
        """批量曲线整备合并总表（第三步，打包不做拟合）：把 out_dir 内各列
        saxs_gap_fill 产物 <原名>_col<k>_filled.dat 的第 k 因变量列拼合为与源
        文件同形的总表 <原名>_filled_book.dat（x 与未合并列 = 源文件 token 照抄，
        合并列 gap 行 = 填充值、其余行与源文件一致），供 Origin 与原始数据
        book 并排拖入对照（行/列数一致，差异仅在 gap 单元）。
        path: 源 N+1 列数据文件；out_dir: 批量填充产物目录（也是总表输出目录，
        相对路径按工作区根解析）；columns: 1-based 因变量列号列表（缺省 = 目录
        内实际存在填充产物的全部列，升序合并）；filled_paths: 显式指定各列
        filled.dat（须与 columns 等长对齐）。
        校验：各列文件行数/列宽与源一致、x 逐行一致，违者报错不合并（P7）。
        返回总表路径 + 逐列 filled_fraction + 合并单元总数。填充值=条件值纪律
        随表头携带。"""
        if not os.path.isfile(path):
            raise ValueError('文件不存在: %s' % path)
        return merge_filled_book(path, out_dir, columns=columns,
                                 filled_paths=filled_paths)
