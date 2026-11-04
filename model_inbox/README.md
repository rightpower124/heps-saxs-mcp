# model_inbox —— 模型入库收件区

把**待入库的模型插件三件套**放入本目录即可被 `saxs_list_models` 自动发现并展示
（zone=inbox，status=inbox）：

```
model_inbox/
└── <模型名>/
    ├── model.py        # 物理代码（散射强度公式 I(q)）
    ├── plugin.py       # 适配层：names/bounds/init/calc/plot_layout 等契约
    └── manifest.json   # 元数据：name/version/modes/labels/refs
```

**纪律（强制）**：

- inbox 区模型**未过验证门**（L1 合成自检 + L2 文献对照 + 人工审核签字），
  `saxs_fit` / `saxs_diag_*` 会**拒绝使用**并提示入库流程；
- `saxs_selftest` 对 inbox 模型只做契约静态检查，不跑冒烟拟合；
- 晋级流程：验证门全过 + 人工签字 → 移入 `model_library/validated/` → 正式可用；
- 本目录内容不入版本库（.gitignore 已排除子目录）。
