# -*- coding: utf-8 -*-
"""模型插件加载器（M2-T1）
扫描 model_library/{validated,candidates}/<模型名>/，以 manifest.json 定位：
  entry 字段（默认 model.py）= 物理代码，零改动
  plugin.py = 适配层，暴露 register(M) -> modes 统一接口：
    names / bounds / init / init_qmax / fixed_keys / calc / derived
    profile / z_range / profile_n / profile_title
    （可选）components / compare / report
物理模块与适配层用唯一模块名加载（spec_from_file_location），
避免不同插件的同名模块（model.py / plugin.py）在 sys.modules 里互相覆盖。
"""
import importlib.util
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.normpath(os.path.join(_HERE, os.pardir, 'model_library'))
# inbox 收件区（部署优化 261104）：用户/开发端把待入库模型放入根目录 model_inbox/
# <模型名>/（三件套同构）。zone='inbox' 仅在 list_models 展示（status='inbox'），
# saxs_fit / saxs_diag_* 拒绝使用——未过验证门的模型不得进入正式分析。
_INBOX = os.path.normpath(os.path.join(_HERE, os.pardir, 'model_inbox'))


def _scan_dirs():
    """yield (zone, model_dir)；zone ∈ validated / candidates / inbox。
    顺序即同名解析优先级：validated > candidates > inbox。"""
    for zone, zdir in (('validated', os.path.join(_LIB, 'validated')),
                       ('candidates', os.path.join(_LIB, 'candidates')),
                       ('inbox', _INBOX)):
        if not os.path.isdir(zdir):
            continue
        for name in sorted(os.listdir(zdir)):
            mdir = os.path.join(zdir, name)
            if os.path.isfile(os.path.join(mdir, 'manifest.json')):
                yield zone, mdir


def list_models():
    """模型库清单（供 saxs_list_models 工具与 Agent 选型）"""
    out = []
    for zone, mdir in _scan_dirs():
        with open(os.path.join(mdir, 'manifest.json'), encoding='utf-8') as f:
            man = json.load(f)
        out.append({'name': man['name'], 'version': man.get('version', ''),
                    'zone': zone, 'status': man.get('status', zone),
                    'description': man.get('description', ''),
                    'modes': sorted(man.get('modes', {})),
                    'recommended_workflow': man.get('recommended_workflow'),
                    'dir': mdir,
                    'inbox_note': ('待入库收件区：未过验证门，禁止正式拟合；'
                                   '走 workflow §7 入库流程'
                                   if zone == 'inbox' else None)})
    # M2.5-4：sasmodels 桥接通道（官方基础形状，开箱即用）
    try:
        import sasmodels_bridge
        out.extend(sasmodels_bridge.available())
    except Exception:
        pass
    return out


def _import(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(name):
    """按 manifest.name 定位模型目录，返回 {'manifest','dir','zone','modes','physics'}"""
    for zone, mdir in _scan_dirs():
        with open(os.path.join(mdir, 'manifest.json'), encoding='utf-8') as f:
            man = json.load(f)
        if man.get('name') != name:
            continue
        physics = _import(os.path.join(mdir, man.get('entry', 'model.py')),
                          'saxs_model_%s' % name)
        plugin = _import(os.path.join(mdir, man.get('plugin', 'plugin.py')),
                         'saxs_plugin_%s' % name)
        return {'manifest': man, 'dir': mdir, 'zone': zone,
                'modes': plugin.register(physics), 'physics': physics}
    # M2.5-4：本地模型库未命中 → sasmodels 桥接通道
    try:
        import sasmodels_bridge
        hit = sasmodels_bridge.lookup(name)
    except Exception:
        hit = None
    if hit is not None:
        return hit
    raise ValueError('模型未找到: %s；可用: %s'
                     % (name, [m['name'] for m in list_models()]))
