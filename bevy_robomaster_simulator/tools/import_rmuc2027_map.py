#!/usr/bin/env python3
"""无参数导入 RMUC2027 的米制 CAD 网格；仅使用 numpy，不修改来源仓库。"""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import struct
import urllib.request

import numpy as np

# ===== 可编辑配置区 =====
SOURCE_DIR = Path('/home/ad/code/RMUC2027-map')  # 更换场地来源时修改，需含 manifest.json 和 LICENSE。
ROOT = Path(__file__).resolve().parents[1]  # 仿真目录迁移时自动跟随脚本位置，通常无需修改。
CACHE = ROOT / '.cache/rmuc2027/field_meshes.npz'  # 已下载的原始网格缓存位置，离线导入时放到这里。
OUTPUT = ROOT / 'assets/maps/rmuc2027'  # 修改生成资源的保存目录时同步修改场景加载路径。
SOURCE_URL = 'https://github.com/keilking66/RMUC2027-map/releases/download/v1.0.0/field_meshes.npz'  # 升级来源版本时修改并更新下方校验值。
SOURCE_SHA256 = 'ff7083cfc157c4836617b157d26213c883d40d988a6f1920e4567ec41ead1465'  # 更换原始网格时填写发布方的 SHA-256，禁止绕过校验。
SOURCE_COMMIT = '5833356ba0595a0ddba5090dc4065f8f99e683c3'  # 升级来源仓库时记录对应提交，便于追溯转换规格。
CAD_CENTER_Y = 1.6243434  # 场地平面对称中心，改变世界坐标原点时调整。
CAD_FLOOR_Z = -1.850872  # CAD 中轴地面基准高程，改变世界高度原点时调整。
CHUNK_BYTES = 40_000_000  # 每个显示 GLB 的目标分块大小，显存或版本库限制变化时调整。
COLLISION_GRID = 0.005  # 机械细节碰撞顶点的米制量化间距；0 保留原网格，5 mm 最大位移 4.34 mm。
DART_MARKER_CAD = [13.312952, -5.083, -0.8427]  # 新发射架上的仿真发射点（CAD 米制坐标）；精确炮管标定时调整，来源未提供功能标记。
# =====================


def cad_to_world(v):
    v = np.asarray(v, dtype=np.float32)
    return np.stack((v[..., 0], v[..., 2] - CAD_FLOOR_Z, CAD_CENTER_Y - v[..., 1]), axis=-1)


def read_glb(path):
    raw = path.read_bytes()
    length, kind = struct.unpack_from('<II', raw, 12)
    assert kind == 0x4e4f534a
    data = json.loads(raw[20:20 + length])
    binary_length = struct.unpack_from('<I', raw, 20 + length)[0]
    return data, raw[28 + length:28 + length + binary_length]


def write_glb(path, data, binary):
    j = json.dumps(data, separators=(',', ':'), ensure_ascii=False).encode()
    j += b' ' * (-len(j) % 4)
    binary += b'\0' * (-len(binary) % 4)
    path.write_bytes(struct.pack('<III', 0x46546c67, 2, 28 + len(j) + len(binary))
                     + struct.pack('<II', len(j), 0x4e4f534a) + j
                     + struct.pack('<II', len(binary), 0x004e4942) + binary)


def matrix(n):
    if 'matrix' in n:
        return np.array(n['matrix']).reshape(4, 4).T
    x, y, z, w = n.get('rotation', [0, 0, 0, 1])
    m = np.eye(4)
    m[:3, :3] = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                           [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                           [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]) @ np.diag(n.get('scale', [1]*3))
    m[:3, 3] = n.get('translation', [0]*3)
    return m


def world_bounds(data, node_indices):
    pts = []
    def walk(i, parent):
        node = data['nodes'][i]
        m = parent @ matrix(node)
        for p in data.get('meshes', [])[node['mesh']]['primitives'] if 'mesh' in node else []:
            a = data['accessors'][p['attributes']['POSITION']]
            lo, hi = a['min'], a['max']
            corners = np.array([[x, y, z, 1] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            pts.extend((corners @ m.T)[:, :3])
        for child in node.get('children', []):
            walk(child, m)
    for i in node_indices:
        walk(i, np.eye(4))
    pts = np.asarray(pts)
    return pts.min(0), pts.max(0)


class Glb:
    def __init__(self):
        self.data = dict(asset=dict(version='2.0', generator='Simulator_aimor RMUC2027 importer'),
                         scene=0, scenes=[dict(nodes=[])], nodes=[], meshes=[],
                         materials=[dict(name='CAD face colors', pbrMetallicRoughness=dict(
                             baseColorFactor=[1]*4, metallicFactor=0, roughnessFactor=0.9), doubleSided=True)],
                         buffers=[dict(byteLength=0)], bufferViews=[], accessors=[])
        self.binary = bytearray()

    def accessor(self, a, kind, component, target):
        a = np.ascontiguousarray(a)
        self.binary.extend(b'\0' * (-len(self.binary) % 4))
        view = len(self.data['bufferViews'])
        self.data['bufferViews'].append(dict(buffer=0, byteOffset=len(self.binary), byteLength=a.nbytes, target=target))
        self.binary.extend(a.tobytes())
        desc = dict(bufferView=view, componentType=component, count=len(a), type=kind)
        if kind == 'VEC3':
            desc.update(min=a.min(0).tolist(), max=a.max(0).tolist())
        if component == 5121:
            desc['normalized'] = True
        self.data['accessors'].append(desc)
        return len(self.data['accessors']) - 1

    def mesh(self, name, v, f, colors=None):
        attr = dict(POSITION=self.accessor(v.astype('<f4'), 'VEC3', 5126, 34962))
        if colors is not None:
            # Split vertices at color boundaries; normals follow the CAD tessellation.
            normals = np.zeros_like(v)
            triangles = v[f]
            face_normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            for corner in range(3):
                np.add.at(normals, f[:, corner], face_normal)
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
            attr['NORMAL'] = self.accessor(normals.astype('<f4'), 'VEC3', 5126, 34962)
            attr['COLOR_0'] = self.accessor(colors.astype('u1'), 'VEC4', 5121, 34962)
        else:
            # Bevy deindexes meshes without NORMAL to generate flat normals; Avian requires
            # an index buffer. This hidden physics-only mesh needs no shading normals.
            normals = np.zeros_like(v)
            normals[:, 1] = 1
            attr['NORMAL'] = self.accessor(normals.astype('<f4'), 'VEC3', 5126, 34962)
        primitive = dict(attributes=attr, indices=self.accessor(f.astype('<u4').ravel(), 'SCALAR', 5125, 34963), material=0)
        i = len(self.data['nodes'])
        self.data['scenes'][0]['nodes'].append(i)
        self.data['nodes'].append(dict(name=name, mesh=len(self.data['meshes'])))
        self.data['meshes'].append(dict(name=name, primitives=[primitive]))

    def save(self, path):
        self.data['buffers'][0]['byteLength'] = len(self.binary)
        write_glb(path, self.data, bytes(self.binary))


def clean_faces(v, f):
    t = v[f]
    area = np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1)
    return area > 1e-12


def source_bounds(parts, ids):
    chosen = [p['bounds_m'] for p in parts if int(p['mesh_id'][1:]) in ids]
    a = cad_to_world(np.array([p['min'] for p in chosen]))
    b = cad_to_world(np.array([p['max'] for p in chosen]))
    return np.minimum(a, b).min(0), np.maximum(a, b).max(0)


def fit_legacy(source, output, groups, parts):
    """Retain named gameplay nodes, fit their envelopes to corresponding CAD assemblies."""
    data, binary = read_glb(ROOT / 'assets' / source)
    original = copy.deepcopy(data)
    roots = []
    report = []
    for names, ids, label in groups:
        nodes = [i for i, n in enumerate(original['nodes']) if n.get('name') in names]
        assert len(nodes) == len(names), (source, names, nodes)
        lo, hi = world_bounds(original, nodes)
        target_lo, target_hi = source_bounds(parts, ids)
        # Uniform planar scale keeps face orientation; vertical scale matches CAD assembly height.
        scale = (target_hi-target_lo)/(hi-lo)
        planar = float((scale[0]+scale[2])/2)
        scale = np.array([planar, scale[1], planar])
        if source in ('POWER.glb', 'OUTPOST.glb'):
            # Preserve metric target dimensions used by vision and the calibrated fixed R glyph.
            scale = np.ones(3)
        translation = (target_hi+target_lo)/2 - scale*(hi+lo)/2
        if source in ('POWER.glb', 'OUTPOST.glb'):
            translation[1] = target_lo[1] - lo[1]
        parent = dict(name=label, children=nodes, scale=scale.tolist(), translation=translation.tolist())
        roots.append(len(data['nodes']))
        data['nodes'].append(parent)
        report.append(dict(name=label, source_nodes=names, replaced_source_parts=[min(ids), max(ids)],
                           translation=translation.tolist(), scale=scale.tolist(),
                           cad_world_bounds=[target_lo.tolist(), target_hi.tolist()]))
    data['scenes'] = [dict(nodes=roots)]
    data['scene'] = 0
    write_glb(OUTPUT / output, data, binary)
    return report


def main():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE.exists():
        print('Downloading', SOURCE_URL, flush=True)
        urllib.request.urlretrieve(SOURCE_URL, CACHE)
    digest = hashlib.sha256(CACHE.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f'来源网格校验失败：{digest}')
    manifest = json.loads((SOURCE_DIR / 'manifest.json').read_text())
    parts = manifest['parts']
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for old in OUTPUT.glob('render_*.glb'):
        old.unlink()
    mesh = np.load(CACHE, allow_pickle=False)
    omitted = set(range(5, 167)) | set(range(174, 260))
    render = Glb()
    collision = Glb()
    files, stats = [], []
    for part in parts:
        key = part['mesh_id']
        number = int(key[1:])
        if number in omitted:
            continue
        v = cad_to_world(mesh[key + '_vertices'])
        f = mesh[key + '_faces']
        keep = clean_faces(v, f)
        colors = np.round(np.clip(mesh[key + '_face_rgba'][keep], 0, 1)*255).astype('u1')
        colors[:, 3] = 255  # Match upstream USD's opaque field export.
        f = f[keep]
        # Source vertices are shared across CAD face colors; split only at color boundaries.
        keys = np.column_stack((f.ravel(), np.repeat(colors, 3, axis=0)))
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        render.mesh(key, v[unique[:, 0]], inverse.reshape(-1, 3), unique[:, 1:])
        cv, cf = v, f
        # Preserve all low-poly terrain exactly; quantize only dense mechanical details.
        quantized = COLLISION_GRID > 0 and len(f) > 15000
        if quantized:
            grid, inv = np.unique(np.rint(v / COLLISION_GRID).astype('i4'), axis=0, return_inverse=True)
            cv = (grid * COLLISION_GRID).astype('f4')
            cf = inv[f]
            cf = cf[clean_faces(cv, cf)]
            cf = np.unique(cf, axis=0)
        collision.mesh(key, cv, cf)
        stats.append(dict(id=key, source_triangles=part['triangles'], render_triangles=len(f),
                          collision_triangles=len(cf), collision_quantized=quantized))
        if len(render.binary) >= CHUNK_BYTES:
            file = f'render_{len(files):02}.glb'
            render.save(OUTPUT / file)
            files.append(file)
            render = Glb()
            print(f'{file}: imported through {key}', flush=True)
    if render.data['meshes']:
        file = f'render_{len(files):02}.glb'
        render.save(OUTPUT / file)
        files.append(file)
    collision.save(OUTPUT / 'collision.glb')

    # The source is a static CAD export. Keep simulator controllers and calibrated R orientation.
    power = fit_legacy('POWER.glb', 'power.glb', [(['ROOT'], range(5, 165), 'RMUC2027_POWER')], parts)
    outposts = fit_legacy('OUTPOST.glb', 'outposts.glb', [
        (['OUTPOST_1'], range(217, 260), 'RMUC2027_RED_OUTPOST'),
        (['OUTPOST_2'], range(174, 217), 'RMUC2027_BLUE_OUTPOST')], parts)
    core_data, _ = read_glb(ROOT / 'assets/TECH_CORE.glb')
    blue = [n['name'] for n in core_data['nodes'] if n.get('name', '').startswith('BLUE_')]
    red = [n['name'] for n in core_data['nodes'] if n.get('name', '').startswith('RED_')]
    core = fit_legacy('TECH_CORE.glb', 'tech_core.glb', [
        (red, range(165, 166), 'RMUC2027_RED_CORE'),
        (blue, range(166, 167), 'RMUC2027_BLUE_CORE')], parts)
    old_ground, _ = read_glb(ROOT / 'assets/GROUND.glb')
    direction = next(n for n in old_ground['nodes'] if n.get('name') == 'DART_LAUNCH_DIRECTION')
    dart_marker = dict(name='DART_LAUNCH_DIRECTION', translation=cad_to_world(DART_MARKER_CAD).tolist(),
                       rotation=direction.get('rotation', [0, 0, 0, 1]))
    write_glb(OUTPUT / 'markers.glb', dict(asset=dict(version='2.0'), scene=0,
              scenes=[dict(nodes=[0])], nodes=[dart_marker]), b'')
    report = dict(source_repository='https://github.com/keilking66/RMUC2027-map', source_version=manifest['version'],
                  source_commit=SOURCE_COMMIT, source_manifest_sha256=hashlib.sha256((SOURCE_DIR / 'manifest.json').read_bytes()).hexdigest(),
                  source_mesh_url=SOURCE_URL, source_sha256=digest,
                  coordinate_transform=dict(formula='[x, z - CAD_FLOOR_Z, CAD_CENTER_Y - y]',
                                            cad_center_y=CAD_CENTER_Y, cad_floor_z=CAD_FLOOR_Z, units='meter', scale=1),
                  render_files=files, source_parts=len(parts), static_parts=len(stats),
                  source_triangles=sum(p['triangles'] for p in parts),
                  render_triangles=sum(p['render_triangles'] for p in stats),
                  collision_triangles=sum(p['collision_triangles'] for p in stats),
                  collision_grid_m=COLLISION_GRID, maximum_collision_vertex_displacement_m=3**0.5*COLLISION_GRID/2,
                  interactive_adaptations=dict(power=power, outposts=outposts, tech_core=core),
                  dart_marker=dict(**dart_marker, calibration='Relocated legacy launch direction; source CAD has no functional muzzle anchor.'), parts=stats)
    (OUTPUT / 'import_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    shutil.copyfile(SOURCE_DIR / 'LICENSE', OUTPUT / 'LICENSE.txt')
    shutil.copyfile(SOURCE_DIR / 'manifest.json', OUTPUT / 'source_manifest.json')
    print(json.dumps({k: report[k] for k in ['static_parts', 'render_triangles', 'collision_triangles', 'render_files']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
