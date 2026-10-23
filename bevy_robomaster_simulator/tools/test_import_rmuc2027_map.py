"""Validate the shipped map against the upstream CAD manifest and gameplay assets."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from import_rmuc2027_map import read_glb

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / 'assets/maps/rmuc2027'


def accessor(data, binary, index):
    a = data['accessors'][index]
    view = data['bufferViews'][a['bufferView']]
    dtype = {5121: 'u1', 5125: '<u4', 5126: '<f4'}[a['componentType']]
    width = {'SCALAR': 1, 'VEC3': 3, 'VEC4': 4}[a['type']]
    start = view.get('byteOffset', 0) + a.get('byteOffset', 0)
    return np.frombuffer(binary, dtype=dtype, count=a['count'] * width, offset=start).reshape(-1, width)


class ImportedMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((MAP / 'import_report.json').read_text())
        cls.source = json.loads((MAP / 'source_manifest.json').read_text())

    def test_source_provenance(self):
        self.assertEqual(self.report['source_manifest_sha256'], hashlib.sha256((MAP / 'source_manifest.json').read_bytes()).hexdigest())
        self.assertEqual(self.source['part_count'] if 'part_count' in self.source else len(self.source['parts']), 644)
        cache = ROOT / '.cache/rmuc2027/field_meshes.npz'
        if cache.exists():
            self.assertEqual(hashlib.sha256(cache.read_bytes()).hexdigest(), self.report['source_sha256'])

    def test_render_bounds_and_triangle_counts_match_cad(self):
        expected = {p['mesh_id']: p for p in self.source['parts']}
        seen = set()
        triangles = 0
        for file in self.report['render_files']:
            data, binary = read_glb(MAP / file)
            for mesh in data['meshes']:
                name = mesh['name']
                self.assertNotIn(name, seen)
                seen.add(name)
                p = mesh['primitives'][0]
                v = accessor(data, binary, p['attributes']['POSITION'])
                f = accessor(data, binary, p['indices']).reshape(-1, 3)
                self.assertTrue(np.isfinite(v).all())
                self.assertLess(int(f.max()), len(v))
                self.assertLessEqual(len(f), expected[name]['triangles'])
                lo, hi = expected[name]['bounds_m']['min'], expected[name]['bounds_m']['max']
                np.testing.assert_allclose(v.min(0), [lo[0], lo[2]+1.850872, 1.6243434-hi[1]], atol=3e-6)
                np.testing.assert_allclose(v.max(0), [hi[0], hi[2]+1.850872, 1.6243434-lo[1]], atol=3e-6)
                triangles += len(f)
        omitted = {f'p{i:05}' for i in list(range(5, 167)) + list(range(174, 260))}
        self.assertEqual(seen | omitted, set(expected))
        self.assertFalse(seen & omitted)
        self.assertEqual(triangles, self.report['render_triangles'])

    def test_collision_buffers_survive_gltf_loading(self):
        data, binary = read_glb(MAP / 'collision.glb')
        count = 0
        for mesh in data['meshes']:
            p = mesh['primitives'][0]
            self.assertIn('NORMAL', p['attributes'])
            v = accessor(data, binary, p['attributes']['POSITION'])
            normals = accessor(data, binary, p['attributes']['NORMAL'])
            self.assertEqual(len(normals), len(v))
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)
            f = accessor(data, binary, p['indices']).reshape(-1, 3)
            self.assertTrue(np.isfinite(v).all())
            self.assertGreater(len(f), 0)
            self.assertLess(int(f.max()), len(v))
            t = v[f]
            self.assertTrue((np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1) > 1e-12).all())
            count += len(f)
        self.assertEqual(count, self.report['collision_triangles'])

    def test_rune_geometry_is_unchanged_and_not_scaled(self):
        _, original = read_glb(ROOT / 'assets/POWER.glb')
        adapted, binary = read_glb(MAP / 'power.glb')
        self.assertEqual(original, binary)
        root = adapted['nodes'][adapted['scenes'][0]['nodes'][0]]
        self.assertEqual(root['scale'], [1, 1, 1])
        outposts, _ = read_glb(MAP / 'outposts.glb')
        for root in outposts['scenes'][0]['nodes']:
            self.assertEqual(outposts['nodes'][root]['scale'], [1, 1, 1])


if __name__ == '__main__':
    unittest.main()
