#!/usr/bin/env python3
"""Инспектор GLB-головы: скелет, веса, морфы и проверка готовности к ARKit (52 blendshapes).

Использование:
    python tools/inspect_glb.py DvaGolovaOK_2_lite.glb
    python tools/inspect_glb.py DvaGolovaOK_2_lite.json        # JSON-обёртка с base64 в поле "data"
    python tools/inspect_glb.py DvaGolovaOK_2_lite_base64.txt  # чистый base64

Зависимости: только numpy.
"""
import base64
import json
import struct
import sys
from collections import Counter

import numpy as np

ARKIT_52 = [
    'eyeBlinkLeft', 'eyeLookDownLeft', 'eyeLookInLeft', 'eyeLookOutLeft', 'eyeLookUpLeft', 'eyeSquintLeft', 'eyeWideLeft',
    'eyeBlinkRight', 'eyeLookDownRight', 'eyeLookInRight', 'eyeLookOutRight', 'eyeLookUpRight', 'eyeSquintRight', 'eyeWideRight',
    'jawForward', 'jawLeft', 'jawRight', 'jawOpen',
    'mouthClose', 'mouthFunnel', 'mouthPucker', 'mouthLeft', 'mouthRight', 'mouthSmileLeft', 'mouthSmileRight',
    'mouthFrownLeft', 'mouthFrownRight', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthPressLeft', 'mouthPressRight',
    'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'noseSneerLeft', 'noseSneerRight', 'tongueOut',
]
CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT2': 4, 'MAT3': 9, 'MAT4': 16}


def read_glb_bytes(path):
    raw = open(path, 'rb').read()
    if raw[:4] == b'glTF':
        return raw
    text = raw.decode('utf-8', errors='ignore').strip()
    if text.startswith('{'):
        return base64.b64decode(json.loads(text)['data'])
    return base64.b64decode(text)


def parse_glb(b):
    magic, version, length = struct.unpack('<4sII', b[:12])
    assert magic == b'glTF', 'не GLB'
    off, chunks = 12, []
    while off < len(b):
        clen, ctype = struct.unpack('<I4s', b[off:off + 8])
        chunks.append((ctype, b[off + 8:off + 8 + clen]))
        off += 8 + clen
    g = json.loads(chunks[0][1])
    binb = chunks[1][1] if len(chunks) > 1 else b''
    return g, binb


def accessor(g, binb, i):
    a = g['accessors'][i]
    n = NC[a['type']]
    dt = CT[a['componentType']]
    if 'bufferView' not in a:
        return np.zeros((a['count'], n), dtype=dt)
    bv = g['bufferViews'][a['bufferView']]
    start = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    item = np.dtype(dt).itemsize * n
    stride = bv.get('byteStride', 0) or item
    buf = np.frombuffer(binb, dtype=np.uint8, count=stride * (a['count'] - 1) + item, offset=start)
    rows = np.lib.stride_tricks.as_strided(buf, shape=(a['count'], item), strides=(stride, 1))
    arr = np.frombuffer(rows.tobytes(), dtype=dt).reshape(a['count'], n)
    if a.get('normalized') and dt != np.float32:
        arr = arr.astype(np.float64) / np.iinfo(dt).max
    return arr


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def local_matrix(n):
    if 'matrix' in n:
        return np.array(n['matrix'], float).reshape(4, 4).T
    M = np.eye(4)
    M[:3, :3] = quat_to_mat(n.get('rotation', [0, 0, 0, 1])) @ np.diag(n.get('scale', [1, 1, 1]))
    M[:3, 3] = n.get('translation', [0, 0, 0])
    return M


def world_matrices(g):
    W = [None] * len(g['nodes'])

    def rec(i, P):
        W[i] = P @ local_matrix(g['nodes'][i])
        for c in g['nodes'][i].get('children', []):
            rec(c, W[i])
    for r in g['scenes'][g.get('scene', 0)]['nodes']:
        rec(r, np.eye(4))
    return W


def image_size(data):
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return struct.unpack('>II', data[16:24])
    if data[:2] == b'\xff\xd8':
        i = 2
        while i < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack('>HH', data[i + 5:i + 9])
                return w, h
            i += 2 + struct.unpack('>H', data[i + 2:i + 4])[0]
    return None


def main(path):
    g, binb = parse_glb(read_glb_bytes(path))
    nodes = g['nodes']
    W = world_matrices(g)
    print(f'Файл: {path}')
    print(f"Генератор: {g['asset'].get('generator')}, glTF {g['asset'].get('version')}")
    print(f"Расширения: {', '.join(g.get('extensionsUsed', [])) or '—'}")

    # ---------------- scene tree
    joints_all = set(j for s in g.get('skins', []) for j in s['joints'])
    print('\n=== Иерархия сцены (J = кость скина) ===')

    def tree(i, d=0, max_children_print=200):
        n = nodes[i]
        tags = []
        if i in joints_all:
            tags.append('J')
        if 'mesh' in n:
            tags.append(f"mesh:{g['meshes'][n['mesh']]['name']}")
        if 'skin' in n:
            tags.append(f"skin:{n['skin']}")
        p = W[i][:3, 3] * 100
        print('  ' * d + f"{n.get('name', i)} [{' '.join(tags)}] world=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) см")
        for c in n.get('children', [])[:max_children_print]:
            tree(c, d + 1)
    for r in g['scenes'][g.get('scene', 0)]['nodes']:
        tree(r)

    # ---------------- skins
    for si, skin in enumerate(g.get('skins', [])):
        joints = skin['joints']
        names = [nodes[j].get('name', str(j)) for j in joints]
        print(f"\n=== Скин {si} '{skin.get('name')}' : {len(joints)} костей ===")
        if 'inverseBindMatrices' in skin:
            ibm = accessor(g, binb, skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
            err = max(np.abs(W[j] @ ibm[k] - np.eye(4)).max() for k, j in enumerate(joints))
            print(f"Bind pose == rest pose: {'да' if err < 1e-4 else 'НЕТ'} (max|W·IBM−I| = {err:.1e})")
        total = np.zeros(len(joints))
        for m in g['meshes']:
            for p in m['primitives']:
                if 'JOINTS_0' not in p['attributes']:
                    continue
                sets = [k for k in p['attributes'] if k.startswith('JOINTS_')]
                J = np.hstack([accessor(g, binb, p['attributes'][f'JOINTS_{k}']) for k in range(len(sets))]).astype(int)
                Wt = np.hstack([accessor(g, binb, p['attributes'][f'WEIGHTS_{k}']) for k in range(len(sets))]).astype(float)
                np.add.at(total, J.ravel(), Wt.ravel())
                hist = Counter((Wt > 1e-4).sum(1).tolist())
                print(f"  {m['name']}: влияний на вершину {dict(sorted(hist.items()))}; сумма весов {Wt.sum(1).min():.3f}…{Wt.sum(1).max():.3f}")
        unused = [names[k] for k in range(len(joints)) if total[k] == 0]
        print(f"Кости без весов ({len(unused)}): {', '.join(unused) or '—'}")

    # ---------------- meshes / morphs
    print('\n=== Меши ===')
    all_targets = []
    for m in g['meshes']:
        tnames = m.get('extras', {}).get('targetNames', [])
        for pi, p in enumerate(m['primitives']):
            pos = g['accessors'][p['attributes']['POSITION']]
            ntri = g['accessors'][p['indices']]['count'] // 3 if 'indices' in p else pos['count'] // 3
            mat = g['materials'][p['material']]['name'] if 'material' in p else '—'
            print(f"  {m['name']}[{pi}] вершин={pos['count']} треугольников={ntri} материал={mat} морфов={len(p.get('targets', []))}")
            print(f"      атрибуты: {', '.join(p['attributes'])}")
            for k in p['attributes']:
                if k.startswith('COLOR_'):
                    c = accessor(g, binb, p['attributes'][k]).astype(float)
                    if np.ptp(c, axis=0).max() == 0:
                        print(f'      {k}: константа {c[0].round(3).tolist()} — бесполезные данные, можно удалить')
        all_targets += tnames
    print(f"Имена морф-таргетов: {', '.join(all_targets) or 'нет ни одного'}")

    # ---------------- materials / images
    print('\n=== Материалы ===')
    for mat in g.get('materials', []):
        pbr = mat.get('pbrMetallicRoughness')
        note = ''
        if pbr is None:
            note = '  <-- нет pbrMetallicRoughness: по спецификации metallic=1, roughness=1 (глаза будут серыми/тёмными)'
        elif 'baseColorTexture' not in pbr:
            note = '  <-- без baseColor текстуры'
        print(f"  {mat['name']}: {json.dumps(pbr) if pbr else '—'}{note}")
    for im in g.get('images', []):
        if 'bufferView' in im:
            bv = g['bufferViews'][im['bufferView']]
            data = binb[bv.get('byteOffset', 0):bv.get('byteOffset', 0) + bv['byteLength']]
            print(f"  image {im.get('name')}: {im.get('mimeType')} {image_size(data)} {bv['byteLength'] // 1024} KB")

    # ---------------- ARKit
    print('\n=== Проверка ARKit (52 blendshapes) ===')
    low = {t.lower().split('.')[-1].replace('arkit_', ''): t for t in all_targets}
    found = [a for a in ARKIT_52 if a.lower() in low]
    missing = [a for a in ARKIT_52 if a.lower() not in low]
    print(f'Найдено {len(found)}/52')
    if missing:
        print('Отсутствуют:', ', '.join(missing))
    verdict = 'ГОТОВ' if not missing else 'НЕ ГОТОВ (нужны морф-таргеты с именами ARKit)'
    print('Итог:', verdict)
    face_bones = [nodes[j].get('name', '') for s in g.get('skins', []) for j in s['joints']]
    if missing and len(face_bones) > 20:
        print(f'Замечание: в скелете {len(face_bones)} костей (лицевой риг) — 52 формы можно запечь из поз костей.')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
