"""Загрузка головы D.Va из GLB в numpy-структуры."""
import io

import numpy as np
from PIL import Image

import common
from glb import load, acc, world_mats
from render import vertex_normals


def load_dva(path=None):
    g, binb = load(str(path or common.GLB))
    W = world_mats(g)
    imgs = []
    for im in g['images']:
        bv = g['bufferViews'][im['bufferView']]
        data = binb[bv.get('byteOffset', 0): bv.get('byteOffset', 0) + bv['byteLength']]
        imgs.append(np.asarray(Image.open(io.BytesIO(data)).convert('RGB'), dtype=float) / 255)
    parts = {}
    for m in g['meshes']:
        for p in m['primitives']:
            mat = g['materials'][p['material']]
            name = {0: 'eyes', 1: 'skin', 2: 'teeth'}[p['material']]
            V = acc(g, binb, p['attributes']['POSITION'])
            F = acc(g, binb, p['indices']).reshape(-1, 3).astype(int)
            UV = acc(g, binb, p['attributes']['TEXCOORD_0'])
            J = acc(g, binb, p['attributes']['JOINTS_0']).astype(int)
            Wt = acc(g, binb, p['attributes']['WEIGHTS_0'])
            tex = None
            bct = mat.get('pbrMetallicRoughness', {}).get('baseColorTexture')
            if bct is not None:
                tex = imgs[g['textures'][bct['index']]['source']]
            parts[name] = dict(V=V, F=F, UV=UV, N=vertex_normals(V, F), J=J, W=Wt, tex=tex, material=mat['name'])
    skin = g['skins'][0]
    joints = skin['joints']
    names = [g['nodes'][j]['name'] for j in joints]
    parent = {c: i for i, n in enumerate(g['nodes']) for c in n.get('children', [])}
    jpos = np.array([W[j][:3, 3] for j in joints])
    jpar = [joints.index(parent[j]) if parent.get(j) in joints else -1 for j in joints]
    return dict(g=g, binb=binb, parts=parts, joint_names=names, joint_pos=jpos, joint_parent=jpar,
                joint_mat=np.array([W[j] for j in joints]))


def joint_group(n):
    """Группа кости для раскраски схемы скелета."""
    if n in ('chestUpper', 'neckLower', 'neckUpper', 'head'):
        return 'spine'
    if n.startswith('Eye.'):
        return 'eyes'
    if n.startswith('tongue'):
        return 'tongue'
    if 'Teeth' in n:
        return 'teeth'
    if n in ('maxi', 'lowerFaceRig', 'Chin') or n.startswith('LipLower') or n.startswith('NasolabialLower'):
        return 'jaw'
    if n.startswith('Brow') or n == 'CenterBrow':
        return 'brows'
    if n.startswith('Eyelid') or n.startswith('Squint'):
        return 'eyelids'
    if n.startswith('Lip') or n.startswith('Nasolabial'):
        return 'mouth'
    if n.startswith('Nose') or n.startswith('Nostril'):
        return 'nose'
    if n.startswith('Cheek') or n.startswith('JawClench'):
        return 'cheeks'
    return 'other'


GROUP_COLORS = {
    'spine': (220, 30, 30), 'eyes': (20, 170, 60), 'tongue': (200, 40, 200), 'teeth': (230, 180, 0),
    'jaw': (255, 120, 0), 'brows': (30, 90, 230), 'eyelids': (0, 190, 210), 'mouth': (240, 40, 110),
    'nose': (120, 70, 20), 'cheeks': (130, 60, 220), 'other': (90, 90, 90),
}
