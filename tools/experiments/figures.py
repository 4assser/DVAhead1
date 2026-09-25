"""Картинки по самой модели (без GNM): виды, скелет+сетка, позы костей, текстура лица.

Результат: docs/img/01_views.jpg, 02_skeleton.jpg, 03_rig_poses.jpg, 04_face_texture.jpg
"""
import numpy as np
from PIL import Image, ImageDraw

from common import IMG, font
from dva import load_dva, joint_group, GROUP_COLORS
from glb import world_mats
from pose import posed_world, skin_part, axis_angle_mat
from render import Canvas, project_ortho, rot_y, vertex_normals

D = load_dva()
P = D['parts']
g, binb = D['g'], D['binb']
allV = np.vstack([P[k]['V'] for k in P])
CENTER = (allV.min(0) + allV.max(0)) / 2


def draw_parts(cv, parts, R, center, scale, W, H):
    for k, p in parts.items():
        Pp = project_ortho(p['V'], R, center, scale, W, H)
        Nr = vertex_normals(p['V'], p['F']) @ R.T
        if k == 'eyes':
            cv.draw(Pp, p['F'], N=Nr, base=(0.93, 0.93, 0.95))
        else:
            cv.draw(Pp, p['F'], N=Nr, uv=p['UV'], tex=p['tex'])


# ------------------------------------------------------------------ 01 / 02
def render_view(R, wire=False, skel=False, fade=False, W=700, H=820, scale=3300.0):
    cv = Canvas(W, H)
    draw_parts(cv, P, R, CENTER, scale, W, H)
    img = cv.image()
    if fade:
        img = Image.blend(img, Image.new('RGB', img.size, 'white'), 0.55)
    dr = ImageDraw.Draw(img)
    if wire:
        p = P['skin']
        Pp = project_ortho(p['V'], R, CENTER, scale, W, H)
        E = np.unique(np.sort(np.vstack([p['F'][:, [0, 1]], p['F'][:, [1, 2]], p['F'][:, [2, 0]]]), 1), axis=0)
        for a, b in E:
            mid = (Pp[a] + Pp[b]) / 2
            xi, yi = int(mid[0]), int(mid[1])
            if 0 <= xi < W and 0 <= yi < H and mid[2] >= cv.depth[yi, xi] - 0.0015:
                dr.line([tuple(Pp[a][:2]), tuple(Pp[b][:2])], fill=(40, 40, 40), width=1)
    if skel:
        J = project_ortho(D['joint_pos'], R, CENTER, scale, W, H)
        for i, (nm, pa) in enumerate(zip(D['joint_names'], D['joint_parent'])):
            if pa >= 0:
                dr.line([tuple(J[pa][:2]), tuple(J[i][:2])], fill=GROUP_COLORS[joint_group(nm)], width=2)
        order = sorted(range(len(D['joint_names'])), key=lambda i: joint_group(D['joint_names'][i]) == 'eyes')
        for i in order:
            grp = joint_group(D['joint_names'][i])
            x, y = J[i][:2]
            r = 7 if grp == 'eyes' else 4
            dr.ellipse([x - r, y - r, x + r, y + r], fill=GROUP_COLORS[grp], outline=(0, 0, 0))
    return img


def hstack(ims):
    out = Image.new('RGB', (sum(i.width for i in ims), max(i.height for i in ims)), 'white')
    x = 0
    for im in ims:
        out.paste(im, (x, 0))
        x += im.width
    return out


hstack([render_view(R) for R in (np.eye(3), rot_y(90), rot_y(35))]).save(IMG / '01_views.jpg', quality=90)

sk = hstack([render_view(np.eye(3), skel=True, fade=True), render_view(rot_y(90), skel=True, fade=True),
             render_view(rot_y(25), wire=True)])
labels = {'spine': 'chestUpper→neck→head (4)', 'eyes': 'Eye.L / Eye.R (2)', 'jaw': 'maxi (челюсть) + низ лица (10)',
          'tongue': 'язык (6)', 'teeth': 'зубы (2)', 'brows': 'брови (9)', 'eyelids': 'веки + прищур (18)',
          'mouth': 'губы/носогубные (13)', 'nose': 'нос (3)', 'cheeks': 'щёки + JawClench (12)'}
f17 = font(17)
out = Image.new('RGB', (sk.width, sk.height + 190), 'white')
out.paste(sk, (0, 0))
dr = ImageDraw.Draw(out)
for i, (k, t) in enumerate(labels.items()):
    x = 40 + (i % 2) * 520
    y = sk.height + 8 + (i // 2) * 32
    dr.ellipse([x, y + 4, x + 16, y + 20], fill=GROUP_COLORS[k], outline=(0, 0, 0))
    dr.text((x + 26, y + 1), t, fill=(0, 0, 0), font=f17)
dr.text((1100, sk.height + 12), 'upperFaceRig, lowerFaceRig — группирующие узлы;\nлинии = связь родитель→ребёнок (лицевые кости\nкрепятся к центру головы, поэтому «звезда»).\nСправа — сетка кожи (~3.8k треугольников).',
        fill=(60, 60, 60), font=f17)
out.save(IMG / '02_skeleton.jpg', quality=90)

# ------------------------------------------------------------------ 03 rig poses
names = [n.get('name') for n in g['nodes']]
W0 = world_mats(g)
parent = {c: i for i, n in enumerate(g['nodes']) for c in n.get('children', [])}


def to_parent_local(node, dw):
    return W0[parent[names.index(node)]][:3, :3].T @ np.asarray(dw)


def both(prefixes, M):
    return {f'{b}.{s}': M for b in prefixes for s in 'LR'}


lids_up = ['EyelidUpper', 'EyelidUpperInner', 'EyelidUpperOuter']
smile = {}
for s_, sg in (('L', 1), ('R', -1)):
    smile[f'LipCorner.{s_}'] = to_parent_local(f'LipCorner.{s_}', [sg * 0.0018, 0.0030, -0.0015])
    smile[f'LipUpperOuter.{s_}'] = to_parent_local(f'LipUpperOuter.{s_}', [sg * 0.0012, 0.0022, -0.0010])
    smile[f'LipLowerOuter.{s_}'] = to_parent_local(f'LipLowerOuter.{s_}', [sg * 0.0010, 0.0016, -0.0008])
    smile[f'CheekUpper.{s_}'] = to_parent_local(f'CheekUpper.{s_}', [0, 0.0015, 0.0008])
    smile[f'NasolabialMouthCorner.{s_}'] = to_parent_local(f'NasolabialMouthCorner.{s_}', [sg * 0.0008, 0.0015, 0.0])
poses = [
    ('rest pose', {}, {}),
    ('jawOpen: maxi −14°', {'maxi': axis_angle_mat([1, 0, 0], -14)}, {}),
    ('eyeBlink: 6 костей век 28°', both(lids_up, axis_angle_mat([1, 0, 0], 28)), {}),
    ('eyeWide: веки −12°', both(lids_up, axis_angle_mat([1, 0, 0], -12)), {}),
    ('mouthSmile: кости уголков губ', {}, smile),
]
Wd, Hd, sc, ctr = 430, 520, 4300.0, np.array([0, 0.087, 0.03])
f18 = font(18)
ims = []
for lab, lr, lt in poses:
    Wp = posed_world(g, lr, lt)
    posed = {k: dict(P[k], V=skin_part(g, binb, P[k], Wp)) for k in ('skin', 'teeth', 'eyes')}
    cv = Canvas(Wd, Hd)
    draw_parts(cv, posed, rot_y(18), ctr, sc, Wd, Hd)
    im = cv.image()
    ImageDraw.Draw(im).text((10, 8), lab, fill=(0, 0, 0), font=f18)
    ims.append(im)
hstack(ims).save(IMG / '03_rig_poses.jpg', quality=90)

# ------------------------------------------------------------------ 04 face texture
tex = (P['skin']['tex'] * 255).astype(np.uint8)
S = tex.shape[0]
u0, u1, v0, v1 = 0.28, 0.50, 0.50, 0.78        # область лица в атласе
crop = Image.fromarray(tex[int(v0 * S):int(v1 * S), int(u0 * S):int(u1 * S)])
k = 4
crop = crop.resize((crop.width * k, crop.height * k), Image.NEAREST)
dr = ImageDraw.Draw(crop)
UV = P['skin']['UV']
for f in P['skin']['F']:
    pts = [((UV[i, 0] - u0) * S * k, (UV[i, 1] - v0) * S * k) for i in f]
    dr.polygon(pts, outline=(0, 220, 255))
lab = Image.new('RGB', (crop.width, crop.height + 60), 'white')
lab.paste(crop, (0, 60))
ImageDraw.Draw(lab).text((8, 6), f'Фрагмент {crop.width // k}×{crop.height // k} px атласа {S}²\n(увеличено ×{k}; голубым — UV-сетка)', fill=(0, 0, 0), font=font(17))
lab.save(IMG / '04_face_texture.jpg', quality=90)
print('figures ->', IMG)
