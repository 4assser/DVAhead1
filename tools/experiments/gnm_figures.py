"""Оценка подгонки GNM и перенос выражений GNM -> меш D.Va.

Нужно сначала запустить fit_gnm.py (он пишет out/fit_sigma*.npz).
Результат: docs/img/05_gnm_fit.jpg, docs/img/06_gnm_expr_transfer.jpg,
           out/fit_report.json, out/transfer_report.json, out/morph_*.npz

    python gnm_figures.py            # перенос с разделением верхней/нижней губы (по умолчанию)
    python gnm_figures.py --naive    # наивный closest-point (рвёт губы — для сравнения)
"""
import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

from common import IMG, OUT, font
from dva import load_dva
from fit_gnm import dva_target, LM_MAP, gnm_landmark_rows, MM
from gnm import GNM, expression_decoder, EXPRESSIONS
from pose import posed_world, skin_part, axis_angle_mat
from render import Canvas, project_ortho, rot_y, vertex_normals

NAIVE = '--naive' in sys.argv
D = load_dva()
G = GNM()
P = D['parts']
surf, bpts, lm = dva_target(D)
skin_all = np.where(G.vg['skin'] > 0.5)[0]
fsk = G.tri[np.isin(G.tri, skin_all).all(1)]
f_eyes = G.tri[np.isin(G.tri, np.where(G.vg['eye_exteriors'] > 0.5)[0]).all(1)]
f_teeth = G.tri[np.isin(G.tri, np.where((G.vg['teeth'] > 0.5) | (G.vg['gums'] > 0.5))[0]).all(1)]
hock = G.vg['hockey_mask']


def colormap(d, vmax=5.0):
    x = np.clip(d / vmax, 0, 1)
    c = np.stack([np.clip(1.5 * x - 0.2, 0, 1), np.clip(1.6 - np.abs(2.2 * x - 1.1) * 1.3, 0, 1), np.clip(1.0 - 2.0 * x, 0, 1)], 1)
    return 0.15 + 0.85 * np.clip(c, 0, 1)


def label(im, text, sub=None):
    dr = ImageDraw.Draw(im)
    dr.text((12, 10), text, fill=(0, 0, 0), font=font(20))
    if sub:
        dr.text((12, 36), sub, fill=(60, 60, 60), font=font(15))
    return im


# ============================================================== 1. метрики подгонки
fits = {}
report = {}
for sg in (3.0, 1.0, 0.3):
    f = np.load(OUT / f'fit_sigma{sg}.npz')
    beta, s, R, t = f['beta'], float(f['s']), f['R'], f['t']
    Vg = G.T + np.tensordot(beta, G.I[:len(beta)], 1)
    Xd = (s * (R @ surf.vertices.T)).T + t
    cp, dist, tid = trimesh.proximity.closest_point(trimesh.Trimesh(Vg, fsk, process=False), Xd)
    face = hock[fsk[tid]].mean(1) > 0.5            # "лицо" = ближайший треугольник GNM в hockey_mask
    d = dist * MM
    L = lambda k: (Vg[gnm_landmark_rows(G, k)[0]] * gnm_landmark_rows(G, k)[1][:, None]).sum(0)
    Ld = lambda k: s * R @ lm[k] + t
    report[sg] = dict(
        scale=s, beta_norm=float(np.linalg.norm(beta)), beta_max=float(np.abs(beta).max()), n_beta_gt3=int((np.abs(beta) > 3).sum()),
        face_n=int(face.sum()), face_mean=float(d[face].mean()), face_median=float(np.median(d[face])),
        face_p95=float(np.percentile(d[face], 95)), face_max=float(d[face].max()),
        lm_res_mm={k: float(np.linalg.norm(L(k) - Ld(k)) * MM) for k in LM_MAP},
        eye_width_gnm=float(np.linalg.norm(L('eye_outer_L') - L('eye_inner_L')) * MM),
        eye_width_dva=float(np.linalg.norm(Ld('eye_outer_L') - Ld('eye_inner_L')) * MM),
        mouth_width_gnm=float(np.linalg.norm(L('mouth_corner_L') - L('mouth_corner_R')) * MM),
        mouth_width_dva=float(np.linalg.norm(Ld('mouth_corner_L') - Ld('mouth_corner_R')) * MM))
    fits[sg] = dict(beta=beta, s=s, R=R, t=t, Vg=Vg, d=d)
json.dump(report, open(OUT / 'fit_report.json', 'w'), indent=1)
for sg, r in report.items():
    print(f"sigma={sg}: |beta|={r['beta_norm']:.1f} max={r['beta_max']:.1f} n>3={r['n_beta_gt3']}  "
          f"face mean/median/p95 = {r['face_mean']:.2f}/{r['face_median']:.2f}/{r['face_p95']:.2f} mm  "
          f"eye {r['eye_width_gnm']:.1f}/{r['eye_width_dva']:.1f}  mouth {r['mouth_width_gnm']:.1f}/{r['mouth_width_dva']:.1f}  "
          f"lm {', '.join(f'{k}={v:.1f}' for k, v in r['lm_res_mm'].items())}")

# ============================================================== 2. лист с подгонкой
Wd, Hd, scale_px = 520, 640, 2600.0
center = np.array([0.0, 0.265, 0.08])
ref = fits[1.0]


def render_gnm(V, R, color=(0.82, 0.74, 0.68)):
    cv = Canvas(Wd, Hd)
    for F_, col in ((fsk, color), (f_eyes, (0.95, 0.95, 0.97))):
        cv.draw(project_ortho(V, R, center, scale_px, Wd, Hd), F_, N=vertex_normals(V, F_) @ R.T, base=col)
    return cv.image()


def render_dva_in_gnm(R, d=None, vmax=5.0):
    cv = Canvas(Wd, Hd)
    tree = cKDTree(surf.vertices)
    for k in ('skin', 'eyes', 'teeth'):
        p = P[k]
        X = (ref['s'] * (ref['R'] @ p['V'].T)).T + ref['t']
        N = vertex_normals(X, p['F']) @ R.T
        Pp = project_ortho(X, R, center, scale_px, Wd, Hd)
        if d is not None and k == 'skin':
            dd, idx = tree.query(p['V'])
            col = colormap(d[idx], vmax)
            col[dd > 1e-5] = 0.7
            cv.draw(Pp, p['F'], N=N, vcolor=col)
        elif k == 'eyes':
            cv.draw(Pp, p['F'], N=N, base=(0.95, 0.95, 0.97))
        else:
            cv.draw(Pp, p['F'], N=N, uv=p['UV'], tex=p['tex'])
    return cv.image()


rows = []
for R in (np.eye(3), rot_y(60)):
    ims = [label(render_dva_in_gnm(R), 'D.Va (оригинал)', 'выровнена в пространство GNM'),
           label(render_gnm(G.T, R), 'GNM: средняя голова', 'identity = 0'),
           label(render_gnm(fits[3.0]['Vg'], R), 'GNM: правдоподобный фит', f"σ=3 мм, |β|={report[3.0]['beta_norm']:.0f}, max {report[3.0]['beta_max']:.1f} sd"),
           label(render_gnm(fits[0.3]['Vg'], R), 'GNM: форсированный фит', f"σ=0.3 мм, |β|={report[0.3]['beta_norm']:.0f}, max {report[0.3]['beta_max']:.1f} sd"),
           label(render_dva_in_gnm(R, d=fits[1.0]['d']), 'ошибка D.Va ↔ фит σ=1', '0 мм (синий) … 5+ мм (красный)')]
    row = Image.new('RGB', (Wd * len(ims), Hd), 'white')
    for i, im in enumerate(ims):
        row.paste(im, (i * Wd, 0))
    rows.append(row)
sheet = Image.new('RGB', (rows[0].width, Hd * len(rows)), 'white')
for i, r in enumerate(rows):
    sheet.paste(r, (0, i * Hd))
sheet.save(IMG / '05_gnm_fit.jpg', quality=90)

# ============================================================== 3. перенос выражений GNM -> D.Va
beta, s, R, t = ref['beta'], ref['s'], ref['R'], ref['t']
Vg = ref['Vg']
lowt = np.where(G.vg['lower_teeth_and_gums'] > 0.5)[0]
gm = trimesh.Trimesh(Vg, fsk, process=False)
toG = lambda V: (s * (R @ V.T)).T + t
sk = P['skin']
m = trimesh.Trimesh(sk['V'], sk['F'], process=False)
nc, lab = connected_components(m.edges_sparse, directed=False)
main = np.argmax(np.bincount(lab))
is_tongue = np.isin(lab, [c for c in range(nc) if c != main and sk['V'][lab == c].mean(0)[1] < 0.08])
Xs = toG(sk['V'])
cp, dist, tid = trimesh.proximity.closest_point(gm, Xs)
names = D['joint_names']
if not NAIVE:
    # Разделяем область рта на верх/низ на обоих мешах, чтобы соответствия не перескакивали через щель губ.
    low_j = [i for i, n in enumerate(names) if n in ('maxi', 'lowerFaceRig', 'Chin', 'lowerTeeth')
             or n.startswith(('LipLower', 'NasolabialLower', 'tongue'))]
    w_low = sum(np.where(np.isin(sk['J'][:, k], low_j), sk['W'][:, k], 0) for k in range(4))
    st = s * R @ ((D['joint_pos'][names.index('LipUpperMiddle')] + D['joint_pos'][names.index('LipLowerMiddle')]) / 2) + t
    near_mouth_d = np.linalg.norm(Xs - st, axis=1) < 0.03
    du = cKDTree(Vg[G.vg['upper_lip'] > 0.5]).query(Vg)[0]
    dl = cKDTree(Vg[G.vg['lower_lip'] > 0.5]).query(Vg)[0]
    tri_low = (dl < du)[fsk].mean(1) > 0.5
    tri_mouth = (np.linalg.norm(Vg - st, axis=1) < 0.035)[fsk].all(1)
    for want_low in (True, False):
        selv = near_mouth_d & ((w_low > 0.5) == want_low)
        allowed = ~tri_mouth | (tri_low == want_low)
        c2, d2, t2 = trimesh.proximity.closest_point(trimesh.Trimesh(Vg, fsk[allowed], process=False), Xs[selv])
        cp[selv], dist[selv], tid[selv] = c2, d2, np.where(allowed)[0][t2]
tri = fsk[tid]
bc = trimesh.triangles.points_to_barycentric(Vg[tri], cp)
print(f'D.Va skin -> GNM: median {np.median(dist) * MM:.2f}, p95 {np.percentile(dist, 95) * MM:.2f}, max {dist.max() * MM:.2f} mm')


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, S_, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    Rk = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return Rk, cb - Rk @ ca


def dva_morph(phi):
    """Дельты выражения GNM -> дельты вершин D.Va (в единицах D.Va). Глаза не двигаются."""
    dE = np.tensordot(phi, G.E, 1)
    d_skin = np.einsum('nk,nkd->nd', bc, dE[tri])
    Rk, tk = kabsch(Vg[lowt], Vg[lowt] + dE[lowt])          # «челюсть» = жёсткое движение нижних зубов GNM
    jaw = lambda X: (Rk @ X.T).T + tk - X
    d_skin[is_tongue] = jaw(Xs[is_tongue])
    Xt = toG(P['teeth']['V'])
    lower = P['teeth']['J'][:, 0] == names.index('lowerTeeth')
    dt = np.zeros_like(Xt)
    dt[lower] = jaw(Xt[lower])
    out = {'skin': d_skin, 'teeth': dt, 'eyes': np.zeros_like(P['eyes']['V'])}
    return {k: (v @ R) / s for k, v in out.items()}, dE


Wd, Hd = 460, 520


def render_dva(morph=None, Rv=rot_y(15), scale_px=4300.0, center=np.array([0, 0.09, 0.03])):
    cv = Canvas(Wd, Hd)
    for k in ('skin', 'teeth', 'eyes'):
        p = P[k]
        V = p['V'] + (morph[k] if morph else 0)
        N = vertex_normals(V, p['F']) @ Rv.T
        Pp = project_ortho(V, Rv, center, scale_px, Wd, Hd)
        if k == 'eyes':
            cv.draw(Pp, p['F'], N=N, base=(0.95, 0.95, 0.97))
        else:
            cv.draw(Pp, p['F'], N=N, uv=p['UV'], tex=p['tex'])
    return cv.image()


def render_gnm_expr(V, Rv=np.eye(3), scale_px=2500.0, center=np.array([0, 0.27, 0.08])):
    cv = Canvas(Wd, Hd)
    for F_, col in ((fsk, (0.82, 0.74, 0.68)), (f_eyes, (0.95, 0.95, 0.97)), (f_teeth, (0.95, 0.93, 0.85))):
        cv.draw(project_ortho(V, Rv, center, scale_px, Wd, Hd), F_, N=vertex_normals(V, F_) @ Rv.T, base=col)
    return cv.image()


def lab_(im, txt):
    ImageDraw.Draw(im).text((10, 8), txt, fill=(0, 0, 0), font=font(19))
    return im


sel = ['HAPPY', 'SURPRISE', 'WINK_LEFT', 'PUCKER', 'SNARL']
top = [lab_(render_gnm_expr(G.T), 'GNM mean: neutral')]
bot = [lab_(render_dva(), 'D.Va: neutral')]
morphs = {}
for nm in sel:
    mo, dE = dva_morph(expression_decoder(EXPRESSIONS.index(nm)))
    morphs[nm] = mo
    top.append(lab_(render_gnm_expr(G.T + dE), f'GNM: {nm.lower()}'))
    bot.append(lab_(render_dva(mo), f'D.Va <- GNM {nm.lower()}'))
    np.savez(OUT / f"morph_{nm}{'_naive' if NAIVE else ''}.npz", **mo)
sheet = Image.new('RGB', (Wd * len(top), Hd * 2), 'white')
for i, (a, b) in enumerate(zip(top, bot)):
    sheet.paste(a, (i * Wd, 0))
    sheet.paste(b, (i * Wd, Hd))
sheet.save(IMG / ('06_gnm_expr_transfer_naive.jpg' if NAIVE else '06_gnm_expr_transfer.jpg'), quality=90)

# ============================================================== 4. насколько закрывается глаз
V = sk['V']
key = np.round(V * 1e6).astype(np.int64)
_, inv = np.unique(key, axis=0, return_inverse=True)
inv = inv.ravel()
E = np.sort(inv[np.vstack([sk['F'][:, [0, 1]], sk['F'][:, [1, 2]], sk['F'][:, [2, 0]]])], 1)
u, c = np.unique(E, axis=0, return_counts=True)
bmask = np.isin(inv, np.unique(u[c == 1].ravel()))
eL = D['joint_pos'][names.index('Eye.L')]
rim = np.where(bmask & (np.linalg.norm(V - eL, axis=1) < 0.03) & (V[:, 0] > 0) & (V[:, 2] > eL[2]) & (np.abs(V[:, 1] - eL[1]) < 0.012))[0]
xc = V[rim, 0].mean()


def aperture(Vx):
    near = np.abs(Vx[rim, 0] - xc) < 0.006
    upper = near & (V[rim, 1] > eL[1] - 0.0005)
    lower = near & (V[rim, 1] <= eL[1] - 0.0005)
    return (Vx[rim][upper, 1].mean() - Vx[rim][lower, 1].mean()) * 1000


g, binb = D['g'], D['binb']
a0 = aperture(V)
a_gnm = aperture(V + morphs['WINK_LEFT']['skin'])
lids = {n: axis_angle_mat([1, 0, 0], 28) for n in ('EyelidUpper.L', 'EyelidUpperInner.L', 'EyelidUpperOuter.L')}
a_bone = aperture(skin_part(g, binb, sk, posed_world(g, lids)))
tr = dict(closest_median_mm=float(np.median(dist) * MM), closest_p95_mm=float(np.percentile(dist, 95) * MM),
          eye_aperture_rest_mm=a0, eye_aperture_gnm_wink_mm=a_gnm, eye_aperture_bone_blink28_mm=a_bone)
json.dump(tr, open(OUT / ('transfer_report_naive.json' if NAIVE else 'transfer_report.json'), 'w'), indent=1)
print(f'left eye aperture (D.Va mm): rest {a0:.2f} | GNM wink {a_gnm:.2f} ({(1 - a_gnm / a0) * 100:.0f}% closed) | '
      f'bone blink 28° {a_bone:.2f} ({(1 - a_bone / a0) * 100:.0f}% closed)')
