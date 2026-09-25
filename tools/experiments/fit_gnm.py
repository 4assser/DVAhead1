"""Fit Google GNM Head (v3) identity to the D.Va head (different topology).

Pipeline: landmarks -> similarity (Umeyama) -> alternating ICP + regularized linear
least squares for GNM head identity coefficients. (Точная копия потребовала бы ещё нерегулярного residual-слоя поверх — здесь не делается.)
"""
import json
import numpy as np, trimesh
import common
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from dva import load_dva
from gnm import GNM
from visibility import dva_skin_visibility

MM = 1000.0


def umeyama(src, dst, w=None, with_scale=True):
    """Find s,R,t minimizing sum w||s R src + t - dst||^2."""
    w = np.ones(len(src)) if w is None else w
    w = w / w.sum()
    mu_s = (w[:, None] * src).sum(0); mu_d = (w[:, None] * dst).sum(0)
    xs = src - mu_s; xd = dst - mu_d
    C = (w[:, None] * xd).T @ xs
    U, S, Vt = np.linalg.svd(C)
    Dm = np.eye(3); Dm[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ Dm @ Vt
    var_s = (w * (xs ** 2).sum(1)).sum()
    s = (S * np.diag(Dm)).sum() / var_s if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def dva_target(D):
    """Exterior D.Va skin surface (main component, visible), plus landmarks."""
    sk = D['parts']['skin']
    m = trimesh.Trimesh(sk['V'], sk['F'], process=False)
    m.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=6)
    ncomp, lab = connected_components(m.edges_sparse, directed=False)
    main = np.argmax(np.bincount(lab))
    vis_orig = dva_skin_visibility(D)
    # map original -> welded
    tree = cKDTree(m.vertices)
    _, o2w = tree.query(sk['V'])
    vis_w = np.zeros(len(m.vertices), bool)
    vis_w[o2w[vis_orig]] = True
    keep_v = (lab == main) & vis_w
    keep_f = keep_v[m.faces].all(1)
    surf = trimesh.Trimesh(m.vertices, m.faces[keep_f], process=False)
    surf.remove_unreferenced_vertices()
    # boundary vertices of the *original* main component (neck/back cut) -> reject correspondences there
    E = np.sort(m.faces[(lab[m.faces[:, 0]] == main)][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), 1)
    u, c = np.unique(E, axis=0, return_counts=True)
    bverts = np.unique(u[c == 1].ravel())
    names = D['joint_names']; jp = D['joint_pos']
    J = lambda n: jp[names.index(n)]
    # eyelid rim loops -> eye corners
    rim = m.vertices[bverts]
    lm = {}
    for side, sgn in (('L', 1), ('R', -1)):
        e = J(f'Eye.{side}')
        sel = rim[(np.linalg.norm(rim - e, axis=1) < 0.03) & (np.sign(rim[:, 0]) == sgn)]
        sel = sel[sel[:, 2] > e[2]]  # front rim only (exclude lashes etc.)
        inner = sel[np.argmin(np.abs(sel[:, 0]))]; outer = sel[np.argmax(np.abs(sel[:, 0]))]
        lm[f'eye_inner_{side}'] = inner; lm[f'eye_outer_{side}'] = outer
    # exterior-vertex snapping helper
    ext_tree = cKDTree(surf.vertices)
    snap = lambda p: surf.vertices[ext_tree.query(p)[1]]
    lm['mouth_corner_L'] = snap(J('LipCorner.L')); lm['mouth_corner_R'] = snap(J('LipCorner.R'))
    mid = surf.vertices[np.abs(surf.vertices[:, 0]) < 0.004]
    nose_c = mid[(np.abs(mid[:, 1] - J('Nose')[1]) < 0.015)]
    lm['nose_tip'] = nose_c[np.argmax(nose_c[:, 2])]
    st = (J('LipUpperMiddle') + J('LipLowerMiddle')) / 2
    cand = mid[np.abs(mid[:, 1] - st[1]) < 0.006]
    lm['stomion'] = cand[np.argmax(cand[:, 2])]
    return surf, m.vertices[bverts], lm


# GNM iBUG-68 indices corresponding to the D.Va landmarks above.
# (iBUG: 36/39 = subject-right eye outer/inner, 42/45 = subject-left inner/outer,
#  48/54 = mouth corners R/L, 30 = nose tip, 62/66 = inner lip midpoints -> stomion)
LM_MAP = {'eye_outer_R': [36], 'eye_inner_R': [39], 'eye_inner_L': [42], 'eye_outer_L': [45],
          'mouth_corner_R': [48], 'mouth_corner_L': [54], 'nose_tip': [30], 'stomion': [62, 66]}


def gnm_landmark_rows(G, key):
    """Return (vertex indices, weights) for a (possibly averaged) GNM landmark."""
    ids = LM_MAP[key]
    idx = np.concatenate([G.lm_idx[i] for i in ids]); w = np.concatenate([G.lm_w[i] for i in ids]) / len(ids)
    return idx, w


def fit(sigma_mm=1.0, n_iter=14, n_id=170, lm_weight=30.0, verbose=True, D=None, G=None):
    D = D or load_dva(); G = G or GNM()
    surf, bpts, lm = dva_target(D)
    btree = cKDTree(bpts)
    keys = list(LM_MAP)
    src = np.array([lm[k] for k in keys])
    dst = np.array([(G.T[gnm_landmark_rows(G, k)[0]] * gnm_landmark_rows(G, k)[1][:, None]).sum(0) for k in keys])
    s, R, t = umeyama(src, dst)
    if verbose:
        print(f'init similarity: scale={s:.3f}  (D.Va is {1/s:.2f}x of GNM mean size)')
    skin_ext = np.where(G.vg['skin_exterior'] > 0.5)[0]
    hockey = G.vg['hockey_mask'][skin_ext]
    gnm_skin_f = G.tri[np.isin(G.tri, skin_ext).all(1)]
    Ib = G.I[:n_id]                                   # (n_id, V, 3)
    beta = np.zeros(n_id)
    inv_s2 = 1.0 / (sigma_mm / MM) ** 2
    hist = []
    for it in range(n_iter):
        Vg = G.T + np.tensordot(beta, Ib, 1)
        Xd = (s * (R @ surf.vertices.T)).T + t          # D.Va in GNM space
        dm = trimesh.Trimesh(Xd, surf.faces, process=False)
        # --- term 1: GNM exterior skin -> closest point on D.Va
        q = Vg[skin_ext]
        cp, dist, tid = trimesh.proximity.closest_point(dm, q)
        gm = trimesh.Trimesh(Vg, gnm_skin_f, process=False)
        gn = gm.vertex_normals[skin_ext]
        dn = dm.face_normals[tid]
        near_b = btree.query(((cp - t) @ R) / s)[0] < 0.008   # near the cut in D.Va space
        thr = max(0.02 * (0.7 ** it), 0.006)
        ok1 = (dist < thr) & (np.abs((gn * dn).sum(1)) > 0.5) & ~near_b
        w1 = np.where(hockey > 0.5, 1.0, 0.4)[ok1]
        # --- term 2: D.Va vertices -> closest point on GNM skin
        cp2, dist2, tid2 = trimesh.proximity.closest_point(gm, Xd)
        ok2 = dist2 < thr
        tri2 = gnm_skin_f[tid2[ok2]]
        bc2 = trimesh.triangles.points_to_barycentric(Vg[tri2], cp2[ok2])
        # --- assemble linear system in beta
        rows_A = []; rows_b = []; rows_w = []
        vi = skin_ext[ok1]
        rows_A.append(Ib[:, vi, :].transpose(1, 2, 0).reshape(-1, n_id))
        rows_b.append((cp[ok1] - G.T[vi]).reshape(-1)); rows_w.append(np.repeat(w1, 3))
        A2 = np.einsum('nk,inkd->ndi', bc2, Ib[:, tri2, :]).reshape(-1, n_id)
        b2 = (Xd[ok2] - np.einsum('nk,nkd->nd', bc2, G.T[tri2])).reshape(-1)
        rows_A.append(A2); rows_b.append(b2); rows_w.append(np.full(len(b2), 1.0))
        for k in keys:
            idx, w = gnm_landmark_rows(G, k)
            Al = np.einsum('k,ikd->di', w, Ib[:, idx, :])
            bl = ((s * R @ lm[k]) + t) - (G.T[idx] * w[:, None]).sum(0)
            rows_A.append(Al); rows_b.append(bl); rows_w.append(np.full(3, lm_weight))
        A = np.vstack(rows_A); b = np.concatenate(rows_b); wv = np.concatenate(rows_w)
        # normalize so that sigma means "per-correspondence noise" for ~1 effective sample per vertex area
        norm = inv_s2 * (len(skin_ext) * 3) / wv.sum() * 0.02
        H = (A * (wv * norm)[:, None]).T @ A + np.eye(n_id)
        beta = np.linalg.solve(H, (A * (wv * norm)[:, None]).T @ b)
        # --- similarity update from both correspondence sets
        Vg = G.T + np.tensordot(beta, Ib, 1)
        src_pts = np.vstack([((cp[ok1] - t) @ R) / s, surf.vertices[ok2]])
        dst_pts = np.vstack([Vg[vi], np.einsum('nk,nkd->nd', bc2, Vg[tri2])])
        s, R, t = umeyama(src_pts, dst_pts)
        rms = np.sqrt(np.mean(dist2[ok2] ** 2)) * MM
        hist.append(dict(it=it, scale=s, rms_mm=rms, n1=int(ok1.sum()), n2=int(ok2.sum()),
                         beta_norm=float(np.linalg.norm(beta)), beta_max=float(np.abs(beta).max())))
        if verbose:
            print(f'it {it:2d}: s={s:.3f} rms(D.Va->GNM)={rms:.2f}mm  |beta|={np.linalg.norm(beta):.1f} max|b_i|={np.abs(beta).max():.2f}  corr={ok1.sum()}/{ok2.sum()}')
    return dict(beta=beta, s=s, R=R, t=t, surf=surf, lm=lm, hist=hist, G=G, D=D)


def evaluate(res):
    G, D, surf = res['G'], res['D'], res['surf']
    s, R, t = res['s'], res['R'], res['t']
    beta = res['beta']
    Vg = G.T + np.tensordot(beta, G.I[:len(beta)], 1)
    skin_ext = np.where(G.vg['skin_exterior'] > 0.5)[0]
    fsk = G.tri[np.isin(G.tri, skin_ext).all(1)]
    gm = trimesh.Trimesh(Vg, fsk, process=False)
    Xd = (s * (R @ surf.vertices.T)).T + t
    _, dist, _ = trimesh.proximity.closest_point(gm, Xd)
    names = D['joint_names']; jp = D['joint_pos']; J = lambda n: jp[names.index(n)]
    P0 = surf.vertices
    regions = {
        'eyes': (np.linalg.norm(P0 - J('Eye.L'), axis=1) < 0.022) | (np.linalg.norm(P0 - J('Eye.R'), axis=1) < 0.022),
        'mouth': np.linalg.norm(P0 - (J('LipUpperMiddle') + J('LipLowerMiddle')) / 2 - np.array([0, 0, -0.005]), axis=1) < 0.022,
        'nose': np.linalg.norm(P0 - J('Nose') - np.array([0, 0.004, -0.008]), axis=1) < 0.016,
    }
    face = P0[:, 2] > 0.0
    regions['face_all'] = face
    regions['other_face'] = face & ~(regions['eyes'] | regions['mouth'] | regions['nose'])
    out = {}
    for k, msk in regions.items():
        d = dist[msk] * MM
        out[k] = dict(n=int(msk.sum()), mean=float(d.mean()), median=float(np.median(d)), p95=float(np.percentile(d, 95)), max=float(d.max()))
    return out, dist * MM, Xd, Vg


if __name__ == '__main__':
    D = load_dva(); G = GNM()
    results = {}
    for sigma in (3.0, 1.0, 0.3):
        print(f'\n=== sigma = {sigma} mm ===')
        r = fit(sigma_mm=sigma, D=D, G=G)
        ev, dist, Xd, Vg = evaluate(r)
        for k, v in ev.items():
            print(f'  {k:11s} n={v["n"]:4d} mean={v["mean"]:.2f} median={v["median"]:.2f} p95={v["p95"]:.2f} max={v["max"]:.2f} mm')
        b = r['beta']
        print(f'  beta: |beta|={np.linalg.norm(b):.1f} (random human ~{np.sqrt(len(b)):.1f}), max|b|={np.abs(b).max():.2f}, #|b|>3: {(np.abs(b)>3).sum()}')
        print('  top components:', [(G.id_names[i], round(float(b[i]), 2)) for i in np.argsort(-np.abs(b))[:8]])
        results[sigma] = dict(eval=ev, beta=b.tolist(), s=r['s'], R=r['R'].tolist(), t=r['t'].tolist(), lm={k: v.tolist() for k, v in r['lm'].items()})
        np.savez(common.OUT / f'fit_sigma{sigma}.npz', beta=b, s=r['s'], R=r['R'], t=r['t'], dist=dist, Xd=Xd, surfV=r['surf'].vertices, surfF=r['surf'].faces)
    json.dump(results, open(common.OUT / 'fit_results.json', 'w'), indent=1)
