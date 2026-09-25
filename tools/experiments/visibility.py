"""Видимость вершин по z-буферу из многих направлений (дешёвая замена ray casting)."""
import numpy as np
import common
from render import Canvas
def look_rot(d):
    """rotation mapping world direction d (from object towards camera) to +Z."""
    d = d/np.linalg.norm(d)
    up = np.array([0,1,0]) if abs(d[1])<0.95 else np.array([1,0,0])
    x = np.cross(up, d); x/=np.linalg.norm(x); y = np.cross(d, x)
    return np.stack([x,y,d])
def visible_vertices(Vq, occluders, dirs, res=420, eps=4e-4):
    allV = np.vstack([v for v,f in occluders]); c = (allV.min(0)+allV.max(0))/2
    ext = np.linalg.norm(allV.max(0)-allV.min(0))
    scale = (res*0.95)/ext
    vis = np.zeros(len(Vq), bool)
    for d in dirs:
        R = look_rot(d)
        cv = Canvas(res,res)
        for V,F in occluders:
            Q = (V-c)@R.T; Pp = np.c_[res/2+Q[:,0]*scale, res/2-Q[:,1]*scale, Q[:,2]]
            cv.draw(Pp, F, shade=False)
        Q = (Vq-c)@R.T; x = (res/2+Q[:,0]*scale).astype(int); y=(res/2-Q[:,1]*scale).astype(int)
        ok = (x>=0)&(x<res)&(y>=0)&(y<res)
        dep = np.full(len(Vq), np.inf); dep[ok] = cv.depth[y[ok],x[ok]]
        # check 3x3 neighbourhood max depth for robustness
        best = dep.copy()
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                xx=np.clip(x+dx,0,res-1); yy=np.clip(y+dy,0,res-1)
                best = np.minimum(best, np.where(ok, np.abs(cv.depth[yy,xx]-Q[:,2]), np.inf))
        vis |= best < eps
    return vis


def view_dirs(n=160):
    """Направления камеры: сфера Фибоначчи без видов сзади и снизу
    (иначе камера смотрит сквозь открытый затылок/шею внутрь головы)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    d = np.c_[np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)]
    return d[(d[:, 2] > -0.25) & (d[:, 1] > -0.45)]


def dva_skin_visibility(D, cache=True):
    """Маска вершин кожи D.Va, видимых снаружи (без внутренностей рта и глазниц)."""
    path = common.OUT / 'dva_skin_visible.npy'
    if cache and path.exists():
        return np.load(path)
    P = D['parts']
    occ = [(P[k]['V'], P[k]['F']) for k in ('skin', 'teeth', 'eyes')]
    vis = visible_vertices(P['skin']['V'], occ, view_dirs())
    np.save(path, vis)
    return vis
