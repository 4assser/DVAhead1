"""Tiny numpy software rasterizer (z-buffer, textured / vertex-colored, smooth Lambert)."""
import numpy as np
from PIL import Image


def rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rot_x(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(n, 1e-12)


def sample_bilinear(img, uv):
    """img float (H,W,3) in [0,1], uv (N,2) glTF convention (v down)."""
    H, W = img.shape[:2]
    x = np.clip(uv[:, 0] % 1.0, 0, 1) * (W - 1)
    y = np.clip(uv[:, 1] % 1.0, 0, 1) * (H - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


class Canvas:
    def __init__(self, W, H, bg=(1, 1, 1)):
        self.W, self.H = W, H
        self.color = np.ones((H, W, 3)) * np.array(bg, dtype=float)
        self.depth = np.full((H, W), -np.inf)

    def draw(self, P, F, N=None, uv=None, tex=None, vcolor=None, base=(0.8, 0.8, 0.8),
             light=(0.3, 0.4, 1.0), ambient=0.35, shade=True, alpha=1.0):
        """P: (V,3) screen-space points: x,y in pixels, z depth (bigger = closer)."""
        light = np.asarray(light, float); light /= np.linalg.norm(light)
        for f in F:
            p = P[f]
            xmin = int(max(np.floor(p[:, 0].min()), 0)); xmax = int(min(np.ceil(p[:, 0].max()), self.W - 1))
            ymin = int(max(np.floor(p[:, 1].min()), 0)); ymax = int(min(np.ceil(p[:, 1].max()), self.H - 1))
            if xmax < xmin or ymax < ymin:
                continue
            xs, ys = np.meshgrid(np.arange(xmin, xmax + 1) + 0.5, np.arange(ymin, ymax + 1) + 0.5)
            x0, y0 = p[0, :2]; x1, y1 = p[1, :2]; x2, y2 = p[2, :2]
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(den) < 1e-12:
                continue
            w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / den
            w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / den
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not inside.any():
                continue
            z = w0 * p[0, 2] + w1 * p[1, 2] + w2 * p[2, 2]
            sub = self.depth[ymin:ymax + 1, xmin:xmax + 1]
            m = inside & (z > sub)
            if not m.any():
                continue
            W3 = np.stack([w0[m], w1[m], w2[m]], 1)
            if tex is not None and uv is not None:
                col = sample_bilinear(tex, W3 @ uv[f])
            elif vcolor is not None:
                col = W3 @ vcolor[f]
            else:
                col = np.tile(np.asarray(base, float), (m.sum(), 1))
            if shade and N is not None:
                n = W3 @ N[f]
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
                lam = np.abs(n @ light)  # two-sided
                col = col * (ambient + (1 - ambient) * lam[:, None])
            cs = self.color[ymin:ymax + 1, xmin:xmax + 1]
            cs[m] = alpha * col + (1 - alpha) * cs[m]
            sub[m] = z[m]

    def image(self):
        return Image.fromarray((np.clip(self.color, 0, 1) * 255).astype(np.uint8))


def project_ortho(V, R, center, scale, W, H):
    """Rotate world points by R around `center`, orthographic projection to pixels."""
    Q = (V - center) @ R.T
    P = np.empty_like(Q)
    P[:, 0] = W / 2 + Q[:, 0] * scale
    P[:, 1] = H / 2 - Q[:, 1] * scale
    P[:, 2] = Q[:, 2]
    return P
