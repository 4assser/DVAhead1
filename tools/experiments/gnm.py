"""Минимальный загрузчик Google GNM Head v3 (numpy) + numpy-реализация semantic expression decoder.

Данные берутся из клона https://github.com/google/GNM (путь: переменная окружения GNM_DIR).
"""
import numpy as np
import common

NPZ = common.GNM_DIR / 'gnm/shape/data/versions/v3_0/gnm_head.npz'
LANDMARKS = common.GNM_DIR / 'gnm/shape/data/landmarks/head_sparse_68.txt'
DECODER = common.GNM_DIR / 'gnm/shape/data/semantic_sampler/expression_decoder_model.h5'

class GNM:
    def __init__(self, path=None):
        path = path or NPZ
        if not path.exists():
            raise FileNotFoundError(f'{path} не найден: git clone https://github.com/google/GNM и укажите GNM_DIR')
        d = np.load(path)
        self.T = d['template_vertex_positions'].astype(np.float64)
        self.I = d['vertex_identity_basis'].astype(np.float64)      # (253,V,3)
        self.E = d['expression_basis'].astype(np.float64)           # (383,V,3)
        self.tri = d['triangles'].astype(int)
        self.quads = d['quads'].astype(int)
        self.tri_uvs = d['triangle_uvs']
        self.J = d['template_joint_positions'].astype(np.float64)
        self.JI = d['joint_identity_basis'].astype(np.float64)
        self.joint_names = [str(x) for x in d['joint_names']]
        self.vg = {str(n): w for n, w in zip(d['vertex_group_names'], d['vertex_groups'])}
        self.id_names = [str(x) for x in d['identity_names']]
        self.ex_names = [str(x) for x in d['expression_names']]
        self.mirror = d['mirror_indices']
        self.skin_w = d['skinning_weights']
        lm = np.loadtxt(LANDMARKS)
        self.lm_idx = lm[:, 0::2].astype(int); self.lm_w = lm[:, 1::2]
    def verts(self, beta=None, phi=None):
        V = self.T.copy()
        if beta is not None: V += np.tensordot(beta, self.I[:len(beta)], 1)
        if phi is not None: V += np.tensordot(phi, self.E[:len(phi)], 1)
        return V
    def joints(self, beta=None):
        Jp = self.J.copy()
        if beta is not None: Jp += np.tensordot(beta, self.JI[:len(beta)], 1)
        return Jp
    def landmarks(self, V):
        return (V[self.lm_idx] * self.lm_w[..., None]).sum(1)

def expression_decoder(label, z=None, path=None):
    """Keras-MLP из semantic_sampler (z64 + one-hot 20 -> 383 коэффициентов выражения), без TensorFlow."""
    import h5py
    f = h5py.File(path or DECODER, 'r'); mw = f['model_weights']
    layers = ['dense_13', 'dense_14', 'dense_15', 'dense_16', 'dense_17']
    z = np.zeros(64) if z is None else z
    oh = np.zeros(20); oh[int(label)] = 1
    x = np.concatenate([z, oh])
    for i, l in enumerate(layers):
        Wk = mw[l][l]['kernel:0'][()]; b = mw[l][l]['bias:0'][()]
        x = x @ Wk + b
        if i < len(layers) - 1: x = np.maximum(x, 0)
    return x
EXPRESSIONS = ['SURPRISE','DISGUST','SUCK','COMPRESS_FACE','STRETCH_FACE','HAPPY','SQUINT','PLATYSMA','BLOW','FUNNELER','SMILE_WIDE','CORNERS_DOWN','PUCKER','WINK_LEFT','WINK_RIGHT','MOUTH_LEFT','MOUTH_RIGHT','LIPS_ROLL_IN','SNARL','TONGUE_CENTER']
