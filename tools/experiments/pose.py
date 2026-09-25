"""Позирование скелета glTF и linear blend skinning (LBS)."""
import common  # noqa: F401
import numpy as np
from glb import acc, local_mat
def axis_angle_mat(axis, deg):
    axis = np.asarray(axis,float); axis/=np.linalg.norm(axis); a=np.radians(deg)
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K
def posed_world(g, local_rot=None, local_trans=None):
    """local_rot: {node_name: 3x3 applied in bone-local space (post-multiplied)}, local_trans: {node_name: (3,) added in parent space}"""
    local_rot = local_rot or {}; local_trans = local_trans or {}
    nodes = g['nodes']; W=[None]*len(nodes)
    def rec(i,P):
        L = local_mat(nodes[i]).copy()
        nm = nodes[i].get('name')
        if nm in local_rot: L[:3,:3] = L[:3,:3] @ local_rot[nm]
        if nm in local_trans: L[:3,3] = L[:3,3] + np.asarray(local_trans[nm])
        W[i] = P @ L
        for c in nodes[i].get('children',[]): rec(c, W[i])
    for r in g['scenes'][0]['nodes']: rec(r, np.eye(4))
    return W
def skin_part(g, binb, part, Wp):
    skin = g['skins'][0]; joints = skin['joints']
    IBM = acc(g,binb,skin['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
    M = np.array([Wp[j] @ IBM[k] for k,j in enumerate(joints)])
    V = part['V']; J = part['J']; Wt = part['W']
    Vh = np.c_[V, np.ones(len(V))]
    out = np.zeros((len(V),4))
    for k in range(4):
        out += Wt[:,k:k+1] * np.einsum('nij,nj->ni', M[J[:,k]], Vh)
    return out[:,:3]
