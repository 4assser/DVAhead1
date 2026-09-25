import json, struct, numpy as np
CT = {5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}
NC = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}
def load(path):
    b = open(path,'rb').read()
    off=12; chunks=[]
    while off < len(b):
        clen, ctype = struct.unpack('<I4s', b[off:off+8]); chunks.append(b[off+8:off+8+clen]); off += 8+clen
    g = json.loads(chunks[0]); binb = chunks[1]
    return g, binb
def acc(g, binb, i):
    a = g['accessors'][i]; bv = g['bufferViews'][a['bufferView']]
    dt = CT[a['componentType']]; n = NC[a['type']]
    start = bv.get('byteOffset',0) + a.get('byteOffset',0)
    stride = bv.get('byteStride', 0)
    itemsize = np.dtype(dt).itemsize*n
    if stride and stride != itemsize:
        raw = np.frombuffer(binb, dtype=np.uint8, count=stride*a['count'], offset=start).reshape(a['count'], stride)[:, :itemsize]
        arr = np.frombuffer(raw.tobytes(), dtype=dt).reshape(a['count'], n)
    else:
        arr = np.frombuffer(binb, dtype=dt, count=a['count']*n, offset=start).reshape(a['count'], n)
    arr = arr.astype(np.float64) if dt==np.float32 else arr.copy()
    if a.get('normalized'):
        arr = arr / np.iinfo(dt).max
    return arr
def quat_to_mat(q):
    x,y,z,w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def local_mat(n):
    M = np.eye(4)
    if 'matrix' in n: return np.array(n['matrix']).reshape(4,4).T
    R = quat_to_mat(n.get('rotation',[0,0,0,1])); S = np.diag(n.get('scale',[1,1,1]))
    M[:3,:3] = R@S; M[:3,3] = n.get('translation',[0,0,0])
    return M
def world_mats(g):
    nodes = g['nodes']; W = [None]*len(nodes)
    def rec(i, P):
        W[i] = P @ local_mat(nodes[i])
        for c in nodes[i].get('children',[]): rec(c, W[i])
    for r in g['scenes'][g.get('scene',0)]['nodes']: rec(r, np.eye(4))
    return W
