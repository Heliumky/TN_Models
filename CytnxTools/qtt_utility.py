import sys
#sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
from ncon import ncon
import numpy as np

I = np.array([[1.,0.],[0.,1.]])
sp = np.array([[0,1],[0,0]])
sm = np.array([[0,0],[1,0]])


def print_bond (bond):
    print(bond.type(), bond.qnums(), bond.getDegeneracies())

def print_bonds (T):
    print(T.labels())
    print(T.shape())
    for i in T.bonds():
        print_bond(i)

def inds_to_x (inds, rescale=1., shift=0.):
    res = inds[0]
    for i in range(1,len(inds)):
        res += inds[i] * 2**i
    return rescale * res + shift

#======================== MPO to MPS ========================
def mpo_to_mps (qtt):
    p = np.zeros((2,2,2))
    p[0,0,0] = 1.
    p[1,1,1] = 1.
    # Reduce to matrix multiplication
    Ms = []                 # Store the matrices after index collapsing
    N = len(qtt)            # The number of tensors in QTT
    for n in range(N):      # For each tensor
        M = qtt[n]
        M = ncon ([M,p], ((-1,1,2,-3), (1,2,-2)))
        Ms.append(M)
    return Ms

#======================== Get Element of MPS / MPO ========================
def get_ele_mps (mps, inds):
    # Reduce to matrix multiplication
    res = mps[0][:,inds[0],:]
    for n in range(1,len(inds)):
        res = res @ mps[n][:,inds[n],:]
    return res    

def get_ele_func (qtt, L, R, inds):
    # Reduce to matrix multiplication
    res = L
    for n in range(len(qtt)):      # For each tensor
        ind = inds[n]              # The index number we want to collapse
        M = qtt[n][:,ind,:]
        res = res @ M
    res = res @ R
    return res    

def get_ele_mpo (mpo, L, R, inds):
    res = L.reshape((1,L.shape[0]))
    for n in range(len(mpo)):
        M = mpo[n]
        M = M[:, inds[n], inds[n], :]
        res = np.dot(res, M)
    res = res @ R.reshape((R.shape[0],1))
    return float(res)

#======================== MPO Contractions with Boundaries ========================
def contract_L (mpo, L):
    Ltmp = L.reshape((1,*L.shape))
    mpo[0] = ncon ([Ltmp,mpo[0]], ((-1,1), (1,-2,-3,-4)))
    return mpo

def contract_R (mpo, R):
    Rtmp = R.reshape((*R.shape,1))
    mpo[-1] = ncon ([Rtmp,mpo[-1]], ((1,-4), (-1,-2,-3,1)))
    return mpo

def contract_LR (mpo, L, R):
    mpo = contract_L (mpo, L)
    mpo = contract_R (mpo, R)
    return mpo

#======================== Conversion to Cytnx Unitensor ========================
def mps_to_uniten (mps):
    res = []
    for i in range(len(mps)):
        A = cytnx.UniTensor (cytnx.from_numpy(mps[i]), rowrank=2)
        A.set_labels(['l','i','r'])
        res.append(A)
    return res

def mpo_to_uniten (mpo,L,R):
    H = []
    for i in range(len(mpo)):
        h = toUniTen(mpo[i])
        h.relabels_(['l','ip','i','r'])
        H.append(h)

    Lr = L.reshape((len(L),1,1))
    Rr = R.reshape((len(R),1,1))
    Lr = toUniTen (Lr)
    Rr = toUniTen (Rr)
    Lr.relabels_(['mid','up','dn'])
    Rr.relabels_(['mid','up','dn'])
    return H, Lr, Rr

def check_same_bonds (b1, b2):
    assert b1.type() == b2.redirect().type()
    assert b1.qnums() == b2.qnums()
    assert b1.getDegeneracies() == b2.getDegeneracies()

def decompose_tensor (T, rowrank, leftU=True, dim=sys.maxsize, cutoff=0.):
    # 1. SVD
    T.set_rowrank_(rowrank)
    s, A1, A2 = cytnx.linalg.Svd_truncate (T, keepdim=dim, err=cutoff)
    # 2. Absort s to A2 or A1
    if leftU:
        A2 = cytnx.Contract(s,A2)
        A1.relabel_('_aux_L','aux')
        A2.relabel_('_aux_L','aux')
    else:
        A1 = cytnx.Contract(s,A1)
        A1.relabel_('_aux_R','aux')
        A2.relabel_('_aux_R','aux')
    return A1, A2

#======================== Tensor Conversion Utilities ========================
def toUniTen (T):
    assert type(T) == np.ndarray
    T = cytnx.from_numpy(T)
    return cytnx.UniTensor (T)

def to_nparray(T):
    assert type(T) == cytnx.UniTensor
    if T.is_blockform():
        tmp = cytnx.UniTensor.zeros(T.shape())
        tmp.convert_from(T)
        T = tmp
    return T.get_block().numpy()

def mps_to_nparray (mps):
    res = []
    for i in range(len(mps)):
        if mps[i].is_blockform():
            A = cytnx.UniTensor.zeros(mps[i].shape())
            A.convert_from(mps[i])
            A = A.get_block().numpy()
        else:
            A = mps[i].get_block().numpy()
        res.append(A)
    return res

#======================== MPO/MPS Local Operator Applications ========================
#       |
#      (A)
#       |
#       | i'
#   ---(h)---
#       | i
#       |
#      (B)
#       |
#
def applyLocal_mpo_tensor (h, A=None, B=None):
    re = ncon([h,A], ((-1,-2,1,-4), (1,-3)))
    re = ncon([re,B], ((-1,1,-3,-4), (-2,1)))
    return re

def applyLocal_mpo (H, A=None, B=None):
    res = []
    for i in range(len(H)):
        h = applyLocal_mpo_tensor (H[i], A, B)
        res.append(h)
    return res

#       |
#      (U)
#       |
#       | i'
#   ---(h)---
#       | i
#       |
#     (Udag)
#       |
#
def applyLocalRot_mpo (H, U):
    Udag = np.conjugate(np.transpose(U))
    res = []
    for i in range(len(H)):
        h = applyLocal_mpo_tensor (H[i], U, Udag)
        res.append(h)
    return res

#       |
#     (op)
#       |
#   ---(A)---
#
def applyLocal_mps (mps, op):
    res = []
    for i in range(len(mps)):
        A = mps[i]
        A = ncon([A,op], ((-1,1,-3), (-2,1)))
        res.append(A)
    return res
