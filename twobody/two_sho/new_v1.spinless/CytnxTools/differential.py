import numpy as np
import matplotlib.pyplot as plt
from ncon import ncon
import qtt_utility as ut

def make_tensorA ():
    A = np.zeros ((3,2,2,3)) # (k1,ipr,i,k2)
    A[0,:,:,0] = ut.I
    A[1,:,:,0] = ut.sp
    A[2,:,:,0] = ut.sm
    A[1,:,:,1] = ut.sm
    A[2,:,:,2] = ut.sp
    return A

def make_LR ():
    L = np.array([-2,1,1])
    R = np.array([1,1,1])
    return L, R

# QTT for a (d/dx)^2 operator
def make_d2dx2_optt (N):
    op_qtt = [make_tensorA() for n in range(N)]        # QTT for a (d/dx)^2 operator
    L = np.array([-2,1,1])
    R = np.array([1,1,1])
    op_qtt[0] = ncon ([L,op_qtt[0]], ((1,), (1,-1,-2,-3)))
    op_qtt[-1] = ncon ([R,op_qtt[-1]], ((1,), (-1,-2,-3,1)))
    return op_qtt

def diff_MPO (N, dx):
    mpo = []
    for i in range(N):
        mpo.append (make_tensorA())
    L = np.array([0.,1.,-1.])
    L *= 0.5/dx
    R = np.array([1.,1.,1.])
    return mpo, L, R


def project_qtt_op (op_qtt, inds):
    res = []
    N = len(op_qtt)
    for n in range(N):
        M = op_qtt[n]
        # Project the operator tensor
        if n == 0:
            M = M[inds[n],:,:]      # (ipr, i, k) --> (i,k)
        elif n == N - 1:
            M = M[:,inds[n],:]      # (k, ipr, i) --> (k,i)
        else:
            M = M[:,inds[n],:,:]    # (k1,ipr,i,k2) --> (k1,i,k2)
        res.append(M)
    return res

def contract_qtt (qtt1, qtt2):
    res = ncon ([qtt1[0],qtt2[0]], ((1,-1), (1,-2)))
    for i in range(1,len(qtt2)-1):
        res = ncon ([res,qtt1[i],qtt2[i]], ((1,2), (1,3,-1), (2,3,-2)))
    res = ncon([res,qtt1[-1],qtt2[-1]], ((1,2), (1,3), (2,3)))
    return res
