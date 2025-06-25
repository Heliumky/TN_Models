import differential as df
import npmps
import linear as lin
from ncon import ncon
import numpy as np

def get_H (V_MPS):
    V_MPO, VL, VR = npmps.mps_func_to_mpo (V_MPS)

    H = []
    for n in range(len(V_MPS)):
        ddx2_tensor = df.make_tensorA()
        hi = npmps.sum_mpo_tensor (ddx2_tensor, V_MPO[n])
        H.append(hi)

    L_ddx2, R_ddx2 = df.make_LR()
    L = np.concatenate ((L_ddx2, VL))
    R = np.concatenate ((R_ddx2, VR))
    return H, L, R

def H_kinetic (N, dx):
    H = []
    for n in range(N):
        ddx2_tensor = df.make_tensorA()
        H.append(ddx2_tensor)
    L, R = df.make_LR()
    L = -0.5*(L/dx**2)
    #L *= -0.5./dx**2
    return H, L, R

def H_trap (N, dx, shift):
    H = []
    for n in range(N):
        x_tensor = lin.make_x_tensor (n, dx)
        x2_tensor = npmps.prod_MPO_tensor (x_tensor, x_tensor)
        H.append(x2_tensor)
    L_x, R_x = lin.make_x_LR (shift)
    L_x2 = ncon([L_x,L_x], ((-1,),(-2,))).reshape(-1,)
    R_x2 = ncon([R_x,R_x], ((-1,),(-2,))).reshape(-1,)
    return H, 0.5*L_x2, R_x2

def Ex_MPO(N,shift):
    rescale = -2*shift/(2**N-1)
    x_mpo, Lx, Rx = lin.make_x_mpo (N, shift, rescale)
    I_ympo,LIy,RIx = npmps.identity_MPO (N, phydim=2)
    Ex, L, R = npmps.product_2MPO(x_mpo, Lx, Rx, I_ympo, LIy, RIx)
    return Ex, L, R

def Lz_MPO (N, shift):
    rescale = -2*shift/(2**N-1)
    p, Ldf, Rdf = df.diff_MPO (N, rescale)
    p, Ldf, Rdf  = npmps.change_dtype (p,Ldf, Rdf, complex)
    p[0] *= -1j
    x, Lx, Rx = lin.make_x_mpo (N, shift, rescale)
    xpy, Lxpy, Rxpy = npmps.product_2MPO(x, Lx, Rx, p, Ldf, Rdf)
    ypx, Lypx, Rypx = npmps.product_2MPO(p, Ldf, Rdf, x,Lx,Rx)
    xpy, Lxpy, Rxpy = npmps.change_dtype (xpy, Lxpy, Rxpy, complex)
    ypx, Lypx, Rypx = npmps.change_dtype (ypx, Lypx, Rypx, complex)
    ypx[0] *= -1
    Lz , LLz, RLz = npmps.sum_2MPO (xpy, Lxpy, Rxpy, ypx, Lypx, Rypx)
    return Lz , LLz, RLz

def Lz2_MPO(N, shift):
    Lz , LLz, RLz = Lz_MPO (N, shift)
    Lz2 = []
    for n in range(len(Lz)):
        Lz2_tensor = npmps.prod_MPO_tensor (Lz[n], Lz[n])
        Lz2.append(Lz2_tensor)
    L = ncon([LLz,LLz], ((-1,),(-2,))).reshape(-1,)
    R = ncon([RLz,RLz], ((-1,),(-2,))).reshape(-1,)
    return Lz2, L, R

def get_H_2D (H_1D, L_1D, R_1D):
    N = len(H_1D)
    H_I, L_I, R_I = npmps.identity_MPO (N, 2)
    H1, L1, R1 = npmps.product_2MPO (H_1D, L_1D, R_1D, H_I, L_I, R_I)
    H2, L2, R2 = npmps.product_2MPO (H_I, L_I, R_I, H_1D, L_1D, R_1D)
    H, L, R = npmps.sum_2MPO (H1, L1, R1, H2, L2, R2)
    return H, L, R


def get_H_3D (H_1D, L_1D, R_1D):
    N = len(H_1D)
    H_I, L_I, R_I = npmps.identity_MPO (N, 2)
    H_2I, L_2I, R_2I = npmps.identity_MPO (2*N, 2)
    H1, L1, R1 = npmps.product_2MPO (H_1D, L_1D, R_1D, H_2I, L_2I, R_2I)
    H2, L2, R2 = npmps.product_2MPO (H_I, L_I, R_I, H_1D, L_1D, R_1D)
    H2, L2, R2 = npmps.product_2MPO (H2, L2, R2, H_I, L_I, R_I)
    H3, L3, R3 = npmps.product_2MPO (H_2I, L_2I, R_2I, H_1D, L_1D, R_1D)
    H, L, R = npmps.sum_2MPO (H1, L1, R1, H2, L2, R2)
    H, L, R = npmps.sum_2MPO (H, L, R, H3, L3, R3)
    return H, L, R
