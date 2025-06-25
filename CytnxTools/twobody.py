import copy, sys
#sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import numpy as np
from ncon import ncon
import qtt_utility as ut
import MPS_utility as mput
import npmps


def phys_index ():
    return cytnx.Bond(cytnx.BD_IN, [[0],[1]], [3,1], [cytnx.Symmetry.Zn(2)])

def make_product_mps (N, cy_dtype=0):
    # The parity for the physical bond is [0,0,0,1]
    ii = phys_index()   # physical bond
    # Virtual bonds
    vb0 = cytnx.Bond(cytnx.BD_IN, [[0]], [1], [cytnx.Symmetry.Zn(2)])
    vb1 = cytnx.Bond(cytnx.BD_IN, [[1]], [1], [cytnx.Symmetry.Zn(2)])

    # Other sites
    re = []
    for i in range(N):
        if i == 0:
            A = cytnx.UniTensor ([vb0, ii, vb1.redirect()], labels=['l','i','r'],dtype= cy_dtype)
            A.at([0,3,0]).value = 1.
        else:
            A = cytnx.UniTensor ([vb1, ii, vb1.redirect()], labels=['l','i','r'],dtype= cy_dtype)
            A.at([0,0,0]).value = 1./3**0.5
            A.at([0,1,0]).value = 1./3**0.5
            A.at([0,2,0]).value = 1./3**0.5
        re.append(A)
    return re

# N is the number of sites for space. The returned MPS has N+1 sites; additional one is for spin.
def make_product_mps_spin (N):
    # The parity for the physical bond is [0,0,0,1]
    ii = cytnx.Bond(cytnx.BD_IN, [[0],[1]], [3,1], [cytnx.Symmetry.Zn(2)])   # physical bond
    # Virtual bonds
    bond = cytnx.Bond(cytnx.BD_IN, [[0],[1]], [1,1], [cytnx.Symmetry.Zn(2)])

    # Other sites
    re = []
    for i in range(N+1):
        if i == 0:
            bondL = cytnx.Bond(cytnx.BD_IN, [[0]], [1], [cytnx.Symmetry.Zn(2)])
            A = cytnx.UniTensor ([bondL, ii, bond.redirect()], labels=['l','i','r'])
            # Mixing of symmetric and anti-symmetric
            # Symmetric
            A.at([0,0,0]).value = 1./3**0.5
            A.at([0,1,0]).value = 1./3**0.5
            A.at([0,2,0]).value = 1./3**0.5
            # Anti-symmetric
            A.at([0,3,1]).value = 1.
            # Normalization
            A *= 1./2**0.5
        elif i != N:
            A = cytnx.UniTensor ([bond, ii, bond.redirect()], labels=['l','i','r'])
            # Symmetric
            A.at([0,0,0]).value = 1./3**0.5
            A.at([0,1,0]).value = 1./3**0.5
            A.at([0,2,0]).value = 1./3**0.5
            # Symmetric
            A.at([1,0,1]).value = 1./3**0.5
            A.at([1,1,1]).value = 1./3**0.5
            A.at([1,2,1]).value = 1./3**0.5
        else:
            # Last site is for the spin
            # Symmetric if the first site is anti-symmetric, and vice versa
            bondR = cytnx.Bond(cytnx.BD_IN, [[1]], [1], [cytnx.Symmetry.Zn(2)])
            A = cytnx.UniTensor ([bond, ii, bondR.redirect()], labels=['l','i','r'])
            # Mixing of symmetric and anti-symmetric
            # Anti-symmetric
            A.at([0,3,0]).value = 1.
            # Symmetric
            A.at([1,0,0]).value = 1./3**0.5
            A.at([1,1,0]).value = 1./3**0.5
            A.at([1,2,0]).value = 1./3**0.5
        re.append(A)
    return re

def make_product_mps (N, parity):
    # The parity for the physical bond is [0,0,0,1]
    ii = phys_index()
    # Virtual bonds
    vb0 = cytnx.Bond(cytnx.BD_IN, [[0]], [1], [cytnx.Symmetry.Zn(2)])
    vb1 = cytnx.Bond(cytnx.BD_IN, [[1]], [1], [cytnx.Symmetry.Zn(2)])

    # Other sites
    re = []
    if parity == 1:
        for i in range(N):
            if i == 0:
                A = cytnx.UniTensor ([vb0, ii, vb1.redirect()], labels=['l','i','r'])
                A.at([0,3,0]).value = 1.
            else:
                A = cytnx.UniTensor ([vb1, ii, vb1.redirect()], labels=['l','i','r'])
                A.at([0,0,0]).value = 1./3**0.5
                A.at([0,1,0]).value = 1./3**0.5
                A.at([0,2,0]).value = 1./3**0.5
            re.append(A)
    else:
        for i in range(N):
            A = cytnx.UniTensor ([vb0, ii, vb0.redirect()], labels=['l','i','r'])
            A.at([0,0,0]).value = 1./3**0.5
            A.at([0,1,0]).value = 1./3**0.5
            A.at([0,2,0]).value = 1./3**0.5
            re.append(A)   
    return re

# The parities are 0,0,0,1 for the states after rotation
def get_swap_U ():
    a = 1/2**0.5
    res = np.zeros((2,2,2,2))
    res[0,0,0,0] = 1.
    res[0,1,0,1] = a
    res[0,1,1,0] = a
    res[1,0,0,1] = a
    res[1,0,1,0] = -a
    res[1,1,1,1] = 1.
    res = res.reshape((4,4))

    # Exchange the 3rd and the 4th columns, so that the +1 paritiy are grouped together
    tmp = copy.copy(res[:,2])
    res[:,2] = res[:,3]
    res[:,3] = tmp
    return res

def H_two_particles (H1, L1, R1, H2, L2, R2):
    I = np.array([[1.,0.],[0.,1.]])

    N = len(H1)
    H = []
    for i in range(N):
        HIi = ncon([H1[i],I], ((-1,-2,-4,-6), (-3,-5)))
        IHi = ncon([H2[i],I], ((-1,-3,-5,-6), (-2,-4)))
        d = H1[i].shape
        HIi = HIi.reshape((d[0], d[1]*2, d[2]*2, d[3]))
        IHi = IHi.reshape((d[0], d[1]*2, d[2]*2, d[3]))
        H.append (npmps.sum_mpo_tensor (HIi, IHi))

    L = np.append(L1,L2)
    R = np.append(R1,R2)
    return H, L, R

def H_merge_two_particle (H):
    N = len(H)
    Hre = []
    for i in range(0,N,2):
        Hi = ncon([H[i],H[i+1]], ((-1,-2,-4,1), (1,-3,-5,-6)))
        d = Hi.shape
        Hi = Hi.reshape((d[0], d[1]*d[2], d[3]*d[4], d[5]))
        Hre.append (Hi)
    return Hre

def corr_matrix (psi):
    mps = []
    for i in range(len(psi)):
        d = psi[i].shape
        mps.append (psi[i].reshape((d[0],2,2,d[2])))

    mpo = []
    for i in range(len(mps)):
        A = ncon([np.conj(mps[i]),mps[i]], ((-1,-3,1,-5), (-2,-4,1,-6)))
        d = A.shape
        A = A.reshape((d[0]*d[1], d[2], d[3], d[4]*d[5]))
        mpo.append (A)
    return mpo

# mpo is a list of tensors as np.array
# if parity ==1, then tot wf is antisym. otherwise etc.
def set_mpo_quantum_number(mpo, L, R, parity=0):
    # Physical bond
    ii = phys_index().redirect()
    iip = ii.redirect()
    dtype = min(ut.toUniTen (mpo[0]).dtype(), ut.toUniTen (L).dtype(), ut.toUniTen (R).dtype())
    #print(dtype)
    # Left and right virtual bonds
    li = cytnx.Bond(cytnx.BD_IN, [[0]], [mpo[0].shape[0]], [cytnx.Symmetry.Zn(2)])
    ri = cytnx.Bond(cytnx.BD_OUT, [[0]], [mpo[-1].shape[-1]], [cytnx.Symmetry.Zn(2)])

    # Convert L, R to UniTensors
    assert L.ndim == 1 and R.ndim == 1
    uL = ut.toUniTen(L.reshape(1, L.shape[0], 1))
    uR = ut.toUniTen(R.reshape(1, R.shape[0], 1))

    vb0 = cytnx.Bond(cytnx.BD_IN, [[0]], [1], [cytnx.Symmetry.Zn(2)])
    vb1 = cytnx.Bond(cytnx.BD_IN, [[1]], [1], [cytnx.Symmetry.Zn(2)])

    qnL = cytnx.UniTensor([li.redirect(), vb0.redirect(), vb0], labels=['mid', 'dn', 'up'], dtype=dtype)
    if parity == 0:
        qnR = cytnx.UniTensor([ri.redirect(), vb0, vb0.redirect()], labels=['mid', 'dn', 'up'], dtype=dtype)
    else:    
        qnR = cytnx.UniTensor([ri.redirect(), vb1, vb1.redirect()], labels=['mid', 'dn', 'up'], dtype=dtype)
    qnL.convert_from(uL)
    qnR.convert_from(uR)

    mpoA = mpo[0]
    re = []

    for i in range(len(mpo) - 1):
        rdim = mpoA.shape[3]
        ri0 = cytnx.Bond(cytnx.BD_OUT, [[0]], [rdim], [cytnx.Symmetry.Zn(2)])
        ri1 = cytnx.Bond(cytnx.BD_OUT, [[1]], [rdim], [cytnx.Symmetry.Zn(2)])

        uT = ut.toUniTen(mpoA)
        qn_T0 = cytnx.UniTensor([li, iip, ii, ri0], labels=["l", "ip", "i", "r"], dtype=dtype)
        qn_T1 = cytnx.UniTensor([li, iip, ii, ri1], labels=["l", "ip", "i", "r"], dtype=dtype)
        qn_T0.convert_from(uT, force=True)
        qn_T1.convert_from(uT, force=True)
        qn_Ts = [qn_T0, qn_T1]

        ri = cytnx.Bond(cytnx.BD_OUT, [[0], [1]], [rdim, rdim], [cytnx.Symmetry.Zn(2)])
        qn_T = cytnx.UniTensor([li, iip, ii, ri], labels=["l", "ip", "i", "r"], dtype=dtype)

        for i1 in range(len(qn_T.bond("l").qnums())):
            for i2 in range(len(qn_T.bond("ip").qnums())):
                for i3 in range(len(qn_T.bond("i").qnums())):
                    for i4 in range(len(qn_T.bond("r").qnums())):
                        blk = qn_Ts[i4].get_block(["l", "ip", "i", "r"], [i1, i2, i3, 0], force=True)
                        if blk.rank() != 0:
                            qn_T.put_block_(blk, ["l", "ip", "i", "r"], [i1, i2, i3, i4])

        rrdim = mpo[i + 1].shape[3]
        mpoA2 = np.zeros((2 * rdim, 4, 4, rrdim), dtype=mpo[i + 1].dtype)
        mpoA2[:rdim, :, :, :] = mpo[i + 1]
        mpoA2[rdim:, :, :, :] = mpo[i + 1]

        qn_T.set_rowrank_(3)
        s, A, vt = cytnx.linalg.Svd_truncate(qn_T, keepdim=2 * rdim, err=1e-12)
        
        #s = s/s.Norm().item()
        s = s.astype(vt.dtype())
        R = cytnx.Contract(s, vt)

        A.relabel_("_aux_L", "r")
        re.append(A)

        TR = cytnx.UniTensor.zeros(R.shape(), dtype=dtype).convert_from(R).get_block().numpy()
        mpoA2 = ncon([TR, mpoA2], ((-1, 1), (1, -4, -5, -6)))

        mpoA = mpoA2
        li = A.bond("r").redirect()

    uT = ut.toUniTen(mpoA)
    ri = cytnx.Bond(cytnx.BD_OUT, [[0]], [mpo[-1].shape[-1]], [cytnx.Symmetry.Zn(2)])
    qn_T = cytnx.UniTensor([li, iip, ii, ri], labels=["l", "ip", "i", "r"], dtype=dtype)
    qn_T.convert_from(uT, force=True)
    re.append(qn_T)

    return re, qnL, qnR

