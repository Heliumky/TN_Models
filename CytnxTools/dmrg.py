import sys
sys.path.insert(0,'/home/chiamin/cytnx_dev/Cytnx_lib/')
import cytnx
import numpy as np
import copy
import qtt_utility as ut
import MPS_utility as mpsut
from copy import deepcopy

# ======================== Environment Tensor Operations ========================

def left_environment_contraction_mpo(E, A1, W, A2):
    """Left environment tensor contraction in MPO environment"""
    E = E.relabels(['mid','dn','up'], ['_mid','_dn','_up'])
    A1 = A1.relabels(['l','i','r'], ['_dn','_i','dn'])
    W = W.relabels(['l','r','ip','i'], ['_mid','mid','_ip','_i'])
    A2 = A2.Dagger().relabels(['l','i','r'], ['_up','_ip','up'])
    
    tmp = cytnx.Contract(E, A1)
    tmp = cytnx.Contract(tmp, W)
    tmp = cytnx.Contract(tmp, A2)
    return tmp

def right_environment_contraction_mpo(E, A1, W, A2):
    """Right environment tensor contraction in MPO environment"""
    E = E.relabels(['mid','dn','up'], ['_mid','_dn','_up'])
    A1 = A1.relabels(['r','i','l'], ['_dn','_i','dn'])
    W = W.relabels(['r','l','ip','i'], ['_mid','mid','_ip','_i'])
    A2 = A2.Dagger().relabels(['r','i','l'], ['_up','_ip','up'])
    
    tmp = cytnx.Contract(E, A1)
    tmp = cytnx.Contract(tmp, W)
    tmp = cytnx.Contract(tmp, A2)
    return tmp

def left_environment_contraction_mps(E, A1, A2):
    """Left environment tensor contraction in MPS environment"""
    A1 = A1.relabels(['l','i','r'], ['_dn','i','dn'])
    A2 = A2.Dagger().relabels(['l','i','r'], ['_up','i','up'])
    E = E.relabels(['up','dn'], ['_up','_dn'])
    tmp = cytnx.Contract(E, A1)
    tmp = cytnx.Contract(tmp, A2)
    return tmp

def right_environment_contraction_mps(E, A1, A2):
    """Right environment tensor contraction in MPS environment"""
    A1 = A1.relabels(['r','i','l'], ['_dn','i','dn'])
    A2 = A2.Dagger().relabels(['r','i','l'], ['_up','i','up'])
    E = E.relabels(['up','dn'], ['_up','_dn'])
    tmp = cytnx.Contract(E, A1)
    tmp = cytnx.Contract(tmp, A2)
    return tmp

# ======================== Environment Tensor Classes ========================

class LR_envir_tensors_mpo:
    """MPO Environment Tensor Manager"""
    def __init__(self, N, L0, R0, verbose=False):
        self.centerL = 0
        self.centerR = N-1
        self.LR = [None] * (N+1)
        self.LR[0] = L0
        self.LR[-1] = R0
        self.verbose = verbose

    def __getitem__(self, i):
        return self.LR[i]
    
    def delete(self, i):
        """Delete environment tensor at specified position"""
        self.centerL = min(self.centerL, i)
        self.centerR = max(self.centerR, i)

    def update_LR(self, mps1, mps2, mpo, centerL, centerR=None):
        """Update left and right environment tensors"""
        if centerR is None:
            centerR = centerL
        
        # Boundary check
        if centerL > centerR + 1:
            raise ValueError(f"centerL ({centerL}) cannot be larger than centerR+1 ({centerR+1})")
        
        # Update left environment
        for p in range(self.centerL, centerL):
            self.LR[p+1] = left_environment_contraction_mpo(self.LR[p], mps1[p], mpo[p], mps2[p])
            if self.verbose:
                print(f"  Updating left environment: {p} -> {p+1}")
        
        # Update right environment
        for p in range(self.centerR, centerR, -1):
            self.LR[p] = right_environment_contraction_mpo(self.LR[p+1], mps1[p], mpo[p], mps2[p])
            if self.verbose:
                print(f"  Updating right environment: {p+1} -> {p}")
        
        self.centerL = centerL
        self.centerR = centerR

class LR_envir_tensors_mps:
    """MPS Environment Tensor Manager"""
    def __init__(self, N, mps1, mps2, verbose=False):
        self.centerL = 0
        self.centerR = N-1
        self.LR = [None] * (N+1)
        self.verbose = verbose
        
        # Initialize boundary tensors
        l1 = mps1[0].bond("l").redirect()
        l2 = mps2[0].bond("l")
        r1 = mps1[-1].bond("r").redirect()
        r2 = mps2[-1].bond("r")
        
        L0 = cytnx.UniTensor([l1, l2], labels=['dn','up'])
        R0 = cytnx.UniTensor([r1, r2], labels=['dn','up'])
        
        assert np.prod(L0.shape()) == np.prod(R0.shape()) == 1
        L0.at([0,0]).value = 1.
        R0.at([0,0]).value = 1.
        
        self.LR[0] = L0
        self.LR[-1] = R0

    def __getitem__(self, i):
        return self.LR[i]
    
    def delete(self, i):
        """Delete environment tensor at specified position"""
        self.centerL = min(self.centerL, i)
        self.centerR = max(self.centerR, i)

    def update_LR(self, mps1, mps2, centerL, centerR=None):
        """Update left and right environment tensors"""
        if centerR is None:
            centerR = centerL
        
        # Boundary check
        if centerL > centerR + 1:
            raise ValueError(f"centerL ({centerL}) cannot be larger than centerR+1 ({centerR+1})")
        
        # Update left environment
        for p in range(self.centerL, centerL):
            self.LR[p+1] = left_environment_contraction_mps(self.LR[p], mps1[p], mps2[p])
            if self.verbose:
                print(f"  Updating left MPS environment: {p} -> {p+1}")
        
        # Update right environment
        for p in range(self.centerR, centerR, -1):
            self.LR[p] = right_environment_contraction_mps(self.LR[p+1], mps1[p], mps2[p])
            if self.verbose:
                print(f"  Updating right MPS environment: {p+1} -> {p}")
        
        self.centerL = centerL
        self.centerR = centerR

# ======================== Effective Hamiltonian Classes ========================

class eff_2s_Hamilt(cytnx.LinOp):
    """Two-site Effective Hamiltonian"""
    def __init__(self, L, M1, M2, R):
        dtype = min(L.dtype(), M1.dtype(), M2.dtype(), R.dtype())
        cytnx.LinOp.__init__(self, "mv", 0, dtype=dtype)
        
        # Define network for H|psi> operation
        self.L = L.relabels(['mid','dn','up'], ['l','ldn','lup'])
        self.M1 = M1.relabels(['l','ip','i','r'], ['l','ip1','i1','mid'])
        self.M2 = M2.relabels(['l','ip','i','r'], ['mid','ip2','i2','r'])
        self.R = R.relabels(['mid','dn','up'], ['r','rdn','rup'])

        # For excited states
        self.anet2 = cytnx.Network()
        self.anet2.FromString(["A1: lup, i1, _",
                               "A2: _, i2, rup",
                               "L: ldn, lup",
                               "R: rdn, rup",
                               "TOUT: ldn, i1, i2, rdn"])
        self.ortho = []
        self.ortho_w = []

    def add_orthogonal(self, L, orthoA1, orthoA2, R, weight):
        self.anet2.PutUniTensor("L", L, ["dn","up"])
        self.anet2.PutUniTensor("R", R, ["dn","up"])
        self.anet2.PutUniTensor("A1", orthoA1.Dagger(), ["l","i","r"])
        self.anet2.PutUniTensor("A2", orthoA2.Dagger(), ["l","i","r"])
        out = self.anet2.Launch()
        out.relabels_(['l','i1','i2','r'])
        self.ortho.append(out)
        self.ortho_w.append(weight)

    # Define H|psi> operation
    def matvec(self, v):
        psi = v.relabels(['l','i1','i2','r'],['ldn','i1','i2','rdn'])

        tmp = cytnx.Contract(self.L, psi)
        tmp = cytnx.Contract(tmp, self.M1)
        tmp = cytnx.Contract(tmp, self.M2)
        tmp = cytnx.Contract(tmp, self.R)

        out = tmp
        out.relabels_(['lup', 'ip1', 'ip2', 'rup'],['l','i1','i2','r'])

        for j in range(len(self.ortho)):
            ortho = self.ortho[j]
            overlap = cytnx.Contract(ortho, v)
            out += self.ortho_w[j] * overlap.item() * ortho.Dagger()
        return out

class eff_1s_Hamilt(cytnx.LinOp):
    """Single-site Effective Hamiltonian (2s-style implementation)"""
    def __init__(self, L, M, R):
        dtype = min(L.dtype(), M.dtype(), R.dtype())
        cytnx.LinOp.__init__(self, "mv", 0, dtype=dtype)

        # Relabel input tensors to standard names
        self.L = L.relabels(['mid','dn','up'], ['l','ldn','lup'])
        self.M = M.relabels(['l','ip','i','r'], ['l','ip','i','r'])
        self.R = R.relabels(['mid','dn','up'], ['r','rdn','rup'])

        # For orthogonal constraints
        self.anet2 = cytnx.Network()
        self.anet2.FromString([
            "A: lup, i, rup",
            "L: ldn, lup",
            "R: rdn, rup",
            "TOUT: ldn, i, rdn"
        ])
        self.ortho = []
        self.ortho_w = []

    def add_orthogonal(self, L, orthoA, R, weight):
        """Add orthogonal state constraint"""
        self.anet2.PutUniTensor("L", L, ["dn", "up"])
        self.anet2.PutUniTensor("R", R, ["dn", "up"])
        self.anet2.PutUniTensor("A", orthoA.Dagger(), ["l", "i", "r"])
        out = self.anet2.Launch()
        out.relabels_(['l', 'i', 'r'])
        self.ortho.append(out)
        self.ortho_w.append(weight)

    def matvec(self, v):
        """Define H|psi> operation"""
        # Standardize input vector labels
        v = v.relabels(v.labels(), ['l','i','r'])
        psi = v.relabels(['l','i','r'], ['ldn','i','rdn'])

        tmp = cytnx.Contract(self.L, psi)
        tmp = cytnx.Contract(tmp, self.M)
        tmp = cytnx.Contract(tmp, self.R)

        out = tmp
        out.relabels_(['lup', 'ip', 'rup'], ['l', 'i', 'r'])

        for j in range(len(self.ortho)):
            ortho = self.ortho[j]
            overlap = cytnx.Contract(ortho, v)
            out += self.ortho_w[j] * overlap.item() * ortho.Dagger()

        return out


# ======================== Helper Functions ========================

def get_sweeping_sites(N, numCenter):
    """Get sweeping site sequence"""
    '''
    it related to ranges and lr in func dmrg.
    if lr==0 then sweep direction :r->l, then, we need toRight of func svd2 to False.
    if lr==1 then sweep l->r, then, we need toRight of func svd2 to True.
    '''
    if numCenter == 2:
        return [range(N-2, -1, -1), range(N-1)]
    elif numCenter == 1:
        return [range(N-1, -1, -1), range(N)]  #lr[0] r->l, lr[1] l->r , 0:False, 1:True
    else:
        raise ValueError(f"Unsupported numCenter: {numCenter}")


def get_eff_H(LR, H, p, numCenter):
    """Get effective Hamiltonian"""
    if numCenter == 2:
        return eff_2s_Hamilt(LR[p], H[p], H[p+1], LR[p+2])
    elif numCenter == 1:
        return eff_1s_Hamilt(LR[p], H[p], LR[p+1])
    else:
        raise ValueError(f"Unsupported numCenter: {numCenter}")


def get_eff_psi(psi, p, numCenter):
    """Get effective wavefunction"""
    if numCenter == 2:
        A1 = psi[p].relabels(['l','i','r'], ['l','i1','mid'])
        A2 = psi[p+1].relabels(['l','i','r'], ['mid','i2','r'])
        phi = cytnx.Contract(A1, A2)
        return phi.relabels(['l','i1','i2','r'])
    elif numCenter == 1:
        return psi[p].clone()
    else:
        raise ValueError(f"Unsupported numCenter: {numCenter}")


def truncate_svd2(T, rowrank, toRight=True, maxdim=None, cutoff=0):
    orig_shape = T.shape()
    T.set_rowrank_(rowrank)
    s, U, Vt, errs = cytnx.linalg.Svd_truncate(T, keepdim=maxdim, err=cutoff, return_err=2)
    s = s / s.Norm().item()
    err = np.sum(ut.to_nparray(errs)) if hasattr(ut, 'to_nparray') else 0.0
    if toRight:
        Vt = cytnx.Contract(s, Vt)  
    else:
        U = cytnx.Contract(U, s)  
    
    if U.rank() == 3:
        U.set_labels(['l', 'i', 'r'])
    elif U.rank() == 2:
        U.set_labels(['l', 'r'])
    
    if Vt.rank() == 3:
        Vt.set_labels(['l', 'i', 'r'])
    elif Vt.rank() == 2:
        Vt.set_labels(['l', 'r'])
    return U, Vt, err



def orthogonalize_MPS_tensor(mps, i, toRight, maxdim=100000000, cutoff=0.0):
    res = copy.copy(mps)
    err = 0.0
    
    if toRight:
        if i != len(mps)-1:
            res[i], R, err = truncate_svd2(res[i], rowrank=2, toRight=toRight, maxdim=maxdim, cutoff=cutoff)
            R.relabels_(['l','r'], ['l','mid'])
            res[i+1] = res[i+1].relabels(['l','i','r'], ['mid','i','r'])
            res[i+1] = cytnx.Contract(R, res[i+1])
            if res[i+1].rank() == 3:
                res[i+1].set_labels(['l','i','r'])
    else:
        if i != 0:
            R, res[i], err = truncate_svd2(res[i], rowrank=1, toRight=toRight, maxdim=maxdim, cutoff=cutoff)
            R.relabels_(['l','r'], ['_','r']) 
            res[i-1] = res[i-1].relabels(['l','i','r'], ['l','i','_'])
            res[i-1] = cytnx.Contract(res[i-1], R)
            if res[i-1].rank() == 3:
                res[i-1].set_labels(['l','i','r'])
    return res, err

# ======================== Main DMRG Function ========================

def dmrg(numCenter, psi, H, L0, R0, maxdims, cutoff, maxIter=10, 
         ortho_mpss=[], weights=[], verbose=True):
    """
    Main DMRG Function
    
    Parameters:
    numCenter : int - Number of center sites (1 or 2)
    psi : list of UniTensor - Initial MPS
    H : list of UniTensor - MPO Hamiltonian
    L0, R0 : UniTensor - Boundary environment tensors
    maxdims : list of int - Maximum bond dimensions for each sweep
    cutoff : float - Truncation threshold
    maxIter : int - Maximum Lanczos iterations
    ortho_mpss : list of list of UniTensor - Orthogonal state MPS list
    weights : list of float - Orthogonal constraint weights
    verbose : bool - Whether to print detailed information
    
    Returns:
    psi : list of UniTensor - Optimized MPS
    ens : list of float - Energy per sweep
    terrs : list of float - Truncation error per sweep
    """
    # Parameter validation
    N = len(psi)
    if N != len(H):
        raise ValueError("MPS and MPO must have the same length")
    
    mpsut.check_mpo_bonds(H, L0, R0)
    mpsut.check_mps_bonds(psi)
    
    # Get sweeping site sequence
    ranges = get_sweeping_sites(N, numCenter)
    
    # Initialize environment tensors with verbose control
    LR = LR_envir_tensors_mpo(N, L0, R0, verbose=False)
    LR.update_LR(psi, psi, H, N-1)
    
    # Initialize orthogonal state environments
    LR_ortho = []
    for omps in ortho_mpss:
        lr = LR_envir_tensors_mps(N, psi, omps, verbose=False)
        lr.update_LR(psi, omps, N-1)
        LR_ortho.append(lr)
    
    # Result storage
    ens, terrs = [], []
    N_update = len(ranges[0]) + len(ranges[1])
    
    # Main sweeping loop
    for k, chi in enumerate(maxdims):
        if verbose:
            print(f'Sweep {k}, chi={chi}, cutoff={cutoff}')
        
        terr = 0.0
        sweep_energy = 0.0
        
        for lr in [0, 1]:
            direction = "right->left" if lr == 1 else "left->right"
            if verbose:
                print(f'  Direction: {direction}')
            
            for p in ranges[lr]:
                toRight = (lr == 1)
                #if verbose:
                    ## Clearer site indication
                #    if numCenter == 2:
                #        print(f"    Optimizing sites {p} and {p+1}")
                #    else:
                #        print(f"    Optimizing site {p}")
                
                # Update orthogonal state environments
                for j in range(len(ortho_mpss)):
                    if numCenter == 2:
                        LR_ortho[j].update_LR(psi, ortho_mpss[j], p, p+1)
                    elif numCenter == 1:
                        LR_ortho[j].update_LR(psi, ortho_mpss[j], p)
                
                # Prepare effective wavefunction
                phi = get_eff_psi(psi, p, numCenter)
                
                # Update environment tensors
                if numCenter == 2:
                    LR.update_LR(psi, psi, H, p, p+1)
                elif numCenter == 1:
                    LR.update_LR(psi, psi, H, p)
                
                # Construct effective Hamiltonian
                effH = get_eff_H(LR, H, p, numCenter)
                
                # Add orthogonal constraints
                for j in range(len(ortho_mpss)):
                    omps = ortho_mpss[j]
                    weight = weights[j]
                    oLR = LR_ortho[j]
                    
                    if numCenter == 2:
                        effH.add_orthogonal(oLR[p], omps[p], omps[p+1], oLR[p+2], weight)
                    elif numCenter == 1:
                        effH.add_orthogonal(oLR[p], omps[p], oLR[p+1], weight)
                
                # Lanczos ground state solver
                enT, phi = cytnx.linalg.Lanczos(
                    effH, phi, method="Gnd", Maxiter=maxIter, CvgCrit=100000
                )
                #phi = phi / phi.Norm().item()
                en = enT.item()
                sweep_energy = en
                
                # Update tensors
                if numCenter == 2:
                    phi.set_labels(['l','i1','i2','r'])
                    U, Vt, err = truncate_svd2(
                        phi, rowrank=2, toRight=toRight, maxdim=chi, cutoff=cutoff
                    )
                    terr += err
                    
                    psi[p] = U
                    psi[p+1] = Vt
                    
                    LR.delete(p)
                    LR.delete(p+1)

                elif numCenter == 1:
                    psi[p] = phi
                    psi, ortho_err = orthogonalize_MPS_tensor(psi, p, toRight, maxdim=chi, cutoff=cutoff)
                    terr += ortho_err
                    # Delete current and adjacent environments
                    LR.delete(p)

        if verbose:
            print(f'  Energy: {sweep_energy:.10f}, Trunc err: {terr / N_update:.2e}')
        ens.append(sweep_energy)
        terrs.append(terr / N_update)
    return psi, ens, terrs
