import os, sys 
#root = os.getenv('/home/chiamin/project/2023/qtt/JhengWei/INSTALL/xfac/build/python/')
#sys.path.insert(0,root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../CytnxTools')))
sys.path.append('/home/jerrychen/xfac/build/python/')
#sys.path.append('/home/jerrychen/Desktop/My_Work/TN_Numerical/qtt_jerry/xfac_cytnx/build/python/')
import xfacpy
import time
import matplotlib.pyplot as plt
import numpy as np

def cc_inds_to_x (inds, rescale, shift):
    res = inds[0]
    for i in range(1,len(inds)):
        res += inds[i] * 2**i
    return rescale * res + shift


def get_dx (inds):
    x1 = inds[::2]
    x2 = inds[1::2]
    x1 = cc_inds_to_x (x1)
    x2 = cc_inds_to_x (x2)
    return abs(x2-x1)

def one_over_r_1D (inds, fac, rescale, shift, cutoff):
    N = len(inds)


    x = cc_inds_to_x (inds, rescale, shift)
    return np.where(
            np.abs(x) < cutoff, 
            -fac/(0.5*cutoff), 
            -fac / np.abs(x)
        )


def one_over_r_2D (inds, fac, rescale, shift, cutoff, R):
    N = len(inds)
    inds_x = inds[:N//2]
    inds_y = inds[N//2:]

    x = cc_inds_to_x (inds_x, rescale, shift)
    y = cc_inds_to_x (inds_y, rescale, shift)
    r = np.sqrt((x-R/2)**2+y**2)
    return np.where(
        r < cutoff, 
        -fac/(np.sqrt(0.5)*cutoff),  # Approximation for small r
        -fac / r      # Standard form for larger r
    )
    
def interact_2D(inds, fac, rescale, shift, cutoff, R):
    xx1 = inds[::2]
    xx2 = inds[1::2]
    #xx1 = inds[:len(inds)//2]
    #xx2 = inds[len(inds)//2:]
    N = len(xx1)//2

    x1 = cc_inds_to_x(xx1[:N], rescale, shift)
    y1 = cc_inds_to_x(xx1[N:], rescale, shift)
    x2 = cc_inds_to_x(xx2[:N], rescale, shift)
    y2 = cc_inds_to_x(xx2[N:], rescale, shift)

    dx = x2 - x1
    dy = y2 - y1
    r = 0.5*(dx**2 + dy**2)
    return r
    
def interact_1D(inds, fac, rescale, shift, cutoff, R):
    xx1 = inds[::2]
    xx2 = inds[1::2]
    N = len(xx1)//2

    x1 = cc_inds_to_x(xx1[:N], rescale, shift)
    x2 = cc_inds_to_x(xx2[:N], rescale, shift)

    dx = x2 - x1
    r = 0.5*(dx**2)
    return r

def one_over_xdr_2D (inds, rescale, shift, cutoff):
    N = len(inds)
    inds_x = inds[:N//2]
    inds_y = inds[N//2:]

    x = cc_inds_to_x (inds_x, rescale, shift)
    y = cc_inds_to_x (inds_y, rescale, shift)
    r = np.sqrt(x*x+y*y)
    return np.where(
        r < cutoff, 
        2*(2*(x/(np.sqrt(0.5)*cutoff))**2-1),  # Approximation for small r
        2*(2*(x / r)**2- 1)      # Standard form for larger r
    )


def one_over_r_3D (inds, fac, rescale, shift, cutoff):
    N = len(inds)
    Ni = N//3
    inds_x = inds[:Ni]
    inds_y = inds[Ni:2*Ni]
    inds_z = inds[2*Ni:]

    x = cc_inds_to_x (inds_x, rescale, shift)
    y = cc_inds_to_x (inds_y, rescale, shift)
    z = cc_inds_to_x (inds_z, rescale, shift)
    r = np.sqrt(x*x+y*y+z*z)
    return np.where(
        r < cutoff, 
        -fac/(np.sqrt(0.75)*cutoff),  # Approximation for small r
        -fac / r      # Standard form for larger r
    )

def dx_inverse (inds, factor, cutoff):
    dx = get_dx (inds)
    if dx < cutoff:
        dx = cutoff
    return factor * dx**(-1)

def funQ1_2D (inds):
    dx,dy = get_dxdy (inds)
    r = np.sqrt(dx**2 + dy**2)
    return r**(-1)


def mps_x1(x0, x1, nsite):
  mps = []
  t0 = (x0/nsite)*np.array([1.,1.])
  dx = x1-x0

  for it in range(nsite):
    ten = np.zeros((2,2,2))
    ten[0,:,0] = [1.,1.]
    ten[1,:,1] = [1.,1.]

    fac = dx/(2**(nsite-it))
    tx = t0 + [0., fac]
    ten[1,:,0] = tx
    mps.append(ten)

  mps[0] = mps[0][1:2,:,:]
  mps[-1] = mps[-1][:,:,0:1]
  return mps 

def inds_to_x (inds, x0, dx):
  s0 = sum([b<<i for i, b in enumerate(inds[::+1])])
  return x0 + dx*s0

def num_to_inds(num, nsite):
  inds = [int(it) for it in np.binary_repr(num, width=nsite)]
  return inds[::-1]

def eval_mps (mps, inds, envL=np.array([1.]), envR=np.array([1.])):
  nsite = len(mps)
  val = envL
  for it in range(nsite):
    ind = inds[it]
    mat = mps[it][:,ind,:]
    val = val @ mat 
  val = val @ envR
  return val 

def xfac_to_npmps (mpsX, nsite):
  mps = [None for i in range(nsite)]
  mps = mpsX.core
  return mps 

def plotF(mps, target_func, rescale, shift):
  plt.rcParams['figure.figsize'] = 6,3 
  SMALL_SIZE = 18
  MEDIUM_SIZE = 20
  BIGGER_SIZE = 16
  plt.rc('font', size=SMALL_SIZE)    
  plt.rc('axes', titlesize=SMALL_SIZE)    
  plt.rc('axes', labelsize=MEDIUM_SIZE)
  plt.rc('xtick', labelsize=SMALL_SIZE)    
  plt.rc('ytick', labelsize=SMALL_SIZE)    
  plt.rc('legend', fontsize=16)    
  plt.rc('figure', titlesize=BIGGER_SIZE)  

  xs,ffs,ffs2 = [],[],[]
  for it in range(2**nsite):
    inds = num_to_inds(it, nsite)
    ff = eval_mps(mps,inds)
    ff2 = target_func(inds)
    xx = cc_inds_to_x(inds, rescale, shift) 
    xs.append(xx)
    ffs.append(ff)
    ffs2.append(ff2)

  plt.plot(xs,ffs, c='r', marker='+', ls='None', markersize=4)
  plt.plot(xs,ffs2, c='k', marker='x', ls='None', markersize=4)

  #plt.axis([x0,x1, 1E-5, +1E5])
  #plt.yscale('log')
  #plt.ylim(ymin=-100)
  plt.show()
  #plt.tight_layout()
  #plt.savefig('tci_inv.pdf', dpi = 600, transparent=True)
  #plt.close()

def write_mps (fname, mps):
    tmp = np.array(mps, dtype=object)
    np.save(fname,tmp, allow_pickle=True)

def load_mps (fname):
    tmp = np.load(fname, allow_pickle=True)
    return list(tmp)

def run_tci (target_func, nsite, rescale, shift, cutoff, factor, R):
    dimP = 2      # Physical dimension
    incD = 2      # increasing bond dimension
    maxD = 80 + incD

    pm = xfacpy.TensorCI2Param()
    pm.pivot1 = np.random.randint(2, size=nsite)
    pm.reltol = 1e-20
    pm.bondDim = 2
    tci = xfacpy.TensorCI2(target_func, [dimP]*nsite, pm)

    while tci.param.bondDim < maxD:
        tci.param.bondDim = tci.param.bondDim + incD

        t0 = time.time()
        tci.iterate(2,2)
        err0 = tci.pivotError[0]
        err1 = tci.pivotError[-1]

        print("{0:20.3e} {1:20.3e} {2:20.3e} {3:20.2e}".
             format(err0, err1, err1/err0, time.time()-t0))

        if (err1/err0 < 1e-8):
            break

    # Output MPS
    mps = xfac_to_npmps (tci.tt, nsite)
    write_mps (f'fit{nsite}_{R}.mps', mps)
    return mps

def run_tci_xdr (target_func, nsite, rescale, shift, cutoff, factor):
    dimP = 2      # Physical dimension
    incD = 2      # increasing bond dimension
    maxD = 80 + incD

    pm = xfacpy.TensorCI2Param()
    pm.pivot1 = np.random.randint(2, size=nsite)
    pm.reltol = 1e-20
    pm.bondDim = 2
    tci = xfacpy.TensorCI2(target_func, [dimP]*nsite, pm)

    while tci.param.bondDim < maxD:
        tci.param.bondDim = tci.param.bondDim + incD

        t0 = time.time()
        tci.iterate(2,2)
        err0 = tci.pivotError[0]
        err1 = tci.pivotError[-1]

        print("{0:20.3e} {1:20.3e} {2:20.3e} {3:20.2e}".
             format(err0, err1, err1/err0, time.time()-t0))

        if (err1/err0 < 1e-8):
            break

    # Output MPS
    mps = xfac_to_npmps (tci.tt, nsite)
    write_mps (f'fit{nsite}_xdr.mps', mps)
    return mps

if __name__ == '__main__':
    nsite = int(sys.argv[1])
    rescale = float(sys.argv[2])
    shift = float(sys.argv[3])
    cutoff = float(sys.argv[4])
    factor = float(sys.argv[5])
    R = float(sys.argv[6])

    if '--1D_one_over_r' in sys.argv:
        def target_func (inds):
            return one_over_r_1D (inds, factor, rescale, shift, cutoff)
        mps = run_tci (target_func, nsite, rescale, shift, cutoff, factor)
        plotF(mps, target_func, rescale, shift)
    elif '--1D_interact' in sys.argv:
        def target_func (inds):
            return interact_1D (inds, factor, rescale, shift, cutoff, R)
        mps = run_tci (target_func, nsite, rescale, shift, cutoff, factor, R)
        #plotF(mps, target_func, rescale, shift)  
    elif '--2D_one_over_r' in sys.argv:
        def target_func (inds):
            return one_over_r_2D (inds, factor, rescale, shift, cutoff, R)
        mps = run_tci (target_func, nsite, rescale, shift, cutoff, factor, R)
        #plotF(mps, target_func, rescale, shift)
    elif '--2D_interact' in sys.argv:
        def target_func (inds):
            return interact_2D (inds, factor, rescale, shift, cutoff, R)
        mps = run_tci (target_func, nsite, rescale, shift, cutoff, factor, R)
        #plotF(mps, target_func, rescale, shift)    
    elif '--2D_one_over_xdr' in sys.argv:
        def target_func (inds):
            return one_over_xdr_2D (inds, rescale, shift, cutoff)
        mps = run_tci_xdr (target_func, nsite, rescale, shift, cutoff, factor)
        #plotF(mps, target_func, rescale, shift)
    elif '--3D_one_over_r' in sys.argv:
        def target_func (inds):
            return one_over_r_3D (inds, factor, rescale, shift, cutoff)
        mps = run_tci (target_func, nsite, rescale, shift, cutoff, factor)
    else:
        raise Exception
