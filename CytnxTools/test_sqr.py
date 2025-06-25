import numpy as np
import npmps
import ncon

loc_mps = np.random.rand(4, 2, 4).astype(dtype)

loc_mpo, Lmpo, Rmpo = npmps.mps_func_to_mpo (loc_tensor)

loc_mpo2, Lmpo2, Rmpo2 = prod_MPO(loc_mpo, Lmpo, Rmpo, loc_mpo, Lmpo, Rmpo)

loc_mpo2, Lmpo2, Rmpo2 = npmps.compress_MPO(loc_mpo2, Lmpo2, Rmpo2,1e-8)
