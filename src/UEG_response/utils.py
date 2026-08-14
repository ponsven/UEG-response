import numpy as np
import numba
from numba import cfunc,carray
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable

# Taken from: https://stackoverflow.com/questions/51109429/how-to-use-numba-to-perform-multiple-integration-in-scipy-with-an-arbitrary-numb
# With thanks to 'Aboottogo'. 
def jit_integrand_function(integrand_function):
    jitted_function = numba.jit(integrand_function, nopython=True)
    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        values = carray(xx,n)
        return jitted_function(values)
    return LowLevelCallable(wrapped.ctypes)

def _norm(v):
  return np.sqrt( np.sum(v*v, axis=1) )

def _cos_angle(v1, v2):
  norm1 = _norm(v1)
  norm2 = _norm(v2)

  tmp = np.zeros(shape=norm1.shape)
  # If one of the vectors are zero, just return 1.0
  idx_zero = np.logical_or(norm1 == 0.0, norm2 == 0.0)
  tmp[idx_zero] = 1.0
  # All other cases
  idx_not_zero = np.logical_not(idx_zero)
  tmp[idx_not_zero] = np.sum(v1[idx_not_zero, :]*v2[idx_not_zero, :], axis=1) / (norm1[idx_not_zero]*norm2[idx_not_zero])
  # Correct for rounding errors
  tmp[tmp >=  1.0] =  1.0
  tmp[tmp <= -1.0] = -1.0
  return tmp