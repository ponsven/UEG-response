# This file is part of the UEG-response code for computations of linear and nonlinear response functions.
# Copyright (C) 2026  Pontus Svensson

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import numpy as np
import numba
from numba import njit
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

def _norm_single(v):
  return np.sqrt( np.sum(v*v) )

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

@njit
def filter_close_values(x, d):
  sorted_x = np.sort(x)
  res = np.empty(len(x), dtype=numba.boolean)
  res[0] = True
  i = 0
  j = 1
  while j < len(x):
    if (sorted_x[j] - sorted_x[i]) <= d:
        res[j] = False
    else:
        res[j] = True
        i = j
    j += 1

  return sorted_x[res]

def _get_points_I(low, high, points_all):
    # Select points in the intervall in question.
    idx = np.logical_and(points_all > low, points_all < high)
    points = points_all[idx]
    if (len(points) == 0):
       return None
    else:
       return points
