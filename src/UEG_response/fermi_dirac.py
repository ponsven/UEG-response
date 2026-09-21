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
from scipy.integrate import quad
from scipy.optimize import root_scalar
from numba import njit

@njit
def fd_0(eta):
    """
    Calculate (normalised) FD integral for k = 0.
    """
    return np.log1p(np.exp(eta))

def f1D_fermi_dirac(qz, mu, hbar, m, beta, ms=2):
  """
    Fermi-Dirac distribution integrated over 2 dimentions including spin degeneracy.
    Argumnsts:
      qz   -- Momentum in remaining dimention, shape (n,)
      mu   -- Chemical potential
      hbar -- Reduced Planks constant
      m    -- Mass of particle
      beta -- Inverse temperature in energy unts.
    Optional:
      ms -- Spin multiplicity of particle, ms=2 appropriet for electrons.
    Output:
      Evaluation of the distribution function, shape (n,).
  """
  # Multiplicity based on spin and pre-factor
  pre = 2.0*np.pi*ms / (2*np.pi*hbar)**3 * (m / beta)

  # Offest in FD integral
  x = beta * ( mu - np.square(qz)/(2*m) )
  return pre * fd_0(x)

def df1D_fermi_dirac(qz, mu, hbar, m, beta, ms=2, dx=1e-4):
  """
    Fermi-Dirac distribution integrated over 2 dimentions including spin degeneracy.
    Argumnsts:
      qz   -- Momentum in remaining dimention, shape (n,)
      mu   -- Chemical potential
      hbar -- Reduced Planks constant
      m    -- Mass of particle
      beta -- Inverse temperature in energy unts.
    Optional:
      ms -- Spin multiplicity of particle, ms=2 appropriet for electrons.
    Output:
      Evaluation of the distribution function, shape (n,).
  """
  # Vector notation
  qz = np.atleast_1d(np.array(qz))

  # Multiplicity based on spin and pre-factor
  pre = -2.0*np.pi*ms / (2*np.pi*hbar)**3 * qz

  # Offest in FD integral
  x = beta * ( mu - np.square(qz)/(2*m) )

  # Finite size difference
  x_u = x * (1 + dx)
  x_d = x * (1 - dx)
  # Correcr in case of x == 0
  idx = (x == 0)
  x_u[idx] =  dx
  x_d[idx] = -dx

  # Result
  res = pre * (fd_0(x_u) - fd_0(x_d))/(x_u - x_d)

  if (len(res) == 1):
    return res[0]
  else:
    return res

def compute_chemical_potential(n, hbar, m, beta, reltol=1e-6, ms=2):
  """
    Computes the chamical potantial for an ideal non-interacting system of spin s particels.
    Argumsnts:
      n    -- Total density
      hbar -- Reduced Plank's constant.
      m    -- Mass of particle
    Optional:
      reltol -- Relative tolerance of solution.
      ms     -- Spin multiplicity of the system, defult 2.
    Output:
      mu -- Chemical potential
  """
  # Relative density difference for given mu.
  density_diff  = lambda mu : 2.0*quad(lambda qz : f1D_fermi_dirac(qz,mu,hbar,m,beta,ms=ms), 
                                       0.0, np.inf, epsrel=0.1*reltol)[0]/n - 1.0

  # Numerical solution. TODO: Add better initial gueess.
  res = root_scalar(density_diff, x0=0.0, x1=1.0, rtol=reltol)
  if not res.converged:
    raise ValueError('Could not compute chemical potential, numeric solver did not converge.')

  mu = res.root
  return mu
