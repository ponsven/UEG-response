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
from .fermi_dirac import compute_chemical_potential
from .I_functions import _I_1_inner, _I_2_inner, _I_2_CV
from .utils import _norm, _cos_angle
from .Maldague_quadratic import _ideal_quadratic_response_Maldague
from .zeroth_harmonic import _chi0_k2_0_Maldague, _chi0_k1_0_k2_0_Maldague

def ideal_linear_response(omega, k, m, hbar, n, beta, ms=2,
                          reltol=1e-6, abstol=1e-8, limit=50, eta_log=1e-4, tol_upper=1e-8, points_n=3, force_output=False):
  """
    Computes the ideal linear response coefficents. Units per energy per volume.

    :param omega:  Angular frequencies for evaluation, shape (n, ) or ()
    :param k:      Wave number for evaluation, shape (n, ) or ()
    :param m:      Mass of particle
    :param hbar:   Reduced Plank's constant.
    :param n:      Number density.
    :param beta:   Inverse temperature in energy units.

    :param ms:        Spin multiplicity of particle (defult: 2).
    :param reltol:    Relative tolerance for solution.
    :param abstol:    Absolute tolerance for solution.
    :param limit:     'limit' passed to 'quad'
    :param eta_log:   Size of region around log-poles which are approximated analytically.
    :param tol_upper: Stop integration when the FD distribution is below this value.
    :param points_n:  Points which helps numerical integration highliting points where FD is steap.
                 Points are given by:  [(mu/EF + n*theta) for n in range(-points_n, points_n+1)]
                 Points are also generated around the log- and sqrt-poles.
    :param force_output: If true, results will be outputted even if convergence is not garanteed.

    :return linear_chi_0: Ideal linear reponse function, shape (n, )
    """
  # Setup for array operations
  omega  = np.atleast_1d(np.array(omega))
  k      = np.atleast_1d(np.array(k))
  omega, k = np.broadcast_arrays(omega, k)

  # Test inputs.
  if (np.any(k <= 0.0)):
      raise ValueError(f'k must be posetive')

  if (np.iscomplexobj(omega)):
      raise ValueError(f"Complex frequncies are not suported.")

  if (n <= 0.0):
    raise ValueError(f"Provided density (%g) must be posetive."%(n))

  if (beta <= 0.0):
    raise ValueError(f"Provided inverse temperature (%g) must be posetive."%(beta))

  if (m <= 0.0):
    raise ValueError(f"Provided mass (%g) must be posetive."%(m))

  if (hbar <= 0.0):
    raise ValueError(f"Provided hbar (%g) must be posetive."%(hbar))

  # Compuet chemical potential.
  eta = beta * compute_chemical_potential(n, hbar, m, beta, reltol=reltol, ms=ms)

  # Compute relevant scales
  qF = (3*np.pi**2*n)**(1/3)
  EF = hbar**2 * qF**2 / (2*m)
  inv_theta = beta * EF

  # Compute the ideal response in terms of I1-integrals.
  linear_chi_0 = np.zeros(shape=omega.shape, dtype=complex)

  # First term
  y = np.abs(k)/qF
  z = hbar*(omega)/EF
  linear_chi_0 -= _I_1_inner(y, z, 1,  qF, EF, eta, inv_theta, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, force_output)

  # Second term
  y = np.abs(-k)/qF
  z = hbar*(-omega)/EF
  linear_chi_0 -= _I_1_inner(y, z, -1, qF, EF, eta, inv_theta, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, force_output)

  return linear_chi_0

# Compute quadratic response susing the direct method.
def _ideal_quadratic_response_direct(omega1, k1_vec, omega2, k2_vec, hbar, qF, EF, eta, inv_theta, eta_pol, eta_sqrt, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, dx, use_parallel, force_output):
  quadratic_chi_0 = np.zeros(shape=omega1.shape, dtype=complex)
  
  # First term:
  y1 = _norm(k2_vec)/qF
  z1 = hbar*(omega2)/EF
  y2 = _norm(k1_vec+k2_vec)/qF
  z2 = hbar*(omega1+omega2)/EF
  csTheta12 = _cos_angle(k2_vec, k1_vec+k2_vec)
  quadratic_chi_0 += 0.5*_I_2_inner(y1, z1, 1, y2, z2, 1, csTheta12, qF, EF, eta, inv_theta, eta_pol, eta_sqrt, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, dx, use_parallel, force_output)

  # Second term:
  y1 = _norm(-k2_vec)/qF
  z1 = hbar*(-omega2)/EF
  y2 = _norm(k1_vec)/qF
  z2 = hbar*(omega1)/EF
  csTheta12 = _cos_angle(-k2_vec, k1_vec)
  quadratic_chi_0 += 0.5*_I_2_inner(y1, z1, -1, y2, z2, 1, csTheta12, qF, EF, eta, inv_theta, eta_pol, eta_sqrt, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, dx, use_parallel, force_output)
  
  # Third term:
  y1 = _norm(-k1_vec-k2_vec)/qF
  z1 = hbar*(-omega1-omega2)/EF
  y2 = _norm(-k1_vec)/qF
  z2 = hbar*(-omega1)/EF
  csTheta12 = _cos_angle(-k1_vec-k2_vec, -k1_vec)
  quadratic_chi_0 += 0.5*_I_2_inner(y1, z1, -1, y2, z2, -1, csTheta12, qF, EF, eta, inv_theta, eta_pol, eta_sqrt, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, dx, use_parallel, force_output)
  
  # Fourth term:
  y1 = _norm(k1_vec)/qF
  z1 = hbar*(omega1)/EF
  y2 = _norm(k2_vec+k1_vec)/qF
  z2 = hbar*(omega2+omega1)/EF
  csTheta12 = _cos_angle(k1_vec, k2_vec+k1_vec)
  quadratic_chi_0 += 0.5*_I_2_inner(y1, z1, 1, y2, z2, 1, csTheta12, qF, EF, eta, inv_theta, eta_pol, eta_sqrt, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, dx, use_parallel, force_output)
  
  # Fift term:
  y1 = _norm(-k1_vec)/qF
  z1 = hbar*(-omega1)/EF
  y2 = _norm(k2_vec)/qF
  z2 = hbar*(omega2)/EF
  csTheta12 = _cos_angle(-k1_vec, k2_vec)
  quadratic_chi_0 += 0.5*_I_2_inner(y1, z1, -1, y2, z2, 1, csTheta12, qF, EF, eta, inv_theta, eta_pol, eta_sqrt, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, dx, use_parallel, force_output)
  
  # Sixth term:
  y1 = _norm(-k2_vec-k1_vec)/qF
  z1 = hbar*(-omega2-omega1)/EF
  y2 = _norm(-k2_vec)/qF
  z2 = hbar*(-omega2)/EF
  csTheta12 = _cos_angle(-k2_vec-k1_vec, -k2_vec)
  quadratic_chi_0 += 0.5*_I_2_inner(y1, z1, -1, y2, z2, -1, csTheta12, qF, EF, eta, inv_theta, eta_pol, eta_sqrt, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, dx, use_parallel, force_output)

  return quadratic_chi_0


def ideal_diagonal_quadratic_response(omega, k, m, hbar, n, beta, ms=2, reltol=1e-6, abstol=1e-8, limit=50, eta_log=1e-4, tol_upper=1e-8, points_n=3, force_output=False):
  """
  Computes the ideal quadratic response function for equal first and second argument. Units per energy**2 per volume.

  :param omega:  Angular frequencies for evaluation, shape (n, ) or ()
  :param k:      Wave number for evaluation, shape (n, ) or ()
  :param m:      Mass of particle
  :param hbar:   Reduced Plank's constant.
  :param n:      Density, for the computation of f1D if not given.
  :param beta:   Inverse temperature in energy units, for the computation of f1D if not given.

  :param ms:           Spin multiplicity of particle, defult 2.
  :param reltol:       Relative tolerance for solution.
  :param abstol:       Absolute tolerance for solution.
  :param limit:        'limit' as passed to 'quad'.
  :param eta_log:      eta_log for the linear response computation, see 'ideal_linear_response'.
  :param tol_upper:    tol_upper for the linear response computation, see 'ideal_linear_response'.
  :param points_n:     points_n for the linear response computation, see 'ideal_linear_response'.
  :param force_output: If true, results will be outputted even if convergence is not garanteed.

  : return quadratic_chi_0: ideal quadratic reponse function, shape (n, ) or ()
  """
  chi_0        = ideal_linear_response( omega,    k, m, hbar, n, beta, ms=ms, reltol=reltol, abstol=abstol, limit=limit, eta_log=eta_log, tol_upper=tol_upper, points_n=points_n, force_output=force_output)
  chi_0_double = ideal_linear_response(2*omega, 2*k, m, hbar, n, beta, ms=ms, reltol=reltol, abstol=abstol, limit=limit, eta_log=eta_log, tol_upper=tol_upper, points_n=points_n, force_output=force_output)

  quadratic_chi_0 = 2*m/(hbar**2*k**2) * (chi_0_double - chi_0)
  return quadratic_chi_0


def ideal_quadratic_response(omega1, k1, omega2, k2, csTheta,
                             m, hbar, n, beta, method='direct', use_parallel=False, direction_k_0='static', ms=2,
                             reltol=1e-6, abstol=1e-8, limit=50, eta_pol=1e-6, eta_sqrt=1e-4, eta_log=1e-4, tol_upper=1e-8, lower=1e-6,
                             dx=1e-4, points_n=3, force_output=False):
  """
  Computes the ideal quadratic response coefficents. Units per energy**2 per volume.

  :param omega1:  First angular frequencies for evaluation, shape (n, ) or ()
  :param k1:      First wave number for evaluation, shape (n, ) or ()
  :param omega2:  Second angular frequencies for evaluation, shape (n, ) or ()
  :param k2:      Second wave number for evaluation, shape (n, ) or ()
  :param csTheta: Angle between k-vectors
  :param m:       Mass of particle
  :param hbar:    Reduced Plank's constant.
  :param n:       Number density.
  :param beta:    Inverse temperature in energy units.

  :param method:        The method used to performe the evaluation (defult: 'direct'). 
  :param use_parallel:  Set to 'True' if implementation for paralle k-vectors should be used (defult: False).
  :param direction_k_0: Either 'static' or 'dynamic'. Determines the treatment when ki==0, omega1==0 and omega2==0.
                        If 'static', ki -> 0 is taken for the static response (defult).
                        If 'dynamic', omegai -> 0 is taken for the response where ki=0.
  :param ms:        Spin multiplicity of particle (defult: 2).
  :param reltol:    Relative tolerance for solution.
  :param abstol:    Absolute tolerance for solution.
  :param limit:     'limit' passed to 'quad'
  :param eta_pol:   eta for principla value evaluation, see 'principal_value_integration_f_over_x'
  :param eta_sqrt:  Size of region around sqrt-poles which are approximated analytically.
  :param eta_log:   Size of region around log-poles which are approximated analytically.
  :param tol_upper: Stop integration when the FD distribution is below this value.
  :param lower:     Lower limit of integration when method='maldague'
  :param dx:        Finite difference parameter used for differentiation of Fermi-Dirac integrals.
  :param points_n:  Points which helps numerical integration highliting points where FD is steap.
               Points are given by:  [(mu/EF + n*theta) for n in range(-points_n, points_n+1)]
               Points are also generated around the log- and sqrt-poles.
  :param force_output: If true, results will be outputted even if convergence is not garanteed.

  :return quadratic_chi_0: Ideal quadratic reponse function, shape (n, )
  """
  # Setup for array operations
  omega1  = np.atleast_1d(np.array(omega1))
  k1      = np.atleast_1d(np.array(k1))
  omega2  = np.atleast_1d(np.array(omega2))
  k2      = np.atleast_1d(np.array(k2))
  csTheta = np.atleast_1d(np.array(csTheta))

  omega1, k1, omega2, k2, csTheta = np.broadcast_arrays(omega1, k1, omega2, k2, csTheta)

  # Test the input.
  if (np.any(k1 < 0.0)):
    raise ValueError(f'k1 must be posetive or zero')
  if (np.any(k2 < 0.0)):
    raise ValueError(f'k2 must be posetive or zero')

  if (np.iscomplexobj(omega1) or np.iscomplexobj(omega2)):
    raise ValueError(f"Complex frequncies are not suported.")

  if (n <= 0.0):
    raise ValueError(f"Provided density (%g) must be posetive."%(n))

  if (beta <= 0.0):
    raise ValueError(f"Provided inverse temperature (%g) must be posetive."%(beta))

  if (m <= 0.0):
    raise ValueError(f"Provided mass (%g) must be posetive."%(m))

  if (hbar <= 0.0):
    raise ValueError(f"Provided hbar (%g) must be posetive."%(hbar))

  if not (direction_k_0 == 'static' or direction_k_0 == 'dynamic'):
    raise ValueError(f"'direction_k_0' (%s) must be either 'static' or 'dynamic'."%(direction_k_0))

  # Compuet chemical potential.
  eta = beta * compute_chemical_potential(n, hbar, m, beta, reltol=reltol, ms=ms)

  # Compute relevant scales
  qF = (3*np.pi**2*n)**(1/3)
  EF = hbar**2 * qF**2 / (2*m)
  inv_theta = beta * EF

  # Angle computation are performed using the vector description.
  # k1 is assumed to align with z-axis
  k1_vec = np.zeros(shape=(len(k1),3))
  k1_vec[:, 2] = k1
  # k2 is assumed to be in the zx-plane
  k2_vec = np.zeros(shape=(len(k2),3))
  k2_vec[:, 0] = k2 * np.sqrt(1.0 - csTheta**2)
  k2_vec[:, 2] = k2 * csTheta

  # Allocate the result
  quadratic_chi_0 = np.zeros(shape=omega1.shape, dtype=complex)

  # Find special cases that are explcitly treated.
  idx_static = np.logical_and( (omega1==0.0), (omega2==0.0) )
  # k1 == 0 and k2 == 0
  idx0 = np.logical_and( np.logical_and((k1==0.0), (k2==0.0)), idx_static)
  if (np.any(idx0)):
    if (direction_k_0 == 'static'):
      quadratic_chi_0[idx0] = _chi0_k1_0_k2_0_Maldague(eta, beta, hbar, m, lower, reltol, abstol, limit, tol_upper, points_n, ms, force_output=force_output)
    else:
      quadratic_chi_0[idx0] = 0.0

  # k1 == 0
  idx1 = np.logical_and( (k1==0.0), idx_static )
  idx1 = np.logical_and( idx1, np.logical_not(idx0) )
  if (np.any(idx1)):
    if (direction_k_0 == 'static'):
      quadratic_chi_0[idx1] = _chi0_k2_0_Maldague(k2[idx1]/qF, qF, eta, beta, hbar, m, lower, reltol, abstol, limit, tol_upper, points_n, ms, force_output=force_output)
    else:
      quadratic_chi_0[idx1] = 0.0

  # k2 == 0
  idx2 = np.logical_and( (k2==0.0), idx_static )
  idx2 = np.logical_and( idx2, np.logical_not(idx0) )
  if (np.any(idx2)):
    if (direction_k_0 == 'static'):
      quadratic_chi_0[idx2] = _chi0_k2_0_Maldague(k1[idx2]/qF, qF, eta, beta, hbar, m, lower, reltol, abstol, limit, tol_upper, points_n, ms, force_output=force_output)
    else:
      quadratic_chi_0[idx2] = 0.0

  # k1 + k2 == 0
  k12 = np.sqrt(np.sum( (k1_vec + k2_vec)**2, axis=1))
  idx3 = np.logical_and( (k12==0.0), idx_static )
  idx3 = np.logical_and( idx3, np.logical_not(idx0) )
  if (np.any(idx3)):
    if (direction_k_0 == 'static'):
      # Use Kalman and Gu’s symmetry.
      quadratic_chi_0[idx3] = _chi0_k2_0_Maldague(k1[idx3]/qF, qF, eta, beta, hbar, m, lower, reltol, abstol, limit, tol_upper, points_n, ms, force_output=force_output)
    else:
      quadratic_chi_0[idx3] = 0.0

  # General case
  idx_other = np.logical_not( np.logical_or(np.logical_or(np.logical_or(idx0, idx1), idx2), idx3) )
  if (np.any(idx_other)):
    if (method == 'direct'):
      quadratic_chi_0[idx_other] = _ideal_quadratic_response_direct(omega1[idx_other], k1_vec[idx_other, :], omega2[idx_other], k2_vec[idx_other, :],
                                                                    hbar, qF, EF, eta, inv_theta,
                                                                    eta_pol, eta_sqrt, eta_log,
                                                                    reltol, abstol, limit, tol_upper, points_n, ms, dx, use_parallel, force_output)
    elif (method == 'maldague'):
      quadratic_chi_0[idx_other] = _ideal_quadratic_response_Maldague(k1_vec[idx_other, :], omega1[idx_other], k2_vec[idx_other, :], omega2[idx_other],
                                                                      eta, beta, hbar, m,
                                                                      lower, reltol, abstol, limit, tol_upper, points_n, ms, force_output=force_output)
    else:
      raise ValueError(f"The 'method' (%s) must be one of: 'direct' or 'maldague'."%(method))

  return quadratic_chi_0

def ground_state_ideal_quadratic_response(omega1, k1, omega2, k2, csTheta, m, hbar, n, ms=2):
    """
    Computes the ideal quadratic response coefficents in the ground state. Units per energy**2 per volume.

    :param omega1:  First angular frequencies for evaluation, shape (n, ) or ()
    :param k1:      First wave number for evaluation, shape (n, ) or ()
    :param omega2:  Second angular frequencies for evaluation, shape (n, ) or ()
    :param k2:      Second wave number for evaluation, shape (n, ) or ()
    :param csTheta: Angle between k-vectors
    :param m:       Mass of particle
    :param hbar:    Reduced Plank's constant.
    :param n:       Number density.

    :param ms:      Spin multiplicity of particle (defult: 2).

    :return quadratic_chi_0: Ideal quadratic reponse function, shape (n, )
    """
    # Setup for array operations
    omega1  = np.atleast_1d(np.array(omega1))
    k1      = np.atleast_1d(np.array(k1))
    omega2  = np.atleast_1d(np.array(omega2))
    k2      = np.atleast_1d(np.array(k2))
    csTheta = np.atleast_1d(np.array(csTheta))
  
    omega1, k1, omega2, k2, csTheta = np.broadcast_arrays(omega1, k1, omega2, k2, csTheta)
  
    # Test the input.
    if (np.any(k1 < 0.0)):
      raise ValueError(f'k1 must be posetive or zero')
    if (np.any(k2 < 0.0)):
      raise ValueError(f'k2 must be posetive or zero')
  
    if (np.iscomplexobj(omega1) or np.iscomplexobj(omega2)):
      raise ValueError(f"Complex frequncies are not suported.")
  
    if (n <= 0.0):
      raise ValueError(f"Provided density (%g) must be posetive."%(n))
  
    if (m <= 0.0):
      raise ValueError(f"Provided mass (%g) must be posetive."%(m))
  
    if (hbar <= 0.0):
      raise ValueError(f"Provided hbar (%g) must be posetive."%(hbar))
  
    # Compute relevant scales
    qF = (3*np.pi**2*n)**(1/3)
    EF = hbar**2 * qF**2 / (2*m)
  
    # Angle computation are performed using the vector description.
    # k1 is assumed to align with z-axis
    k1_vec = np.zeros(shape=(len(k1),3))
    k1_vec[:, 2] = k1
    # k2 is assumed to be in the zx-plane
    k2_vec = np.zeros(shape=(len(k2),3))
    k2_vec[:, 0] = k2 * np.sqrt(1.0 - csTheta**2)
    k2_vec[:, 2] = k2 * csTheta

    quadratic_chi_0 = np.zeros(shape=omega1.shape, dtype=complex)
      
    # First term:
    y1 = _norm(k2_vec)/qF
    z1 = hbar*(omega2)/EF
    y2 = _norm(k1_vec+k2_vec)/qF
    z2 = hbar*(omega1+omega2)/EF
    csTheta12 = _cos_angle(k2_vec, k1_vec+k2_vec)
    quadratic_chi_0 += 0.5*_I_2_CV(y1, z1, 1, y2, z2, 1, csTheta12, qF, EF, ms)
    
    # Second term:
    y1 = _norm(-k2_vec)/qF
    z1 = hbar*(-omega2)/EF
    y2 = _norm(k1_vec)/qF
    z2 = hbar*(omega1)/EF
    csTheta12 = _cos_angle(-k2_vec, k1_vec)
    quadratic_chi_0 += 0.5*_I_2_CV(y1, z1, -1, y2, z2, 1, csTheta12, qF, EF, ms)
      
    # Third term:
    y1 = _norm(-k1_vec-k2_vec)/qF
    z1 = hbar*(-omega1-omega2)/EF
    y2 = _norm(-k1_vec)/qF
    z2 = hbar*(-omega1)/EF
    csTheta12 = _cos_angle(-k1_vec-k2_vec, -k1_vec)
    quadratic_chi_0 += 0.5*_I_2_CV(y1, z1, -1, y2, z2, -1, csTheta12, qF, EF, ms)
      
    # Fourth term:
    y1 = _norm(k1_vec)/qF
    z1 = hbar*(omega1)/EF
    y2 = _norm(k2_vec+k1_vec)/qF
    z2 = hbar*(omega2+omega1)/EF
    csTheta12 = _cos_angle(k1_vec, k2_vec+k1_vec)
    quadratic_chi_0 += 0.5*_I_2_CV(y1, z1, 1, y2, z2, 1, csTheta12, qF, EF, ms)
      
    # Fift term:
    y1 = _norm(-k1_vec)/qF
    z1 = hbar*(-omega1)/EF
    y2 = _norm(k2_vec)/qF
    z2 = hbar*(omega2)/EF
    csTheta12 = _cos_angle(-k1_vec, k2_vec)
    quadratic_chi_0 += 0.5*_I_2_CV(y1, z1, -1, y2, z2, 1, csTheta12, qF, EF, ms)
      
    # Sixth term:
    y1 = _norm(-k2_vec-k1_vec)/qF
    z1 = hbar*(-omega2-omega1)/EF
    y2 = _norm(-k2_vec)/qF
    z2 = hbar*(-omega2)/EF
    csTheta12 = _cos_angle(-k2_vec-k1_vec, -k2_vec)
    quadratic_chi_0 += 0.5*_I_2_CV(y1, z1, -1, y2, z2, -1, csTheta12, qF, EF, ms)

    return quadratic_chi_0

def no_G_linear(omega, k):
  return 0.0

def no_theta_quadratic(omega1, k1, omega2, k2, csTheta):
  return 0.0

def quadratic_response(omega1, k1, omega2, k2, csTheta,
                       m, hbar, e, eps0, n, beta, method='direct',
                       G_linear=no_G_linear, theta_quadratic=no_theta_quadratic, ms=2,
                       reltol=1e-6, abstol=1e-8, eta_pol=1e-6, eta_sqrt=1e-4, eta_log=1e-4, tol_upper=1e-8,
                       dx=1e-4, points_n=3,
                       ideal=False, use_diag=True, force_output=False):
  """
    Computes the quadratic response function. Units per energy**2 per volume. It defults to a RPA description if
    no 'G_linear' or 'theta_quadratic' is given.
    Arguments:
      omega1    -- First angular frequencies for evaluation, shape (n, ) or ()
      k1        -- First wave number for evaluation, shape (n, ) or ()
      omega2    -- Second angular frequencies for evaluation, shape (n, ) or ()
      k2        -- Second wave number for evaluation, shape (n, ) or ()
      csTheta   -- Cos of angle between k-vectors, shape (n, ) or ()
      m         -- Mass of particle
      hbar      -- Reduced Plank's constant.
      e         -- Electric charge of particle
      eps0      -- Dielectric constant of free space.
      n         -- Density, for the computation of f1D if not given.
      beta      -- Inverse temperature in energy units, for the computation of f1D if not given.
    Optional: Either f1D or n and beta nust be given. If not n is given, qF must be given.
      method          -- The method used to performe the ideal evaluation. 
      G_linear        -- Local field correction, dimentionless. Callabale with (omega, k)
      theta_quadratic -- Quadratic local field corection, units energy * volume**2. Callabale with (omega1, k1, omega2, k2, costheta)
      ms        -- Spin multiplicity of particle, defult 2.
      reltol    -- Relative tolerance for solution.
      abstol    -- Absolute tolerance for solution.
      eta_pol   -- eta for principla value evaluation, see 'principal_value_integration_f_over_x'
      eta_sqrt  -- Size of region around sqrt-poles which are approximated analytically.
      eta_log   -- Size of region around log-poles which are approximated analytically.
      tol_upper -- Stop integration when the FD distribution is below this value.
      dx        -- Finite difference parameter used for differentiation of Fermi-Dirac integrals.
      points_n  -- Points which helps numerical integration highliting points where FD is steap.
                   Points are given by:  [(mu/EF + n*theta) for n in range(-points_n, points_n+1)]
                   Points are also generated around the log- and sqrt-poles.
      ideal     -- Performe ideal response function calculation.
      use_diag  -- Use diagonal implementation if possible. Defult True.
      force_output -- If true, the result is given even when convergence is not garanteed. Defult False.
    Output:
      quadratic_chi -- quadratic reponse function, shape (n, ) or ()
  """

  # If ideal, inforce ideal behaviour.
  if (ideal):
    G_linear = no_G_linear
    theta_quadratic = no_theta_quadratic

  # Setup for array operations
  omega1  = np.atleast_1d(np.array(omega1))
  k1      = np.atleast_1d(np.array(k1))
  omega2  = np.atleast_1d(np.array(omega2))
  k2      = np.atleast_1d(np.array(k2))
  csTheta = np.atleast_1d(np.array(csTheta))

  omega1, k1, omega2, k2, csTheta = np.broadcast_arrays(omega1, k1, omega2, k2, csTheta)

  # For the angle computation, we also introduced a vector description of k1 and k2.
  # k1 is assumed to align with z-axis
  k1_vec = np.zeros(shape=(len(k1),3))
  k1_vec[:, 2] = k1
  # k2 is assumed to be in the zx-plane
  k2_vec = np.zeros(shape=(len(k2),3))
  k2_vec[:, 0] = k2 * np.sqrt(1.0 - csTheta**2)
  k2_vec[:, 2] = k2 * csTheta

  # Compute ideal response
  if (np.all(k1 == k2) and np.all(omega1 == omega2) and np.all(csTheta == 1.0) and use_diag ):
    quadratic_chi_0 = ideal_diagonal_quadratic_response(omega1, k1, m, hbar, n, beta, ms=ms, reltol=reltol, abstol=abstol, eta_log=eta_log, tol_upper=tol_upper, points_n=points_n, force_output=force_output)
  else:
    quadratic_chi_0 = ideal_quadratic_response(omega1, k1, omega2, k2, csTheta,
                                              m, hbar, n, beta, ms=ms, method=method,
                                              reltol=reltol, abstol=abstol, eta_pol=eta_pol, eta_sqrt=eta_sqrt, eta_log=eta_log, tol_upper=tol_upper,
                                              dx=dx, points_n=points_n, force_output=force_output)
  chi_0_1 = ideal_linear_response(omega1, k1, m, hbar, n, beta, ms=ms, reltol=reltol, abstol=abstol, eta_log=eta_log, tol_upper=tol_upper, points_n=points_n, force_output=force_output)
  chi_0_2 = ideal_linear_response(omega2, k2, m, hbar, n, beta, ms=ms, reltol=reltol, abstol=abstol, eta_log=eta_log, tol_upper=tol_upper, points_n=points_n, force_output=force_output)
  chi_0_12 = ideal_linear_response(omega1+omega2, _norm(k1_vec+k2_vec),  m, hbar, n, beta, ms=ms, reltol=reltol, abstol=abstol, eta_log=eta_log, tol_upper=tol_upper, points_n=points_n, force_output=force_output)

  # Dielectric constants
  if (ideal):
    eps_1  = 1.0
    eps_2  = 1.0
    eps_12 = 1.0
  else:
    # Compute theta functions
    theta_1  = e**2/(eps0*k1**2)      * (1.0 - G_linear(omega1, k1))
    theta_2  = e**2/(eps0*k2**2)      * (1.0 - G_linear(omega2, k2))
    theta_12 = e**2/(eps0*_norm(k1_vec+k2_vec)**2) * (1.0 - G_linear(omega1+omega2, _norm(k1_vec+k2_vec)))

    eps_1  = 1.0 - theta_1  * chi_0_1
    eps_2  = 1.0 - theta_2  * chi_0_2
    eps_12 = 1.0 - theta_12 * chi_0_12

  # Quadratic LFC
  quadratic_theta = theta_quadratic(omega1, k1, omega2, k2, csTheta)

  quadratic_chi = ( quadratic_chi_0 + chi_0_12*chi_0_1*chi_0_2*quadratic_theta ) / (eps_1*eps_2*eps_12)
  return quadratic_chi
