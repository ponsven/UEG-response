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
from numba import njit
import warnings
from .utils import jit_integrand_function, _get_points_I
from .principal_value_integration import principal_value_integration_f_over_x
from .fermi_dirac import f1D_fermi_dirac, df1D_fermi_dirac

### General ###

# Heaviside function
@njit
def heaviside(x):
    if (x > 0.0):
        return 1.0
    if (x < 0.0):
        return 0.0
    return 0.5

# Fermi-Dirac function in momentum space using diemntionless units.
@njit
def _reduced_FD(x, inv_theta, eta):
    return 1.0 / ( np.exp(inv_theta*x**2 - eta) + 1.0 )

### Linear order ###

# Real part of phi1-function
@njit
def _phi_1_corrected_real_single(x, A):
    return x * np.log(np.abs((A+x)/(A-x)))

# Real part of phi1-function
_phi_1_corrected_real = np.vectorize(_phi_1_corrected_real_single)

# Imaginary part of phi1-function
@njit
def _phi_1_corrected_imag_single(x, A, sng):
    return x * np.sign(sng) * (heaviside(-A+x) - heaviside(-A-x)) * np.pi

_phi_1_corrected_imag = np.vectorize(_phi_1_corrected_imag_single)

# Implement the phi-function
def phi_1_corrected(x, A, sng):
    return _phi_1_corrected_real(x, A) + 1j * _phi_1_corrected_imag(x, A, sng)

@jit_integrand_function
def _real_I_1_integrad(X):
    x         = X[0]
    A         = X[1]
    inv_theta = X[2]
    eta       = X[3]
    return _phi_1_corrected_real_single(x, A) * _reduced_FD(x, inv_theta, eta)

@jit_integrand_function
def _imag_I_1_integrad(X):
    x         = X[0]
    A         = X[1]
    sng       = X[2]
    inv_theta = X[3]
    eta       = X[4]
    return _phi_1_corrected_imag_single(x, A, sng) * _reduced_FD(x, inv_theta, eta)

def _generate_all_points_I(eta, inv_theta, points_n):
    # Generate relevant poinst
    points_tmp = np.array( [(eta + n)/inv_theta for n in range(-points_n, points_n+1)] )
    points_all = np.sqrt(points_tmp[points_tmp >= 0.0])
    return points_all

def _I_1_inner_single(y, z, sng, qF, EF, eta, inv_theta, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, force_output):
  # Additional special points of the FD-function
  points_all = _generate_all_points_I(eta, inv_theta, points_n)
  # Upper bound of integral based on the FD
  max_high = np.sqrt((max(0.0,eta) + np.log(1/tol_upper))/inv_theta)

  # Compute integral in the seperate intervals.
  # Accumelator for the result without pre-factor.
  res = 0.0 + 1j * 0.0
  err = 0.0 + 1j * 0.0

  # Compute the integration intervals
  A  = (-z - y**2)/(2*y)

  # Setuo integration and treat special cases
  xBreak = np.abs(A)
  # First intervall:  [0,               xBreak - eta_log]
  lows  = [0.0]
  highs = [min(xBreak - eta_log, max_high)]
  log_divergences = []
  # Second intervall: [xLow + eta_log,  xHigh - eta_log]
  if ((xBreak + eta_log) < max_high):
    lows.append(xBreak + eta_log)
    highs.append(max_high)
    log_divergences.append(xBreak)

  # Add additional interesting points:
  points_all_additional = []
  for n in range(2, points_n+2):
    points_all_additional.append(xBreak-n*eta_log)
    points_all_additional.append(xBreak+n*eta_log)
  points_all = np.concatenate((points_all, points_all_additional))


  # Make sure the 'eta' regions around each pole is suffucently small.
  if (np.any(np.array(lows) >= np.array(highs))):
    warnings.warn("Invalid ordering of integration bounds occured, attempt reducing the eta_sqrt and eta_log values.", RuntimeWarning)
    return np.nan + 1j*np.nan

  # Scale the absolut error based on pre-factors
  abstol_scaled = abstol / np.abs(ms * qF**3 / ( EF * (2*np.pi)**2 * 2 * y))

  # Perform numerical integration.
  for i_intervall, (low, high) in enumerate(zip(lows, highs)):
    # Real
    quad_output = quad(_real_I_1_integrad, low, high,
                        args=(A,inv_theta,eta), full_output=1,
                        points=_get_points_I(low,high,points_all), epsabs=abstol_scaled/2, epsrel=reltol/2, limit=limit)
    if (len(quad_output) > 3):
      message = quad_output[3]
      if not (force_output):
        raise ValueError("Integrator 'quad' failed when computing real part in interval [%.2f, %.2f] (intervall %d/%d) with error:\n %s"%(low,high,i_intervall+1,len(lows),message))
      else:
        warnings.warn("Integrator 'quad' failed when computing real part with error:\n %s"%(message), RuntimeWarning)
        res += quad_output[0]
        err += quad_output[1]
    else:
      res += quad_output[0]
      err += quad_output[1]

    # Imag
    if ( low < xBreak ):
      # We are in the first region, with no imaginary part.
      continue

    quad_output = quad(_imag_I_1_integrad, low, high,
                        args=(A,sng,inv_theta,eta), full_output=1,
                        points=_get_points_I(low,high,points_all), epsabs=abstol_scaled/2, epsrel=reltol/2, limit=limit)
    if (len(quad_output) > 3):
      message = quad_output[3]
      if not (force_output):
        raise ValueError("Integrator 'quad' failed when computing imaginary part in interval [%.2f, %.2f] (intervall %d/%d) with error:\n %s"%(low,high,i_intervall+1,len(lows),message))
      else:
        warnings.warn("Integrator 'quad' failed when computing imaginary part with error:\n %s"%(message), RuntimeWarning)
        res += 1j * quad_output[0]
        err += 1j * quad_output[1]
    else:
      res += 1j * quad_output[0]
      err += 1j * quad_output[1]

  # Compute integral around the divergeing point analytically.
  # Logarithmic divergence
  for xlog in log_divergences:
    flog = _reduced_FD(xlog, inv_theta, eta)
    # Real part has a divergence.
    res += flog * A * 2*eta_log * (np.log(np.abs(2*A)) - np.log(eta_log) + 1)
    # Imaginary part is non-divergent, but have a step.
    res += 1j * flog * np.abs(A) * np.pi * np.sign(sng) * eta_log

  # Scale the result
  res *= -ms * qF**3 / ( EF * (2*np.pi)**2 * 2 * y)
  err *= -ms * qF**3 / ( EF * (2*np.pi)**2 * 2 * y)

  return res

def _I_1_inner(y, z, sng, qF, EF, eta, inv_theta, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, force_output):
    tmp = np.zeros(shape=y.shape, dtype=complex)
    for i, (y_, z_) in enumerate(zip(y, z)):
        tmp[i] = _I_1_inner_single(y_, z_, sng, qF, EF, eta, inv_theta, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, force_output)
    return tmp

### Quadratic order ###

# Real part of phi-function
@njit
def _phi_2_corrected_real_single(x, A, sng1, B, sng2, csTheta):
    # Pre-computation
    G2 = A**2 - 2*A*B*csTheta + B**2
    snTheta2 = 1 - csTheta**2

    if ((G2 - x**2*snTheta2) > 0.0):
        ##### First branch ######
        # Real contribution
        real_1 = A*B - x**2*csTheta
        real_2 = x * np.sqrt(G2 - x**2*snTheta2)
        phi = np.log(np.abs((real_1+real_2)/(real_1-real_2)))
    else:
        ##### Second branch #######
        arg = ( x**2 * csTheta - A*B ) / np.sqrt((x**2 - A**2) * (x**2 - B**2))
        phi = 2*np.arccos(-arg) - np.abs( -np.sign(sng2) - np.sign(sng1) ) * np.pi

    # Scale overall result
    phi *= x / np.sqrt(np.abs(G2 - x**2*snTheta2))

    return phi

# Real part of phi-function
_phi_2_corrected_real = np.vectorize(_phi_2_corrected_real_single)


# Imaginary part without pre-factor
@njit
def _phi_2_corrected_imag_wo_pre_single(x, A, sng1, B, sng2, csTheta):
    # Pre-computation
    G2 = A**2 - 2*A*B*csTheta + B**2
    snTheta2 = 1 - csTheta**2

    if ((G2 - x**2*snTheta2) > 0.0):
        ##### First branch #######
        if ((A*B - x**2*csTheta) > 0.0): # Divergence in denumerator
            if ((A+B) > 0.0):
                bar = 0.0
            else:
                bar = 2.0*(np.sign(sng1)+np.sign(sng2))
            return (  np.sign(sng1)*( heaviside(-A+x) + heaviside(-A-x) )
                    + np.sign(sng2)*( heaviside(-B+x) + heaviside(-B-x) ) - bar) * np.pi
        else: # Divergence in numerator
            if ((A-B) > 0.0):
                bar = -2.0*np.sign(sng2)
            else:
                bar = -2.0*np.sign(sng1)
            return (- np.sign(sng1)*( heaviside(-A+x) + heaviside(-A-x) )
                    - np.sign(sng2)*( heaviside(-B+x) + heaviside(-B-x) ) - bar) * np.pi
    else:
        ##### Second branch #######
        return 0.0

_phi_2_corrected_imag_wo_pre = np.vectorize(_phi_2_corrected_imag_wo_pre_single)

# Imaginary part of phi function
@njit
def _phi_2_corrected_imag_single(x, A, sng1, B, sng2, csTheta):
    # Pre-computation
    G2 = A**2 - 2*A*B*csTheta + B**2
    snTheta2 = 1 - csTheta**2

    # Inner computation
    phi = _phi_2_corrected_imag_wo_pre_single(x, A, sng1, B, sng2, csTheta)

    # Scale overall result
    phi *= x / np.sqrt(np.abs(G2 - x**2*snTheta2))

    return phi

_phi_2_corrected_imag = np.vectorize(_phi_2_corrected_imag_single)

# Implement the phi-function
def phi_2_corrected(x, A, sng1, B, sng2, csTheta):
    return _phi_2_corrected_real(x, A, sng1, B, sng2, csTheta) + 1j * _phi_2_corrected_imag(x, A, sng1, B, sng2, csTheta)

@jit_integrand_function
def _real_I_2_integrad(X):
    x         = X[0]
    A         = X[1]
    sng1      = X[2]
    B         = X[3]
    sng2      = X[4]
    csTheta   = X[5]
    inv_theta = X[6]
    eta       = X[7]
    return _phi_2_corrected_real_single(x, A, sng1, B, sng2, csTheta) * _reduced_FD(x, inv_theta, eta)


@jit_integrand_function
def _imag_I_2_integrad(X):
    x         = X[0]
    A         = X[1]
    sng1      = X[2]
    B         = X[3]
    sng2      = X[4]
    csTheta   = X[5]
    inv_theta = X[6]
    eta       = X[7]
    return _phi_2_corrected_imag_single(x, A, sng1, B, sng2, csTheta) * _reduced_FD(x, inv_theta, eta)

def _I_2_inner_full(y1, z1, sng1, y2, z2, sng2, csTheta, qF, EF, eta, inv_theta, eta_sqrt, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, force_output):
  # Additional special points of the FD-function
  points_all = _generate_all_points_I(eta, inv_theta, points_n)
  # Upper bound of integral based on the FD
  max_high = np.sqrt((max(0.0,eta) + np.log(1/tol_upper))/inv_theta)

  # Compute integral in the seperate intervals.
  # Accumelator for the result without pre-factor.
  res = 0.0 + 1j * 0.0
  err = 0.0 + 1j * 0.0

  # Compute the integration intervals
  A  = (-z1 - y1**2)/(2*y1)
  B  = (-z2 - y2**2)/(2*y2)
  G2 = A**2 - 2*A*B*csTheta + B**2
  snTheta2 = 1 - csTheta**2
  if (np.abs(csTheta) != 1.0):
      xG = np.sqrt(G2/snTheta2)
  else:
      xG = np.inf

  # Treat special cases
  if ( np.abs(A - B) < 2*eta_log ): # A = B
    # print("A = B")
    if (csTheta == 1.0):
      raise ValueError("Implementation don't treat both A = B and cos(theta) = 1.")
    B = A
    xlog = np.abs(A)
    # First intervall:  [0,              xlog - eta_log]
    lows  = [0.0]
    highs = [min(xlog - eta_log, max_high)]
    # Second intervall: [xlog + eta_log, xG - eta_sqrt]
    if ((xlog + eta_log) < max_high):
      lows.append(xlog + eta_log)
      highs.append(min(xG - eta_sqrt, max_high))
      quadratic_log_divergence  = True
    else:
      quadratic_log_divergence  = False
    # Third intervall: [xG + eta_sqrt, np.inf]
    if ((xG + eta_sqrt) < max_high):
      lows.append(xG + eta_sqrt)
      highs.append(max_high)
      has_sqrt_divergence = True
    else:
      has_sqrt_divergence = False
    has_sqrt_divergence2 = False
    log_divergences = []
    quadratic_log_divergence2 = False

    # Add additional interesting points:
    points_all_additional = []
    for n in range(2, points_n+2):
      points_all_additional.append(xlog-n*eta_log)
      points_all_additional.append(xlog+n*eta_log)
      points_all_additional.append(xG-n*eta_sqrt)
      points_all_additional.append(xG+n*eta_sqrt)
    points_all = np.concatenate((points_all, points_all_additional))
  elif ( np.abs(A + B) < 2*eta_log ): # A = -B
    # print("A = -B")
    if (csTheta == -1.0):
      raise ValueError("Implementation don't treat both A = -B and cos(theta) = -1.")
    B = -A
    xlog = np.abs(A)
    # First intervall:  [0,              xlog - eta_log]
    lows  = [0.0]
    highs = [min(xlog - eta_log, max_high)]
    # Second intervall: [xlog + eta_log, xG - eta_sqrt]
    if ((xlog + eta_log) < max_high):
      lows.append(xlog + eta_log)
      highs.append(min(xG - eta_sqrt, max_high))
      quadratic_log_divergence2  = True
    else:
      quadratic_log_divergence2  = False
    # Third intervall: [xG + eta_sqrt, np.inf]
    if ((xG + eta_sqrt) < max_high):
      lows.append(xG + eta_sqrt)
      highs.append(max_high)
      has_sqrt_divergence = True
    else:
      has_sqrt_divergence = False
    has_sqrt_divergence2 = False
    log_divergences = []
    quadratic_log_divergence = False

    # Add additional interesting points:
    points_all_additional = []
    for n in range(2, points_n+2):
      points_all_additional.append(xlog-n*eta_log)
      points_all_additional.append(xlog+n*eta_log)
      points_all_additional.append(xG-n*eta_sqrt)
      points_all_additional.append(xG+n*eta_sqrt)
    points_all = np.concatenate((points_all, points_all_additional))
  elif (np.abs(xG - max(np.abs(A), np.abs(B))) < (eta_log+eta_sqrt) ): # xG = xHigh
    # print("xG = xHigh")
    # Case where xHigh = xG
    xLow, xHigh = sorted( (np.abs(A), np.abs(B)) )
    # First intervall:  [0,               xLow - eta_log]
    lows  = [0.0]
    highs = [min(xLow - eta_log, max_high)]
    log_divergences = []
    # Second intervall: [xLow + eta_log,  xG - eta_sqrt]
    if ((xLow + eta_log) < max_high):
      lows.append(xLow + eta_log)
      highs.append(min(xG - eta_sqrt, max_high))
      log_divergences.append(xLow)
    # Third intervall:  [xG + eta_sqrt,   np.inf]
    if ((xG + eta_sqrt) < max_high):
      lows.append(xG + eta_sqrt)
      highs.append(max_high)
      has_sqrt_divergence2 = True
    else:
      has_sqrt_divergence2 = False
    has_sqrt_divergence = False
    quadratic_log_divergence  = False
    quadratic_log_divergence2 = False

    # Add additional interesting points:
    points_all_additional = []
    for n in range(2, points_n+2):
      points_all_additional.append(xLow-n*eta_log)
      points_all_additional.append(xLow+n*eta_log)
      points_all_additional.append(xG-n*eta_sqrt)
      points_all_additional.append(xG+n*eta_sqrt)
    points_all = np.concatenate((points_all, points_all_additional))
  else: # General case
    # print("General")
    xLow, xHigh = sorted( (np.abs(A), np.abs(B)) )
    # First intervall:  [0,               xLow - eta_log]
    lows  = [0.0]
    highs = [min(xLow - eta_log, max_high)]
    log_divergences = []
    # Second intervall: [xLow + eta_log,  xHigh - eta_log]
    if ((xLow + eta_log) < max_high):
      lows.append(xLow + eta_log)
      highs.append(min(xHigh - eta_log, max_high))
      log_divergences.append(xLow)
    # Third intervall:  [xHigh + eta_log, xG - eta_sqrt]
    if ((xHigh + eta_log) < max_high):
      lows.append(xHigh + eta_log)
      highs.append(min(xG - eta_sqrt, max_high))
      log_divergences.append(xHigh)
    # Fourth intervall: [xG + eta_sqrt, np.inf]
    if ((xG + eta_sqrt) < max_high):
      lows.append(xG + eta_sqrt)
      highs.append(max_high)
      has_sqrt_divergence = True
    else:
      has_sqrt_divergence = False
    has_sqrt_divergence2 = False
    quadratic_log_divergence  = False
    quadratic_log_divergence2 = False

    # Add additional interesting points:
    points_all_additional = []
    for n in range(2, points_n+2):
      points_all_additional.append(xLow-n*eta_log)
      points_all_additional.append(xLow+n*eta_log)
      points_all_additional.append(xHigh-n*eta_log)
      points_all_additional.append(xHigh+n*eta_log)
      points_all_additional.append(xG-n*eta_sqrt)
      points_all_additional.append(xG+n*eta_sqrt)
    points_all = np.concatenate((points_all, points_all_additional))


  # Make sure the 'eta' regions around each pole is suffucently small.
  if (np.any(np.array(lows) >= np.array(highs))):
    warnings.warn("Invalid ordering of integration bounds occured, attempt reducing the eta_sqrt and eta_log values.", RuntimeWarning)
    return np.nan + 1j*np.nan

  # Scale the absolute error based on pre-factors
  abstol_scaled = abstol / np.abs(ms * qF**3 / ( EF**2 * (2*np.pi)**2 * 4 * y1 * y2))

  # Perform numerical integration.
  for i_intervall, (low, high) in enumerate(zip(lows, highs)):
    # Real
    quad_output = quad(_real_I_2_integrad, low, high,
                        args=(A,sng1,B,sng2,csTheta,inv_theta,eta), full_output=1,
                        points=_get_points_I(low,high,points_all), epsabs=abstol_scaled/6, epsrel=reltol/6, limit=limit)
    if (len(quad_output) > 3):
      message = quad_output[3]
      if not (force_output):
        raise ValueError("Integrator 'quad' failed when computing real part in interval [%.2f, %.2f] (intervall %d/%d) with error:\n %s"%(low,high,i_intervall+1,len(lows),message))
      else:
        warnings.warn("Integrator 'quad' failed when computing real part with error:\n %s"%(message), RuntimeWarning)
        res += quad_output[0]
        err += quad_output[1]
    else:
      res += quad_output[0]
      err += quad_output[1]

    # Imag
    if ( low > xG ):
      # We are in the last region, with no imaginary part.
      continue

    quad_output = quad(_imag_I_2_integrad, low, high,
                        args=(A,sng1,B,sng2,csTheta,inv_theta,eta), full_output=1,
                        points=_get_points_I(low,high,points_all), epsabs=abstol/6, epsrel=reltol/6, limit=limit)
    if (len(quad_output) > 3):
      message = quad_output[3]
      if not (force_output):
        raise ValueError("Integrator 'quad' failed when computing imaginary part in interval [%.2f, %.2f] (intervall %d/%d) with error:\n %s"%(low,high,i_intervall+1,len(lows),message))
      else:
        warnings.warn("Integrator 'quad' failed when computing imaginary part with error:\n %s"%(message), RuntimeWarning)
        res += 1j * quad_output[0]
        err += 1j * quad_output[1]
    else:
      res += 1j * quad_output[0]
      err += 1j * quad_output[1]

  # Compute integral around the divergeing points analytically.
  # Logarithmic divergences
  for xlog in log_divergences:
    flog = _reduced_FD(xlog, inv_theta, eta)
    # Real part has a divergence.
    val = A*B - xlog**2 * csTheta
    if (val < 0):
        log_val = (-2*xlog*csTheta + (G2 - 2*xlog**2*snTheta2)/np.sqrt(G2 - xlog**2 *snTheta2)) / (val - np.abs(val))
        int_val = 2*eta_log * (np.log(np.abs(log_val)) + (np.log(eta_log)-1))
    else:
        log_val = (val + np.abs(val)) / (-2*xlog*csTheta - (G2 - 2*xlog**2*snTheta2)/np.sqrt(G2 - xlog**2 *snTheta2))
        int_val = 2*eta_log * (np.log(np.abs(log_val)) - (np.log(eta_log)-1))
    int_val*= flog * xlog / np.sqrt(G2 - xlog**2*snTheta2)
    res += int_val
    # Imaginary part is non-divergent, but might have a step.
    sum_m_factor = _phi_2_corrected_imag_wo_pre_single(xlog-(eta_log/2), A, sng1, B, sng2, csTheta) + _phi_2_corrected_imag_wo_pre_single(xlog+(eta_log/2), A, sng1, B, sng2, csTheta)
    res += 1j * flog * xlog / np.sqrt(G2 - xlog**2*snTheta2) * sum_m_factor * eta_log

  # Logaritmic divergence in the case that A = B.
  if (quadratic_log_divergence):
    xlog = np.abs(A)
    flog = _reduced_FD(xlog, inv_theta, eta)
    # Real part has a logaritmic divergence.
    log_val = 2*A**2*(1-csTheta) / (1/2 * snTheta2/(1-csTheta) * (3 + snTheta2/(1 - csTheta)**2) - csTheta)
    int_val = ( np.log(np.abs(log_val)) - 2*(np.log(eta_log) - 1) ) * 2*eta_log
    int_val *= flog * xlog / np.sqrt(G2 - xlog**2*snTheta2)
    res += int_val
    # Imaginary part is non-divergent, but might have a step.
    sum_m_factor = _phi_2_corrected_imag_wo_pre_single(xlog-(eta_log/2), A, sng1, B, sng2, csTheta) + _phi_2_corrected_imag_wo_pre_single(xlog+(eta_log/2), A, sng1, B, sng2, csTheta)
    res += 1j * flog * xlog / np.sqrt(G2 - xlog**2*snTheta2) * sum_m_factor * eta_log

  # Logaritmic divergence in the case that A = -B.
  if (quadratic_log_divergence2):
    xlog = np.abs(A)
    flog = _reduced_FD(xlog, inv_theta, eta)
    # Real part has a logaritmic divergence.
    log_val = (1/2 * snTheta2/(1+csTheta) * (3 + snTheta2/(1 + csTheta)**2) + csTheta) / (2*A**2*(1+csTheta))
    int_val = ( np.log(np.abs(log_val)) + 2*(np.log(eta_log) - 1) ) * 2*eta_log
    int_val *= flog * xlog / np.sqrt(G2 - xlog**2*snTheta2)
    res += int_val
    # Imaginary part is non-divergent, but might have a step.
    sum_m_factor = _phi_2_corrected_imag_wo_pre_single(xlog-(eta_log/2), A, sng1, B, sng2, csTheta) + _phi_2_corrected_imag_wo_pre_single(xlog+(eta_log/2), A, sng1, B, sng2, csTheta)
    res += 1j * flog * xlog / np.sqrt(G2 - xlog**2*snTheta2) * sum_m_factor * eta_log

  # Sqrt divergence
  if (has_sqrt_divergence):
    # Only treat this point if we integrate over it.
    fG = _reduced_FD(xG, inv_theta, eta)
    # The sqrt-divergence low side
    res += fG * (2.0*xG**2 / (A*B - xG**2 * csTheta)) * eta_sqrt
    res += 1j * fG * (xG/np.sqrt(2*xG*snTheta2)) * _phi_2_corrected_imag_wo_pre_single(xG-(eta_sqrt/2), A, sng1, B, sng2, csTheta) * 2*np.sqrt(eta_sqrt)

    # The sqrt-divergence high side
    C = (A - B*csTheta)*(A*csTheta - B)
    m_part_wo_pre = np.pi*(np.sign(C) + 1)  - np.abs( -np.sign(sng2) - np.sign(sng1) ) * np.pi
    # Both a eta^(1/2) and eta^1 contribution to the real part, but no imaginary contribution.
    res += fG * xG/np.sqrt(2*xG*snTheta2) * m_part_wo_pre * 2*np.sqrt(eta_sqrt)
    res += fG * (2.0*xG**2 / (A*B - xG**2 * csTheta)) * eta_sqrt

  # Sqrt divergence where xHigh = xG
  if (has_sqrt_divergence2):
    # Only treat this point if we integrate over it.
    fG = _reduced_FD(xG, inv_theta, eta)
    # The sqrt-divergence low side
    res += fG * 2.0*csTheta/snTheta2 * eta_sqrt
    res += 1j * fG * (xG/np.sqrt(2*xG*snTheta2)) * _phi_2_corrected_imag_wo_pre_single(xG-(eta_sqrt/2), A, sng1, B, sng2, csTheta) * 2*np.sqrt(eta_sqrt)

    # The sqrt-divergence high side
    m_part_wo_pre = (1  - np.abs( -np.sign(sng2) - np.sign(sng1) )) * np.pi
    # Both a eta^(1/2) and eta^1 contribution to the real part, but no imaginary contribution.
    res += fG * xG/np.sqrt(2*xG*snTheta2) * m_part_wo_pre * 2*np.sqrt(eta_sqrt)
    res += fG * 2.0*xG*csTheta/np.sqrt((xG**2 - xLow**2) * snTheta2) * eta_sqrt

  # Scale the result
  res *= ms * qF**3 / ( EF**2 * (2*np.pi)**2 * 4 * y1 * y2)
  err *= ms * qF**3 / ( EF**2 * (2*np.pi)**2 * 4 * y1 * y2)

  return res

def _I_2_inner_parallel(y1, z1, sng1, y2, z2, sng2, qF, EF, eta, inv_theta, eta_pol, reltol, abstol, limit, tol_upper, points_n, ms, dx, force_output):
  # Correct eps sign.
  sng1 *= np.sign(y1)
  sng2 *= np.sign(y2)

  # Positions of the poles
  p1 = y1/2 + z1/(2*y1)
  p2 = y2/2 + z2/(2*y2)

  # Points and bounderies for integration
  eta_or_zero = max(0.0, eta)
  points_tmp = np.sqrt((eta_or_zero + np.arange(-np.floor(eta_or_zero), -np.floor(eta_or_zero)+points_n+1))/inv_theta)
  high = max(np.abs(p1), np.abs(p2)) + np.sqrt( (max(eta,0.0) + np.log(1/tol_upper - 1))/inv_theta)

  # Scale abs error based on pre-factors
  abstol_scaled = abstol / np.abs(qF**3/(4*EF**2*y1*y2))

  if (p1 == p2):
    # Case where the pols are the same.
    if (sng1*sng2 < 0.0):
      raise ValueError('Incompatable sign!')
    f = lambda t: df1D_fermi_dirac(p1+t, eta/inv_theta, 1.0, 0.5, inv_theta, ms=ms, dx=dx)
    # Find numerical hints for integrator
    points_FD = np.concatenate( (points_tmp - p1, points_tmp + p1))
    tmp = principal_value_integration_f_over_x(f, eta=eta_pol, reltol=reltol/6, abstol=abstol_scaled/6, points=points_FD, limit=limit, high=high, force_output=force_output) + 1j*np.pi * f(0.0) * sng1
  else:
    # Case where the two pols are distinct.
    pol_factor = 1/(p1-p2)
    f = lambda t: pol_factor * ( f1D_fermi_dirac(p1+t, eta/inv_theta, 1.0, 0.5, inv_theta, ms=ms)
                               - f1D_fermi_dirac(p2+t, eta/inv_theta, 1.0, 0.5, inv_theta, ms=ms) )
    # Find numerical hints for integrator
    points_FD = np.concatenate( (points_tmp - p1, points_tmp + p1, points_tmp - p2, points_tmp + p2) )
    tmp = principal_value_integration_f_over_x(f, eta=eta_pol, reltol=reltol/6, abstol=abstol_scaled/6, points=points_FD, limit=limit, high=high, force_output=force_output) \
             + 1j*np.pi * pol_factor * ( sng1*f1D_fermi_dirac(p1, eta/inv_theta, 1.0, 0.5, inv_theta, ms=ms) \
                                       - sng2*f1D_fermi_dirac(p2, eta/inv_theta, 1.0, 0.5, inv_theta, ms=ms) )

  tmp *= qF**3/(4*EF**2*y1*y2)
  return tmp

def _I_2_inner(y1, z1, sng1, y2, z2, sng2, csTheta, qF, EF, eta, inv_theta, eta_pol, eta_sqrt, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, dx, use_parallel, force_output):
    tmp = np.zeros(shape=y1.shape, dtype=complex)
    for i, (y1_, z1_, y2_, z2_, csTheta_) in enumerate(zip(y1, z1, y2, z2, csTheta)):
        if (y1_ == 0.0): # First k-vector is zero
          tmp[i] = _I_1_inner_single(y2_, z2_, sng2, qF, EF, eta, inv_theta, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, force_output)/(z1_*EF)
        elif (y2_ == 0.0): # Second k-vector is zero
          tmp[i] = _I_1_inner_single(y1_, z1_, sng1, qF, EF, eta, inv_theta, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, force_output)/(z2_*EF)
        elif (csTheta_ == 1.0 and use_parallel): # The vectors are parallel
            tmp[i] = _I_2_inner_parallel(y1_, z1_, sng1,  y2_, z2_, sng2, qF, EF, eta, inv_theta, eta_pol, reltol, abstol, limit, tol_upper, points_n, ms, dx, force_output)
        elif (csTheta_ == -1.0 and use_parallel): # The vectors are anti-parallel, correct the sign
            tmp[i] = _I_2_inner_parallel(y1_, z1_, sng1, -y2_, z2_, sng2, qF, EF, eta, inv_theta, eta_pol, reltol, abstol, limit, tol_upper, points_n, ms, dx, force_output)
        else: # The general case
            tmp[i] = _I_2_inner_full(y1_, z1_, sng1, y2_, z2_, sng2, csTheta_, qF, EF, eta, inv_theta, eta_sqrt, eta_log, reltol, abstol, limit, tol_upper, points_n, ms, force_output)
    return tmp

@njit
def _I_2_CV_real_single(y1, z1, sng1, y2, z2, sng2, csTheta, qF, EF, ms):
    if (csTheta > -1.0 and csTheta < 1.0):
        snTheta2 = 1 - csTheta**2
        pre_Cenni = ms * (0.5)**2 / ((2*np.pi)**2 * y1 * y2 * snTheta2) * qF**3 / EF**2
        A  = (-z1 - y1**2)/(2*y1)
        B  = (-z2 - y2**2)/(2*y2)
        G2 = A**2 - 2*A*B*csTheta + B**2
        kF = 1.0


        phi_real = _phi_2_corrected_real_single(kF, A, sng1, B, sng2, csTheta)

        I_Cenni_real = pre_Cenni * ( (A*csTheta-B)*np.log(np.abs((A-kF)/(A+kF))) 
                                + (B*csTheta-A)*np.log(np.abs((B-kF)/(B+kF)))
                                - (G2 - kF**2*snTheta2)/kF * phi_real )
    elif (csTheta >= 1.0):
        A  = (-z1 - y1**2)/(2*y1)
        B  = (-z2 - y2**2)/(2*y2)
        G2 = A**2 - 2*A*B*1.0 + B**2
        snTheta2 = 0.0
        kF = 1.0

        pre_Cenni = - ms/(2*(4*np.pi)**2 * y1*y2) * qF**3/EF**2 
        if (np.abs(A-B) < 1e-10):
            I_Cenni_real = pre_Cenni * 2 * ( 2 + A*np.log(np.abs((A-1)/(A+1))) )  
        else:
            I_Cenni_real = pre_Cenni * (2 + A*np.log(np.abs((A-1)/(A+1))) 
                                        + B*np.log(np.abs((B-1)/(B+1))) 
                                        + (A*B - 1)/np.abs(A-B) * np.log(np.abs( (A*B - 1 + np.abs(A-B)) / (A*B - 1 - np.abs(A-B)) )) )
    else:
        A  = (-z1 - y1**2)/(2*y1)
        B  = (-z2 - y2**2)/(2*y2)
        G2 = A**2 + 2*A*B*1.0 + B**2
        snTheta2 = 0.0
        kF = 1.0

        pre_Cenni = ms/(2*(4*np.pi)**2 * y1*y2) * qF**3/EF**2 
        if (np.abs(A+B) < 1e-10):
            I_Cenni_real = pre_Cenni * 2 * ( 2 + A*np.log(np.abs((A-1)/(A+1))) )  
        else:
            I_Cenni_real = pre_Cenni * (2 + A*np.log(np.abs((A-1)/(A+1))) 
                                          + B*np.log(np.abs((B-1)/(B+1))) 
                                          + (A*B + 1)/np.abs(A+B) * np.log(np.abs( (A*B + 1 + np.abs(A+B)) / (A*B + 1 - np.abs(A+B)) )) )
    
    return I_Cenni_real

_I_2_CV_real = np.vectorize(_I_2_CV_real_single)

@njit
def _I_2_CV_imag_single(y1, z1, sng1, y2, z2, sng2, csTheta, qF, EF, ms):
    if (csTheta > -1.0 and csTheta < 1.0):
        snTheta2 = 1 - csTheta**2
        pre_Cenni = ms * (0.5)**2 / ((2*np.pi)**2 * y1 * y2 * snTheta2) * qF**3 / EF**2
        A  = (-z1 - y1**2)/(2*y1)
        B  = (-z2 - y2**2)/(2*y2)
        G2 = A**2 - 2*A*B*csTheta + B**2
        kF = 1.0

        phi_imag = _phi_2_corrected_imag_single(kF, A, sng1, B, sng2, csTheta)

        I_Cenni_imag = pre_Cenni * ( (A*csTheta-B)*(-sng1)*np.pi*(kF > np.abs(A))
                                + (B*csTheta-A)*(-sng2)*np.pi*(kF > np.abs(B))
                                - (G2 - kF**2*snTheta2)/kF * phi_imag )  
    elif (csTheta >= 1.0):
        A  = (-z1 - y1**2)/(2*y1)
        B  = (-z2 - y2**2)/(2*y2)
        G2 = A**2 - 2*A*B*1.0 + B**2
        snTheta2 = 0.0
        kF = 1.0

        pre_Cenni = - ms/(2*(4*np.pi)**2 * y1*y2) * qF**3/EF**2 
        
        if (np.abs(A-B) < 1e-10):
            I_Cenni_imag = pre_Cenni * (  A*(-sng1)*np.pi*(kF > np.abs(A)) 
                                        + B*(-sng2)*np.pi*(kF > np.abs(B)) )
        else:
            I_Cenni_imag = pre_Cenni * (  A*(-sng1)*np.pi*(kF > np.abs(A)) 
                                        + B*(-sng2)*np.pi*(kF > np.abs(B)) 
                                        + (A*B-1)/np.abs(A-B) * _phi_2_corrected_imag_wo_pre_single(1.0, A, sng1, B, sng2, 1.0))
    else:
        A  = (-z1 - y1**2)/(2*y1)
        B  = (-z2 - y2**2)/(2*y2)
        G2 = A**2 + 2*A*B*1.0 + B**2
        snTheta2 = 0.0
        kF = 1.0

        pre_Cenni = ms/(2*(4*np.pi)**2 * y1*y2) * qF**3/EF**2 
        
        if (np.abs(A+B) < 1e-10):
            I_Cenni_imag = pre_Cenni * (  A*(-sng1)*np.pi*(kF > np.abs(A)) 
                                        + B*(-sng2)*np.pi*(kF > np.abs(B)) )
        else:
            I_Cenni_imag = pre_Cenni * (  A*(-sng1)*np.pi*(kF > np.abs(A)) 
                                        + B*(-sng2)*np.pi*(kF > np.abs(B)) 
                                        + (A*B+1)/np.abs(A+B) * _phi_2_corrected_imag_wo_pre_single(1.0, A, sng1, B, sng2, -1.0))
    
    return I_Cenni_imag

_I_2_CV_imag = np.vectorize(_I_2_CV_imag_single)

def _I_2_CV(y1, z1, sng1, y2, z2, sng2, csTheta, qF, EF, ms):
   return _I_2_CV_real(y1, z1, sng1, y2, z2, sng2, csTheta, qF, EF, ms) + 1j * _I_2_CV_imag(y1, z1, sng1, y2, z2, sng2, csTheta, qF, EF, ms)
