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
import warnings
from .utils import _get_points_I

def principal_value_integration_f_over_x(f, eta=1e-6, reltol=1e-6, abstol=1e-8, args=None, points=None, limit=50, high=None, force_output=False):
  """
    Computes the principal value integral (latex notation)
      P \\int_{-\\infty}^{\\infty} f(z)/z dz
    via a numerical integration away from the singularity and based on taylor approximation
    close to the divergent point at z = 0.
    The numerical error e in the full integration i, is
      e <= max(abstol, i*reltol)
    Arguments:
      f -- Function in the integral (not including 1/z), callable object.
    Optional:
      eta    -- The size of the region in which the taylor approximation is used, [-eta, eta].
      reltol -- Relative tolerance for integration.
      abstol -- Absolute tolerance for integration.
      points -- 'Importent' points for the integration.
      limit  -- 'limit' passed to quad, defult 50.
      high   -- If 'points' are given, giv this upper bound for the computation.
    Output:
      res -- Numerical estimate for principle value of integral.
  """
  # Bulk of integration
  if (args is None):
    g = lambda x : (f(x) - f(-x))/x
  else:
    g = lambda x : (f(x, *args) - f(-x, *args))/x
  if (points is None):
    if (high is None):
      high = np.inf
  else:
    if (high is None):
      raise ValueError("If 'points' are given, manually set upper bound 'high'.")

  quad_output = quad(g, eta, high, epsrel=0.1*reltol, epsabs=0.1*abstol, points=_get_points_I(eta,high,points), limit=limit, full_output=1)

  if (len(quad_output) > 3):
    message = quad_output[3]
    if not (force_output):
      raise ValueError("Integrator 'quad' failed with error:\n %s"%(message))
    else:
      warnings.warn("Integrator 'quad' failed when computing principal value with error:\n %s"%(message), RuntimeWarning)
  y = quad_output[0]
  abserr = quad_output[1]

  # Singulerity integration.
  if (args is None):
    fp1 = f(eta)
    fn1 = f(-eta)
    y_sing = fp1 - fn1
    y_sing_abserr = np.abs((f(2.0*eta) - 2.0*fp1 + 2*fn1 - f(-2.0*eta))/18.0)
  else:
    fp1 = f(eta, *args)
    fn1 = f(-eta, *args)
    y_sing = fp1 - fn1
    y_sing_abserr = np.abs((f(2.0*eta, *args) - 2.0*fp1 + 2*fn1 - f(-2.0*eta, *args))/18.0)

  res = y + y_sing
  res_abserr = abserr + y_sing_abserr

  # Error estimate
  if (res_abserr > max(abstol, reltol*np.abs(res))):
    res_relerr = res_abserr / np.abs(res)
    if not (force_output):
      raise ValueError(f'Relative error {res_relerr} or absolute error {res_abserr} exccceds relative tolerence {reltol} or absolut tolerance {abstol}.')
    else:
      warnings.warn(f'Relative error {res_relerr} or absolute error {res_abserr} exccceds relative tolerence {reltol} or absolut tolerance {abstol}.')
    

  return res