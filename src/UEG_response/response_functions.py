import numpy as np
from .fermi_dirac import compute_chemical_potential
from .I_functions import _I_1_inner, _I_2_inner
from .utils import _norm, _cos_angle

def ideal_linear_response(omega, k, m, hbar, n, beta, ms=2,
                          reltol=1e-6, abstol=1e-8, eta_log=1e-4, tol_upper=1e-8, points_n=3, force_output=False):
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
  linear_chi_0 -= _I_1_inner(y, z, 1,  qF, EF, eta, inv_theta, eta_log, reltol, abstol, tol_upper, points_n, ms, force_output)

  # Second term
  y = np.abs(-k)/qF
  z = hbar*(-omega)/EF
  linear_chi_0 -= _I_1_inner(y, z, -1, qF, EF, eta, inv_theta, eta_log, reltol, abstol, tol_upper, points_n, ms, force_output)

  return linear_chi_0

# Compute quadratic response susing the direct method.
def _ideal_quadratic_response_direct(omega1, k1_vec, omega2, k2_vec, hbar, qF, EF, eta, inv_theta, eta_pol, eta_sqrt, eta_log, reltol, abstol, tol_upper, points_n, ms, dx, force_output):
  quadratic_chi_0 = np.zeros(shape=omega1.shape, dtype=complex)
  
  # First term:
  y1 = _norm(k2_vec)/qF
  z1 = hbar*(omega2)/EF
  y2 = _norm(k1_vec+k2_vec)/qF
  z2 = hbar*(omega1+omega2)/EF
  csTheta12 = _cos_angle(k2_vec, k1_vec+k2_vec)
  quadratic_chi_0 += 0.5*_I_2_inner(y1, z1, 1, y2, z2, 1, csTheta12, qF, EF, eta, inv_theta, eta_pol, eta_sqrt, eta_log, reltol, abstol, tol_upper, points_n, ms, dx, force_output)

  # Second term:
  y1 = _norm(-k2_vec)/qF
  z1 = hbar*(-omega2)/EF
  y2 = _norm(k1_vec)/qF
  z2 = hbar*(omega1)/EF
  csTheta12 = _cos_angle(-k2_vec, k1_vec)
  quadratic_chi_0 += 0.5*_I_2_inner(y1, z1, -1, y2, z2, 1, csTheta12, qF, EF, eta, inv_theta, eta_pol, eta_sqrt, eta_log, reltol, abstol, tol_upper, points_n, ms, dx, force_output)
  
  # Third term:
  y1 = _norm(-k1_vec-k2_vec)/qF
  z1 = hbar*(-omega1-omega2)/EF
  y2 = _norm(-k1_vec)/qF
  z2 = hbar*(-omega1)/EF
  csTheta12 = _cos_angle(-k1_vec-k2_vec, -k1_vec)
  quadratic_chi_0 += 0.5*_I_2_inner(y1, z1, -1, y2, z2, -1, csTheta12, qF, EF, eta, inv_theta, eta_pol, eta_sqrt, eta_log, reltol, abstol, tol_upper, points_n, ms, dx, force_output)
  
  # Fourth term:
  y1 = _norm(k1_vec)/qF
  z1 = hbar*(omega1)/EF
  y2 = _norm(k2_vec+k1_vec)/qF
  z2 = hbar*(omega2+omega1)/EF
  csTheta12 = _cos_angle(k1_vec, k2_vec+k1_vec)
  quadratic_chi_0 += 0.5*_I_2_inner(y1, z1, 1, y2, z2, 1, csTheta12, qF, EF, eta, inv_theta, eta_pol, eta_sqrt, eta_log, reltol, abstol, tol_upper, points_n, ms, dx, force_output)
  
  # Fift term:
  y1 = _norm(-k1_vec)/qF
  z1 = hbar*(-omega1)/EF
  y2 = _norm(k2_vec)/qF
  z2 = hbar*(omega2)/EF
  csTheta12 = _cos_angle(-k1_vec, k2_vec)
  quadratic_chi_0 += 0.5*_I_2_inner(y1, z1, -1, y2, z2, 1, csTheta12, qF, EF, eta, inv_theta, eta_pol, eta_sqrt, eta_log, reltol, abstol, tol_upper, points_n, ms, dx, force_output)
  
  # Sixth term:
  y1 = _norm(-k2_vec-k1_vec)/qF
  z1 = hbar*(-omega2-omega1)/EF
  y2 = _norm(-k2_vec)/qF
  z2 = hbar*(-omega2)/EF
  csTheta12 = _cos_angle(-k2_vec-k1_vec, -k2_vec)
  quadratic_chi_0 += 0.5*_I_2_inner(y1, z1, -1, y2, z2, -1, csTheta12, qF, EF, eta, inv_theta, eta_pol, eta_sqrt, eta_log, reltol, abstol, tol_upper, points_n, ms, dx, force_output)

  return quadratic_chi_0


def ideal_diagonal_quadratic_response(omega, k, m, hbar, n, beta, ms=2, reltol=1e-6, abstol=1e-8, eta_log=1e-4, tol_upper=1e-8, points_n=3, force_output=False):
  """
    Computes the ideal quadratic response function for equal first and second argument. Units per energy**2 per volume.
    Arguments:
      omega -- Angular frequencies for evaluation, shape (n, ) or ()
      k     -- Wave number for evaluation, shape (n, ) or ()
      m     -- Mass of particle
      hbar  -- Reduced Plank's constant.
      n      -- Density, for the computation of f1D if not given.
      beta   -- Inverse temperature in energy units, for the computation of f1D if not given.
    Optional:
      ms           -- Spin multiplicity of particle, defult 2.
      reltol       -- Relative tolerance for solution.
      abstol       -- Absolute tolerance for solution.
      eta_log      -- eta_log for the linear response computation, see 'ideal_linear_response'.
      tol_upper    -- tol_upper for the linear response computation, see 'ideal_linear_response'.
      points_n     -- points_n for the linear response computation, see 'ideal_linear_response'.
      force_output -- If true, results will be outputted even if convergence is not garanteed.
    Output:
      quadratic_chi_0 -- ideal quadratic reponse function, shape (n, ) or ()
  """
  chi_0        = ideal_linear_response( omega,    k, m, hbar, n, beta, ms=ms, reltol=reltol, abstol=abstol, eta_log=eta_log, tol_upper=tol_upper, points_n=points_n, force_output=force_output)
  chi_0_double = ideal_linear_response(2*omega, 2*k, m, hbar, n, beta, ms=ms, reltol=reltol, abstol=abstol, eta_log=eta_log, tol_upper=tol_upper, points_n=points_n, force_output=force_output)

  quadratic_chi_0 = 2*m/(hbar**2*k**2) * (chi_0_double - chi_0)
  return quadratic_chi_0


def ideal_quadratic_response(omega1, k1, omega2, k2, csTheta,
                             m, hbar, n, beta, method='direct', ms=2,
                             reltol=1e-6, abstol=1e-8, eta_pol=1e-6, eta_sqrt=1e-4, eta_log=1e-4, tol_upper=1e-8,
                             dx=1e-4, points_n=3, force_output=False):
  """
    Computes the ideal quadratic response coefficents. Units per energy**2 per volume.
    Arguments:
      omega1  -- First angular frequencies for evaluation, shape (n, ) or ()
      k1      -- First wave number for evaluation, shape (n, ) or ()
      omega2  -- Second angular frequencies for evaluation, shape (n, ) or ()
      k2      -- Second wave number for evaluation, shape (n, ) or ()
      csTheta -- Angle between k-vectors
      m       -- Mass of particle
      hbar    -- Reduced Plank's constant.
      n       -- Density, for the computation of eta and inv_theta if not given.
      beta    -- Inverse temperature in energy units, for the computation of eta and inv_theta if not given.
    Optional: Either eta and inv_theta or n and beta nust be given. If not n is given, qF must be given.
      method    -- The method used to performe the evaluation. 
      ms        -- Spin multiplicity of particle.
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
      force_output -- If true, results will be outputted even if convergence is not garanteed.
    Output:
      quadratic_chi_0 -- ideal quadratic reponse function, shape (n, ) or ()
      eta             -- Chemical potential of uniform system in units of kB T.
      inv_theta       -- Fermi energy in units of kB T.
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

  if (method == 'direct'):
    quadratic_chi_0 = _ideal_quadratic_response_direct(omega1, k1_vec, omega2, k2_vec,
                                                       hbar, qF, EF, eta, inv_theta,
                                                       eta_pol, eta_sqrt, eta_log,
                                                       reltol, abstol, tol_upper, points_n, ms, dx, force_output)
  elif (method == 'maldague'):
    raise ValueError("TODO: implement")
  else:
    raise ValueError(f"The 'method' (%s) must be one of: 'direct' or 'maldague'."%(method))

  return quadratic_chi_0

def no_G_linear(omega, k):
  return 0.0

def no_theta_quadratic(omega1, k1, omega2, k2, csTheta):
  return 0.0

def quadratic_response(omega1, k1, omega2, k2, csTheta,
                       m, hbar, e, eps0, n, beta,
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
                                              m, hbar, n, beta, ms=ms,
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
