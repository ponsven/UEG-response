import numpy as np
from numba import njit
import warnings
from scipy.integrate import quad
from .I_functions import _I_2_CV_real_single, _I_2_CV_imag_single
from .utils import jit_integrand_function, _norm, _cos_angle, filter_close_values, _get_points_I


@njit
def _real_chi_2_0_CV_single(eta_bar, k1s, omega1s, sng1s, k2s, omega2s, sng2s, csThetas, beta, hbar, m, ms):
    EF = eta_bar / beta
    qF = np.sqrt(2*m*EF)/hbar

    I_tot = 0.0
    for i in range(6):
        y1 = k1s[i]/qF
        z1 = hbar*omega1s[i]/EF
        y2 = k2s[i]/qF
        z2 = hbar*omega2s[i]/EF
        I_tot += _I_2_CV_real_single(y1, z1, sng1s[i], y2, z2, sng2s[i], csThetas[i], qF, EF, ms)

    return 0.5 * I_tot

_real_chi_2_0_CV = np.vectorize(_real_chi_2_0_CV_single, excluded=np.arange(1,12))

@njit
def _imag_chi_2_0_CV_single(eta_bar, k1s, omega1s, sng1s, k2s, omega2s, sng2s, csThetas, beta, hbar, m, ms):
    EF = eta_bar / beta
    qF = np.sqrt(2*m*EF)/hbar

    I_tot = 0.0
    for i in range(6):
        y1 = k1s[i]/qF
        z1 = hbar*omega1s[i]/EF
        y2 = k2s[i]/qF
        z2 = hbar*omega2s[i]/EF
        I_tot += _I_2_CV_imag_single(y1, z1, sng1s[i], y2, z2, sng2s[i], csThetas[i], qF, EF, ms)

    return 0.5 * I_tot

_imag_chi_2_0_CV = np.vectorize(_imag_chi_2_0_CV_single, excluded=np.arange(1,12))

@jit_integrand_function
def _real_chi_2_0_integrand_Maldague(X):
    eta_bar  = X[0]
    k1s      = X[1:7]
    omega1s  = X[7:13]
    sng1s    = X[13:19]
    k2s      = X[19:25]
    omega2s  = X[25:31]
    sng2s    = X[31:37]
    csThetas = X[37:43]
    beta        = X[43]
    eta         = X[44]
    hbar        = X[45]
    m           = X[46]
    ms          = X[47] 

    tmp = _real_chi_2_0_CV_single(eta_bar, k1s, omega1s, sng1s, k2s, omega2s, sng2s, csThetas, beta, hbar, m, ms)

    return tmp / (4 * np.cosh((eta_bar - eta)/2)**2)

@jit_integrand_function
def _imag_chi_2_0_integrand_Maldague(X):
    eta_bar  = X[0]
    k1s      = X[1:7]
    omega1s  = X[7:13]
    sng1s    = X[13:19]
    k2s      = X[19:25]
    omega2s  = X[25:31]
    sng2s    = X[31:37]
    csThetas = X[37:43]
    beta        = X[43]
    eta         = X[44]
    hbar        = X[45]
    m           = X[46]
    ms          = X[47] 

    tmp = _imag_chi_2_0_CV_single(eta_bar, k1s, omega1s, sng1s, k2s, omega2s, sng2s, csThetas, beta, hbar, m, ms)

    return tmp / (4 * np.cosh((eta_bar - eta)/2)**2)

def _setup_computations_Maldague(k1_vec, omega1, k2_vec, omega2):
    n, _ = k1_vec.shape

    k1s        = np.zeros(shape=(n, 6))
    omega1s    = np.zeros(shape=(n, 6))
    sng1s      = np.zeros(shape=(n, 6))
    k2s        = np.zeros(shape=(n, 6))
    omega2s    = np.zeros(shape=(n, 6))
    sng2s      = np.zeros(shape=(n, 6))
    csTheta12s = np.zeros(shape=(n, 6))


    # First term:
    k1s[:, 0]        = _norm(k2_vec)
    omega1s[:, 0]    = omega2
    sng1s[:, 0]      = 1
    k2s[:, 0]        = _norm(k1_vec+k2_vec)
    omega2s[:, 0]    = omega1+omega2
    sng2s[:, 0]      = 1
    csTheta12s[:, 0] = _cos_angle(k2_vec, k1_vec+k2_vec)
   
    # Second term:
    k1s[:, 1]        = _norm(-k2_vec)
    omega1s[:, 1]    = -omega2
    sng1s[:, 1]      = -1
    k2s[:, 1]        = _norm(k1_vec)
    omega2s[:, 1]    = omega1
    sng2s[:, 1]      = 1
    csTheta12s[:, 1] = _cos_angle(-k2_vec, k1_vec)

    # Third term:
    k1s[:, 2]        = _norm(-k1_vec-k2_vec)
    omega1s[:, 2]    = -omega1-omega2
    sng1s[:, 2]      = -1
    k2s[:, 2]        = _norm(-k1_vec)
    omega2s[:, 2]    = -omega1
    sng2s[:, 2]      = -1
    csTheta12s[:, 2] = _cos_angle(-k1_vec-k2_vec, -k1_vec)

    # Fourth term:
    k1s[:, 3]        = _norm(k1_vec)
    omega1s[:, 3]    = omega1
    sng1s[:, 3]      = 1
    k2s[:, 3]        = _norm(k2_vec+k1_vec)
    omega2s[:, 3]    = omega2+omega1
    sng2s[:, 3]      = 1
    csTheta12s[:, 3] = _cos_angle(k1_vec, k2_vec+k1_vec)

    # Fift term:
    k1s[:, 4]        = _norm(-k1_vec)
    omega1s[:, 4]    = -omega1
    sng1s[:, 4]      = -1
    k2s[:, 4]        = _norm(k2_vec)
    omega2s[:, 4]    = omega2
    sng2s[:, 4]      = 1
    csTheta12s[:, 4] = _cos_angle(-k1_vec, k2_vec)

    # Sixth term:
    k1s[:, 5]        = _norm(-k2_vec-k1_vec)
    omega1s[:, 5]    = -omega2-omega1
    sng1s[:, 5]      = -1
    k2s[:, 5]        = _norm(-k2_vec)
    omega2s[:, 5]    = -omega2
    sng2s[:, 5]      = -1
    csTheta12s[:, 5] = _cos_angle(-k2_vec-k1_vec, -k2_vec)

    return k1s, omega1s, sng1s, k2s, omega2s, sng2s, csTheta12s

def _generate_I_points_Maldague(k1, omega1, k2, omega2, csTheta, beta, hbar, m):
        # Special points in I
        A_tilde  = -np.sqrt(2*m)/hbar * (hbar*omega1 + hbar**2 * k1**2/(2*m)) / (2*k1)
        B_tilde  = -np.sqrt(2*m)/hbar * (hbar*omega2 + hbar**2 * k2**2/(2*m)) / (2*k2)
        G2_tilde = A_tilde**2 - 2*A_tilde*B_tilde*csTheta + B_tilde**2

        if (np.abs(csTheta) == 1.0):
            points_I = beta * np.array([A_tilde**2, B_tilde**2, 2*min(A_tilde**2, B_tilde**2), np.inf])
        else:
            points_I = beta * np.array([A_tilde**2, B_tilde**2, 2*min(A_tilde**2, B_tilde**2), G2_tilde/(1 - csTheta**2)])
        
        return points_I

def _Maldague_chi_2_0_CV(eta_bar, k1, omega1, k2, omega2, csTheta, beta, hbar, m, ms):
    # Setup for array operations
    omega1  = np.atleast_1d(np.array(omega1))
    k1      = np.atleast_1d(np.array(k1))
    omega2  = np.atleast_1d(np.array(omega2))
    k2      = np.atleast_1d(np.array(k2))
    csTheta = np.atleast_1d(np.array(csTheta))

    omega1, k1, omega2, k2, csTheta = np.broadcast_arrays(omega1, k1, omega2, k2, csTheta)

    # Angle computation are performed using the vector description.
    # k1 is assumed to align with z-axis
    k1_vec = np.zeros(shape=(len(k1),3))
    k1_vec[:, 2] = k1
    # k2 is assumed to be in the zx-plane
    k2_vec = np.zeros(shape=(len(k2),3))
    k2_vec[:, 0] = k2 * np.sqrt(1.0 - csTheta**2)
    k2_vec[:, 2] = k2 * csTheta
    
    k1s, omega1s, sng1s, k2s, omega2s, sng2s, csTheta12s = _setup_computations_Maldague(k1_vec, omega1, k2_vec, omega2)
    val = _real_chi_2_0_CV(eta_bar, k1s[0, :], omega1s[0, :], sng1s[0, :], k2s[0, :], omega2s[0, :], sng2s[0, :], csTheta12s[0, :], beta, hbar, m, ms) + 1j * _imag_chi_2_0_CV(eta_bar, k1s[0, :], omega1s[0, :], sng1s[0, :], k2s[0, :], omega2s[0, :], sng2s[0, :], csTheta12s[0, :], beta, hbar, m, ms)

    # Points of intereset
    points_I = np.zeros(shape=(6, 4))
    for i in range(6):
        points_I[i, :] = _generate_I_points_Maldague(k1s[0, i], omega1s[0, i], k2s[0, i], omega2s[0, i], csTheta12s[0, i], beta, hbar, m)

    return val, points_I

def _ideal_quadratic_response_Maldague(k1_vec, omega1, k2_vec, omega2, csTheta, eta, beta, hbar, m, lower, reltol, abstol, limit, tol_upper, points_n, ms, force_output=True):
    # Allocate data
    res = np.zeros(shape=(len(csTheta), ), dtype=complex)
    err = np.zeros(shape=(len(csTheta), ), dtype=complex)

    # Pre-compute angle computations and vector setup
    k1s, omega1s, sng1s, k2s, omega2s, sng2s, csTheta12s = _setup_computations_Maldague(k1_vec, omega1, k2_vec, omega2) 
    X = np.zeros(shape=(len(csTheta), 47))
    X[:, 0:42] = np.concatenate( (k1s, omega1s, sng1s, k2s, omega2s, sng2s, csTheta12s), axis=1 )
    X[:, 42] = beta
    X[:, 43] = eta
    X[:, 44] = hbar
    X[:, 45] = m
    X[:, 46] = ms

    # Set upper limit of supression.
    max_suppression = 1e-30

    # Points of intereset
    points_I = np.zeros(shape=(6, 4))
    points_FD = np.array( [(max(0.0, eta) + n) for n in range(-points_n, points_n+1)] ) # Special points from thermal factor.

    # Loop over all computations
    for idx in range(len(csTheta)):
        # Compute points
        for i in range(6):
            points_I[i, :] = _generate_I_points_Maldague(k1s[idx, i], omega1s[idx, i], k2s[idx, i], omega2s[idx, i], csTheta12s[idx, i], beta, hbar, m)

        # Set interval
        eta_or_zero = max(0.0, eta)
        low  = max(lower, eta_or_zero + np.log(tol_upper))
        high = min( max(eta_or_zero - np.log(tol_upper), np.max(points_I[:, 0])+3, np.max(points_I[:, 1])+3), eta - np.log(max_suppression)) # Make sure non-zero part of imag is included but also limit max suppression.

        points_all = np.concatenate( (points_I.flatten(), points_FD) )
        points_all = filter_close_values(points_all,  1e-10) # Remove duplicate points

        # Perform numerical integration
        points = _get_points_I(low, high, points_all)
        quad_output = quad(_real_chi_2_0_integrand_Maldague, low, high,
                                args=tuple(X[idx, :]), 
                                points=points, full_output=1, epsabs=abstol, epsrel=reltol, limit=limit)

        if (len(quad_output) > 3):
          message = quad_output[3]
          if not (force_output):
            raise ValueError("Integrator 'quad' failed when computing real part with error:\n %s"%(message))
          else:
            warnings.warn("Integrator 'quad' failed when computing real part with error:\n %s"%(message), RuntimeWarning)
            res[idx] += quad_output[0]
            err[idx] += quad_output[1]
        else:
          res[idx] += quad_output[0]
          err[idx] += quad_output[1]

        quad_output = quad(_imag_chi_2_0_integrand_Maldague, low, high,
                                args=tuple(X[idx, :]), 
                                points=points, full_output=1, epsabs=abstol, epsrel=reltol, limit=limit)

        if (len(quad_output) > 3):
          message = quad_output[3]
          if not (force_output):
            raise ValueError("Integrator 'quad' failed when computing imag part with error:\n %s"%(message))
          else:
            warnings.warn("Integrator 'quad' failed when computing imag part with error:\n %s"%(message), RuntimeWarning)
            res[idx] += 1j * quad_output[0]
            err[idx] += 1j * quad_output[1]
        else:
          res[idx] += 1j * quad_output[0]
          err[idx] += 1j * quad_output[1]
    
    return res

