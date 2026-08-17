import numpy as np
from scipy.integrate import quad
import warnings
from .utils import jit_integrand_function, _get_points_I

@jit_integrand_function
def _chi0_k2_0_integrad_Maldague(X):
    eta_bar = X[0]
    k1      = X[1]
    beta    = X[2]
    eta     = X[3]
    hbar    = X[4]
    m       = X[5]
    ms      = X[6]

    y1_int = hbar * k1 / np.sqrt(2*m*eta_bar/beta)
    pre = (2*m)**(3/2)/(hbar**3) * ms/(4*(2*np.pi)**2)
    val = pre/np.sqrt(eta_bar/beta) * np.log(np.abs((2+y1_int)/(2-y1_int))) / y1_int

    return val/(4*np.cosh((eta_bar - eta)/2)**2)

def _generate_all_points_chi0_k2_0(k1, eta, beta, hbar, m, points_n):
    points_I = [0.25*beta*hbar**2*k1**2/(2*m)]
    points_FD = np.array( [(max(0.0, eta) + n) for n in range(-points_n, points_n+1)] )
    return np.concatenate((points_I, points_FD))
    
    
def _chi0_k2_0_Maldague(y1, qF, eta, beta, hbar, m, lower, reltol, abstol, limit, tol_upper, points_n, ms, force_output=True):
    # Work in dimentionfull units.
    k1     = qF * y1

    # Allocate data
    res = np.zeros(shape=(len(y1), ))
    err = np.zeros(shape=(len(y1), ))

    # Set upper limit of supression.
    max_suppression = 1e-30

    # Loop over all computations
    for idx in range(len(y1)):
        # Compute points
        points_all = _generate_all_points_chi0_k2_0(k1[idx], eta, beta, hbar, m, points_n)

        # Set interval
        eta_or_zero = max(0.0, eta)
        low  = max(lower, eta_or_zero + np.log(tol_upper))
        high = min( eta_or_zero - np.log(tol_upper), eta - np.log(max_suppression) ) # Make sure non-zero part of imag is included but also limit max suppression.

        # Remove duplicate points
        if ( (np.abs(points_all[0]) - np.abs(points_all[1])) < 1e-10 ):
            points_all = np.delete(points_all, 1)

        # Perform numerical integration
        points = _get_points_I(low, high, points_all)
        quad_output = quad(_chi0_k2_0_integrad_Maldague, low, high,
                        args=(k1[idx], beta, eta, hbar, m, ms), 
                        points=points, full_output=1, epsabs=abstol, epsrel=reltol, limit=limit)

        if (len(quad_output) > 3):
          message = quad_output[3]
          if not (force_output):
            raise ValueError("Integrator 'quad' failed when computing real part with error:\n %s"%(message))
          else:
            warnings.warn("Integrator 'quad' failed when computing real part with error:\n %s"%(message), RuntimeWarning)
            res[idx] = quad_output[0]
            err[idx] = quad_output[1]
        else:
          res[idx] = quad_output[0]
          err[idx] = quad_output[1]
    
    return res


@jit_integrand_function
def _chi0_k1_0_k2_0_integrad_Maldague(X):
    eta_bar = X[0]
    beta    = X[1]
    eta     = X[2]
    hbar    = X[3]
    m       = X[4]
    ms      = X[5]

    pre = (2*m)**(3/2)/(hbar**3) * ms/(4*(2*np.pi)**2)
    val = pre/np.sqrt(eta_bar/beta)

    return val/(4*np.cosh((eta_bar - eta)/2)**2)

def _generate_all_points_chi0_k1_0_k2_0(eta, points_n):
    points_FD = np.array( [(max(0.0, eta) + n) for n in range(-points_n, points_n+1)] )
    return points_FD
    
    
def _chi0_k1_0_k2_0_Maldague(eta, beta, hbar, m, lower, reltol, abstol, limit, tol_upper, points_n, ms, force_output=True):
    # Set upper limit of supression.
    max_suppression = 1e-30

    # Compute points
    points_all = _generate_all_points_chi0_k1_0_k2_0(eta, points_n)

    # Set interval
    eta_or_zero = max(0.0, eta)
    low  = max(lower, eta_or_zero + np.log(tol_upper))
    high = min( eta_or_zero - np.log(tol_upper), eta - np.log(max_suppression) ) # Make sure non-zero part of imag is included but also limit max suppression.

    # Perform numerical integration
    points = _get_points_I(low, high, points_all)
    quad_output = quad(_chi0_k1_0_k2_0_integrad_Maldague, low, high,
                    args=(beta, eta, hbar, m, ms), 
                    points=points, full_output=1, epsabs=abstol, epsrel=reltol, limit=limit)

    if (len(quad_output) > 3):
      message = quad_output[3]
      if not (force_output):
        raise ValueError("Integrator 'quad' failed when computing real part with error:\n %s"%(message))
      else:
        warnings.warn("Integrator 'quad' failed when computing real part with error:\n %s"%(message), RuntimeWarning)
        res = quad_output[0]
        err = quad_output[1]
    else:
      res = quad_output[0]
      err = quad_output[1]
    
    return res


