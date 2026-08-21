import math
import numpy as np
from scipy.special import wofz
from finitediff import get_weights
from .principal_value_integration import principal_value_integration_f_over_x
from .utils import _norm_single

def plasma_dispersion_function(z):
    return 1j*np.sqrt(np.pi) * wofz(z)

def _Y_integrand_numerator_centered_real(x, a, b, c, m):
    xp = c + x 
    Z = plasma_dispersion_function(a + b*xp)
    return np.exp(-xp**2) * xp**m * np.real(Z)

def _Y_integrand_numerator_centered_imag(x, a, b, c, m):
    xp = c + x 
    Z = plasma_dispersion_function(a + b*xp)
    return np.exp(-xp**2) * xp**m * np.imag(Z)

def _generlized_plasma_dispersion_function_m_1(a, b, c, sgn_c, m, eta_pol=1e-4, reltol=1e-6, abstol=1e-8, tol_upper=1e-8, n_points=5):
    # Evaluation of principal value part
    high = c + np.sqrt(-np.log(tol_upper))
    points = [n + c for n in range(-n_points, n_points+1)]
    if (np.abs(b) > 1e-6):
        points.append((1.0-a-b*c)/b)
    if not (np.abs(c) < 1e-6):
        points = points + [n - c for n in range(-n_points, n_points+1)]
        if (np.abs(b) > 1e-6):
            points.append((1.0-a+b*c)/b)
    points = np.array(points)
    val_real = principal_value_integration_f_over_x(_Y_integrand_numerator_centered_real, args=(a,b,c,m), 
                                                    eta=eta_pol, reltol=reltol, abstol=abstol, high=high, points=points) 
    val_imag = principal_value_integration_f_over_x(_Y_integrand_numerator_centered_imag, args=(a,b,c,m), 
                                                    eta=eta_pol, reltol=reltol, abstol=abstol, high=high, points=points)
    principal_value = val_real + 1j*val_imag


    # evaluation of pool contribution.
    pol_contribution = np.sign(sgn_c) * 1j * np.pi * ( _Y_integrand_numerator_centered_real(0.0,a,b,c,m)
                                                  + 1j*_Y_integrand_numerator_centered_imag(0.0,a,b,c,m) )

    return (principal_value + pol_contribution) / np.sqrt(np.pi)

def generlized_plasma_dispersion_function_m_1(a, b, c, sgn_c, m, eta_pol=1e-4, reltol=1e-6, abstol=1e-8):
    # Setup for array operations
    a = np.atleast_1d(np.array(a))
    b = np.atleast_1d(np.array(b))
    c = np.atleast_1d(np.array(c))

    a, b, c = np.broadcast_arrays(a, b, c)
    res = np.zeros(shape=a.shape, dtype=complex)

    for i, (_a, _b, _c) in enumerate(zip(a, b, c)):
        res[i] = _generlized_plasma_dispersion_function_m_1(_a, _b, _c, sgn_c, m, eta_pol=eta_pol, reltol=reltol, abstol=abstol)
    
    return res

def generlized_plasma_dispersion_function_m_n(a, b, c, sgn_c, m, n, dc=1e-4, eta_pol=1e-4, reltol=1e-6, abstol=1e-8):
    if (np.abs(int(n) - n) > 1e-16):
        raise ValueError(f"n parameter must be an integer.")
    if (np.abs(int(m) - m) > 1e-16):
        raise ValueError(f"m parameter must be an integer.")
    
    # Clean input
    m = int(m)
    n = int(n)

    # Perform computation based on the different cases of n:
    if (n < 0):
        raise ValueError(f"Value of parameter 'n' must be greater than 0.")
    elif (n == 0):
        raise ValueError(f"The case of 'n = 0' is not implemented.")
    elif (n == 1):
        return generlized_plasma_dispersion_function_m_1(a, b, c, sgn_c, m, 
                                                         eta_pol=eta_pol, reltol=reltol, abstol=abstol)
    else:
        # Compute higher order functions based on finite difference derivatives.

        # Setup for array operations
        a = np.atleast_1d(np.array(a))
        b = np.atleast_1d(np.array(b))
        c = np.atleast_1d(np.array(c))

        a, b, c = np.broadcast_arrays(a, b, c)

        # Setup weights for finte difference
        order = n - 1
        if (order % 2 == 0): # even case
            dcs = dc*np.array([i for i in range(-(order//2),order//2 + 1)])
        else:
            dcs = dc*np.array([i for i in range(-(order+1)//2,(order+1)//2+1) if not (i == 0)])
        weights = get_weights(dcs, 0.0, maxorder=order)[:, -1]


        # Compute values for Y_(m,1)
        a_s = a[:, None] + 0.0 * dcs[None, :] # Make sure a and b, has the same shape as c
        b_s = b[:, None] + 0.0 * dcs[None, :] # Make sure a and b, has the same shape as c
        c_s = c[:, None] + dcs[None, :]
        vals = generlized_plasma_dispersion_function_m_1(a_s.flatten(), 
                                                         b_s.flatten(), 
                                                         c_s.flatten(), 
                                                         sgn_c, m, eta_pol=eta_pol, reltol=reltol, abstol=abstol)

        return np.sum( weights[None, :] * np.reshape(vals, shape=(len(c),len(dcs))), axis=1) / math.factorial(n-1)

def M_sum_1_2(k1_vec, omega1, k2_vec, omega2, n, vth, dc=1e-4, eta_pol=1e-4, reltol=1e-6, abstol=1e-8):
    omega = omega1 + omega2
    k_vec = k1_vec + k2_vec

    k_cross_k1 = _norm_single(np.cross(k_vec, k1_vec))
    norm_k     = _norm_single(k_vec)

    a1 = omega1 * norm_k / (np.sqrt(2.0) * vth * k_cross_k1)
    b1 = - np.dot(k1_vec, k_vec) / k_cross_k1

    a2 = omega2 * norm_k / (np.sqrt(2.0) * vth * k_cross_k1)
    b2 = - np.dot(k2_vec, k_vec) / k_cross_k1

    # s = -a/b
    s = omega / (np.sqrt(2.0) * vth * norm_k)

    pre = n / (4*vth**4 * k_cross_k1 * norm_k**2)
    return pre * ( generlized_plasma_dispersion_function_m_n(a1, b1, s, 1.0, 0, 3, dc=dc, eta_pol=eta_pol, reltol=reltol, abstol=abstol) 
                 + generlized_plasma_dispersion_function_m_n(a2, b2, s, 1.0, 0, 3, dc=dc, eta_pol=eta_pol, reltol=reltol, abstol=abstol) )

def M_1_sum_2(k1_vec, omega1, k2_vec, omega2, n, vth, dc=1e-4, eta_pol=1e-4, reltol=1e-6, abstol=1e-8):
    omega = omega1 + omega2
    k_vec = k1_vec + k2_vec

    k_cross_k1 = _norm_single(np.cross(k_vec, k1_vec))
    norm_k1    = _norm_single(k1_vec)

    a2 = omega2 * norm_k1 / (np.sqrt(2) * vth * k_cross_k1)
    b2 = - np.dot(k1_vec, k2_vec) / k_cross_k1

    a  = omega * norm_k1 / (np.sqrt(2) * vth * k_cross_k1)
    b  = - np.dot(k1_vec, k_vec) / k_cross_k1

    # s = -a1/b1
    s = omega1 / (np.sqrt(2.0) * vth * norm_k1)

    pre = n / (4*vth**4 * k_cross_k1 * norm_k1**2)
    return pre * ( generlized_plasma_dispersion_function_m_n(a2, b2, s, 1.0, 0, 3, dc=dc, eta_pol=eta_pol, reltol=reltol, abstol=abstol) 
                 - generlized_plasma_dispersion_function_m_n(a,  b,  s, 1.0, 0, 3, dc=dc, eta_pol=eta_pol, reltol=reltol, abstol=abstol) )

def _classical_ideal_quadratic_response(k1_vec, omega1, k2_vec, omega2, n, beta, m, dc, eta_pol, reltol, abstol):
    vth = 1.0/np.sqrt(m*beta) 
    k_vec = k1_vec + k2_vec

    return (1.0/(2.0*m**2)) * ( np.dot(k1_vec,k2_vec)*np.dot(k_vec,k_vec)  * M_sum_1_2(k1_vec, omega1, k2_vec, omega2, n, vth, dc=dc, eta_pol=eta_pol, reltol=reltol, abstol=abstol)
                              + np.dot(k_vec,k2_vec)*np.dot(k1_vec,k1_vec) * M_1_sum_2(k1_vec, omega1, k2_vec, omega2, n, vth, dc=dc, eta_pol=eta_pol, reltol=reltol, abstol=abstol)
                              + np.dot(k1_vec,k_vec)*np.dot(k2_vec,k2_vec) * M_1_sum_2(k2_vec, omega2, k1_vec, omega1, n, vth, dc=dc, eta_pol=eta_pol, reltol=reltol, abstol=abstol) ) 

def classical_ideal_quadratic_response(k1, omega1, k2, omega2, csTheta, n, beta, m, dc=1e-4, eta_pol=1e-4, reltol=1e-6, abstol=1e-8):

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

    classical_quadratic_chi_0 = np.zeros(shape=omega1.shape, dtype=complex)
    for i, (_omega1, _k1, _omega2, _k2, _csTheta) in enumerate(zip(omega1, k1, omega2, k2, csTheta)):
        k1_vec = np.array([0, 0, 1]) * _k1
        k2_vec = np.array([0, np.sqrt(1 - _csTheta**2), _csTheta]) * _k2
        classical_quadratic_chi_0[i] = _classical_ideal_quadratic_response(k1_vec, _omega1, k2_vec, _omega2, n, beta, m, dc, eta_pol, reltol, abstol)[0]

    return classical_quadratic_chi_0

