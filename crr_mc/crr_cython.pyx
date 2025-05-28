# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX

cdef double rand_uniform():
    return rand() / (<double>RAND_MAX)

cdef double compute_payoff(double[:] trajectory, int T):
    cdef double max_val = trajectory[0]
    cdef int i
    for i in range(1, T + 1):
        if trajectory[i] > max_val:
            max_val = trajectory[i]
    return max_val

def evaluate_mc_cython(int M, double initial_value, double up_factor, 
                      double down_factor, double p_star, double interest_rate, 
                      int T):
    cdef int i, j
    cdef double[:] payoffs = np.empty(M, dtype=np.float64)
    cdef double[:] trajectory = np.empty(T + 1, dtype=np.float64)
    cdef double discount_factor = (1 + interest_rate) ** (-T)
    
    for i in range(M):
        trajectory[0] = initial_value
        for j in range(1, T + 1):
            if rand_uniform() < p_star:
                trajectory[j] = trajectory[j-1] * up_factor
            else:
                trajectory[j] = trajectory[j-1] * down_factor
        payoffs[i] = compute_payoff(trajectory, T)
    
    cdef double[:] cumsum_payoffs = np.cumsum(np.asarray(payoffs)) * discount_factor
    cdef double[:] result = np.empty(M, dtype=np.float64)
    for i in range(M):
        result[i] = cumsum_payoffs[i] / (i + 1)
    
    return np.asarray(result)