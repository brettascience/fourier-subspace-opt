'''Functions for finding the optimal 1-dimensional subspace of a high dimensional space,
where optimal means that the projection has the largest amplitude for a given frequency
of interest'''

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize


def projection_fourier_amplitude(w, X, times, period_length, return_complex=False):
    '''
    Computes Fourier power coefficient for data projection `w` and
    frequency `period_length`
    
    w: projection vector (,dimensions) <- parameters to fit
    X: data matrix (time points, dimensions)
    times: time points in minutes associated with each data point (time points,)
    period_length: period (1/frequency) in days for which we are trying to maximize amplitude
    return_complex: Boolean which allows for returning the complex fourier coefficient
        instead of the amplitude (absolute value). Useful for getting phase
    '''    
    proj = np.dot(X, w)
    
    mins_per_day = 60*24
    cos_t = np.cos(times*2*np.pi/(period_length*mins_per_day))
    sin_t = np.sin(times*2*np.pi/(period_length*mins_per_day))
    
    complex_sum = np.sum(proj*cos_t + 1j*proj*sin_t)
    
    if return_complex is False:
        return -np.abs(complex_sum) # must be negative for minimization
    elif return_complex is True:
        return complex_sum

    
def optimize_projection(vector_df, times, period_length, standardize=True):
    '''
    Find direction of optimal projection for a high-dimensional dataset
    that maximizes variance at a requested time frequency.
    
    vector_df: Data Frame of vectors at each time point (time points, dimensions)
    times: time points associated with each data point (time points,)
    period_length: frequency for which we are trying to maximize projection amplitude
    standardize: Boolean. When True, dataset is standardized (zero mean, unit variance)
    '''
    # randomly initialize projection
    w_0 = np.random.normal(0, 1, vector_df.shape[1])
    w_0 = w_0 / np.sqrt(np.sum(w**2)) # unit-vectorize
    
    # constrain the solution so that w is a unit vector
    constraints = ( {'type': 'eq', 'fun': lambda x:  np.sqrt(np.sum(x**2))-1} )
    
    if standardize is True:
        scaler = StandardScaler()
        vector_df = scaler.fit_transform(vector_df)
    
    res = minimize(
                projection_fourier_amplitude,
                w_0,
                args=(vector_df, times, period_length),
                constraints=constraints
                )
    
    # add the optimal phase and the data projected onto w into the results object
    res.phase = np.angle(projection_fourier_amplitude(res.x,
                      vector_df,
                      times,
                      period_length,
                      return_complex=True)
                      )
    res.projection = np.dot(vector_df, res.x)
    
    return res
