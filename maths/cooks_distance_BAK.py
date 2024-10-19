"""
Cook's Distance is used to estimate the influence of a data point in 
in least squares regression. 

Cook's Distance removes each data point and measures the effect of removing the
data point. 

The algorithm works as follows:
For each data point in the regression, remove the point from the set 
    and calculate the effect of removing that point. 

D_i = (sum over all other points(y_actual - y_observed)^2) / (rank * MSE^2) 


https://en.wikipedia.org/wiki/Cook's_distance
"""
from machine_learning.loss_functions.mean_squared_error import mean_squared_error
import numpy as np

def calculate_cooks_distance(y_observed: array, y_fitted: array, rank: int) -> array:
    """Calculate Cook's Distance
        Input: 
            y_observed: numpy array of observed y values
            y_fitted: numpy array of fitted y values from linear regression model
            rank: int representing the number of coefficients
        Output:
            cooks_distance: numpy array of Cook's distance for each y value.
         
    """
    import numpy as np
    _mse = mean_squared_error(y_observed, y_fitted)
    _y_difference_squared = (y_observed - y_fitted)**2

    if isinstance(rank) is not int:
        msg = f"Rank is an integer representing the number of predictors. Input: {rank}"
        raise TypeError(msg)
    
    if len(y_observed) != len(y_fitted):
        msg = f"The arrays of observed and fitted values must be equal length. Currently 
            observed = {len(y_observed)} and fitted = {len(y_fitted)}"
        raise ValueError(msg)

    if len(y_observed) == 0:
        raise ValueError("The y value arrays must not be empty")

    _summed_difference = sum(_y_difference_squared)
    for item in np.nditer(_y_difference_squared):
        k = (_summed_difference - item) / (rank * _mse)
        
