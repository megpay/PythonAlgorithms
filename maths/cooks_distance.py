"""
Cook's Distance is used to estimate the influence of a data point in 
in least squares regression. 

Cook's Distance removes each data point one at a time, and measures the effect. Large
Cook's Distance values for an individual data point indicates that data point should 
be further investigated. A cutoff for what is large needs to be decided upon, and 1 is
often used. 

The algorithm works as follows:
For each data point in the regression, remove the point from the set 
    and calculate the effect of removing that point. 

D_i = (sum over all other points(y_actual - y_observed)^2) / (rank * MSE^2) 


https://en.wikipedia.org/wiki/Cook's_distance
"""

import numpy as np
from machine_learning.loss_functions import mean_squared_error
from sklearn import datasets
from sklearn.linear_model import LinearRegression

def calculate_cooks_distance(y_observed: np.ndarray, y_fitted: np.ndarray, rank: int) -> np.ndarray:
    """Calculate Cook's Distance
        Input: 
            y_observed: numpy array of observed y values
            y_fitted: numpy array of fitted y values from linear regression model
            rank: int representing the number of coefficients
        Output:
            cooks_distance: numpy array of Cook's distance for each y value.
         
    """
    mse = mean_squared_error(y_observed, y_fitted)
    y_difference_squared = (y_observed - y_fitted)**2

    if isinstance(rank) is not int:
        msg = f"Rank is an integer representing the number of predictors. Input: {rank}"
        raise TypeError(msg)
    
    if len(y_observed) != len(y_fitted) or len(y_observed) == 0:
        msg = f"The arrays of observed and fitted values must be equal length and non-empty. 
            Currently observed = {len(y_observed)} and fitted = {len(y_fitted)}"
        raise ValueError(msg)

    # This is leave one out, so summing over all and then individually subtracting.
    summed_difference = sum(y_difference_squared)
    for item in np.nditer(y_difference_squared):
        k = (summed_difference - item) / (rank * mse)
        
if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
    mdl = LinearRegression()
    mdl.fit(df.Jumps.shape(-1, 1), df.Pulse)

    main()

