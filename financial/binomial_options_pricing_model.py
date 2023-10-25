"""

https://en.wikipedia.org/wiki/Binomial_options_pricing_model
"""

from math import exp, sqrt
import numpy as np

class BinomialOptionsPrice:
    """
    This calculates the binomial options price given a call or put option. 
    Input:
        expiration_time: int This is the time until the option expires
        stock_price: float This is the stock price today
        strike_price: float This is the strike price at the time the option expires
        interest_rate: float The interest rate for the option
        sigma: float The volitility of the stock
        tree_height: int The number of levels of the tree
        option_type: str Either 'call' or 'put'. Any other value raises an error
    Output:
        options_price: float 
    """
    def __init__(self, 
        expiration_time: int, 
        stock_price: float, 
        strike_price: float, 
        interest_rate: float, 
        sigma: float, 
        tree_height: int,
        option_type: str) -> None:
        self.tree_height = tree_height
        self.sigma = sigma
        self.interest_rate = interest_rate
        self.strike_price = strike_price
        self.stock_price = stock_price
        self.expiration_time = expiration_time
        self.option_type = option_type

        if self.expiration_time / self.tree_height >= self.sigma **2 / self.interest_rate**2:
            raise ValueError("Time step too big. This will cause the probability to be outside [0, 1]")

    def calculate_down(self) -> float:
        """Calculates the down factor"""
        return exp(self.sigma * sqrt(self.expiration_time/self.tree_height))
    
    def calculate_up(self) -> float:
        """Calculates the up factor"""
        return 1/self.calculate_down()

    def calculate_rate_delta_t(self) -> float:
        """Calculates"""
        return exp(-self.interest_rate * self.expiration_time/self.tree_height)
    
    def calculate_s_n(self, placement):
        return self.stock_price * self.calculate_up()**(self.tree_height - placement)

    def calculate_leaf_values(self):
        _leaf_list = []
        for i in range(self.tree_height + 1):
            if self.option_type == 'call':
                _leaf_list.append(max(self.calculate_s_n(i) - self.strike_price, 0))
            elif self.option_type == 'put':
                _leaf_list.append(max(self.strike_price - self.calculate_s_n(i), 0))
            else:
                raise ValueError("Option type must be either 'call' or 'put'")
        return np.array(_leaf_list)
    
    def calculate_node_values(self, i):
        
        return [i] * (self.tree_height + 1)

    def generate_other_rows(self):
        _nodes = np.zeros((self.tree_height, self.tree_height + 1))
        _leaves = self.calculate_leaf_values()
        # calculate the nodes here
        i = self.tree_height - 1
        while i >= 0:
            _nodes[i, ] = self.calculate_node_values(i)
            i -= 1
        return np.vstack((_leaves, _nodes))  



if __name__ == "__main__":
    import doctest

    doctest.testmod()
    tmp = BinomialOptionsPrice(1, 4.5, 5.6, 0.23, 1, 5, 'call')
    print(tmp.generate_other_rows())