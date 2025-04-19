import numpy as np 
from scipy.stats import norm
from scipy.optimize import brentq


class Solver:

    def __init__(self):
        pass
        
    def computeImpliedVolatility(self, price, omega, spot, timeToMaturity, strike, rate, dividend):
        forward = spot * np.exp((rate - dividend) * timeToMaturity)

        def f(x):
            factor = x * np.sqrt(timeToMaturity)
            d1 = np.log(forward / strike) / factor + 0.5 * factor
            d2 = d1 - factor

            return omega * np.exp(- rate * timeToMaturity) * (forward * norm.cdf(omega * d1) - strike * norm.cdf(omega * d2)) - price

        try:
            implied_vol = brentq(f, 0.00001, 20)
            return implied_vol
        except ValueError as e:
            print("Erreur lors du calcul de la volatilité implicite:", e)
            return None
        


    def computeBasketImpliedVolatility(self, price, omega, timeToMaturity, strike, forward, rate):

        def f(x):
            factor = x * np.sqrt(timeToMaturity)
            d1 = np.log(forward / strike) / factor + 0.5 * factor
            d2 = d1 - factor

            return omega * np.exp(- rate * timeToMaturity) * (forward * norm.cdf(omega * d1) - strike * norm.cdf(omega * d2)) - price

        try:
            implied_vol = brentq(f, 0.000000001, 20)
            return implied_vol
        except ValueError as e:
            print("Erreur lors du calcul de la volatilité implicite:", e)
            return None
        
        
        
    def getVega(self, forward, sigma, timeToMaturity, strike, rate):
        factor = sigma * np.sqrt(timeToMaturity)
        d1 = np.log(forward / strike) / factor + 0.5 * factor
        d2 = d1 - factor

        phi = np.exp(-0.5 * d2 * d2) / np.sqrt(2 * np.pi)

        return strike * np.exp(- rate * timeToMaturity) * phi * np.sqrt(timeToMaturity)