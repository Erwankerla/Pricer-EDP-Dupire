import numpy as np
import pandas as pd 
from scipy.stats import norm
from scipy.integrate import quad_vec

from Model import *
from Option import *

class Price:

    def __init__(self, price, stdError=None, delta=None, gamma=None):
        self.price = price
        self.stdError = stdError
        self.delta = delta
        self.gamma = gamma

 
class PricerMC: 

    def __init__(self, model: IModel, nbPaths, nbTimeSteps, varianceReductionMethod = None):
        self.model = model
        self.nbPaths = nbPaths
        self.nbTimeSteps = nbTimeSteps
        self.varianceReductionMethod = varianceReductionMethod

    def Price(self, option: IOption):
        timeToMaturity = option.timeToMaturity
        rate = self.model.rate

        simulations = self.model.runSimulation(self.nbPaths, self.nbTimeSteps, timeToMaturity, self.varianceReductionMethod)

        if self.varianceReductionMethod is None:
            S = simulations.paths
            lastSpots = S[:, -1]
            payoffs = np.array([option.payoff(lastSpots[i]) for i in range(len(lastSpots))])

            meanPayoffs = np.mean(payoffs)
            stdError = np.std(payoffs, ddof=1) / np.sqrt(self.nbPaths)

            return Price(np.exp(- rate * timeToMaturity) * meanPayoffs, stdError)
        
        elif self.varianceReductionMethod == VarianceReductionMethod.ANTITHETIQUE:

            S, S_antithetique = simulations.paths, simulations.varianceReductionPaths[0] #[0] car c'est un tuple avec args*
            lastSpots, lastSpotsAntithetique = S[:, -1], S_antithetique[:, -1]

            payoffs = np.array([option.payoff(lastSpots[i]) for i in range(len(lastSpots))])
            payoffsAntithetique = np.array([option.payoff(lastSpotsAntithetique[i]) for i in range(len(lastSpotsAntithetique))])

            combined_payoffs = (payoffs + payoffsAntithetique) / 2
            meanPayoffs = np.mean(combined_payoffs)
            stdError = np.std(combined_payoffs, ddof=1) / np.sqrt(self.nbPaths)

            return Price(np.exp(-rate * timeToMaturity) * meanPayoffs, stdError)
        else:
            return Price(0, 0)


    def getSrtikeForward(self, startDate):
        
        if (pd.to_datetime(startDate).normalize() - pd.Timestamp.now().normalize()).days > 0:
            timeToMaturity = (pd.to_datetime(startDate).normalize - pd.Timestamp.now().normalize()).days / 365.25

            simulations = self.model.runSimulation(self.nbPaths, self.nbTimeSteps, timeToMaturity)

            S = simulations.paths
            lastSpots = S[:, -1]

            return np.mean(lastSpots)
        
        elif (pd.to_datetime(startDate).normalize() - pd.Timestamp.now().normalize()).days == 0:
            return self.model.spot
        
        else:
            raise ValueError("La date de début doit être dans le futur pour avoir le strike forward. Rentrez des srtikes manuellement s'il vous plait.")

    


class PricerCF:

    def __init__(self, model: IModel):
        self.model = model

    def Price(self, option: IOption):
        if isinstance(option, VanillaOption):
            return self.PriceVanillaOption(option)
        else:
            return Price(0, 0)
        
    def PriceVanillaOption(self, option: IOption):
        if isinstance(self.model, BSModel):
            rate = self.model.rate
            sigma = self.model.sigma
            dividend = self.model.dividend
            
            if isinstance(option.maturityDate, float):
                timeToMaturity = option.maturityDate
            else:
                timeToMaturity = option.timeToMaturity
            
            strike = option.strike
            omega = option.omega
            spot = self.model.spot

            d1 = (np.log(spot / strike) + (rate - dividend + 0.5 * sigma**2) * timeToMaturity) / (sigma * np.sqrt(timeToMaturity))
            d2 = d1 - sigma * np.sqrt(timeToMaturity)

            priceCF = omega * (np.exp(- dividend * timeToMaturity) * spot * norm.cdf(omega * d1) - strike * np.exp(-rate * timeToMaturity) * norm.cdf(omega * d2))

            return Price(priceCF, 0)
        
        elif isinstance(self.model, HestonModel):
            rate = self.model.rate
            sigma = self.model.sigma
            dividend = self.model.dividend
            
            if isinstance(option.maturityDate, float):
                timeToMaturity = option.maturityDate
            else:
                timeToMaturity = option.timeToMaturity
            
            strike = option.strike
            omega = option.omega
            spot = self.model.spot
            forward = spot * np.exp((rate - dividend) * timeToMaturity)
            
            integrand = lambda u: np.real(np.exp(-1j*u*np.log(strike)) / (1j*u) * (self.model.phi(u - 1j, timeToMaturity) - strike * self.model.phi(u, timeToMaturity)))
            integral = quad_vec(integrand, 0, np.inf)[0]

            price = np.exp(-rate * timeToMaturity) * (0.5 * (forward - strike) + integral / np.pi)

            if omega == -1:
                price = price - np.exp(-rate * timeToMaturity) * (forward - strike)

            return Price(price, 0)