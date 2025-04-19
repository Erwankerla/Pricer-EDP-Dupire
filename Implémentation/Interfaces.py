from abc import ABC, abstractmethod


class IOption(ABC):
    
    @abstractmethod
    def __init__(self, strike, option_type, maturity):
        self.strike = strike
        self.type = option_type
        self.maturityDate = maturity

    @abstractmethod
    def payoff(self, spot):
        pass


class IModel(ABC):
    
    @abstractmethod
    def __init__(self, spot, rate, dividend):
        self.spot = spot
        self.rate = rate
        self.dividend = dividend

    @abstractmethod
    def runSimulation(self, nbPaths, nbTimeSteps, timeToMaturity, varianceReductionMethod = None):
        pass