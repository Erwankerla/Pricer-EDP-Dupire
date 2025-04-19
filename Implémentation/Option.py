import numpy as np 
import pandas as pd 
from enum import Enum
from typing import List


from Interfaces import IOption


class OptionType(Enum):
    CALL = 'call'
    PUT = 'put'


class Quote:

    def __init__(self, strike, impliedVol, optionType):
        self.strike = strike
        self.impliedVol = impliedVol
        self.optionType = optionType



class VanillaOption(IOption):

    def __init__(self, type, maturityDate, strike = None, startDate = None):
        super().__init__(strike, type, maturityDate)
        
        if not isinstance(maturityDate, float):
            if startDate is None:
                self.timeToMaturity = (pd.to_datetime(maturityDate).normalize() - pd.Timestamp.now().normalize()).days / 365.25
            else:
                self.timeToMaturity = (pd.to_datetime(maturityDate).normalize() - pd.to_datetime(startDate).normalize()).days / 365.25
        elif isinstance(maturityDate, (float, int)):
            self.timeToMaturity = maturityDate
        else:
            raise ValueError('Choose a correct format for maturityDate (datetime, float or int)')

        self.omega = 1 if self.type == OptionType.CALL else -1

    def payoff(self,spot):
        return np.maximum(self.omega*(spot - self.strike), 0)
    

class CallSpread(IOption):

    def __init__(self, strike, strike1, maturityDate, startDate = None):
        self.strike = strike
        self.strike1 = strike1
        self.maturityDate = maturityDate

        if not isinstance(self.maturityDate, float):
            if startDate is None:
                self.timeToMaturity = (pd.to_datetime(maturityDate).normalize() - pd.Timestamp.now().normalize()).days / 365.25
            else:
                self.timeToMaturity = (pd.to_datetime(maturityDate).normalize() - pd.to_datetime(startDate).normalize()).days / 365.25
        elif isinstance(maturityDate, (float, int)):
            self.timeToMaturity = maturityDate
        else:
            raise ValueError('Choose a correct format for maturityDate (datetime, float or int)')

    def payoff(self, spot):
        if spot < self.strike:
            return 0
        elif self.strike <= spot <= self.strike1:
            return spot - self.strike
        return self.strike1 - self.strike



class OptionStrategy:

    def __init__(self, options: List[IOption]):
        self.options = options

    def payoff(self, spot): #pour stategy il faut que payoff retourne une liste de cashFlow
        return np.sum(self.options[i].payoff(spot) for i in range(len(self.options)))
    
    