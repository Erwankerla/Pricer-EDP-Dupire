import numpy as np
from scipy.optimize import minimize
from typing import List

from Solver import Solver
from Option import Quote, OptionType




def FirstDerivateAOA(k, theta, phiTheta, rho):
    return 0.5 * theta * phiTheta * (rho + (phiTheta*k + rho) / np.sqrt((phiTheta*k + rho)**2 + 1 - rho**2))
    
def SecondeDerivativeAOA(k, theta, phiTheta, rho):
    return 0.5 * theta * phiTheta * phiTheta * ((1-rho**2) / ((phiTheta*k + rho)**2 + 1 - rho**2)**(3/2))


def phi(theta, etha, gamma):
    return etha / (theta**gamma * (1 + theta)**(1-gamma))

def SVIAOA(k, timeToMaturity, theta, phiTheta, rho):
    return np.sqrt(0.5 * theta * (1 + rho * phiTheta * k + np.sqrt((phiTheta * k + rho)**2 + 1 - rho**2)) / timeToMaturity)



class SVI:

    def __init__(self, dicQuotes: dict[float, List[Quote]], forwardsRatesDivs: dict[float, List]) :
        self.dicQuotes = dicQuotes
        self.forwardsRatesDivs = forwardsRatesDivs
        self.timeToMaturities = list(dicQuotes.keys())
        self.vegasExpiries = {}
        max_vegas = []
        self.solver = Solver()

        self.maxVol = 2.0
        forward, rate, div = forwardsRatesDivs[self.timeToMaturities[0]]
        self.strikes = [0.01 * quote.strike * forward for quote in dicQuotes[list(dicQuotes.keys())[0]]]

        for timeToMaturity in self.timeToMaturities:
            forward, rate = forwardsRatesDivs[timeToMaturity][0], forwardsRatesDivs[timeToMaturity][1]
            quotes = dicQuotes[timeToMaturity]
            vegaExpiry = []
            
            for quote in quotes:
                strike = 0.01 * quote.strike * forward
                vega = self.solver.getVega(forward, quote.impliedVol, timeToMaturity, strike, rate)
                vegaExpiry.append(vega)

            self.vegasExpiries[timeToMaturity] = vegaExpiry
            max_vegas.append(max(vegaExpiry)) 

        self.penality = max(max_vegas) if max_vegas else 0

        self.Numberquotes = sum(len(quotes) for quotes in dicQuotes.values())

        self.parameters = {}
        self.surface = {}



    def SVICalibrationAOA(self, quotesMaturity, timeToMaturity):

        forward, rate, div = self.forwardsRatesDivs[timeToMaturity]
        implied_vols = np.array([quote.impliedVol for quote in quotesMaturity])
        log_strikes = np.log(np.array(self.strikes) / forward)

        vegas = self.vegasExpiries[timeToMaturity]
        
        def costFunction(parameters):
            sum_error = 0

            theta, etha, gamma, rho = parameters
            
            if theta <= 0:
                return 1e8
            
            if gamma < 0 or gamma > 0.5:
                return 1e8

            if etha > 2/np.sqrt(1 + np.abs(rho)):
                return 1e8
            
            if np.abs(rho) > 1:
                return 1e8
            
            phiTheta = phi(theta, etha, gamma)
            
            for i, k in enumerate(log_strikes):
                svi_value = 0.5 * theta * (1 + rho * phiTheta * k + np.sqrt((phiTheta * k + rho)**2 + 1 - rho**2)) / timeToMaturity
                vol_svi = np.sqrt(svi_value)
                sum_error += (vol_svi - implied_vols[i]) ** 2

            return sum_error / self.Numberquotes
        
        point1 = np.array([0.2, 1, 0.01, -0.5])
        point2 = np.array([0.1, 1.5, 0.5, -0.2])
        point3 = np.array([0.15, 0.5, 0.25, -0.7])
        point4 = np.array([0.25, 0.7, 0.12, -0.1])
        point5 = np.array([0.18, 1.4, 0.37, -0.8])

        initial_simplex = [point1, point2, point3, point4, point5]

        result = minimize(costFunction, x0=initial_simplex[0], method='Nelder-Mead', options={'initial_simplex': initial_simplex, 'maxiter': 1e3})

        return result.x


    def GetSVIValuesForMaturityAOA(self, quotesMaturity, strikes, timeToMaturity):
        result = []
        
        forward, rate, div = self.forwardsRatesDivs[timeToMaturity]
        
        parameters = self.SVICalibrationAOA(quotesMaturity, timeToMaturity)
        self.parameters[timeToMaturity] = parameters

        #print(parameters)
        
        theta, etha, gamma, rho = parameters
        phiTheta = phi(theta, etha, gamma)

    
        for strike in strikes:
            log_strike = np.log(strike / forward)
            implied_volatility = SVIAOA(log_strike, timeToMaturity, theta, phiTheta, rho)
            quote = next((q for q in quotesMaturity if q.strike == strike), None)

            if quote is None:
                optionType = OptionType.CALL if strike > forward else OptionType.PUT
            else:
                optionType = quote.optionType

            result.append(Quote(strike, implied_volatility, optionType))

        return result
    

    def CreateSurfaceSVIAOA(self, strikes=None):

        if strikes is None:
            min_strike = min(q.strike for quotes in self.dicQuotes.values() for q in quotes)
            max_strike = max(q.strike for quotes in self.dicQuotes.values() for q in quotes)
            strikes = [float(K) for K in range(int(1 * min_strike), int(1 * max_strike), 10)]

        strikes.sort()
        #changer pour mettre directement les logStrikes comme ca ca s'adapte à n'importe quels strikes
        for timeToMaturity in self.timeToMaturities:
            quotes_maturity = self.dicQuotes[timeToMaturity]
            
            if timeToMaturity > 0:
                self.surface[timeToMaturity] = self.GetSVIValuesForMaturityAOA(quotes_maturity, strikes, timeToMaturity)






































"""    def SVICalibration2D(self, quotesMaturity, timeToMaturity):
        
        vegas = np.array(self.vegasExpiries[timeToMaturity])
        implied_vols = np.array([quote.impliedVol for quote in quotesMaturity])
        forward, rate, div = self.forwardsRatesDivs[timeToMaturity]

        strikes = np.array([quote.strike for quote in quotesMaturity])
        log_strikes = np.log(strikes / forward)
        totalVariances = timeToMaturity * implied_vols ** 2
        Weigths = np.diag(vegas / vegas) # on prend les poids égaux à 1

        minVol = np.min(implied_vols)
        minLogStrikes, maxLogStrikes = np.min(log_strikes), np.max(log_strikes)

        def cost_function(params):

            sigma, m = params
            if sigma < minVol or sigma > self.maxVol or m < minLogStrikes or m > maxLogStrikes:
                return 1e8

            X = np.zeros((len(quotesMaturity), 3))
            for i, k in enumerate(log_strikes):
                y = (k - m) / sigma
                z = np.sqrt(1 + y ** 2)
                X[i, 0] = 1
                X[i, 1] = y
                X[i, 2] = z

            XT_W_X = X.T @ Weigths @ X

            if np.linalg.det(XT_W_X) == 0:
                return 1e8
            
            B = np.linalg.inv(XT_W_X) @ X.T @ Weigths @ totalVariances
            a, c, d = B[0], B[1], B[2]
            b = d / sigma
            rho = c / (b * sigma)

            if a + b * sigma * np.sqrt(1 - rho ** 2) < 0 or rho < -1 or rho > 1 or b < 0:
                return 1e8

            sum_error = 0
            for i, k in enumerate(log_strikes):
                svi_value = (a + b * (rho * (k - m) + np.sqrt(sigma ** 2 + (k - m) ** 2))) / timeToMaturity
                vol_svi = np.sqrt(svi_value)
                sum_error += (vol_svi - implied_vols[i]) ** 2

            return sum_error / self.Numberquotes

        point1 = np.array([0.42 * (self.maxVol + minVol), 0.42 * (maxLogStrikes + minLogStrikes)])
        point2 = np.array([0.55 * (self.maxVol + minVol), 0.55 * (maxLogStrikes + minLogStrikes)])
        point3 = np.array([0.5 * (self.maxVol + minVol), 0.5 * (maxLogStrikes + minLogStrikes)])

        initial_simplex = [point1, point2, point3]

        bounds = [[0, self.maxVol], [minLogStrikes, maxLogStrikes]]

        result = minimize(cost_function, x0=initial_simplex[0], method='Nelder-Mead', options={'initial_simplex': initial_simplex, 'maxiter': 1e3}, bounds=bounds)

        sigma_opt, m_opt = result.x
        X_opt = np.zeros((len(quotesMaturity), 3))
        for i, k in enumerate(log_strikes):
            y = (k - m_opt) / sigma_opt
            z = np.sqrt(1 + y ** 2)
            X_opt[i, 0] = 1
            X_opt[i, 1] = y
            X_opt[i, 2] = z

        B_opt = np.linalg.inv(X_opt.T @ Weigths @ X_opt) @ X_opt.T @ Weigths @ totalVariances
        a_opt, c_opt, d_opt = B_opt[0], B_opt[1], B_opt[2]
        b_opt = d_opt / sigma_opt
        rho_opt = c_opt / (b_opt * sigma_opt)

        return [a_opt, b_opt, rho_opt, m_opt, sigma_opt]
    

    def GetSVIValuesForMaturity(self, quotesMaturity, strikes, timeToMaturity):
        result = []
        
        forward, rate, div = self.forwardsRatesDivs[timeToMaturity]
        
        parameters = self.SVICalibration2D(quotesMaturity, timeToMaturity)
        self.parameters[timeToMaturity] = parameters

        print(parameters)
        
        a, b, rho, m, sigma = parameters
        def SVI(log_strike):
            return np.sqrt((a + b * (rho * (log_strike - m) + np.sqrt(sigma ** 2 + (log_strike - m) ** 2))) / timeToMaturity)

        for k in strikes:
            log_strike = np.log(k / forward)
            implied_volatility = SVI(log_strike)
            quote = next((q for q in quotesMaturity if q.strike == k), None)

            if quote is None:
                optionType = OptionType.CALL if k > forward else OptionType.PUT
            else:
                optionType = quote.optionType

            result.append(Quote(k, implied_volatility, optionType))

        return result
    

    def CreateSurfaceSVI(self, strikes=None):

        if strikes is None:
            min_strike = min(q.strike for quotes in self.dicQuotes.values() for q in quotes)
            max_strike = max(q.strike for quotes in self.dicQuotes.values() for q in quotes)
            strikes = [float(K) for K in range(int(0.5 * min_strike), int(1.25 * max_strike) + 1)]

        strikes.sort()

        for timeToMaturity in self.timeToMaturities:
            quotes_maturity = self.dicQuotes[timeToMaturity]
            
            if timeToMaturity > 0:
                self.surface[timeToMaturity] = self.GetSVIValuesForMaturity(quotes_maturity, strikes, timeToMaturity)"""
