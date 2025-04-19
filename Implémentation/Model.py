import numpy as np
from enum import Enum
from typing import List
from scipy.interpolate import griddata
import matplotlib.pyplot as plt 
from Interfaces import IModel


from LocalSurface import *


class InterpolationMethodSurface(Enum):
    LINEAR = "linear"
    NEAREST = "nearest"
    CUBIC = "cubic"

class VarianceReductionMethod(Enum):
    ANTITHETIQUE = 'antithetique'

class SimulationResult:
    
    def __init__(self, paths, *args):
        self.paths = paths
        self.varianceReductionPaths = args


class BSModel(IModel):

    def __init__(self, spot, sigma, rate, dividend, dW = None):
        super().__init__(spot, rate, dividend)
        self.sigma = sigma
        self.dW = dW

    def runSimulation(self, nbPaths, nbTimeSteps, timeToMaturity, varianceReductionMethod = None):
        dt = timeToMaturity / nbTimeSteps

        rate = self.rate
        sigma = self.sigma
        dividend = self.dividend
        spot = self.spot

        if self.dW is None :
            dW = np.random.randn(nbPaths, nbTimeSteps)
        
        else:
            dW = self.dW

        if varianceReductionMethod is None:
            S = np.zeros((nbPaths, nbTimeSteps + 1))
            S[:, 0] = spot
            for i in range(1, nbTimeSteps + 1):
                
                S[:, i] = S[:, i-1] * ( 1 + (rate - dividend) * dt + sigma * np.sqrt(dt) * dW[:, i-1])

            return SimulationResult(S)
        
        elif varianceReductionMethod == VarianceReductionMethod.ANTITHETIQUE:
            S = np.zeros((nbPaths, nbTimeSteps + 1))
            S_antithetic = np.zeros((nbPaths, nbTimeSteps + 1))
            S[:, 0], S_antithetic[:, 0] = spot, spot
            dW_antithetic = -dW

            for i in range(1, nbTimeSteps + 1):
                S[:, i] = S[:, i-1] * ( 1 + (rate - dividend) * dt + sigma * np.sqrt(dt) * dW[:, i-1])
                S_antithetic[:, i] = S_antithetic[:, i-1] * ( 1 + (rate - dividend) * dt + sigma * np.sqrt(dt) * dW_antithetic[:, i-1])

            return SimulationResult(S, S_antithetic)
        
        else:
            return SimulationResult([])
        

class HestonModel(IModel):

    def __init__(self, spot, variance, rate, dividend, kappa, theta, sigma, rho, dW = None, dZ = None):
        super().__init__(spot, rate, dividend)

        if 2 * kappa * theta >= sigma * sigma:
            self.variance = variance
            self.kappa = kappa
            self.theta = theta
            self.sigma = sigma
            self.rho = rho
            self.dW = dW
            self.dZ = dZ
        else:
            raise ValueError('Feller condition is not verified, please change your inputs')

    def runSimulation(self, nbPaths, nbTimeSteps, timeToMaturity, varianceReductionMethod = None):
        dt = timeToMaturity / nbTimeSteps

        rate = self.rate
        variance = self.variance
        sigma = self.sigma
        dividend = self.dividend
        spot = self.spot
        rho = self.rho
        kappa = self.kappa
        theta = self.theta

        if self.dW is None and self.dZ is None :
            dW = np.random.randn(nbPaths, nbTimeSteps)
            dZ = np.random.randn(nbPaths, nbTimeSteps)
            
        elif self.dW is not None and self.dZ is not None:
            dW = self.dW
            dZ = self.dZ
    
        elif self.dW is not None and self.dZ is None:
            print("You have only one array of gaussian simulation, make sure it is your wish")
            self.dW = dW
            dZ = np.random.randn(nbPaths, nbTimeSteps)
        else:
            print("You have only one array of gaussian simulation, make sure it is your wish")
            self.dZ = dZ
            dW = np.random.randn(nbPaths, nbTimeSteps)

        dZ1 = rho * dZ + np.sqrt(1 - rho * rho) * dW

        if varianceReductionMethod is None:
            S = np.zeros((nbPaths, nbTimeSteps + 1))
            V = np.zeros((nbPaths, nbTimeSteps + 1))
            S[:, 0] = spot
            V[:, 0] = variance
            for i in range(1, nbTimeSteps + 1):

                V[:, i] = ((np.sqrt(V[:, i - 1]) + 0.5 * sigma * np.sqrt(dt) * dZ[:, i - 1])**2 + (kappa * theta - 0.25 * sigma * sigma) * dt) / (1 + kappa * dt)
                S[:, i] = S[:, i - 1] * ( 1 + (rate - dividend) * dt + np.sqrt(V[:, i - 1] * dt) * dZ1[:, i - 1])

            return SimulationResult(S)
        
        elif varianceReductionMethod == VarianceReductionMethod.ANTITHETIQUE:
            S, V = np.zeros((nbPaths, nbTimeSteps + 1)), np.zeros((nbPaths, nbTimeSteps + 1))
            S_antithetic, V_antithetique = np.zeros((nbPaths, nbTimeSteps + 1)), np.zeros((nbPaths, nbTimeSteps + 1))
            S[:, 0], S_antithetic[:, 0] = spot, spot
            V[:, 0], V_antithetique[:, 0] = variance, variance
            dZ_antithetique, dZ1_antithetique = -dZ, -dZ1

            for i in range(1, nbTimeSteps + 1):

                V[:, i] = ((np.sqrt(V[:, i - 1]) + 0.5 * sigma * np.sqrt(dt) * dZ[:, i - 1])**2 + (kappa * theta - 0.25 * sigma * sigma) * dt) / (1 + kappa * dt)
                S[:, i] = S[:, i - 1] * ( 1 + (rate - dividend) * dt + np.sqrt(V[:, i - 1] * dt) * dZ1[:, i - 1])

                V_antithetique[:, i] = ((np.sqrt(V_antithetique[:, i - 1]) + 0.5 * sigma * np.sqrt(dt) * dZ_antithetique[:, i - 1])**2 + (kappa * theta - 0.25 * sigma * sigma) * dt) / (1 + kappa * dt)
                S_antithetic[:, i] = S_antithetic[:, i - 1] * ( 1 + (rate - dividend) * dt + np.sqrt(V_antithetique[:, i - 1] * dt) * dZ1_antithetique[:, i - 1])
            
            return SimulationResult(S, S_antithetic)
        
        else:
            return SimulationResult([])
    
    def phi(self, u, t):
        rate = self.rate
        variance = self.variance
        sigma = self.sigma
        dividend = self.dividend
        spot = self.spot
        rho = self.rho
        kappa = self.kappa
        theta = self.theta

        d = np.sqrt((rho * sigma * 1j * u - kappa) ** 2 + sigma**2 * (1j * u + u**2))
        g = (kappa - rho * sigma * 1j * u - d) / (kappa - rho * sigma * 1j * u + d)

        return np.exp(1j * u * (np.log(spot) + (rate - dividend) * t)
                      + theta * kappa / (sigma**2) * ((kappa - rho * sigma * 1j * u - d) * t - 2 * np.log((1 - g * np.exp(-d * t)) / (1 - g)))
                      + variance / sigma**2 * (kappa - rho * sigma * 1j * u - d) * (1 - np.exp(-d * t)) / (1 - g * np.exp(-d * t))) 





class DupireModel(IModel):

    def __init__(self, spot, rate, dividend, surface: dict[float, List[Quote]], forwardsRatesDivs: dict[float, List], parameters: dict[float, List] = None, isTotalVarianceSurface: bool = None, dW=None, interpolation_method=InterpolationMethod.LINEAR):
        super().__init__(spot, rate, dividend)

        self.surface = surface
        self.forwardsRatesDivs = forwardsRatesDivs
        self.parameters = parameters
        self.isTotalVarianceSurface = isTotalVarianceSurface
        self.dW = dW
        self.interpolation_method = interpolation_method
        
        self.timeToMaturities = list(surface.keys())
        self.forwards = {ttm: spot * np.exp((rate - dividend) * ttm) for ttm in self.timeToMaturities}        
        self.strikes = [quote.strike for quote in surface[self.timeToMaturities[0]]]
        self.refForward = self.forwards[self.timeToMaturities[0]]


        self._minStrikes = np.min(self.strikes)
        self._maxStrikes = np.max(self.strikes)

        self._minTime = np.min(self.timeToMaturities)
        self._maxTime = np.max(self.timeToMaturities)

        self.maxK = np.max(self.strikes)
        self.minK = np.min(self.strikes)

        if isTotalVarianceSurface is True:
            self.preprocessed_quotes = {strike: {} for strike in self.strikes}
            for ttm in self.timeToMaturities:
                for quote in self.surface[ttm]:
                    if quote.strike in self.preprocessed_quotes:
                        value = quote.impliedVol ** 2 * ttm
                        self.preprocessed_quotes[quote.strike][ttm] = value

            self.TSplines = {
                strike: self.create_interpolator(sorted(ttm_vol_map.items()))
                for strike, ttm_vol_map in self.preprocessed_quotes.items() if ttm_vol_map
            }

        self.catch = {}

    
    def create_interpolator(self, data):
        ttm_values, vol_values = zip(*data)
        smoothedValues = self.smooth_values(vol_values)
        return CubicSpline(ttm_values, smoothedValues, bc_type="natural")


    def GetAllDerivative(self, timeToMaturities):
        allDerivatives = {}

        for strike in self.strikes:
            splines = self.TSplines[strike]
            
            listeDerivatives = [GetDerivative(timeToMaturity, splines, self._minTime, self._maxTime) for timeToMaturity in timeToMaturities]

            smoothed_values = self.smooth_values(listeDerivatives)
            allDerivatives[strike] = {timeToMaturity: smoothed_value for timeToMaturity, smoothed_value in zip(timeToMaturities, smoothed_values)}


        transposed = {}
        for strike, time_values in allDerivatives.items():
            for timeToMaturity, value in time_values.items():
                if timeToMaturity not in transposed:
                    transposed[timeToMaturity] = {}
                transposed[timeToMaturity][strike] = value

        splinesDerivatives = {}
        for timeToMaturity, strikesValues in transposed.items():
            strikes, derivatives = list(strikesValues.keys()), list(strikesValues.values())
            splinesDerivatives[timeToMaturity] = CubicSpline(strikes, derivatives, bc_type='natural')

        return splinesDerivatives


    def smooth_values(self, values, window_size=5, polyorder=3):
        """Applique un filtre de Savitzky-Golay pour lisser les valeurs, tout en conservant la taille de l'entrée."""
        if len(values) < window_size:
            return values  # Pas assez de données pour lisser
        # Assurez-vous que window_size est impair et supérieur à polyorder
        if window_size % 2 == 0:
            window_size += 1
        if window_size <= polyorder:
            raise ValueError("window_size must be greater than polyorder")
        
        smoothed = savgol_filter(values, window_size, polyorder, mode='nearest')
        return smoothed
    

    def closestTimeToMaturity(self, t):
        closest = None
        min_diff = float('inf')  # Initialiser à l'infini

        for time in self.timeToMaturities:
            diff = abs(time - t)  # Calculer la différence absolue
            if diff < min_diff:  # Vérifier si c'est la plus petite différence
                min_diff = diff
                closest = time  # Mettre à jour le timeToMaturity le plus proche

        return closest


    
    def runSimulation(self, nbPaths, nbTimeSteps, timeToMaturity, varianceReductionMethod=None):
        dt = timeToMaturity / nbTimeSteps
        sqrt_dt = np.sqrt(dt)
        drift = (self.rate - self.dividend) * dt

        if self.dW is None:
            dW = np.random.randn(nbPaths, nbTimeSteps)
        else:
            dW = self.dW


        if varianceReductionMethod is None:

            S = np.zeros((nbPaths, nbTimeSteps + 1))
            S[:, 0] = self.spot

            for j in range(1, nbTimeSteps + 1):
                t = j * dt
                localVols = np.array([self.getLocalVolatility(t, s) for s in S[:, j - 1]])

                S[:, j] = S[:, j - 1] * (1 + drift + localVols * sqrt_dt * dW[:, j - 1])

            return SimulationResult(S)
            
        elif varianceReductionMethod == VarianceReductionMethod.ANTITHETIQUE:
            S = np.zeros((nbPaths, nbTimeSteps + 1))
            S_antithetique = np.zeros((nbPaths, nbTimeSteps + 1))
            S[:, 0] = self.spot
            S_antithetique[:, 0] = self.spot
            dW_antithetique = -dW

            for j in range(1, nbTimeSteps + 1):
                t = j * dt
                localVols = np.array([self.getLocalVolatility(t, s) for s in S[:, j - 1]])
                localVolsAntithetique = np.array([self.getLocalVolatility(t, s) for s in S_antithetique[:, j - 1]])

                S[:, j] = S[:, j - 1] * (1 + drift + localVols * sqrt_dt * dW[:, j - 1])
                S_antithetique[:, j] = S_antithetique[:, j - 1] * (1 + drift + localVolsAntithetique * sqrt_dt * dW_antithetique[:, j - 1])

            plt.plot(S.T)
            plt.show()
            return SimulationResult(S, S_antithetique)

    

    def getLocalVolatility(self, t, S):

        S = np.round(S, 2)
        if (t, S) in self.catch:
            return self.catch[(t, S)]
        
        vol = self.volatilityBilinearityInterpolation(t, S)
        self.catch[(t, S)] = vol

        return vol
    

    def getForward(self, t):

        closest_ttm = min(self.timeToMaturities, key=lambda x: abs(x - t))
        return self.forwards[closest_ttm]


    def volatilityBilinearityInterpolation(self, t, S):

        forward = self.forwards[self.timeToMaturities[0]]
        S = np.log(S / forward)
        
        if t in self.timeToMaturities and S in self.strikes:
            return next(lq.impliedVol for lq in self.surface[t] if lq.strike == S)

        elif t not in self.timeToMaturities and S not in self.strikes:
            if S <= self._minStrikes or S >= self._maxStrikes and self._minTime <= t <= self._maxTime:
                x1 = max([tm for tm in self.timeToMaturities if tm <= t])
                x2 = min([tm for tm in self.timeToMaturities if tm >= t])
                y1 = next(lq.impliedVol for lq in self.surface[x1] if lq.strike == (self._minStrikes if S <= self._minStrikes else self._maxStrikes))
                y2 = next(lq.impliedVol for lq in self.surface[x2] if lq.strike == (self._minStrikes if S <= self._minStrikes else self._maxStrikes))

                return y1 + (y2 - y1) * (t - x1) / (x2 - x1)

            elif self._minStrikes <= S <= self._maxStrikes and self._minTime <= t <= self._maxTime:
                x1 = max([K for K in self.strikes if K <= S])
                x2 = min([K for K in self.strikes if K >= S])
                y1 = max([tm for tm in self.timeToMaturities if tm <= t])
                y2 = min([tm for tm in self.timeToMaturities if tm >= t])

                f11 = next(lq.impliedVol for lq in self.surface[y1] if lq.strike == x1)
                f21 = next(lq.impliedVol for lq in self.surface[y1] if lq.strike == x2)
                f12 = next(lq.impliedVol for lq in self.surface[y2] if lq.strike == x1)
                f22 = next(lq.impliedVol for lq in self.surface[y2] if lq.strike == x2)

                dx = S - x1
                dy = t - y1

                deltaX = x2 - x1
                deltaY = y2 - y1
                deltaFx = f21 - f11
                deltaFy = f12 - f11
                deltaFxy = f11 + f22 - f21 - f12

                return deltaFx * dx / deltaX + deltaFy * dy / deltaY + deltaFxy * dx * dy / (deltaX * deltaY) + f11

            elif self._minStrikes <= S <= self._maxStrikes and t <= self._minTime or t >= self._maxTime:
                x1 = max([K for K in self.strikes if K <= S])
                x2 = min([K for K in self.strikes if K >= S])
                y1 = next(lq.impliedVol for lq in self.surface[self._minTime if t <= self._minTime else self._maxTime] if lq.strike == x1)
                y2 = next(lq.impliedVol for lq in self.surface[self._minTime if t <= self._minTime else self._maxTime] if lq.strike == x2)

                return y1 + (y2 - y1) * (S - x1) / (x2 - x1)

            else:
                return next(lq.impliedVol for lq in self.surface[self._minTime if t <= self._minTime else self._maxTime] if lq.strike == (self._minStrikes if S <= self._minStrikes else self._maxStrikes))

        elif t in self.timeToMaturities and S not in self.strikes:
            if self._minStrikes <= S <= self._maxStrikes:
                x1 = max([K for K in self.strikes if K <= S])
                x2 = min([K for K in self.strikes if K >= S])
                y1 = next(lq.impliedVol for lq in self.surface[t] if lq.strike == x1)
                y2 = next(lq.impliedVol for lq in self.surface[t] if lq.strike == x2)

                return y1 + (y2 - y1) * (S - x1) / (x2 - x1)

            else:
                return next(lq.impliedVol for lq in self.surface[t] if lq.strike == (self._minStrikes if S <= self._minStrikes else self._maxStrikes))

        else:
            if self._minTime <= t <= self._maxTime:
                x1 = max([tm for tm in self.timeToMaturities if tm <= t])
                x2 = min([tm for tm in self.timeToMaturities if tm >= t])
                y1 = next(lq.impliedVol for lq in self.surface[x1] if lq.strike == S)
                y2 = next(lq.impliedVol for lq in self.surface[x2] if lq.strike == S)

                return y1 + (y2 - y1) * (t - x1) / (x2 - x1)

            else:
                return next(lq.impliedVol for lq in self.surface[self._minTime if t <= self._minTime else self._maxTime] if lq.strike == S)

