from enum import Enum
import numpy as np
from scipy.interpolate import CubicSpline, interp1d, PchipInterpolator, Akima1DInterpolator, UnivariateSpline
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import math

from SVI import *

class InterpolationMethod(Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    PCHIP = "pchip"
    AKIMA = "akima"
    CUBIC_CLAMPED = "cubic_clamped"
    CUBIC_NOT_A_KNOT = "cubic_not_a_knot"
    CUBIC_PERIODIC = "cubic_periodic"
    CUBIC_NATURAL = 'cubic_natural'

class LocalSurfaceType(Enum):
    SVI = 'svi'
    IMPLIED = 'implied'



EPSILON = 1e-6



def GetDerivative(timeToMaturity, spline, minT, maxT):

        if timeToMaturity <= minT:
            derivative = (spline(timeToMaturity + EPSILON) - spline(timeToMaturity)) / EPSILON
        elif timeToMaturity >= maxT:
            derivative = (spline(timeToMaturity) - spline(timeToMaturity - EPSILON)) / EPSILON
        else:
            derivative = (spline(timeToMaturity + EPSILON) - spline(timeToMaturity - EPSILON)) / (2 * EPSILON)

        return derivative


class LocalSurface:

    def __init__(self, quotesDict: dict[float, List], forwardsRatesDivs: dict[float, List], surfaceType: LocalSurfaceType, parameters: dict[float, List] = None, interpolation_method = None):

        if interpolation_method is None:
            raise ValueError('Please choose interpolation method for the construction of the local volatility surface')

        self.quotesDic = quotesDict
        self.forwardsRatesDivs = forwardsRatesDivs
        self.parameters = parameters
        self.surfaceType = surfaceType
        self.interpolation_method = interpolation_method

        if surfaceType is LocalSurfaceType.SVI and parameters is None:
            raise ValueError('Please give SVI parameters for the construction of the local volatility surface')
    

        self.timeToMaturities = list(quotesDict.keys())
        self.minT = np.min(self.timeToMaturities)
        self.maxT = np.max(self.timeToMaturities)

        initial_quotes = self.quotesDic[self.timeToMaturities[0]]
        self.strikes = [quote.strike for quote in initial_quotes]
        self.maxK = np.max(self.strikes)
        self.minK = np.min(self.strikes)
        self.surface = {}

        self.refForward = forwardsRatesDivs[self.timeToMaturities[0]][0]
        if self.minK > 0: #alors notre surface n'est pas en log strike        
            self.initialSurfaceLogStrike = {ttm : [Quote(np.log(quote.strike / self.refForward), quote.impliedVol, quote.optionType) for quote in quotesDict[ttm]] for ttm in self.timeToMaturities}


        self.preprocessed_quotes = {strike: {} for strike in self.strikes}
        for ttm in self.timeToMaturities:
            for quote in self.quotesDic.get(ttm, []):
                if quote.strike in self.preprocessed_quotes:
                    value = quote.impliedVol ** 2 * ttm if surfaceType == LocalSurfaceType.SVI else quote.impliedVol
                    self.preprocessed_quotes[quote.strike][ttm] = value

        self.TSplines = {
            strike: self.create_interpolator(sorted(ttm_vol_map.items()))
            for strike, ttm_vol_map in self.preprocessed_quotes.items() if ttm_vol_map
        }

        if surfaceType == LocalSurfaceType.IMPLIED:
            self.kSplines = {}
            for ttm in self.timeToMaturities:
                forward, rate, div = self.forwardsRatesDivs[ttm]
                ttm_quotes = quotesDict.get(ttm, [])
                k_values, vol_values = zip(*[(np.log(quote.strike / forward), quote.impliedVol) for quote in ttm_quotes])
                self.kSplines[ttm] = self.create_interpolator(zip(k_values, vol_values))


        self.allDerivatives, self.allDerivativesRaw = self.GetAllDerivative()
        if surfaceType == LocalSurfaceType.IMPLIED:
            self.allDerivativesK, self.allDerivativesKK = self.GetAllDerivativeK()

        #self.plot_surface()


    def plot_surface(self):
        strikes = sorted(self.allDerivatives.keys())
        timeToMaturities = sorted(self.timeToMaturities)

        X, Y = np.meshgrid(timeToMaturities, strikes)
        Z = np.zeros_like(X)
        Z_raw = np.zeros_like(X)

        for i, strike in enumerate(strikes):
            for j, timeToMaturity in enumerate(timeToMaturities):
                Z[i, j] = self.allDerivatives[strike].get(timeToMaturity, np.nan)
                Z_raw[i, j] = self.allDerivativesRaw[strike].get(timeToMaturity, np.nan)

        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X, Y, Z, cmap='viridis')
        ax1.set_xlabel('Time to Maturity')
        ax1.set_ylabel('Strike')
        ax1.set_zlabel('Smoothed Derivative Value')
        ax1.set_title('Smoothed Surface of Derivatives')

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(X, Y, Z_raw, cmap='viridis')
        ax2.set_xlabel('Time to Maturity')
        ax2.set_ylabel('Strike')
        ax2.set_zlabel('Raw Derivative Value')
        ax2.set_title('Raw Surface of Derivatives')

        plt.tight_layout()
        plt.show()


    def create_interpolator(self, data):
        """Create interpolator based on chosen method."""
        ttm_values, vol_values = zip(*data)

        smoothedValues = self.smooth_values(vol_values)
        
        if self.interpolation_method == InterpolationMethod.LINEAR:
            return interp1d(ttm_values, smoothedValues, kind="linear", fill_value="extrapolate")
        elif self.interpolation_method == InterpolationMethod.QUADRATIC:
            return interp1d(ttm_values, smoothedValues, kind="quadratic", fill_value="extrapolate")
        elif self.interpolation_method == InterpolationMethod.PCHIP:
            return PchipInterpolator(ttm_values, smoothedValues)
        elif self.interpolation_method == InterpolationMethod.AKIMA:
            return Akima1DInterpolator(ttm_values, smoothedValues)
        elif self.interpolation_method == InterpolationMethod.CUBIC_CLAMPED:
            return CubicSpline(ttm_values, smoothedValues, bc_type="clamped")
        elif self.interpolation_method == InterpolationMethod.CUBIC_NOT_A_KNOT:
            return CubicSpline(ttm_values, smoothedValues, bc_type="not-a-knot")
        elif self.interpolation_method == InterpolationMethod.CUBIC_PERIODIC:
            return CubicSpline(ttm_values, smoothedValues, bc_type="periodic")
        elif self.interpolation_method == InterpolationMethod.CUBIC_NATURAL:
            return CubicSpline(ttm_values, smoothedValues, bc_type="natural")
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation_method}")
    

    def GetAllDerivative(self):
        allDerivatives = {}
        allDerivativesRaw = {}

        for strike in self.strikes:
            splines = self.TSplines[strike]
            
            listeDerivatives = [GetDerivative(timeToMaturity, splines, self.minT, self.maxT) for timeToMaturity in self.timeToMaturities]
            
            allDerivativesRaw[strike] = {timeToMaturity: value for timeToMaturity, value in zip(self.timeToMaturities, listeDerivatives)}

            smoothed_values = self.smooth_values(listeDerivatives)
            allDerivatives[strike] = {timeToMaturity: smoothed_value for timeToMaturity, smoothed_value in zip(self.timeToMaturities, smoothed_values)}

        return allDerivatives, allDerivativesRaw




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
    

    def GetAllDerivativeK(self):
        allDerivativesK = {}
        allDerivativesKK = {}

        for timeToMaturity in self.timeToMaturities:
            quotes = self.quotesDic[timeToMaturity]
            forward, rate, div = self.forwardsRatesDivs[timeToMaturity]
            Kspline = self.kSplines[timeToMaturity]
            
            allDerivativesK[timeToMaturity] = {}
            allDerivativesKK[timeToMaturity] = {}

            K_values = []
            KK_values = []

            for strike in self.strikes:
                quote = next((q for q in quotes if q.strike == strike), None)
                
                #strike1 = strike * forward / 100
                k = np.log(strike / forward)
                k_plus = k + EPSILON
                k_minus = k - EPSILON
                
                impliedVol = quote.impliedVol
                impliedVol_plus = Kspline(k_plus)
                impliedVol_minus = Kspline(k_minus)

                K_derivative = (impliedVol_plus - impliedVol) / EPSILON
                KK_derivative = (impliedVol_plus + impliedVol_minus - 2 * impliedVol) / (EPSILON**2)

                allDerivativesK[timeToMaturity][strike] = K_derivative
                allDerivativesKK[timeToMaturity][strike] = KK_derivative

                K_values.append(K_derivative)
                KK_values.append(KK_derivative)

            # Appliquer le lissage
            if K_values:
                smoothed_K_values = self.smooth_values(K_values)
                for i, strike in enumerate(self.strikes[len(self.strikes) - len(smoothed_K_values):]):
                    allDerivativesK[timeToMaturity][strike] = smoothed_K_values[i]
            
            if KK_values:
                smoothed_KK_values = self.smooth_values(KK_values)
                for i, strike in enumerate(self.strikes[len(self.strikes) - len(smoothed_KK_values):]):
                    allDerivativesKK[timeToMaturity][strike] = smoothed_KK_values[i]

        return allDerivativesK, allDerivativesKK



    def CreateSurface(self, logStrike:bool):
        
        if self.surfaceType is LocalSurfaceType.SVI:
            referenceForward = self.forwardsRatesDivs[self.timeToMaturities[0]][0]

            liste_t = []
            for timeToMaturity in self.timeToMaturities:
                quotes = self.quotesDic[timeToMaturity]
                parameters = self.parameters[timeToMaturity]
                theta, etha, gamma, rho = parameters
                phiTheta = phi(theta, etha, gamma)

                forward, rate, div = self.forwardsRatesDivs[timeToMaturity]

                localQuotes = []
                for strike in self.strikes:
                    quote = next((q for q in quotes if q.strike == strike), None)
                    totalVariance = quote.impliedVol * quote.impliedVol * timeToMaturity

                    k = np.log(strike / forward)

                    T_derivative = self.allDerivatives[strike][timeToMaturity]
                    K_derivative = FirstDerivateAOA(k, theta, phiTheta, rho)
                    KK_derivative = SecondeDerivativeAOA(k, theta, phiTheta, rho)
                    denom = ((1 - k * K_derivative / (2 * totalVariance))**2 + 0.5 * KK_derivative - (1/(4 * totalVariance) + 1/16) * K_derivative**2)

                    if T_derivative / denom < 0:
                        localVol = math.sqrt(np.abs(T_derivative / denom))
                        print('Invalid Value in sqrt')
                    else:
                        localVol = math.sqrt(T_derivative / denom)
                    
                    if logStrike is True:
                        localQuotes.append(Quote(k, localVol, quote.optionType))
                    else:
                        localQuotes.append(Quote(strike, localVol, quote.optionType))
                
                self.surface[timeToMaturity] = localQuotes
        
        else:
            referenceForward = self.forwardsRatesDivs[self.timeToMaturities[0]][0]

            for timeToMaturity in self.timeToMaturities:
                quotes = self.quotesDic[timeToMaturity]
                forward, rate, div = self.forwardsRatesDivs[timeToMaturity]

                localQuotes = []
                for strike in self.strikes:
                    quote = next((q for q in quotes if q.strike == strike), None)

                    #strike1 = strike * forward / 100
                    k = np.log(strike / forward)

                    impliedVol = quote.impliedVol

                    T_derivative = self.allDerivativesRaw[strike][timeToMaturity]
                    K_derivative = self.allDerivativesK[timeToMaturity][strike]
                    KK_derivative = self.allDerivativesKK[timeToMaturity][strike]

                    num = 2 * impliedVol * timeToMaturity * T_derivative + impliedVol * impliedVol
                    denom = impliedVol * timeToMaturity * KK_derivative - 0.25 * impliedVol * impliedVol * timeToMaturity * timeToMaturity * K_derivative * K_derivative + (1 - k * K_derivative / impliedVol)**2
                    
                    if num < 0 or denom < 0  :
                        localVol = np.sqrt(np.abs(num) / np.abs(denom)) 
                        print('no good')
                    else:
                        localVol = np.sqrt(num / denom)
                    
                    if localVol < 0 :
                        print('ok')
                    
                    if logStrike is True:
                        localQuotes.append(Quote(np.log(strike/referenceForward), localVol, quote.optionType))
                    else:
                        localQuotes.append(Quote(strike, localVol, quote.optionType))

                self.surface[timeToMaturity] = localQuotes

    


