import numpy as np
from Model import *
from Option import VanillaOption, IOption, CallSpread
from scipy.interpolate import CubicSpline
from Pricer import Price


EPSILON = 0.01


class SchemePDE(Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    CRANK_NICHOLSON = "crank-nicholson"





class PricerPDE:

    def __init__(self, scheme:SchemePDE, model, y_max, y_numbers, x_max, x_numbers=None, CFL=None, localSurface:dict[float, List[Quote]]=None, forwardsRatesDivs: dict[float, List]=None):
        self.model = model
        self.scheme = scheme

        if scheme is SchemePDE.EXPLICIT and CFL is None:
            raise ValueError('Please enter a CFL number to ensure the stability of the computation (0 < CFL <= 1)')

        if x_numbers is None and CFL is not None:
            self.x_numbers = int((x_max * y_numbers**2 * self.model.sigma**2) / CFL)
        else:
            self.x_numbers = x_numbers

        self.y_numbers = y_numbers
        self.x_max = x_max
        self.y_max = y_max

        self.x_step = x_max / self.x_numbers
        self.y_step = y_max / self.y_numbers

        self.y_numbers = y_numbers + 1
        self.x_values = np.array([x_max - i * self.x_step for i in range(self.x_numbers)]).reshape(-1, 1)
        self.y_values = np.array([y_max - i * self.y_step for i in range(self.y_numbers)]).reshape(-1, 1)
        self.y_values_square = self.y_values ** 2

        self.y_values1d = self.y_values.flatten()
        self.y_values_square1d = self.y_values_square.flatten()

        #print(f"\n{(self.x_step / (self.y_step * self.y_step)) / ( 1 / (self.model.sigma**2 * self.y_max**2))}")

        self.CS = None
        self.lastPayoffs = None


        self.localSurface = localSurface
        if localSurface is not None and forwardsRatesDivs is not None:
            self.localSurface = localSurface
            
            self.timeToMaturities = list(localSurface.keys())

            self.strikes = [quote.strike for quote in self.localSurface[self.timeToMaturities[0]]]
            self.selectedSurface = {} # the part of the surface we keep for our pricing


            if x_max > self.timeToMaturities[-1]:
                raise ValueError("The Option maturity is higher than the max maturity on the local surface")
            else:
                self.preprocessed_quotes = {strike: {} for strike in self.strikes}
                for ttm in self.timeToMaturities:

                    quotes = self.localSurface[ttm]
                    for quote in quotes:
                        if quote.strike in self.preprocessed_quotes:
                            value = quote.impliedVol
                            self.preprocessed_quotes[quote.strike][ttm] = value
                
                quotes_xmax = []
                for strike in self.strikes:
                    cs = CubicSpline(self.timeToMaturities, list(self.preprocessed_quotes[strike].values()))
                    cs(self.x_max)

                    quotes_xmax.append(Quote(strike, cs(self.x_max), OptionType.CALL)) #Ici l'option ne nous intéresse pas donc on met tout a CALL
                
                relevantTimeToMaturities = [ttm for ttm in self.timeToMaturities if ttm < self.x_max]

                for ttm in relevantTimeToMaturities:
                    self.selectedSurface[ttm] = self.localSurface[ttm]

                self.selectedSurface[self.x_max] = quotes_xmax

                self.PDESmiles = {}
                rate, dividend = model.rate, model.dividend
                forwardSurface, forwardPDE = 0, 0
                
                
                for i, ttm in enumerate(list(self.selectedSurface.keys())):
                    forwardSurface = forwardsRatesDivs[self.timeToMaturities[i]][0]
                    forwardPDE = model.spot * math.exp((rate - dividend) * ttm)
                    quotes = self.selectedSurface[ttm]
                    surfaceLogStrikes, localsVols = [], []
                    for i, strike in enumerate(self.strikes):
                        surfaceLogStrikes.append(math.log(strike / forwardSurface))
                        localsVols.append(quotes[i].impliedVol)
                    
                    cs = CubicSpline(surfaceLogStrikes, localsVols)
                    
                   
                    #self.PDESmiles[ttm] = [cs(math.log(S / forwardPDE)) for S in self.y_values1d] #ici si on a des valeurs or range de la spline il extrapole par défaut

                    listeVolsInterpolated = []
                    for S in self.y_values1d:
                        log = math.log(S / forwardPDE)
                        if log >= surfaceLogStrikes[-1]: # Supérieur au max des log strikes
                            listeVolsInterpolated.append(localsVols[-1])
                        elif log <= surfaceLogStrikes[0]: # Inférieur au min des log strikes
                            listeVolsInterpolated.append(localsVols[0])
                        else:
                            listeVolsInterpolated.append(cs(log))
                    self.PDESmiles[ttm] = listeVolsInterpolated
                

    def Price(self, option:VanillaOption):
        strike = option.strike
        spot = self.model.spot

        if spot < 0 or spot > self.y_max:
            raise ValueError("Spot is out of the specified range")

        I = np.identity(self.y_numbers)
        rI = self.model.rate * I

        S = np.diag(self.y_values1d)
        D1 = np.zeros((self.y_numbers, self.y_numbers))

        # Calcul des coefficients D1 (différences finies pour la dérivée première)
        for i in range(1, self.y_numbers - 1):
            D1[i, i - 1] = 1 / (2 * self.y_step)
            D1[i, i + 1] = -1 / (2 * self.y_step)
        D1[0, 0], D1[0, 1] = 1 / self.y_step, -1 / self.y_step
        D1[-1, -2], D1[-1, -1] = 1 / self.y_step, -1 / self.y_step

        rSD1 = (self.model.rate - self.model.dividend) * S @ D1


        # Calcul des coefficients D2 (différences finies pour la dérivée seconde)
        squareS = np.diag(self.y_values_square1d)
        D2 = np.zeros((self.y_numbers, self.y_numbers))

        for i in range(1, self.y_numbers - 1):
            D2[i, i - 1] = 1 / (self.y_step ** 2)
            D2[i, i] = -2 / (self.y_step ** 2)
            D2[i, i + 1] = 1 / (self.y_step ** 2)
        D2[0, 0], D2[0, 1], D2[0, 2] = 1 / (self.y_step ** 2), -2 / (self.y_step ** 2), 1 / (self.y_step ** 2)
        D2[-1, -3], D2[-1, -2], D2[-1, -1] = 1 / (self.y_step ** 2), -2 / (self.y_step ** 2), 1 / (self.y_step ** 2)


        payoffs = np.array([option.payoff(self.y_values[i, 0]) for i in range(self.y_numbers)]).reshape(-1, 1)
        self.initialPayoffs = payoffs[:, 0][::-1]

        if self.localSurface is None:
            factor = 0.5 * self.model.sigma ** 2
            factorSquareSD2 = factor * squareS @ D2
            C = -rI + rSD1 + factorSquareSD2

            if self.scheme is SchemePDE.EXPLICIT:
                E = I + self.x_step * C
            elif self.scheme is SchemePDE.IMPLICIT:
                E = np.linalg.inv(I - self.x_step * C)
            else: # Crank-Nicholson scheme
                E = np.linalg.inv(I - 0.5 * self.x_step * C) @ (I + 0.5 * self.x_step * C)
        
            for _ in range(self.x_numbers):
                payoffs = E @ payoffs
                payoffs[0, 0] = self.y_max * np.exp(-self.model.dividend * _ * self.x_step) - strike * np.exp(-self.model.rate * _ * self.x_step)
                payoffs[-1, 0] = 0
                #payoffs[0, 0] = 2 * payoffs[1, 0] - payoffs[2, 0]
                #payoffs[-1, 0] = 2 * payoffs[-2, 0] - payoffs[-3, 0]

            cs = CubicSpline(self.y_values1d[::-1], payoffs[:, 0][::-1])
            self.CS = cs
            self.lastPayoffs = payoffs[:, 0][::-1]
            
            price = cs(spot)
            delta = (cs(spot + EPSILON) - cs(spot - EPSILON)) / (2 * EPSILON)
            gamma = (cs(spot + EPSILON) + cs(spot - EPSILON) - 2 * cs(spot)) / (EPSILON * EPSILON)
            return Price(price=price, delta=delta, gamma=gamma)

        else:
            ttmsPDESmiles = list(self.PDESmiles.keys())
            for _ in range(self.x_numbers):
                ttm = _ * self.x_step
                # Trouver l'élément le plus proche de ttm
                closest_element = min(ttmsPDESmiles, key=lambda x: abs(x - ttm))

                localsVols = np.array(self.PDESmiles[closest_element]) ** 2

                factorSquareSD2 = 0.5 * localsVols * squareS @ D2
                C = -rI + rSD1 + factorSquareSD2

                if self.scheme is SchemePDE.EXPLICIT:
                    E = I + self.x_step * C
                elif self.scheme is SchemePDE.IMPLICIT:
                    E = np.linalg.inv(I - self.x_step * C)
                else: # Crank-Nicholson scheme
                    E = np.linalg.inv(I - 0.5 * self.x_step * C) @ (I + 0.5 * self.x_step * C)
            
                payoffs = E @ payoffs
                payoffs[0, 0] = self.y_max * np.exp(-self.model.dividend * ttm) - strike * np.exp(-self.model.rate * ttm)
                payoffs[-1, 0] = 0
            
            cs = CubicSpline(self.y_values1d[::-1], payoffs[:, 0][::-1])
            self.CS = cs
            self.lastPayoffs = payoffs[:, 0][::-1]

            price = cs(spot)
            delta = (cs(spot + EPSILON) - cs(spot - EPSILON)) / (2 * EPSILON)
            gamma = (cs(spot + EPSILON) + cs(spot - EPSILON) - 2 * cs(spot)) / (EPSILON * EPSILON)
            return Price(price, delta=delta, gamma=gamma)


    def Plots(self, option:IOption = None):

        if self.CS is not None:
            cs = self.CS
            payoffs = self.lastPayoffs
            initPayoffs = self.initialPayoffs
            spot = self.model.spot

            deltas, gammas = [], []
            for i in range(1, len(self.y_values1d)-1):
                p_minus, p, p_plus = payoffs[i-1], payoffs[i], payoffs[i+1]
                if i == 0:
                    delta = (p_plus - p) / self.y_step
                    gamma = (p_plus - 2 * p + p_minus) / (2 * self.y_step * self.y_step)
                elif i == len(self.y_values1d) - 1:
                    delta = (p - p_minus) / self.y_step
                    gamma = (p_plus - 2 * p + p_minus) / (2 * self.y_step * self.y_step)
                else:
                    delta = (p_plus - p_minus) / (2 * self.y_step)
                    gamma = (p_plus - 2 * p + p_minus) / (self.y_step * self.y_step)
                deltas.append(delta)
                gammas.append(gamma)

                
            # Configuration de la figure
            plt.figure(figsize=(12, 10))

            # Prix en %
            plt.subplot(3, 1, 1)
            plt.title('Prix en %', fontsize=16)
            plt.plot(self.y_values1d[::-1], 100 * payoffs / spot, color='blue', linewidth=2, label='Prix (%)')
            plt.plot(self.y_values1d[::-1], 100 * initPayoffs * (1 if option is None else np.exp(-self.model.dividend * option.maturityDate)) / spot, color='black', linewidth=2, label='Payoff (%)')
            plt.axvline(x=spot, color='red', linestyle='--', label='Spot', linewidth=1.5)
            plt.xlabel('Valeur de l’actif sous-jacent', fontsize=14)
            plt.ylabel('Prix (%)', fontsize=14)
            plt.legend()
            plt.grid(True)

            # Delta
            plt.subplot(3, 1, 2)
            plt.title('Delta', fontsize=16)

            # Calcul des deltas en évitant les indices extrêmes
            plt.plot(self.y_values1d[::-1][1:-1], deltas, color='green', linewidth=2, label='Delta')
            plt.axvline(x=spot, color='red', linestyle='--', label='Spot', linewidth=1.5)
            plt.xlabel('Valeur de l’actif sous-jacent', fontsize=14)
            plt.ylabel('Delta', fontsize=14)
            plt.legend()
            plt.grid(True)

            # Gamma
            plt.subplot(3, 1, 3)
            plt.title('Gamma', fontsize=16)
            plt.plot(self.y_values1d[::-1][1:-1], gammas, color='purple', linewidth=2, label='Gamma')
            plt.axvline(x=spot, color='red', linestyle='--', label='Spot', linewidth=1.5)
            plt.xlabel('Valeur de l’actif sous-jacent', fontsize=14)
            plt.ylabel('Gamma', fontsize=14)
            plt.legend()
            plt.grid(True)

            # Ajustement de l'affichage
            plt.tight_layout()
            plt.show()
        
        else:
            raise ValueError('Lancer la fonction Price avant de vouloir afficher les sensibilités !')






class OptimalPricerPDE:

    def __init__(self, scheme:SchemePDE, model, y_max, y_numbers, x_max, x_numbers=None, CFL=None, localSurface:dict[float, List[Quote]]=None, forwardsRatesDivs: dict[float, List]=None):
        self.model = model
        self.scheme = scheme

        if scheme is SchemePDE.EXPLICIT and CFL is None:
            raise ValueError('Please enter a CFL number to ensure the stability of the computation (0 < CFL <= 1)')

        if x_numbers is None and CFL is not None:
            self.x_numbers = int((x_max * y_numbers**2 * self.model.sigma**2) / CFL)
        else:
            self.x_numbers = x_numbers

        self.y_numbers = y_numbers
        self.x_max = x_max
        self.y_max = y_max

        self.x_step = x_max / self.x_numbers
        self.y_step = y_max / self.y_numbers

        self.y_numbers = y_numbers + 1

        self.x_values = np.array([x_max - i * self.x_step for i in range(self.x_numbers)]).reshape(-1, 1)
        self.y_values = np.array([y_max - i * self.y_step for i in range(self.y_numbers)]).reshape(-1, 1)
        self.y_values_square = self.y_values ** 2

        self.y_values1d = self.y_values.flatten()
        self.y_values_square1d = self.y_values_square.flatten()


        self.CS = None # cubicSpline pour les sensibilités
        self.lastPayoffs = None # Pour avoir la courbe des prix
        self.initialPayoffs = None #pour tracer le payoff du call spread

        #print(f"\n{(self.x_step / (self.y_step * self.y_step)) / ( 1 / (self.model.sigma**2 * self.y_max**2))}")



        I = np.identity(self.y_numbers)
        rI = self.model.rate * I

        S = np.diag(self.y_values1d)
        D1 = np.zeros((self.y_numbers, self.y_numbers))

        # Calcul des coefficients D1 (différences finies pour la dérivée première)
        for i in range(1, self.y_numbers - 1):
            D1[i, i - 1] = 1 / (2 * self.y_step)
            D1[i, i + 1] = -1 / (2 * self.y_step)
        D1[0, 0], D1[0, 1] = 1 / self.y_step, -1 / self.y_step
        D1[-1, -2], D1[-1, -1] = 1 / self.y_step, -1 / self.y_step

        self.rI = rI
        self.rSD1 = (self.model.rate - self.model.dividend) * S @ D1


        # Calcul des coefficients D2 (différences finies pour la dérivée seconde)
        squareS = np.diag(self.y_values_square1d)
        D2 = np.zeros((self.y_numbers, self.y_numbers))

        for i in range(1, self.y_numbers - 1):
            D2[i, i - 1] = 1 / (self.y_step ** 2)
            D2[i, i] = -2 / (self.y_step ** 2)
            D2[i, i + 1] = 1 / (self.y_step ** 2)
        D2[0, 0], D2[0, 1], D2[0, 2] = 1 / (self.y_step ** 2), -2 / (self.y_step ** 2), 1 / (self.y_step ** 2)
        D2[-1, -3], D2[-1, -2], D2[-1, -1] = 1 / (self.y_step ** 2), -2 / (self.y_step ** 2), 1 / (self.y_step ** 2)
        
        self.squareS = squareS
        self.D2 = D2


        self.localSurface = localSurface
        if localSurface is not None and forwardsRatesDivs is not None:
            self.localSurface = localSurface
            
            self.timeToMaturities = list(localSurface.keys())

            self.strikes = [quote.strike for quote in self.localSurface[self.timeToMaturities[0]]]
            self.selectedSurface = {} # the part of the surface we keep for our pricing


            if x_max > self.timeToMaturities[-1]:
                raise ValueError("The Option maturity is higher than the max maturity on the local surface")
            else:
                self.preprocessed_quotes = {strike: {} for strike in self.strikes}
                for ttm in self.timeToMaturities:

                    quotes = self.localSurface[ttm]
                    for quote in quotes:
                        if quote.strike in self.preprocessed_quotes:
                            value = quote.impliedVol
                            self.preprocessed_quotes[quote.strike][ttm] = value
                
                quotes_xmax = []
                for strike in self.strikes:
                    cs = CubicSpline(self.timeToMaturities, list(self.preprocessed_quotes[strike].values()))

                    quotes_xmax.append(Quote(strike, cs(self.x_max), OptionType.CALL)) #Ici le type de l'option ne nous intéresse pas donc on met tout a CALL
                
                relevantTimeToMaturities = [ttm for ttm in self.timeToMaturities if ttm < self.x_max]

                for ttm in relevantTimeToMaturities:
                    self.selectedSurface[ttm] = self.localSurface[ttm]

                self.selectedSurface[self.x_max] = quotes_xmax

                self.PDESmiles = {}
                rate, dividend = model.rate, model.dividend
                forwardSurface, forwardPDE = 0, 0
                
                
                for i, ttm in enumerate(list(self.selectedSurface.keys())):
                    forwardSurface = forwardsRatesDivs[self.timeToMaturities[i]][0]
                    forwardPDE = model.spot * math.exp((rate - dividend) * ttm)
                    quotes = self.selectedSurface[ttm]
                    surfaceLogStrikes, localsVols = [], []
                    for i, strike in enumerate(self.strikes):
                        surfaceLogStrikes.append(math.log(strike / forwardSurface))
                        localsVols.append(quotes[i].impliedVol)
                    
                    cs = CubicSpline(surfaceLogStrikes, localsVols)
                    
                    listeVolsInterpolated = []
                    for S in self.y_values1d:
                        log = math.log(S / forwardPDE) if S > 0 else -np.inf
                        if log >= surfaceLogStrikes[-1]: # Supérieur au max des log strikes
                            listeVolsInterpolated.append(localsVols[-1])
                        elif log <= surfaceLogStrikes[0]: # Inférieur au min des log strikes
                            listeVolsInterpolated.append(localsVols[0])
                        else:
                            listeVolsInterpolated.append(cs(log))
                    self.PDESmiles[ttm] = listeVolsInterpolated
                
                
                self.E = []
                ttmsPDESmiles = list(self.PDESmiles.keys())
                for _ in range(self.x_numbers):
                    ttm = _ * self.x_step
                    # Trouver l'élément le plus proche de ttm
                    closest_element = min(ttmsPDESmiles, key=lambda x: abs(x - ttm))

                    localsVols = np.array(self.PDESmiles[closest_element]) ** 2

                    factorSquareSD2 = 0.5 * localsVols * squareS @ D2
                    C = -rI + self.rSD1 + factorSquareSD2

                    if self.scheme is SchemePDE.EXPLICIT:
                        self.E.append(I + self.x_step * C)
                    elif self.scheme is SchemePDE.IMPLICIT:
                        self.E.append(np.linalg.inv(I - self.x_step * C))
                    else: # Crank-Nicholson scheme
                        self.E.append(np.linalg.inv(I - 0.5 * self.x_step * C) @ (I + 0.5 * self.x_step * C))


    def Price(self, option:IOption):

        if isinstance(option, VanillaOption):
            strike = option.strike
            spot = self.model.spot
            I = np.identity(self.y_numbers)

            if spot < 0 or spot > self.y_max:
                raise ValueError("Spot is out of the specified range")

            payoffs = np.array([option.payoff(self.y_values[i, 0]) for i in range(self.y_numbers)]).reshape(-1, 1)
            self.initialPayoffs = payoffs[:, 0][::-1] * np.exp(-self.model.rate * option.maturityDate)

            if self.localSurface is None:
                factor = 0.5 * self.model.sigma ** 2
                factorSquareSD2 = factor * self.squareS @ self.D2
                C = -self.rI + self.rSD1 + factorSquareSD2

                if self.scheme is SchemePDE.EXPLICIT:
                    E = I + self.x_step * C
                elif self.scheme is SchemePDE.IMPLICIT:
                    E = np.linalg.inv(I - self.x_step * C)
                else: # Crank-Nicholson scheme
                    E = np.linalg.inv(I - 0.5 * self.x_step * C) @ (I + 0.5 * self.x_step * C)
            
                for _ in range(self.x_numbers):
                    payoffs = E @ payoffs
                    payoffs[0, 0] = self.y_max * np.exp(-self.model.dividend * _ * self.x_step) - strike * np.exp(-self.model.rate * _ * self.x_step)
                    payoffs[-1, 0] = 0

                cs = CubicSpline(self.y_values1d[::-1], payoffs[:, 0][::-1])
                self.CS = cs
                self.lastPayoffs = payoffs[:, 0][::-1]

                price = cs(spot)
                delta = (cs(spot + EPSILON) - cs(spot - EPSILON)) / (2 * EPSILON)
                gamma = (cs(spot + EPSILON) + cs(spot - EPSILON) - 2 * cs(spot)) / (EPSILON * EPSILON)
                return Price(price=price, delta=delta, gamma=gamma)

            else:
                
                for i in range(self.x_numbers):
                    payoffs = self.E[i] @ payoffs
                    payoffs[0, 0] = self.y_max * np.exp(-self.model.dividend * i * self.x_step) - strike * np.exp(-self.model.rate * i * self.x_step)
                    payoffs[-1, 0] = 0 #payoffs[-2, 0]
                
                cs = CubicSpline(self.y_values1d[::-1], payoffs[:, 0][::-1])
                self.CS = cs
                self.lastPayoffs = payoffs[:, 0][::-1]

                price_plus, price, price_minus = cs(spot + EPSILON), cs(spot), cs(spot - EPSILON)
                delta = (price_plus - price_minus) / (2 * EPSILON)
                gamma = (price_plus + price_minus - 2 * price) / (EPSILON * EPSILON)
                return Price(price=price, delta=delta, gamma=gamma)
        

        elif isinstance(option, CallSpread):

            strike = option.strike
            strike1 = option.strike1
            spot = self.model.spot
            I = np.identity(self.y_numbers)

            if spot < 0 or spot > self.y_max:
                raise ValueError("Spot is out of the specified range")

            payoffs = np.array([option.payoff(self.y_values[i, 0]) for i in range(self.y_numbers)]).reshape(-1, 1)
            self.initialPayoffs = payoffs[:, 0][::-1] * np.exp(-self.model.rate * option.maturityDate)

            if self.localSurface is None:
                factor = 0.5 * self.model.sigma ** 2
                factorSquareSD2 = factor * self.squareS @ self.D2
                C = -self.rI + self.rSD1 + factorSquareSD2

                if self.scheme is SchemePDE.EXPLICIT:
                    E = I + self.x_step * C
                elif self.scheme is SchemePDE.IMPLICIT:
                    E = np.linalg.inv(I - self.x_step * C)
                else: # Crank-Nicholson scheme
                    E = np.linalg.inv(I - 0.5 * self.x_step * C) @ (I + 0.5 * self.x_step * C)
            
                for _ in range(self.x_numbers):
                    payoffs = E @ payoffs
                    payoffs[0, 0] = (strike1 - strike) * np.exp(-self.model.rate * _ * self.x_step)
                    payoffs[-1, 0] = 0

                cs = CubicSpline(self.y_values1d[::-1], payoffs[:, 0][::-1])
                self.CS = cs
                self.lastPayoffs = payoffs[:, 0][::-1]

                price = cs(spot)
                delta = (cs(spot + EPSILON) - cs(spot - EPSILON)) / (2 * EPSILON)
                gamma = (cs(spot + EPSILON) + cs(spot - EPSILON) - 2 * cs(spot)) / (EPSILON * EPSILON)
                return Price(price=price, delta=delta, gamma=gamma)

            else:
                
                for i in range(self.x_numbers):
                    payoffs = self.E[i] @ payoffs
                    payoffs[0, 0] = (strike1 - strike) * np.exp(-self.model.rate * i * self.x_step) 
                    payoffs[-1, 0] = 0 #payoffs[-2, 0]
                
                cs = CubicSpline(self.y_values1d[::-1], payoffs[:, 0][::-1])
                self.CS = cs
                self.lastPayoffs = payoffs[:, 0][::-1]

                price_plus, price, price_minus = cs(spot + EPSILON), cs(spot), cs(spot - EPSILON)
                delta = (price_plus - price_minus) / (2 * EPSILON)
                gamma = (price_plus + price_minus - 2 * price) / (EPSILON * EPSILON)
                return Price(price=price, delta=delta, gamma=gamma)
        
        else:
            return Price(0)
    

    def Plots(self, option:IOption = None):

        if self.CS is not None:
            cs = self.CS
            payoffs = self.lastPayoffs
            initPayoffs = self.initialPayoffs
            spot = self.model.spot

            deltas, gammas = [], []
            for i, s in enumerate(self.y_values1d[::-1]):
                if i == 0:
                    delta = (cs(s + EPSILON) - cs(s)) / EPSILON
                    gamma = (cs(s + 2 * EPSILON) - 2 * cs(s + EPSILON) + cs(s)) / (2 * EPSILON * EPSILON)
                elif i == len(self.y_values1d) - 1:
                    delta = (cs(s) - cs(s - EPSILON)) / EPSILON
                    gamma = (cs(s) - 2 * cs(s - EPSILON) + cs(s - 2 * EPSILON)) / (2 * EPSILON * EPSILON)
                else:
                    delta = (cs(s + EPSILON) - cs(s - EPSILON)) / (2 * EPSILON)
                    gamma = (cs(s + EPSILON) + cs(s - EPSILON) - 2 * cs(s)) / (EPSILON * EPSILON)
                deltas.append(delta)
                gammas.append(gamma)

                
            # Configuration de la figure
            plt.figure(figsize=(12, 10))

            # Prix en %
            plt.subplot(3, 1, 1)
            plt.title('Prix en %', fontsize=16)
            plt.plot(self.y_values1d[::-1], payoffs, color='blue', linewidth=2, label='Prix (%)')
            plt.plot(self.y_values1d[::-1], initPayoffs * (1 if option is None else np.exp(-self.model.dividend * option.maturityDate)), color='black', linewidth=2, label='Payoff (%)')
            plt.axvline(x=spot, color='red', linestyle='--', label='Spot', linewidth=1.5)
            plt.xlabel('Valeur de l’actif sous-jacent', fontsize=14)
            plt.ylabel('Prix (%)', fontsize=14)
            plt.legend()
            plt.grid(True)

            # Delta
            plt.subplot(3, 1, 2)
            plt.title('Delta', fontsize=16)

            # Calcul des deltas en évitant les indices extrêmes
            plt.plot(self.y_values1d[::-1], deltas, color='green', linewidth=2, label='Delta')
            plt.axvline(x=spot, color='red', linestyle='--', label='Spot', linewidth=1.5)
            plt.xlabel('Valeur de l’actif sous-jacent', fontsize=14)
            plt.ylabel('Delta', fontsize=14)
            plt.legend()
            plt.grid(True)

            # Gamma
            plt.subplot(3, 1, 3)
            plt.title('Gamma', fontsize=16)
            plt.plot(self.y_values1d[::-1], gammas, color='purple', linewidth=2, label='Gamma')
            plt.axvline(x=spot, color='red', linestyle='--', label='Spot', linewidth=1.5)
            plt.xlabel('Valeur de l’actif sous-jacent', fontsize=14)
            plt.ylabel('Gamma', fontsize=14)
            plt.legend()
            plt.grid(True)

            # Ajustement de l'affichage
            plt.tight_layout()
            plt.show()
        
        else:
            raise ValueError('Lancer la fonction Price avant de vouloir afficher les sensibilités !')
