from Option import *
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SurfacesConstructor:

    def __init__(self, path):

        all_sheets = pd.read_excel(path, sheet_name=None)
        sheet_names = list(all_sheets.keys())
        sheets_from_third = {name: all_sheets[name] for name in sheet_names[2:]}
        self.surfaces = sheets_from_third

        self.keys = self.surfaces.keys()

        self.timeToMaturities = []
        self.minMaturities = []
        self.maxMaturities = []

        self.allSurfaces = self.getSurfaces()

        self.maxOfMinMaturities = np.max(self.minMaturities)
        self.minOfMaxMaturities = np.min(self.maxMaturities)

        self.rangeTimeToMaturities = [ttm for ttm in self.timeToMaturities if ttm >= self.maxOfMinMaturities or ttm <= self.minOfMaxMaturities]
        
        self.reconstructedSurfaces = self.ReConstructedSurfaces()

        #self.plot_reconstructed_surfaces_3D()
        print('SURFACES SUCCESFULLY RECONSTRUCTED')


    def ReConstructedSurfaces(self):
        reconstructedSurfaces = {}
        for key in self.keys:
            quotesDic, forwardsRatesDivs = self.allSurfaces[key]
            filteredQuotesDic = {ttm: quotes for ttm, quotes in quotesDic.items() if ttm in self.rangeTimeToMaturities}
            keysFilteredQuotesDic = filteredQuotesDic.keys()
            strikes = [quote.strike for quote in filteredQuotesDic[self.rangeTimeToMaturities[0]]]

            missingKeys = [ttm for ttm in self.rangeTimeToMaturities if ttm not in keysFilteredQuotesDic]

            if missingKeys: 
                splines = {}
                for strike in strikes:#alors il faut interpoler pour les maturités manquantes pour chaque strike
                    points = []
                    for ttm, quotes in filteredQuotesDic.items():
                        implied_vol = next((q.impliedVol for q in quotes if q.strike == strike), None)

                        if implied_vol is not None:
                            points.append((ttm, implied_vol))
                
                    if len(points) > 1:
                        ttm_values, vol_values = zip(*points)
                        cs = CubicSpline(ttm_values, vol_values, bc_type='natural')
                        splines[strike] = cs
                    else:
                        print(f'Not enough points to create spline for strike {strike}')

                # Remplir quotesDic et forwardsRatesDivs pour les maturités manquantes en utilisant les splines
                for timeToMaturity in missingKeys:
                    quotesDic[timeToMaturity] = []
                    forwardsRatesDivs[timeToMaturity] = []

                    existing_ttm = sorted(filteredQuotesDic.keys())
                    existing_forwards = []
                    existing_rates = []
                    existing_divs = []

                    for ttm in existing_ttm:
                        forward, rate, div = forwardsRatesDivs[ttm]
                        existing_forwards.append(forward)
                        existing_rates.append(rate)
                        existing_divs.append(div)

                    if len(existing_forwards) > 1:
                        forward_interp = interp1d(existing_ttm, existing_forwards, fill_value="extrapolate")
                        forwardsRatesDivs[timeToMaturity].append(forward_interp(timeToMaturity))
                    else:
                        forwardsRatesDivs[timeToMaturity].append(None)

                    if len(existing_rates) > 1:
                        rate_interp = interp1d(existing_ttm, existing_rates, fill_value="extrapolate")
                        forwardsRatesDivs[timeToMaturity].append(rate_interp(timeToMaturity))
                    else:
                        forwardsRatesDivs[timeToMaturity].append(None)

                    if len(existing_divs) > 1:
                        div_interp = interp1d(existing_ttm, existing_divs, fill_value="extrapolate")
                        forwardsRatesDivs[timeToMaturity].append(div_interp(timeToMaturity))
                    else:
                        forwardsRatesDivs[timeToMaturity].append(None)

                    forward, rate, div = forwardsRatesDivs[self.rangeTimeToMaturities[0]]
                    for strike in strikes:
                        if strike in splines:
                            implied_vol = splines[strike](timeToMaturity)
                            optionType = OptionType.CALL if strike >= forward else OptionType.PUT
                            new_quote = Quote(strike=strike, impliedVol=implied_vol, optionType=optionType)
                            quotesDic[timeToMaturity].append(new_quote)

            quotesDic = {ttm: quotesDic[ttm] for ttm in sorted(quotesDic.keys())}
            forwardsRatesDivs = {ttm: forwardsRatesDivs[ttm] for ttm in sorted(forwardsRatesDivs.keys())}
            reconstructedSurfaces[key] = [quotesDic, forwardsRatesDivs]
    
        return reconstructedSurfaces


    def plot_reconstructed_surfaces_3D(self):
        for key, (quotesDic, forwardsRatesDivs) in self.reconstructedSurfaces.items():
            # Préparer les données pour la volatilité implicite
            strikes = sorted(set(quote.strike for quotes in quotesDic.values() for quote in quotes))
            ttm_values = sorted(quotesDic.keys())
            print(len(ttm_values))
            # Créer une grille pour la surface
            X, Y = np.meshgrid(ttm_values, strikes)
            Z = np.full(X.shape, np.nan)  # Initier avec NaN

            for i, strike in enumerate(strikes):
                for j, ttm in enumerate(ttm_values):
                    quotes = quotesDic.get(ttm, [])
                    for quote in quotes:
                        if quote.strike == strike:
                            Z[i, j] = quote.impliedVol

            # Tracer la surface 3D pour les quotes
            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.7)
            ax.set_title(f'Implied Volatility Surface for {key}')
            ax.set_xlabel('Time to Maturity (TTM)')
            ax.set_ylabel('Strike')
            ax.set_zlabel('Implied Volatility')
            plt.show()



    def getDataSurfaces(self, df):
        quotesDic, forwardsRatesDivs = {}, {}

        for index, row in df.iterrows():
            timeToMaturity = row['timeToMaturity']
            forward = row['forward']
            rate = row['rate']
            dividend = row['dividend']

            if timeToMaturity not in self.timeToMaturities:
                self.timeToMaturities.append(timeToMaturity)

            forwardsRatesDivs[timeToMaturity] = [forward, rate, dividend]

            if timeToMaturity not in quotesDic:
                quotesDic[timeToMaturity] = []

            quotes = []
            
            existing_strikes = df.columns[5:]

            for strike in existing_strikes:
                try:
                    impliedVol = float(row[strike])
                    strike_value = float(strike)

                    if strike_value > forward:
                        quote = Quote(strike_value, impliedVol, OptionType.CALL)
                    else:
                        quote = Quote(strike_value, impliedVol, OptionType.PUT)

                    quotes.append(quote)
                    
                    quotesDic[timeToMaturity].append(quote)

                except ValueError:
                    print(f"Skipping invalid strike value: {strike}")
        
        
        timeToMaturities = list(quotesDic.keys())
        self.minMaturities.append(np.min(timeToMaturities))
        self.maxMaturities.append(np.max(timeToMaturities))

        return [quotesDic, forwardsRatesDivs]


    def getSurfaces(self):
        surfaces = {}
        sheetsDic = self.surfaces
        keys = list(sheetsDic.keys())
        for key in keys:
            surfaces[key] = self.getDataSurfaces(pd.DataFrame(sheetsDic[key]))
        return surfaces