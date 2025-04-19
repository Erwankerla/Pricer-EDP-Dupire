from datetime import datetime
from DataFetcher import DataFetcher
from Pricer import *
from SurfacesConstructor import *
from PDE import *


############################################### Récupération des Datas #######################################################################
startDate = datetime.now() - pd.offsets.BDay(1) # Aujourd'hui moins un jour ouvré
maturityDate = '2030-11-12'
path = "Inputs-Outputs\CACSDSurfaces.xlsx"
data = pd.read_excel(path, sheet_name="Data")
CACSpot, rate, dividend = data.iloc[0, 1], data.iloc[0, 2], data.iloc[0, 3]

 
dataFetcher = DataFetcher(path=path, sheet_name="Surface")
dataFetcher.getData() 
forwardsRatesDivs = dataFetcher.forwardsRatesDivs
quotesDic = dataFetcher.quotesDic
expiries = dataFetcher.expiries


############################################### SVI for all volatilities smiles ###############################################################
quotes = [quote for quote in quotesDic[expiries[2]]]
strikes_svi = [0.01 * quote.strike * forwardsRatesDivs[expiries[0]][0] for quote in quotes]
svi = SVI(quotesDic, forwardsRatesDivs)
svi.CreateSurfaceSVIAOA(strikes_svi)
surface = svi.surface
parameters = svi.parameters


############################################### Local Volatility Surface Constructor ###########################################################
localSurfaceCreator = LocalSurface(surface, forwardsRatesDivs, LocalSurfaceType.SVI, parameters=parameters, interpolation_method=InterpolationMethod.CUBIC_NATURAL)
localSurfaceCreator.CreateSurface(logStrike=False)
localSurface = localSurfaceCreator.surface


############################################### Pricing for Call Spread on CAC40 ###############################################################
T = (pd.to_datetime(maturityDate).normalize() - pd.to_datetime(startDate).normalize()).days / 365.25 #time to maturity
S = CACSpot #niveau du CAC à la date de valo
d = 953.3521/S/T #divendes forcées
rate = 0.02070543


bmod = BSModel(spot=S, sigma=0.18, rate=rate, dividend=d)
p = PricerCF(bmod)
pde = OptimalPricerPDE(scheme=SchemePDE.CRANK_NICHOLSON, model=bmod, y_max=4*S, y_numbers=100, x_max=T, x_numbers=None, CFL=0.5, localSurface=localSurface, forwardsRatesDivs=forwardsRatesDivs)


Call100 = VanillaOption(OptionType.CALL, T, strike = 7226.98)
Call135 = VanillaOption(OptionType.CALL, T, strike = 1.35 * 7226.98)
cs = CallSpread(strike = 7226.98, strike1=1.35 * 7226.98, maturityDate=T)

priceCall100, priceCall135 = p.Price(Call100).price, p.Price(Call135).price
priceCS = pde.Price(cs)


print(f"\nCall-Spread BS : {np.round(priceCall100 - priceCall135, 2)}")
print(f"Call-Spread bis Local EDP : {np.round(100 * ( priceCS.price / S), 2)}%")
print(f"Delta Call-Spread bis Local EDP : {priceCS.delta}")
print(f"Gamma Call-Spread bis Local EDP : {priceCS.gamma}") 
pde.Plots()

input('\nAppuyez pour quitter...')