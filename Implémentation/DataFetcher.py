from Option import *
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt



class DataFetcher:

    def __init__(self, path, sheet_name=None):
        
        if sheet_name is None:
            self.df = pd.read_excel(path)
        else:
            self.df = pd.read_excel(path, sheet_name=sheet_name)

        self.quotesDic = {}
        self.forwardsRatesDivs = {}
        self.expiries = []
        

    def getData(self):
        df = self.df

        for index, row in df.iterrows():
            timeToMaturity = row['timeToMaturity']
            forward = row['forward']
            rate = row['rate']
            dividend = row['dividend']
            
            self.expiries.append(timeToMaturity)
            self.forwardsRatesDivs[timeToMaturity] = [forward, rate, dividend]

            if timeToMaturity not in self.quotesDic:
                self.quotesDic[timeToMaturity] = []
            
            existing_strikes = df.columns[5:]

            for strike in existing_strikes:
                try:
                    impliedVol = float(row[strike])
                    strike_value = float(strike)

                    if strike_value > forward:
                        quote = Quote(strike_value, impliedVol, OptionType.CALL)
                    else:
                        quote = Quote(strike_value, impliedVol, OptionType.PUT)
                    
                    self.quotesDic[timeToMaturity].append(quote)

                except ValueError:
                    print(f"Skipping invalid strike value: {strike}")

        print("DATA HAS BEEN FETCHED")


    



