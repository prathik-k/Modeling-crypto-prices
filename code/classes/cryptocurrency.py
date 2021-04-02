import pandas as pd 

class Crypto:
    def __init__(self,symbol):
        data_path = "data/"+symbol.upper()+"_USD.csv"
        self.price_df = pd.read_csv(data_path)
        self.n_datapoints = len(self.price_df)

    @property
    def get_df(self):
        return self.price_df
    
