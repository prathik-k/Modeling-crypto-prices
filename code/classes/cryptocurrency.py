import pandas as pd 

class Crypto:
    def __init__(self,symbol):
        data_path = "data/"+symbol.upper()+".csv"
        self.symbol = symbol.upper()
        self.price_df = pd.read_csv(data_path)
        self.n_datapoints = len(self.price_df)

    @property
    def get_df(self):
        return self.price_df
    
