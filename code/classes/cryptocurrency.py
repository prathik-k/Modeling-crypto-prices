import pandas as pd 
from datetime import timedelta

class Crypto:
    def __init__(self,symbol):
        data_path = "data/"+symbol.upper()+"_USD.csv"
        attention_data_path = "data/"+symbol.upper()+".csv"
        self.symbol = symbol.upper()
        self.price_df = pd.read_csv(data_path)
        self.attention_df = pd.read_csv(attention_data_path)
        self.n_datapoints = len(self.price_df)

    @property
    def get_df(self):
        return self.price_df

    @property
    def get_attention_df(self):
        return self.attention_df
    
    '''
    def return_prices_over_range(self,start_date='2018-05-01',end_date='2019-05-01'):
        if isinstance(end_date,int):
            end_date = start_date+timedelta(days=end_date)
        self.price_df['Date'] = pd.to_datetime(self.price_df['Date'])
        mask = (self.price_df['Date'] > start_date) & (self.price_df['Date'] <= end_date)
        return self.price_df.loc[mask]
    '''
    def return_prices_over_range(self,start_date='2018-05-01',end_date='2019-05-01'):
        if isinstance(end_date,int):
            end_date = start_date+timedelta(days=end_date)
        self.price_df['Date'] = pd.to_datetime(self.price_df['Date'])
        mask = (self.price_df['Date'] > start_date) & (self.price_df['Date'] <= end_date)
        return self.price_df.loc[mask]['Closing Price (USD)']