import pandas as pd 
import numpy as np 
import copy

from classes.cryptocurrency import Crypto
from classes.GBM_base import GBM_base


class Simulation:
    def __init__(self,currencies=['btc','xrp','ltc'],start_date='2018-05-01',end_date='2019-05-01',start_capital=1e5):
        self.capital = start_capital
        self.crypto_set = {}
        for c in currencies:
            curr_crypto = Crypto(c)
            self.crypto_set[c] = (copy.deepcopy(curr_crypto),curr_crypto.return_prices_over_range(start_date,end_date))
            
    
        
        

if __name__=='__main__':
    initial_capital = 1e5
    btc,xrp,ltc = Crypto('btc'),Crypto('xrp'),Crypto('ltc')
    btc_range = btc.return_prices_over_range(start_date='2018-05-01',end_date='2019-05-01')
    xrp_range = xrp.return_prices_over_range(start_date='2018-05-01',end_date='2019-05-01')
    ltc_range = btc.return_prices_over_range(start_date='2018-05-01',end_date='2019-05-01')
    print(btc_range)



