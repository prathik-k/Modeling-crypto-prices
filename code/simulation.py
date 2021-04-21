import pandas as pd 
import numpy as np 
from classes.cryptocurrency import Crypto
from classes.GBM_base import GBM_base

if __name__=='__main__':
    initial_capital = 1e5
    btc,xrp,ltc = Crypto('btc'),Crypto('xrp'),Crypto('ltc')
    btc_range = btc.return_prices_over_range(start_date='2018-05-01',end_date='2019-05-01')
    xrp_range = xrp.return_prices_over_range(start_date='2018-05-01',end_date='2019-05-01')
    ltc_range = btc.return_prices_over_range(start_date='2018-05-01',end_date='2019-05-01')
    print(btc_range)


