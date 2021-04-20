import pandas as pd 
import numpy as np 
from classes.cryptocurrency import Crypto
from classes.GBM_base import GBM_base

if __name__=='__main__':
    initial_capital = 1e5
    btc,xrp,ltc = Crypto('btc'),Crypto('xrp'),Crypto('ltc')

    