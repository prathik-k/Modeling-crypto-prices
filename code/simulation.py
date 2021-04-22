import pandas as pd 
import numpy as np 
import copy
from datetime import timedelta


from classes.cryptocurrency import Crypto
from classes.GBM_base import GBM_base


class Simulation:
    def __init__(self,currencies=['btc','xrp','ltc'],start_date='2018-05-01',time_steps=10):
        self.capital = None
        self.crypto_set = {}
        self.holdings = {}
        self.n_time_steps = time_steps

        for c in currencies:
            curr_crypto = Crypto(c)
            model = GBM_base(curr_crypto,pred_type='rolling',hist_range = [pd.to_datetime(start_date),pd.to_datetime(start_date)+timedelta(days=30)]
            ,n_pred_periods=self.n_time_steps)
            model.make_predictions_base(n_pred_paths=40)
            S = model.S
            S_end_vals = np.zeros(len(S)+1)
            S_end_vals[0] = model.S0_vals[0]
            for i in range(1,len(S)+1):
                S_end_vals[i] = np.mean(S[i-1],axis=0)[-1]
            
            self.crypto_set[c] = (copy.deepcopy(curr_crypto),S_end_vals)
            self.holdings[c] = 0
    
    def simulate(self,h=1,start_capital=1e5,trade_amount = 5e3):
        '''
        Function to simulate model performance. 
        Arguments:
        h: Minimum threshold value (as percentage change from previous price) to perform a buy or sell action.
        '''
        step = 1
        self.capital = start_capital
        while step<self.n_time_steps+1:
            p_max,n_max=0,0
            for c in self.crypto_set.keys():
                prev_price = self.crypto_set[c][1][step-1]
                curr_price = self.crypto_set[c][1][step]
                diff = curr_price-prev_price
                if diff>=0:
                    if diff>p_max:
                        p_max = diff
                        trading_curr_pos = (c,p_max)                    
                else:
                    if diff<n_max:
                        n_max = diff
                        trading_curr_neg = (c,n_max)
            
            if p_max == 0 or abs(n_max)>abs(p_max):
                trade_c,_ = trading_curr_neg
                if (h/100)*self.crypto_set[trade_c][1][step-1]>abs(n_max) or self.holdings[trade_c]<=trade_amount:
                    print('The portfolio was held at the current state.')
                else:
                    self.holdings[trade_c]-= trade_amount
                    self.capital += trade_amount
                    print('{} of {} was sold. The remaining balance of {} is {}.'.format(trade_amount,trade_c,trade_c,self.holdings[trade_c]))
                    
            elif n_max == 0 or abs(p_max)>abs(n_max):
                trade_c,_ = trading_curr_pos
                if (h/100)*self.crypto_set[trade_c][1][step-1]>abs(p_max) or self.capital<=trade_amount:
                    print('The portfolio was held at the current state.')
                else:
                    self.holdings[trade_c]+= trade_amount
                    self.capital -= trade_amount
                    print('{} of {} was bought. The remaining balance of {} is {}.'.format(trade_amount,trade_c,trade_c,self.holdings[trade_c]))
                    
            step += 1
        
        if self.capital<=trade_amount:
            print('Remaining capital is {}. The model lost the simulation'.format(self.capital))
        else:
            print('Remaining capital is {}. The model succeeded in the simulation'.format(self.capital))              
            
    
        
        

if __name__=='__main__':
    sim = Simulation()
    sim.simulate()



