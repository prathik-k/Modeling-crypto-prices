import pandas as pd
import numpy as np
import copy
from datetime import timedelta


from classes.cryptocurrency import Crypto
from classes.GBM_base import GBM_base, GBM_attention

import matplotlib.pyplot as plt


class Simulation:
    def __init__(self,currencies=['btc','xrp','ltc'],start_date='2018-05-01',time_steps=20,period=15):

        self.capital = None
        self.crypto_set = {}
        self.holdings = {}
        self.n_time_steps = time_steps
        self.period = period

        for c in currencies:
            curr_crypto = Crypto(c)
            model = GBM_base(curr_crypto,pred_type='rolling',hist_range = [pd.to_datetime(start_date),pd.to_datetime(start_date)+timedelta(days=30)],period=self.period,n_pred_periods=self.n_time_steps)
            model.make_predictions_base(n_pred_paths=40)
            S = model.S
            S_end_vals = np.zeros(len(S)+1)
            S_end_vals[0] = model.S0_vals[0]
            for i in range(1,len(S)+1):
                S_end_vals[i] = np.mean(S[i-1],axis=0)[-1]

            self.crypto_set[c] = (copy.deepcopy(curr_crypto),S_end_vals)
            self.holdings[c] = 0

    def simulate(self,h=0.1,start_capital=1e5,trade_amount = 5000):
        '''
        Function to simulate model performance.
        Arguments:
        h: Minimum threshold value (as percentage change from previous price) to perform a buy or sell action.
        '''
        step = 1

        track_cap = np.zeros((self.n_time_steps+1,1))
        current_hold = np.zeros((self.n_time_steps+1,3))
        track_cap[0,0] = start_capital
        self.capital = start_capital
        while step<self.n_time_steps+1:
            p_max,n_max=0,0

            for c in self.crypto_set.keys():
                prev_price = self.crypto_set[c][1][step-1]
                curr_price = self.crypto_set[c][1][step]
                diff_perc = (curr_price-prev_price)*100/prev_price
                if diff_perc>=0:
                    if diff_perc>p_max:
                        p_max = diff_perc
                        trading_curr_pos = (c,p_max)
                else:
                    if diff_perc<n_max:
                        n_max = diff_perc
                        trading_curr_neg = (c,n_max)

            print(self.holdings)

            if p_max == 0 or abs(n_max)>abs(p_max):
                trade_c,_ = trading_curr_neg
                trading_curr_price = self.crypto_set[trade_c][1][step]

                if (h/100)*self.crypto_set[trade_c][1][step-1]>abs(n_max) or self.holdings[trade_c]*trading_curr_price-trade_amount<0:
                    print('The portfolio was held at the current state.')


                else:
                    self.holdings[trade_c] -= trade_amount/self.crypto_set[trade_c][1][step]
                    self.capital += trade_amount
                    print('${} worth of {} was sold. The {} wallet has {} {} in it. The remaining cash balance is {}.'.format(trade_amount,
                    trade_c,trade_c,self.holdings[trade_c],trade_c,self.capital))


            elif n_max == 0 or abs(p_max)>abs(n_max):
                trade_c,_ = trading_curr_pos
                if (h/100)*self.crypto_set[trade_c][1][step-1]>abs(p_max) or self.capital<=trade_amount:
                    print('The portfolio was held at the current state.')

                else:
                    self.holdings[trade_c]+= trade_amount/self.crypto_set[trade_c][1][step]
                    self.capital -= trade_amount
                    print('${} worth of {} was bought. The {} wallet has {} {} in it. The remaining cash balance is {}.'.format(trade_amount,
                    trade_c,trade_c,self.holdings[trade_c],trade_c,self.capital))


            total_usd = 0
            cn = 0
            for c in self.crypto_set.keys():
                current_hold[step,cn] = self.holdings[c]
                total_usd += self.holdings[c]*self.crypto_set[c][1][step]
                cn += 1
            total_usd += self.capital
            track_cap[step,0] = total_usd
            step += 1


        self.disp_holdings()


        xl = np.arange(self.n_time_steps+1)

        plt.rcParams.update({'font.size': 16})
        plt.scatter(xl,track_cap[:,0],marker='d',color='red',s=80)
        plt.axhline(y = 1e5,color='k',linestyle='dashed')
        plt.xlabel('Trade number')
        plt.ylabel('Portfolio capital is USD')
        plt.ylim([np.min(track_cap[:,0])-10000,np.max(track_cap[:,0])+10000])
        plt.legend(['Initial capital','Current portfolio value'],loc='upper right')
        plt.grid()
        plt.show()

        fig, axs = plt.subplots(3)

        axs[0].stem(xl,current_hold[:,0],basefmt=" ")
        axs[0].set_ylabel('BTC')
        axs[1].stem(xl,current_hold[:,1],basefmt=" ")
        axs[1].set_ylabel('XRP')
        axs[2].stem(xl,current_hold[:,2],basefmt=" ")
        axs[2].set_ylabel('LTC')
        axs[2].set_xlabel('Trade number')
        plt.show()


    def disp_holdings(self,step=-1):
        print('\n\nPortfolio consists of:')
        total_usd = 0
        for c in self.crypto_set.keys():
            total_usd += self.holdings[c]*self.crypto_set[c][1][step]
            print('{} balance = {}..........${}'.format(c,self.holdings[c],self.holdings[c]*self.crypto_set[c][1][step]))
        total_usd += self.capital
        print('Total value of the portfolio is ${}'.format(total_usd))




if __name__=='__main__':
    sim = Simulation()
    sim.simulate()
