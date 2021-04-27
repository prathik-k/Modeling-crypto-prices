import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import warnings
from classes.cryptocurrency import Crypto
from classes.GBM_base import GBM_base


# Child Class GBM_attention from parent GBM_base
class GBM_attention(GBM_base):

    def __init__(self,crypto,pred_type,hist_range=None):        
        super().__init__(crypto,hist_range,pred_type='single')
        self.crypto = crypto
        self.pred_type = pred_type if pred_type=='rolling' else 'single'
        self.prices = crypto.get_df['Closing Price (USD)']
        self.dates = pd.to_datetime(crypto.price_df['Date'])  
        self.attention = crypto.get_attention_df['attention_scaled']
        self.hist_range = [pd.to_datetime(hist_range[0]),pd.to_datetime(hist_range[1])]
        self.train_set, self.train_set_a = self.__get_train_set()
        self.mu_a, self.sigma_a = self.compute_params(self.train_set_a)
        self.P0 = float(self.train_set_a.iloc[-1])
        if self.pred_type=='single':
            '''
            Defining more initialization parameters of the class.
            Arguments:
            hist_range: Historic range (range of data points) for training set
            train_set: The closing prices for dates within the historic range
            S0: The last price in train_set; used as a root for the future predictions
            mu, sigma: The mean and standard deviation computed from prices in train_set
            returns: Percent daily returns for all dates in train_set
            '''
            self.mu_a, self.sigma_a = self.compute_params(self.train_set_a)
            self.P0 = float(self.train_set_a.iloc[-1])
            #print(self.mu_a)
            #print(self.mu_a)

            self.mu,self.sigma = self.compute_params(self.train_set)
            self.S0 = float(self.train_set.iloc[-1])
            '''
            Declaring the parameters that will be used by the member functions.
            n_pred: Number of future prediction points
            n_pred_paths: Number of future prediction paths
            pred_dates: Range of future dates
            b, W: Brownian motion parameters
            S: Set of predictions made by all paths
            test_set: THe actual values observed in the future time range
            lower_conf, upper_conf: Lower and upper 95% confidence intervals based on the lognormal distribution of S in the future
            '''
            self.n_pred = None
            self.n_pred_paths = None
            self.pred_dates = None
            self.b = None
            self.W = None

            self.S = None
            self.P = None
            self.expected_S = None
            self.test_set = None
            self.lower_conf = None
            self.upper_conf = None


    def make_predictions_base(self, n_pred_paths, n_pred):

        if self.pred_type=='single':
            self.n_pred = n_pred
            if self.n_pred+len(self.train_set)>len(self.prices):
                warnings.warn("Number of predictions desired is greater than the amount of test data available. Changing size...")
                self.n_pred = len(self.prices)-len(self.train_set)
            self.n_pred_paths = n_pred_paths
            self.compute_brownian_params(self.n_pred)
            self.pred_dates = np.arange(self.n_pred)+1
            drift_a = (self.mu_a - 0.5 * self.sigma_a**2) * self.pred_dates
            diffusion_a = self.sigma_a * self.W
            self.P = self.P0*np.exp(drift_a + diffusion_a)

            #Chaging mu and sigma parameters based in attention parameter P
            self.mu = self.mu*self.P
            self.sigma = self.sigma*(self.P**(1/2))

            drift = (self.mu - 0.5 * (self.sigma)**2) * self.pred_dates
            diffusion = self.sigma*self.W
            self.S = self.S0*np.exp(drift+diffusion)

            self.expected_S,self.lower_conf,self.upper_conf = self.get_confidence_intervals(self.S0,self.mu*self.P,self.sigma*(self.P**(1/2)),drift,self.pred_dates)
            self.test_set = self.crypto.return_prices_over_range(pd.to_datetime(self.hist_range[1]),pd.to_datetime(self.hist_range[1])+timedelta(days=self.n_pred))

        self.P = np.array(self.P)       
        self.S = np.array(self.S)


    def plot_predictions(self, savefig=True):
        crypto = self.crypto.symbol.upper()+' ($)'
        if self.pred_type=='single':
            fig,ax = plt.subplots()
            xvals = np.arange(self.n_pred)
            S_mean = np.mean(self.S,axis=0)
            mape = self.get_error_metrics(self.test_set,self.S)
            ax.plot(xvals,self.S[0],c='tab:blue',label='Trials')
            for i in range(1,len(self.S)):
                ax.plot(xvals,self.S[i],c='tab:blue',label='_')
            ax.plot(xvals,self.test_set,c='k',label='Actual S')
            ax.plot(xvals,S_mean,'y',label='Mean S')
            ax.annotate('MAPE: {:.3f}'.format(mape),(0.2, 0.95),
            xycoords='axes fraction',arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=8,horizontalalignment='right', verticalalignment='top')
            ax.set_xlabel('Days')
            ax.set_ylabel(crypto)
            ax.grid()
            ax.legend(loc =3)


        if savefig:
             save_path = 'results/{}_{}paths_{}'.format(self.crypto.symbol,str(self.n_pred_paths),self.pred_type)
             fig.savefig(save_path)
        plt.show()



    def __get_train_set(self):

        '''
        Function to generate and return the train set for both 'single' and 'rolling' prediction cases.
        '''
        try:
            if not isinstance(self.hist_range[0],datetime) or not isinstance(self.hist_range[1],datetime):
                start_date,end_date = pd.to_datetime(self.hist_range[0]),pd.to_datetime(self.hist_range[1])
            else:
                start_date,end_date = self.hist_range[0],self.hist_range[1]
        except:
            train_set = self.prices[0:200]
            train_set_a = self.attention[0:200]
            self.hist_range = [self.dates[0],self.dates[199]]
            return train_set, train_set_a

        if isinstance(self.hist_range,list) and len(self.hist_range)==2 and \
            start_date>=self.dates[0] and end_date<self.dates.iloc[-1]:
            start_idx,end_idx = self.dates[self.dates==start_date].index[0],self.dates[self.dates==end_date].index[0]
            train_set = self.prices[start_idx:end_idx]
            train_set_a = self.attention[start_idx:end_idx]
        else:
            train_set = self.prices[0:200]
            train_set_a = self.attention[0:200]
            self.hist_range = [self.crypto.price_df['Date'][0],self.crypto.price_df['Date'][199]]
        return train_set, train_set_a
