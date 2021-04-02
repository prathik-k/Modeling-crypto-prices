import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
import warnings

class GBM_base(object):
    def __init__(self,crypto,hist_range=None):        
        self.crypto = crypto
        self.prices = crypto.get_df['Closing Price (USD)']       

        self.hist_range = hist_range
        self.train_set = self.__get_train_set()  
        self.compute_params(self.train_set)
        self.S0 = float(self.train_set.iloc[-1])

        self.n_pred = None
        self.n_pred_paths = None    
        self.pred_dates = None
        self.b = None    
        self.W = None
        self.S = None
        self.test_set = None
           
    
    def compute_params(self,train_set):
        self.returns = ((train_set-train_set.shift(1))/train_set.shift(1)).dropna()    
        self.mu = self.returns.mean()
        self.sigma = self.returns.std()
    
    def compute_brownian_params(self):
        self.b = np.random.normal(0, 1, (self.n_pred_paths,self.n_pred))
        self.W = np.cumsum(self.b, axis=1)
    
    def make_predictions(self,n_pred=50,n_pred_paths=2):               
        self.n_pred = n_pred     
        if self.n_pred+len(self.train_set)>len(self.prices):
            warnings.warn("Number of predictions desired is greater than the amount of test data available. Changing size...")
            self.n_pred = len(self.prices)-len(self.train_set)
        
        self.n_pred_paths = n_pred_paths
        self.compute_brownian_params()
        self.pred_dates = np.arange(self.n_pred)+1
        drift = (self.mu - 0.5 * self.sigma**2) * self.pred_dates
        diffusion = self.sigma * self.W

        self.S = self.S0*np.exp(drift+diffusion)     

        self.test_set = self.prices[self.hist_range[1]:self.hist_range[1]+self.n_pred]
        #self.plot_predictions(S)
        
    def plot_predictions(self):
        fig,ax = plt.subplots()
        xvals = np.arange(self.n_pred)
        for i in range(len(self.S)):
            ax.plot(xvals,self.S[i],c='b')
        ax.plot(xvals,self.test_set,c='r')
        plt.show()       

    
    def __get_train_set(self):
        if isinstance(self.hist_range,list) and len(self.hist_range)==2 and isinstance(self.hist_range[0],int) and isinstance(self.hist_range[1],int) \
            and self.hist_range[0]>=0 and self.hist_range[1]<len(self.prices)-1:
            train_set = self.prices[self.hist_range[0]:self.hist_range[1]]
        else:
            train_set = self.prices[0:200]
            self.hist_range = [0,200]
        return train_set       
