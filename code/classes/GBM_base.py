import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
import warnings
from classes.cryptocurrency import Crypto 

class GBM_base(object):
    def __init__(self,crypto,hist_range=None,pred_type='single',period=30,n_pred_periods=10):
        '''
        Arguments:
        crypto: Crypto() object for the asset
        hist_range: Historic range (range of data points) for training set
        pred_type: 'single' - Computation of mu and sigma performed over training set, followed by prediction on all test points.
                   'rolling' - Computation of mu and sigma for successive segments with no. of days specified by 'period'.
        period: See above. Ignored for pred_type='single'.
        n_pred_periods: The number of successive periods for computation of mu and sigma. Ignored for pred_type='single'.
        '''
        '''
            Defining some initialization parameters of the class.

            crypto: Crypto() object for the asset
            prices: Closing price data for the relevant cryptocurrency
        '''

        self.crypto = crypto
        self.pred_type = pred_type if pred_type=='rolling' else 'single'
        self.prices = crypto.get_df['Closing Price (USD)'] 
        self.hist_range = hist_range
        self.train_set = self.__get_train_set()
        
          

        if self.pred_type=='single':
            '''
            Defining more initialization parameters of the class.

            hist_range: Historic range (range of data points) for training set
            train_set: The closing prices for dates within the historic range
            S0: The last price in train_set; used as a root for the future predictions        
            mu, sigma: The mean and standard deviation computed from prices in train_set
            returns: Percent daily returns for all dates in train_set
            '''                     
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
            self.expected_S = None
            self.test_set = None
            self.lower_conf = None
            self.upper_conf = None
        
        elif self.pred_type=='rolling':
            self.period = period
            self.n_pred_periods = n_pred_periods
            self.test_sets = self.__get_test_sets()
            self.S = []
            self.b = []
            self.W = []
            mu,sigma = self.compute_params(self.train_set)
            self.mu_vals,self.sigma_vals = [mu],[sigma]
            self.S0_vals = [float(self.train_set.iloc[-1])]
            
                            
    
    def compute_params(self,train_set):
        returns = ((train_set-train_set.shift(1))/train_set.shift(1)).dropna()    
        mu = returns.mean()
        sigma = returns.std()
        return mu,sigma
    
    #Computes the parameters b and W for the GBM model
    def compute_brownian_params(self,n_pred):        
        b = np.zeros((self.n_pred_paths,self.n_pred))
        for i in range(len(b)):
            b[i] = np.random.normal(0,1,self.n_pred)
        if self.pred_type=='single':
            self.b = b
            #self.b = np.random.normal(0, 1, (self.n_pred_paths,self.n_pred))
            self.W = np.cumsum(self.b, axis=1)
        else:
            self.b.append(b)
            self.W.append(np.cumsum(b, axis=1))

    #Function to make predictions for the base GBM model    
    def make_predictions_base(self,n_pred_paths=2,n_pred=50):    
        if self.pred_type=='single':           
            self.n_pred = n_pred     
            if self.n_pred+len(self.train_set)>len(self.prices):
                warnings.warn("Number of predictions desired is greater than the amount of test data available. Changing size...")
                self.n_pred = len(self.prices)-len(self.train_set)            
            self.n_pred_paths = n_pred_paths
            self.compute_brownian_params(self.n_pred)
            self.pred_dates = np.arange(self.n_pred)+1
            drift = (self.mu - 0.5 * self.sigma**2) * self.pred_dates
            diffusion = self.sigma * self.W
            self.S = self.S0*np.exp(drift+diffusion)  
            self.expected_S = self.S0*np.exp((self.mu+0.5*(self.sigma**2))*self.pred_dates)            
            self.lower_conf = np.exp(np.log(self.S0)+drift-1.96*self.sigma*np.sqrt(self.pred_dates))   
            self.upper_conf = np.exp(np.log(self.S0)+drift+1.96*self.sigma*np.sqrt(self.pred_dates))            
            self.test_set = self.prices[self.hist_range[1]:self.hist_range[1]+self.n_pred]

        elif self.pred_type=='rolling':
            for i,test in enumerate(self.test_sets):
                if i>0:
                    self.S0_vals.append(np.mean(self.S[i-1],axis=0))
                    next_mu,next_sigma = self.compute_params(self.test_sets[i-1])
                    self.mu_vals.append(next_mu)
                    self.sigma_vals.append(next_sigma)
                self.n_pred = len(test)
                self.n_pred_paths = n_pred_paths
                self.compute_brownian_params(self.n_pred)
                self.pred_dates = np.arange(self.n_pred)+1

                drift = (self.mu_vals[i] - 0.5 * self.sigma_vals[i]**2) * self.pred_dates
                diffusion = self.sigma_vals[i] * self.W[i]
                self.S.append(self.S0_vals[i]*np.exp(drift+diffusion))
        self.S = np.array(self.S)


    #Function to plot predictions        
    def plot_predictions(self):
        if self.pred_type=='single':
            _,ax = plt.subplots()
            xvals = np.arange(self.n_pred)
            S_mean = np.mean(self.S,axis=0)
            ax.plot(xvals,self.S[0],c='b',label='Trials')
            for i in range(1,len(self.S)):
                ax.plot(xvals,self.S[i],c='b',label='_')
            ax.plot(xvals,self.test_set,c='r',label='Actual S')
            ax.plot(xvals,self.expected_S,'k',label='Expected S')
            ax.plot(xvals,S_mean,'y',label='Mean S')
            ax.fill_between(xvals,self.lower_conf,self.upper_conf,color='b', alpha=.1)
            ax.legend()
            plt.show()       
        elif self.pred_type=='rolling':
            tot_plots = len(self.test_sets)
            cols = 2
            rows = tot_plots//cols+tot_plots%cols
            fig,ax = plt.subplots(rows,cols, figsize=(20, 20))
            fig.subplots_adjust(hspace = 1, wspace=0.6)
            ax = ax.ravel()
            xvals = np.arange(self.n_pred)
            for t,test in enumerate(self.test_sets):
                S_mean = np.mean(self.S[t],axis=0)
                ax[t].plot(self.S[t][0],label='Trials')
                for i in range(1,len(self.S[t])):
                    ax[t].plot(xvals,self.S[t][i],c='b',label='_')
                ax[t].plot(xvals,test,c='r',label='Actual S')
                ax[t].plot(xvals,S_mean,'y',label='Mean S')
                ax[t].legend()
            plt.show()   




    #Returning train set and validating historical range input
    def __get_train_set(self):
        if isinstance(self.hist_range,list) and len(self.hist_range)==2 and \
            isinstance(self.hist_range[0],int) and isinstance(self.hist_range[1],int) \
            and self.hist_range[0]>=0 and self.hist_range[1]<len(self.prices)-1:
            train_set = self.prices[self.hist_range[0]:self.hist_range[1]]
        else:
            train_set = self.prices[0:200]
            self.hist_range = [0,200]
        return train_set       
    
    def __get_test_sets(self):
        train_end_idx = self.hist_range[-1]
        available_test_range = len(self.prices)-train_end_idx-1
        if self.period*self.n_pred_periods<=available_test_range:
            full_test_set = self.prices[train_end_idx:train_end_idx+self.period*self.n_pred_periods]
            test_sets = np.split(full_test_set,self.n_pred_periods)
            return test_sets
        else:
            full_test_set = self.prices[train_end_idx:(train_end_idx+int((available_test_range//self.period)*self.period))]
            test_sets = np.split(full_test_set,self.n_pred_periods)
            return test_sets


            





    