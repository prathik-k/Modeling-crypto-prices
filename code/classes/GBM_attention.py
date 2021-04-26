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

    def __init__(self,crypto,pred_type,hist_range, period, n_pred_periods):
        
        super().__init__(crypto,hist_range=None,pred_type='single',period=30,n_pred_periods=10)
        #super().__get_confidence_intervals(S0,mu,sigma,drift,pred_dates)
        

        self.crypto = crypto
        self.dates = pd.to_datetime(crypto.get_attention_df['Date'])
        self.pred_type = pred_type if pred_type=='rolling' else 'single'
        self.prices = crypto.get_df['Closing Price (USD)']
        self.attention = crypto.get_attention_df['attention_scaled']
        self.hist_range = hist_range
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
            print(self.mu_a)
            print(self.mu_a)

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

        elif self.pred_type=='rolling':
            self.period = period
            self.n_pred_periods = n_pred_periods
            self.test_sets, self.test_sets_a = self.__get_test_sets()
            self.S = []
            self.P = []
            self.expected_S = []
            self.lower_conf = []
            self.upper_conf = []
            self.b_a = None
            self.W_a = None
            self.b = []
            self.W = []
            mu,sigma = self.compute_params(self.train_set)
            mu_a,sigma_a = self.compute_params(self.train_set_a)
            self.mu_vals,self.sigma_vals = [mu],[sigma]
            self.mu_vals_a,self.sigma_vals_a = [mu_a],[sigma_a]
            self.S0_vals = [float(self.train_set.iloc[-1])]
            self.P0_vals = [float(self.train_set_a.iloc[-1])]



    # def compute_params(self, train_set):
    #     #super().compute_params(train_set)
    #     '''
    #     Function to compute mu and sigma over a given training set
    #     Arguments:
    #     train_set: The training set (i.e., list of closing prices as a pandas dataframe)
    #     '''
    #     returns = ((train_set-train_set.shift(1))/train_set.shift(1)).dropna()
    #     mu = returns.mean()
    #     sigma = returns.std()
    #     return mu,sigma

    # def compute_brownian_params(self, n_pred):
    #     #super().compute_brownian_params(n_pred)
    #     b = np.zeros((self.n_pred_paths,self.n_pred))
    #     for i in range(len(b)):
    #         b[i] = np.random.normal(0,1,self.n_pred)
    #     if self.pred_type=='single':
    #         self.b = b
    #         #self.b = np.random.normal(0, 1, (self.n_pred_paths,self.n_pred))
    #         self.W = np.cumsum(self.b, axis=1)
    #     else:
    #         self.b.append(b)
    #         self.b_a = b
    #         self.W_a = np.cumsum(self.b_a, axis=1)
    #         self.W.append(np.cumsum(b, axis=1))

    def make_predictions_base(self, n_pred_paths, n_pred):
        #super().make_predictions_base(n_pred_paths=2,n_pred=50)

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


            self.mu = self.mu*self.P
            self.sigma = self.sigma*(self.P**(1/2))

            drift = (self.mu - 0.5 * (self.sigma)**2) * self.pred_dates
            diffusion = self.sigma*self.W
            self.S = self.S0*np.exp(drift+diffusion)
            '''
            self.expected_S = self.S0*np.exp((self.mu+0.5*(self.sigma**2))*self.pred_dates)
            self.lower_conf = np.exp(np.log(self.S0)+drift-1.96*self.sigma*np.sqrt(self.pred_dates))
            self.upper_conf = np.exp(np.log(self.S0)+drift+1.96*self.sigma*np.sqrt(self.pred_dates))
            '''
            self.expected_S,self.lower_conf,self.upper_conf = self.__get_confidence_intervals(self.S0,self.mu,self.sigma,drift,self.pred_dates)
            print(type(self.hist_range[1]))
            self.test_set = self.crypto.return_prices_over_range(self.hist_range[1],self.hist_range[1]+timedelta(days=self.n_pred))

        elif self.pred_type=='rolling':
            # self.n_pred = n_pred
            # #self.period = period
            # #self.n_pred_periods = n_pred_periods
            # if self.n_pred+len(self.train_set)>len(self.prices):
            #     warnings.warn("Number of predictions desired is greater than the amount of test data available. Changing size...")
            #     self.n_pred = len(self.prices)-len(self.train_set)
            # self.n_pred_paths = n_pred_paths
            # self.compute_brownian_params(self.n_pred)
            # self.pred_dates = np.arange(self.n_pred)+1
            # drift_a = (self.mu_a - 0.5 * self.sigma_a**2) * self.pred_dates
            # diffusion_a = self.sigma_a * self.W_a
            # self.P = self.P0*np.exp(drift_a + diffusion_a)

            for i,test in enumerate(self.test_sets_a):
                # if i>0:
                #     self.P0_vals.append(np.mean(self.P[i-1],axis=0))
                #     next_mu_a,next_sigma_a = self.compute_params(self.test_sets_a[i-1])
                #     self.mu_vals_a.append(next_mu_a)
                #     self.sigma_vals_a.append(next_sigma_a)

                    # self.S0_vals.append(np.mean(self.S[i-1],axis=0))
                    # next_mu,next_sigma = self.compute_params(self.test_sets[i-1])
                    # self.mu_vals.append(next_mu)
                    # self.sigma_vals.append(next_sigma)

                self.n_pred = len(test)
                self.n_pred_paths = n_pred_paths
                self.compute_brownian_params(self.n_pred)
                self.pred_dates = np.arange(self.n_pred) + 1

                drift_a = (self.mu_a - 0.5 * self.sigma_a**2) * self.pred_dates
                diffusion_a = self.sigma_a * self.W_a
                self.P = self.P0*np.exp(drift_a + diffusion_a)
                
                # self.mu_vals[i] = self.mu_vals[i]*self.P[i]
                # self.sigma_vals[i] = self.sigma_vals[i]*(self.P[i]**(1/2))
                # drift = (self.mu_vals[i] - 0.5 * self.sigma_vals[i]**2) * self.pred_dates
                # diffusion = self.sigma_vals[i] * self.W[i]
                # self.S.append(self.S0_vals[i]*np.exp(drift+diffusion))
            
            #self.P = np.array(self.P)
            #self.P = preprocessing.normalize(self.P)

            for i,test in enumerate(self.test_sets):
                if i>0:

                    self.S0_vals.append(np.mean(self.S[i-1],axis=0))
                    next_mu,next_sigma = self.compute_params(self.test_sets[i-1])
                    self.mu_vals.append(next_mu)
                    self.sigma_vals.append(next_sigma)

                self.n_pred = len(test)
                self.n_pred_paths = n_pred_paths
                self.compute_brownian_params(self.n_pred)
                self.pred_dates = np.arange(self.n_pred) + 1

                #print(self.mu_vals[i].shape)

                self.mu_vals[i] = np.multiply(self.mu_vals[i],self.P)
                self.sigma_vals[i] = np.multiply(self.sigma_vals[i],(self.P**(1/2)))
                print(self.mu_vals[i].shape)
                print(type(self.P))
                print(self.P.shape)
                

                drift = (self.mu_vals[i] - 0.5 * self.sigma_vals[i]**2) * self.pred_dates
                diffusion = self.sigma_vals[i] * self.W[i]
                self.S.append(self.S0_vals[i]*np.exp(drift+diffusion))



                exp_S, lower, upper = self.__get_confidence_intervals(self.S0_vals[i],self.mu_vals[i],self.sigma_vals[i],drift,self.pred_dates)
                self.expected_S.append(exp_S)
                self.lower_conf.append(lower)
                self.upper_conf.append(upper)

        
        self.P = np.array(self.P)
        
        #df.to_csv("P.csv")
        #self.P = preprocessing.normalize(self.P)        
        
        np.dtype(np.float64)
        self.S = np.array(self.S)

        #self.S = np.reshape(self.S, (-1, self.period, self.n_pred_periods))
        print(self.S.shape)
        #print(self.mu_vals[i].shape)
        #print(self.sigma_vals[i].shape)
        #print(self.P.shape)

        #print(self.P[i].shape)
        #print(self.S0_vals[i].shape)
        #print(self.mu_vals[i].shape)


    def __get_confidence_intervals(self,S0,mu,sigma,drift,pred_dates):
        #super().__get_confidence_intervals(S0,mu,sigma,drift,pred_dates)
        expected_S = S0*np.exp((mu+0.5*(sigma**2))*pred_dates)
        lower_conf = np.exp(np.log(S0)+drift-1.96*sigma*np.sqrt(pred_dates))
        upper_conf = np.exp(np.log(S0)+drift+1.96*sigma*np.sqrt(pred_dates))
        return (expected_S,lower_conf,upper_conf)

    # def plot_predictions(self, savefig=True):
    # #     super().plot_predictions(savefig=True)
    #     crypto = self.crypto.symbol.upper()+' ($)'
    #     if self.pred_type=='single':
    #         fig,ax = plt.subplots()
    #         xvals = np.arange(self.n_pred)
    #         S_mean = np.mean(self.S,axis=0)
    #         mape = self.get_error_metrics(self.test_set,self.S)
    #         ax.plot(xvals,self.S[0],c='tab:blue',label='Trials')
    #         for i in range(1,len(self.S)):
    #             ax.plot(xvals,self.S[i],c='tab:blue',label='_')
    #         ax.plot(xvals,self.test_set,c='k',label='Actual S')
    #         #ax.plot(xvals,self.expected_S,'tab:red',label='Expected S')
    #         ax.plot(xvals,S_mean,'y',label='Mean S')
    #         #ax.fill_between(xvals,self.lower_conf,self.upper_conf,color='b', alpha=.1)
    #         ax.annotate('MAPE: {:.3f}'.format(mape),(0.2, 0.95),
    #         xycoords='axes fraction',arrowprops=dict(facecolor='black', shrink=0.05),
    #         fontsize=8,horizontalalignment='right', verticalalignment='top')
    #         ax.set_xlabel('Days')
    #         ax.set_ylabel(crypto)
    #         ax.grid()
    #         ax.legend(loc =3)

    #     elif self.pred_type=='rolling':
    #         tot_plots = len(self.test_sets)
    #         cols = 2
    #         rows = tot_plots//cols+tot_plots%cols
    #         fig,ax = plt.subplots(rows,cols, figsize=(20, 20))
    #         fig.subplots_adjust(hspace = 1, wspace=0.6)
    #         ax = ax.ravel()
    #         xvals = np.arange(self.n_pred)
    #         for t,test in enumerate(self.test_sets):
    #             S_mean = np.mean(self.S[t],axis=0)
    #             print(self.S[t].shape)
    #             print(test.shape)
    #             mape = self.get_error_metrics(test,self.S[t])
    #             ax[t].plot(self.S[t][0],label='Trials')
    #             for i in range(1,len(self.S[t])):
    #                 ax[t].plot(xvals,self.S[t][i],c='tab:blue',label='_')
    #             ax[t].plot(xvals,test,c='k',label='Actual S')
    #             #ax[t].plot(xvals,self.expected_S[t],'tab:red',label='Expected S')
    #             ax[t].plot(xvals,S_mean,'y',label='Mean S')
    #             ax[t].annotate('MAPE: {:.3f}'.format(mape),(0.2, 0.95),
    #             xycoords='axes fraction',arrowprops=dict(facecolor='black', shrink=0.05),
    #             fontsize=8,horizontalalignment='right', verticalalignment='top')
    #             if t==0:
    #                 ax[t].legend(loc =3)
    #             ax[t].grid()
    #             ax[t].set_xlabel('Days')
    #             ax[t].set_ylabel(crypto)

    #     if savefig:
    #          save_path = 'results/{}_{}paths_{}'.format(self.crypto.symbol,str(self.n_pred_paths),self.pred_type)
    #          fig.savefig(save_path)
    #     plt.show()

    # def get_error_metrics(self,actual,predicted_set):
    #     #super().get_error_metrics(actual, predicted_set

    #     S_mean = np.mean(predicted_set,axis=0)
    #     mape = np.mean(np.abs((S_mean - actual)/actual))*100
    #     return mape
            #Can add other error metrics if necessary...

    def __get_test_sets(self):
        #super().__get_test_sets

        '''
        Function to generate and return multiple test sets for the rolling prediction case.
        '''
        train_end_idx = self.dates[self.dates==self.hist_range[-1]].index[0].astype(int)
        available_test_range = len(self.prices)-train_end_idx-1

        if self.period*self.n_pred_periods<=available_test_range:
            full_test_set = self.prices[train_end_idx:train_end_idx+self.period*self.n_pred_periods]
            test_sets = np.split(full_test_set,self.n_pred_periods)

            full_test_set_a = self.attention[train_end_idx:train_end_idx+self.period*self.n_pred_periods]
            test_sets_a = np.split(full_test_set_a,self.n_pred_periods)

            return test_sets, test_sets_a
        else:
            full_test_set = self.prices[train_end_idx:(train_end_idx+int((available_test_range//self.period)*self.period))]
            test_sets = np.split(full_test_set,self.n_pred_periods)

            full_test_set_a = self.attention[train_end_idx:(train_end_idx+int((available_test_range//self.period)*self.period))]
            test_sets_a = np.split(full_test_set_a,self.n_pred_periods)

            return test_sets, test_sets_a

    def __get_train_set(self):
        #super().__get_train_set

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
            isinstance(self.hist_range[0],datetime) and isinstance(self.hist_range[1],datetime) \
            and start_date>=self.dates[0]>=0 and end_date<self.dates.iloc[-1]:
            start_idx,end_idx = self.dates[self.dates==start_date].index[0],self.dates[self.dates==end_date].index[0]
            train_set = self.prices[start_idx:end_idx]
            train_set_a = self.attention[start_idx:end_idx]
        else:
            train_set = self.prices[0:200]
            train_set_a = self.attention[0:200]
            self.hist_range = [self.crypto.price_df['Date'][0],self.crypto.price_df['Date'][199]]
        return train_set, train_set_a