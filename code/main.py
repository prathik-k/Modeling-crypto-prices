
from classes.cryptocurrency import Crypto 
from classes.GBM_base import GBM_base

if __name__=='__main__':
    btc = Crypto('BTC')
    gbm = GBM_base(crypto=btc,hist_range=[0,1000],
    pred_type='rolling',period=30,n_pred_periods=10)
    gbm.make_predictions_base(n_pred_paths=100,n_pred=50)
    gbm.plot_predictions(savefig=True)