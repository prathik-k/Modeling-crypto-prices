
from classes.cryptocurrency import Crypto 
from classes.GBM_base import GBM_base

if __name__=='__main__':
    btc = Crypto('BTC')
    gbm = GBM_base(btc,[0,1000])
    gbm.make_predictions(n_pred=50,n_pred_paths=30)
    gbm.plot_predictions()