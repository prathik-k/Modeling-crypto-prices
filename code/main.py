
from classes.cryptocurrency import Crypto 
from classes.GBM_base import GBM_base

if __name__=='__main__':
    btc = Crypto('BTC')
    gbm = GBM_base(btc,[0,1400])
    gbm.make_predictions_base(n_pred=100,n_pred_paths=5)
    gbm.plot_predictions()