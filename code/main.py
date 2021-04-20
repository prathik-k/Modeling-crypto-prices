
from classes.cryptocurrency import Crypto
from classes.GBM_base import *
#from classes.GBM_attention import GBM_attention

if __name__=='__main__':
    btc = Crypto('bitcoin')
    gbm = GBM_attention(crypto=btc,hist_range=[0,1000],
    pred_type='single',period=30,n_pred_periods=10)
    gbm.make_predictions_base(n_pred_paths=100,n_pred=300)
    #gbm.plot_predictions(savefig=True)

    # 100/3; 75/4; 50/6; 30/10;
