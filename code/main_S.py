
from classes.cryptocurrency import Crypto
from classes.GBM_base import GBM_base, GBM_attention

if __name__=='__main__':
    btc = Crypto('xrp')



    gbm = GBM_attention(crypto=btc,hist_range=[0,500],
    pred_type='single',period=100,n_pred_periods=3)
    gbm.make_predictions_base(n_pred_paths=100,n_pred=30)

    gbm.plot_predictions(savefig=True)
