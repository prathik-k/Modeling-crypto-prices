
from classes.cryptocurrency import Crypto
from classes.GBM_base import GBM_base
from classes.GBM_attention import GBM_attention

if __name__=='__main__':
    btc = Crypto('btc')
    gbm = GBM_attention(crypto=btc,hist_range=[0,1000],pred_type='single')
    gbm.make_predictions_base(n_pred_paths=100,n_pred=30)
    
    gbm.plot_predictions(savefig=True)
