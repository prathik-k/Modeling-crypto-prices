from classes.cryptocurrency import Crypto
from classes.GBM_base import GBM_base
from classes.GBM_attention import GBM_attention
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="""Enter the cryptocurrency,
     historic range, type of prediction and prediction details.""")

    '''
    Sample command for a single prediction and base model:
    python main.py --crypto btc --model base --prediction single --n_paths 100 --h_range [0,1000] --n_pred_pts 100 --save_res Y
    Sample command for a rolling prediction and base model:
    python main.py --crypto btc --model base --prediction rolling --n_paths 100 --h_range [0,1000] --n_periods 10 --period 30 --save_res Y
    Sample command for a single prediction and attention model:
    python main.py --crypto btc --model attention --prediction single --n_paths 100 --h_range [0,1000] --n_pred_pts 30 --save_res Y

    '''
    
    parser.add_argument('--crypto',action='store',type=str,required=True,help='Cryptocurrency')
    parser.add_argument('--model',action='store',type=str,required=True,help='Type of Model (Base/Attention)')
    parser.add_argument('--prediction',action='store',default='single',type=str,required=True,help='Type of prediction (single/rolling)')
    parser.add_argument('--n_paths',action='store',default=100,type=int,required=True,help='Number of random paths generated')
    parser.add_argument('--h_range',action='store',default=[0,300], type = list,required=False,
    help='History range for training set, specified as list like [0,1000] means taking 1000 days')
    parser.add_argument('--period',action='store',default=30,type=int,
    required=False,help='Time period for each prediction for the rolling case')
    parser.add_argument('--n_periods',action='store',default=10,type=int,
    required=False,help='Number of periods for the rolling case')
    parser.add_argument('--n_pred_pts',action='store',default=100,type=int,required=False,
    help='Number of prediction points for the single case.')
    parser.add_argument('--save_res',action='store',default='Y',type=str,required=False,
    help='Save plot of prediction results? (Y/N)')
    args=parser.parse_args()

    crypto,model,pred_type,n_paths,save_res=args.crypto,args.model,args.prediction,args.n_paths,args.save_res

    index = Crypto(crypto.upper())
    if pred_type=='single' and model=="base":
        hist_range,n_pred_pts = args.h_range,args.n_pred_pts        
        gbm = GBM_base(crypto=index,hist_range=hist_range,pred_type=pred_type)
        gbm.make_predictions_base(n_pred_paths=n_paths,n_pred=n_pred_pts)
        gbm.plot_predictions(savefig=True) if save_res.upper()=='Y' else gbm.plot_predictions(savefig=False)
    
    elif pred_type=='rolling' and model=="base":
        hist_range,period,n_periods = args.h_range,args.period,args.n_periods
        
        gbm = GBM_base(crypto=index,hist_range=hist_range,period=period,n_pred_periods=n_periods,pred_type=pred_type)
        gbm.make_predictions_base(n_pred_paths=n_paths)
        gbm.plot_predictions(savefig=True) if save_res.upper()=='Y' else gbm.plot_predictions(savefig=False)

    elif pred_type=='single' and model=="attention":
        hist_range,n_pred_pts = args.h_range,args.n_pred_pts
        gbm = GBM_attention(crypto=index,hist_range=hist_range,pred_type=pred_type)
        gbm.make_predictions_base(n_pred_paths=n_paths,n_pred=n_pred_pts)
        gbm.plot_predictions(savefig=True) if save_res.upper()=='Y' else gbm.plot_predictions(savefig=False)

'''
    btc = Crypto('ETH')
    gbm = GBM_base(crypto=btc,hist_range=[0,1000],
    pred_type='rolling',period=30,n_pred_periods=10)
    gbm.make_predictions_base(n_pred_paths=100,n_pred=300)
    gbm.plot_predictions(savefig=True)
'''

#type=lambda s: [str(item) for item in s.split(',')]
