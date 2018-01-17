# with early-stopping, test2 data is averaged. train2 n_estimators is average of those found in generating test2 data
from data import *
import xgboost,sys
from bayes_opt import BayesianOptimization
import argparse,random
from sklearn.pipeline import Pipeline
parser = argparse.ArgumentParser()
parser.add_argument('-np',  type=int,  default=-1)
parser.add_argument('-n',   type=int,  default=10)
parser.add_argument('-nbags',   type=int,  default=3)
parser.add_argument("--verbose", help="increase output verbosity",action="store_true")
parser.add_argument("--second", action="store_true")
args = parser.parse_args()
nthread = args.np
nbags=args.nbags
patience=20  #for early stopping
featureType,targetType=('i','log')
xgbparams ={'silent': 1, 'objective': 'reg:linear',
            'learning_rate': 0.03,
            'booster': 'gbtree'}

if featureType in ['c','0']: X,y,X_test,_=load_cat_cont()
elif featureType=='1': X,y,X_test,_=load_cat1_cont()
elif featureType=='2': X,y,X_test,_=load_cat2_cont()
elif featureType=='i': X,y,X_test,_=load_catint_cont()

base= 'xgb_'+featureType+'__'+targetType
p_path = opath+base+".csv"
print base, p_path, nthread

p_init={'colsample_bytree': 0.7,'max_depth': 8,'alpha': 1,'subsample': 0.7,}
discreteP=['max_depth']
if targetType=='raw':
    yforw=lambda x,d={}:x
    yback=lambda x,d={}:x
elif 'log' in targetType:
    yforw=lambda x,d={'y__shift':0}:np.log(x+d['y__shift'])
    yback=lambda x,d={'y__shift':0}:np.exp(x)-d['y__shift']
elif targetType=='sqrt':
    yforw=lambda x,d={'y__shift':0}:np.sqrt(x+d['y__shift'])
    yback=lambda x,d={'y__shift':0}:np.power(x,2)-d['y__shift']
    
# def fair_obj(labels, preds):
def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = preds-labels
    con = np.average(np.abs(x))
    grad = con*x / (np.abs(x)+con)
    hess = con**2 / (np.abs(x)+con)**2
    return grad, hess
def es_eval(preds, dtrain):  #for xgboost evaluation on val set for early stopping
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(yback(labels),yback(preds))

p_prev,_ = load_save_config(p_path,default=p_init,maxrows=args.n,remove_extremes=False)
train2,test2 = load_save_2()
for score in random.sample(p_prev.keys(),len(p_prev)):
    params=p_prev[score]
    colname = 'xgb_'+str(abs(round(score,2)))
    if colname in train2.columns: continue
    for k in params.keys():
      if k in discreteP:
        params[k]=int(params[k])
    for j in random.sample(range(nbags),nbags):
      colname_sub=colname+'_'+str(j)
      if colname_sub in train2.columns:
        if (not args.second) or any(train2[colname_sub]):
          continue
      print colname_sub,params
      y_=yforw(y,params)
      test2col =0
      train2col =0*y
      if colname_sub not in train2.columns:
        load_save_2(train2col,test2col,colname_sub)
      nrounds=0
      nroundslist=[]
      p=xgbparams.copy()
      p.update(params)
      p.update({'seed':j})
      for train_idx, val_idx in kftrain.split(X):
        X_train,y_train_=X[train_idx],y_[train_idx]
        X_val  ,y_val_  =X[val_idx]  ,y_[val_idx]
        d_train = xgboost.DMatrix(X_train, label=y_train_)
        d_valid = xgboost.DMatrix(X_val, label=y_val_)
        if not nrounds:
          model = xgboost.train(p,d_train,
                                num_boost_round=100000,
                                evals=[(d_valid, 'eval')],
                                early_stopping_rounds=patience,
                                feval=es_eval,
                                obj=fair_obj,
                                verbose_eval=args.verbose)
          nroundslist.append(model.best_ntree_limit)
        else:
          model = xgboost.train(p,d_train,
                                num_boost_round=nrounds,
                                obj=fair_obj,
                                verbose_eval=args.verbose)
        train2col[val_idx]+=yback(model.predict(d_valid),params)
      nrounds=int(np.average(nroundslist))
      d_train = xgboost.DMatrix(X, label=y_)
      d_test = xgboost.DMatrix(X_test)
      p=xgbparams.copy()
      p.update(params)
      model = xgboost.train(p,d_train,
                            num_boost_round=int(nrounds*kftrain.get_n_splits()/(kftrain.get_n_splits()-1.)),
                            obj=fair_obj,
                            verbose_eval=args.verbose)
      test2col=yback(model.predict(d_test),params)
      print j, mean_absolute_error(train2col,y)
      train2,test2=load_save_2(np.clip(train2col,0,np.inf),np.clip(test2col,0,np.inf),colname_sub)
    bagcols = [colname+'_'+str(j) for j in xrange(nbags)]
    if train2[bagcols].mean().min()>0:
      train2col=train2[bagcols].mean(1)
      test2col=test2[bagcols].mean(1)
      load_save_2(train2col,test2col,colname,delcols=bagcols)
      print mean_absolute_error(train2col,y)
      os.execv(sys.executable,[sys.executable]+sys.argv)

os.execv(sys.executable,[sys.executable]+sys.argv+['--second'])