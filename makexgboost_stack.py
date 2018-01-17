# with early-stopping, testthis data is averaged. trainthis n_estimators is average of those found in generating testthis data
from data import *
thislevel=load_save_2
nextlevel=load_save_3
picklefile='/work/cascades/hyhui/xgb2.dat'
base = 'xgb2'
from bayes_opt import BayesianOptimization
import argparse,random,xgboost,sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import cPickle as pickle
tmpfile = '/tmp/' + str(time.time()).replace(".","") + str(random.random())
parser = argparse.ArgumentParser()
parser.add_argument('-np',  type=int,  default=-1)
parser.add_argument('-n',   type=int,  default=10)
parser.add_argument('-nbags',   type=int,  default=5)
parser.add_argument("--debug", action="store_true")
parser.add_argument('-t',   type=str,  default=random.choice(['log']))
parser.add_argument("--verbose", help="increase output verbosity",action="store_true")
parser.add_argument("--second", action="store_true")
patience=20  #for early stopping
args = parser.parse_args()
nthread = args.np
targetType = args.t
nbags=args.nbags
maxepoch=1000 if not args.debug else 5
xgbparams ={'silent': 1, 'objective': 'reg:linear',
            'learning_rate': 0.01,
            'booster': 'gbtree'}

train1 = pd.read_csv(path+"train.csv",index_col=0,usecols=['id','loss'])
trainthis,testthis = thislevel()

with open(picklefile,'rb') as infile:
  (featimp,p,n_rounds_without_improve,n_wait)=pickle.load(infile)
discreteP=['max_depth','ncols']

if True: #data,y,make_model preparation
  if targetType=='raw':
      yforw=lambda x,d={}:x
      yback=lambda x,d={}:x
  elif 'log' in targetType:
      yforw=lambda x,d={'shift':0}:np.log(np.clip(x+d['shift'],1,np.inf))
      yback=lambda x,d={'shift':0}:np.exp(x)-d['shift']
  elif targetType=='sqrt':
      yforw=lambda x,d={'shift':0}:np.sqrt(np.clip(x+d['shift'],0,np.inf))
      yback=lambda x,d={'shift':0}:np.power(x,2)-d['shift']
  
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
trainnext,testnext = nextlevel()
for i in random.sample(range(1,args.n+1),args.n):
    r=featimp.iloc[-i]
    params = p[round(r.name,4)]
    chosen_feat = list(r.dropna().index)
    colname = base+'_'+str(round(abs(r.name),3))
    if colname in testnext.columns: continue
    for k in params.keys():
      if k in discreteP:
        params[k]=int(params[k])
    params['colsample_bytree']=max(2./len(chosen_feat),params['colsample_bytree'])
    for j in random.sample(range(nbags),nbags):
      colname_sub=colname+'_'+str(j)
      if colname_sub in testnext.columns or colname in trainnext.columns:
        if (not args.second) or any(testnext[colname_sub]):
          continue
      print i,j,r.name,params
      
      train_y = yforw(train1['loss'],params)
      train_x = yforw(trainthis,params).ix[:,chosen_feat]
      test_x = yforw(testthis,params).ix[:,chosen_feat]
      testcolnext =0
      traincolnext =0*train_y
      # if colname_sub not in testnext.columns:
        # nextlevel(traincolnext,testcolnext,colname_sub)
      nrounds=[]
      nroundslist=[]
      p=xgbparams.copy()
      p.update(params)
      p.update({'seed':j})
      for train_idx, val_idx in kftrain.split(train_x):
        X_train, X_val = train_x.iloc[train_idx], train_x.iloc[val_idx]
        y_train, y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]
        d_train = xgboost.DMatrix(X_train, label=y_train)
        d_valid = xgboost.DMatrix(X_val, label=y_val)
        if not nrounds:
          model = xgboost.train(p,
                                d_train,
                                num_boost_round=maxepoch,
                                evals=[(d_valid, 'eval')],
                                feval=es_eval,
                                early_stopping_rounds=patience,
                                obj=fair_obj,
                                verbose_eval=args.verbose)
          nroundslist.append(model.best_ntree_limit)
        else:
          model = xgboost.train(p,
                                d_train,
                                num_boost_round=nrounds,
                                obj=fair_obj,
                                verbose_eval=args.verbose)
        traincolnext.iloc[val_idx]=yback(model.predict(d_valid),params)
      nrounds=int(np.average(nroundslist))
      d_train = xgboost.DMatrix(train_x, label=train_y)
      d_test = xgboost.DMatrix(test_x)
      p=xgbparams.copy()
      p.update(params)
      model = xgboost.train(p,
                            d_train,
                            num_boost_round=int(nrounds*kftrain.get_n_splits()/(kftrain.get_n_splits()-1.)),
                            obj=fair_obj,
                            verbose_eval=args.verbose)
      testcolnext=yback(model.predict(d_test),params)
      print j,mean_absolute_error(traincolnext,train1['loss'])
      trainnext,testnext=nextlevel(np.clip(traincolnext,0,np.inf),np.clip(testcolnext,0,np.inf),colname_sub)
    bagcols = [colname+'_'+str(j) for j in xrange(nbags)]
    if all([e in trainnext.columns for e in bagcols]):
      if trainnext.ix[:,bagcols].mean().min()>0:
        traincolnext=trainnext.ix[:,bagcols].mean(1)
        testcolnext=testnext.ix[:,bagcols].mean(1)
        nextlevel(traincolnext,testcolnext,colname,delcols=bagcols)
        print mean_absolute_error(traincolnext,train1['loss'])
        os.execv(sys.executable,[sys.executable]+sys.argv)

if not args.second: os.execv(sys.executable,[sys.executable]+sys.argv+['--second'])
