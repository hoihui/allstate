# with early-stopping, test2 data is averaged. train2 n_estimators is average of those found in generating test2 data
from data import *
import xgboost
from bayes_opt import BayesianOptimization
import argparse,random,xgboost,xgbfir,heapq
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import cPickle as pickle
picklefile=path+'xgb1.dat'
trainfile=path+"train_exp.csv"
tmpfile = '/tmp/' + str(time.time()).replace(".","") + str(random.random())
musthave=[]
mustretain=['cont%d'%e for e in xrange(1,15)]+['cat%d' % i for i in xrange(1,117)]
ncolsrange=(100,1000)
patience = 15
bestn = 10   #making sure these best rows has all their features
xgbparams ={'silent': 1, 'objective': 'reg:linear',
            'learning_rate': 0.1, 'eta': 0.1,
            'booster': 'gbtree'}

def col_keep(colsin):
  global featimp
  cols = list( colsin )
  featimpmean = featimp.mean()
  for c1 in cols:
    if '_' in c1:
      f1,f2=c.split('_')
      c2=f2+'_'+f1
      if c1 in cols and c2 in cols and featimpmean[[c1,c2]].notnull().all():
        if featimpmean[c1]>featimpmean[c2]:
          c.remove(c2)
        else:
          c.remove(c1)
  return cols
  
Nrows=188318 #sum(1 for line in open(trainfile)) - 1
nrows=188318
parser = argparse.ArgumentParser()
parser.add_argument('-np',  type=int,  default=-1)
parser.add_argument('-acq', type=str,  default=random.choice(['poi','ei']))
parser.add_argument("--preload", help="load all columns first",action="store_true")
parser.add_argument("--trunc", help="force truncate featimp",action="store_true")
parser.add_argument("--verbose", help="force truncate featimp",action="store_true")
parser.add_argument('-init', type=int, default=1)
parser.add_argument('-iter', type=int, default=4)
args = parser.parse_args()
if args.np>0: xgbparams['nthread'] = args.np
def gen_featimpmean():
  if featimp.shape[0]<5:return featimp.mean()
  weights = .01+.99*(featimp.index-featimp.index.min())/(featimp.index.max()-featimp.index.min())
  weights = pd.Series(weights.values,index=featimp.index)
  featimpmean = featimp.mul(weights,axis=0).sum()/featimp.notnull().mul(weights,axis=0).sum()
  scoreimp = featimp.replace(0,np.nan).notnull().mul(weights,axis=0).sum()/featimp.replace(0,np.nan).notnull().sum()  #if chosen but fscore=0, ignore that
  featimpmean /= featimpmean.sum()
  scoreimp /= scoreimp.sum()
  result = featimpmean
  return result/result.sum()
def weighted_featimp(chosen):
  if featimp.shape[0]<5 or not chosen:return featimp.drop(chosen,1).mean()
  weights = featimp[chosen].count(1)
  weights = .01+.99*weights/weights.max()
  unchosen = featimp.drop(chosen,1)
  weightedmean = unchosen.mul(weights,axis=0).sum()/unchosen.notnull().mul(weights,axis=0).sum()
  weightedmean /= weightedmean.sum()
  return weightedmean

if not os.path.isfile(picklefile):
  n_rounds_without_improve = 0
  n_wait = 0  #for reference in future in determining cutoff to n_rounds
  featimp=pd.read_csv(trainfile,nrows=0,index_col=0)
  featimp=featimp.drop('loss',1)
  p = {}
  with open(picklefile,'wb') as outfile:
    pickle.dump((featimp,p,n_rounds_without_improve,n_wait),outfile, pickle.HIGHEST_PROTOCOL)
  time.sleep(5)
with open(picklefile,'rb') as infile:
  (featimp,p,n_rounds_without_improve,n_wait)=pickle.load(infile)

if args.preload:
  try:
    with open(traindat,'rb') as infile:
      train_exp = pickle.load(infile)
  except:
    train_exp = pd.read_csv(trainfile,index_col=0,usecols=['id','loss']+list(featimp.columns))
    try:
      with open(traindat,'wb') as outfile:
        pickle.dump(train_exp,outfile,pickle.HIGHEST_PROTOCOL)
    except:pass
    
if True: #parameters initialization
  discreteP=['max_depth','ncols']
  p_range={    'alpha': (0,10),
    'colsample_bytree': (0.01,1),
           'max_depth': (5.5,11.5),
           'subsample': (.8,1),
               'ncols': (ncolsrange[0],min(ncolsrange[1],featimp.shape[1])),
               'shift': (0,400)
          }
      
  yforw=lambda x,d={'shift':0}:np.log(np.clip(x+d['shift'],1,np.inf))
  yback=lambda x,d={'shift':0}:np.exp(x)-d['shift']
  
if True: #optimization score & dependencies
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
  def score(**params):
      global featimp
      for k in params.keys():
        if k in discreteP:
          params[k]=int(params[k])
      featimpmean=gen_featimpmean()
      chosen_feat=[]
      while len(chosen_feat)<min(params['ncols'],featimp.shape[1]):
        candfeat=weighted_featimp(chosen_feat).fillna(featimpmean.fillna(1.))
        candfeat=candfeat.fillna(1./featimp.shape[1])
        candfeat=candfeat.replace(0,candfeat[candfeat>0].min())
        theone = np.random.choice(candfeat.index,p=candfeat.values/np.sum(candfeat.values))
        chosen_feat.append( theone )
      chosen_feat=list(set(col_keep(chosen_feat)+musthave))
      p=xgbparams.copy()
      p.update(params)
      skip = sorted(random.sample(xrange(1,Nrows+1),Nrows-nrows))
      if args.preload: train = train_exp.ix[:,chosen_feat+['loss']]
      else: train = pd.read_csv(path+"train_exp.csv",index_col=0,usecols=['id','loss']+chosen_feat,skiprows=skip )
      print train.shape
      train_y = yforw(train['loss'],params)
      train_x = train.drop('loss',1)
      y_pred = 0*train_y
      fscores=dict((el,0) for el in chosen_feat) 
      for train_idx, val_idx in kftune.split(train_x):
        X_train, X_val = train_x.iloc[train_idx], train_x.iloc[val_idx]
        y_train, y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]
        d_train = xgboost.DMatrix(X_train, label=y_train)
        d_valid = xgboost.DMatrix(X_val, label=y_val)
        model = xgboost.train(p,
                              d_train,
                              num_boost_round=100000,
                              evals=[(d_valid, 'eval')],
                              early_stopping_rounds=patience,
                              feval=es_eval,
                              # obj=fair_obj,
                              verbose_eval=False)
        y_pred.iloc[val_idx]=model.predict(d_valid,ntree_limit=model.best_ntree_limit)
        xgbfir.saveXgbFI(model,OutputXlsxFile=tmpfile,TopK=p_range['ncols'][1])
        time.sleep(5)
        fi=pd.read_excel(tmpfile,index_col=0)
        fscore = fi['Expected Gain'].to_dict() #Gain, FScore, wFScore, Average wFScore, Average Gain, Expected Gain
        meanscore = np.average(fscore.values())
        for k in fscore.keys(): fscore[k]/=meanscore*len(fscore)
        featimpmean=featimpmean.fillna(1./featimp.shape[1])
        normalization = featimpmean[chosen_feat].sum()/featimpmean.sum()/np.sum(fscore.values())/kftune.get_n_splits()
        for k,v in fscore.iteritems():
          fscores[k]+=normalization*v
      curscore = -mean_absolute_error(yback(y_pred,params),yback(train_y,params))
      featimp = featimp.append(pd.Series(fscores,name=round(curscore,4)))
      return curscore

while True:
  init_points=args.init
  n_iter=args.iter
  bo = BayesianOptimization(score, p_range)
  if p: bo.initialize(p)
  else: init_points,n_iter=1,0
  if args.trunc: init_points,n_iter=0,0
  bo.maximize(init_points=init_points, n_iter=n_iter, acq=args.acq)
  featimp_cur=featimp
  p_new = {}
  for i in xrange(len(bo.Y)):
    if bo.Y[i] not in bo.y_init:
      p_new[bo.Y[i].round(4)]={bo.keys[j]:bo.X[i,j].round(4) for j in xrange(len(bo.keys))}

  if not os.path.isfile(picklefile): break
  with open(picklefile,'rb') as infile:
      try:
        (featimp,p_now,n_rounds_without_improve,n_wait)=pickle.load(infile)
        p.update(p_now)
      except: featimp=featimp_cur
  if featimp.shape[1]!=featimp_cur.shape[1]: break
  i1=featimp.index
  i2=featimp_cur.index
  oldi1=[i for i in xrange(len(i1)) if i1[i] not in i2]
  featimp=pd.concat((featimp.iloc[oldi1],featimp_cur)).sort_index()

  if p_new and len(p)>5:
    if max(p_new.keys())<=sorted(p.keys())[-5]: 
      n_nonboundary=sum([all([v2 not in p_range[k2] for k2,v2 in v.iteritems()]) for _,v in p_new.iteritems()])
      if args.acq!="ei":n_rounds_without_improve+=n_nonboundary-init_points
    else: 
      n_wait,n_rounds_without_improve = max(n_wait,n_rounds_without_improve),0
  p.update(p_new)

  if n_rounds_without_improve>50 or args.trunc or featimp.shape[0]>1000:
    cols = list(featimp.tail(bestn).dropna(1,'all').columns)
    cols = list(set(cols+mustretain))
    i=6
    while len(cols)<featimp.shape[1]*0.8 and i<featimp.shape[0]:
      r = list(featimp.iloc[-i].replace(0,np.nan).dropna().index)
      cols = list(set(r+cols))
      i+=1
      print i,len(cols)
    featimp = featimp.tail(500).ix[:,cols]
    p = {k:p[k] for k in heapq.nlargest(5, p)}
    n_rounds_without_improve = 0
    n_wait = 0

  with open(picklefile,'wb') as outfile:
    pickle.dump((featimp,p,n_rounds_without_improve,n_wait),outfile, pickle.HIGHEST_PROTOCOL)
  print (featimp.shape,len(p),n_rounds_without_improve,n_wait)
  
  args.trunc=False
  #break
import sys
os.execv(sys.executable,[sys.executable]+sys.argv)

