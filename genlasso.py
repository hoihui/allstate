#exhaustive (no hyperparameter)
from data import *
import argparse,random
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
# parser = argparse.ArgumentParser()
# parser.add_argument('-f',   type=str,  default=random.choice(['c','1']))
# parser.add_argument('-t',   type=str,  default=random.choice(['raw','log','sqrt']))
# args = parser.parse_args()
# featureType = args.f
# targetType = args.t
for featureType in ['c','1']:
  for targetType in ['raw','log','sqrt']:
    if featureType in ['c','0']: X,y,X_test=load_cat_cont()
    elif featureType=='1': X,y,X_test=load_cat1_cont()
    elif featureType=='2': X,y,X_test=load_cat2_cont()
    print featureType,targetType
    evals=4
    maxrounds=1
    base= 'lasso_'+featureType+'__'+targetType

    if True: #parameters initialization
      discreteP=[]
      cur_p= {}
      p_dict={}
      if targetType!='raw':
          cur_p['y__shift']=np.random.uniform(100,300)
          p_dict['y__shift']=lambda x: hp.uniform('y__shift',max(x-100,1),min(x+100,1000))

    if True: #data,y,make_model preparation
      if targetType=='raw':
          yforw=lambda x,d={}:x
          yback=lambda x,d={}:x
      elif targetType=='log':
          yforw=lambda x,d={'y__shift':0}:np.log(x+d['y__shift'])
          yback=lambda x,d={'y__shift':0}:np.exp(x)-d['y__shift']
      elif targetType=='sqrt':
          yforw=lambda x,d={'y__shift':0}:np.sqrt(x+d['y__shift'])
          yback=lambda x,d={'y__shift':0}:np.power(x,2)-d['y__shift']

      def make_model(model=None,params={}):
        if not model:
          if featureType=='c':
              model=Pipeline([
                  ('enc',Cat2Cont(range(len(cat)),na_action='average')),
                  ('std', StandardScaler()),
                  ('lasso',LassoCV(max_iter=1000000,n_jobs=1,cv=n_splits))
              ])
          else:
              model=Pipeline([
                  ('lasso',LassoCV(max_iter=1000000))
              ])
        # model.set_params(**{k:v for k,v in params.iteritems() if 'y__' not in k})
        return model

    if True and targetType!='raw': #optimization
      def scoring(est,x,y):
          return mean_absolute_error(yback(y),yback(est.predict(x)))
      def score(params):
          for k in params.keys():
            if any([disc in k for disc in discreteP]):
              params[k]=int(params[k])
          sX,sy=shuffle(X,y)
          sy_=yforw(sy,params)
          model= make_model()
          c=cross_val_score(model,sX,sy_,cv=n_splits,n_jobs=1,scoring=scoring)
          s=c.mean()
          sys.stdout.write("\r{0} | {1}\t".format(s,params)); sys.stdout.flush()
          return s
      hcv=hyperoptCoordSearchCV(cur_p,p_dict,score,evals=evals,maxrounds=maxrounds)
      new_p=hcv.fit()
      for k in new_p.keys():
        if any([disc in k for disc in discreteP]):
          new_p[k]=int(new_p[k])
      cur_p.update(new_p)
      
    if True: #prepare and output next-level
      y_=yforw(y,cur_p)
      
      model = make_model()
      model.fit(X,y_)
      test2col=model.predict(X_test)
      test2col=yback(test2col,cur_p)

      train2col=cross_val_predict(model,X,y_,cv=kf,n_jobs=1,method='predict',verbose=2)
      train2col=yback(train2col,cur_p)

      cur_score=mean_absolute_error(y,train2col)

      save_2(train2col,test2col,base,cur_p,cur_score)
        
      print cur_score
  
  