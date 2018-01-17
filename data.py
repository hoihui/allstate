import os
from bayes_opt import BayesianOptimization
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,cross_val_score,cross_val_predict,KFold
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import sys,random
import scipy.sparse, time
import numbers

path=opath = "/path/to/data/and/scripts/"
        
numeric=['cont%d'%e for e in xrange(1,15)]
cat=['cat%d' % i for i in xrange(1,117)] # categorical feature(s)
# cat_small_nunique=['cat%d' % i for i in xrange(1,117) if train['cat%d' % i].nunique()<15]
#cat1 to cat72 have only two labels A and B. In most of the cases, B has very few entries
#cat73 to cat 108 have more than two labels
#cat109 to cat116 have many labels
target='loss'
kftune = KFold(n_splits=5,shuffle=True)
kftrain = KFold(n_splits=8,shuffle=False)

class Cat2Cont(BaseEstimator, TransformerMixin):
    def __init__(self,catLabelPos,na_action=None,exclude_current=False):
        self.catLabelPos=catLabelPos
        self.sumcount={}
        self.na_action=na_action
        self.exclude_current=exclude_current
    def fit(self, X, y=None):
        self.y=y
        df=pd.DataFrame(X[:,self.catLabelPos],copy=True)
        df['y']=self.y
        for c in range(len(self.catLabelPos)):
            self.sumcount[c]=df.groupby(c)['y'].agg(['sum','count'])
        return self
    def transform(self,X,y=None):
        df=pd.DataFrame(X[:,self.catLabelPos],copy=True)
        if self.exclude_current:
            for c in df.columns:
                df[c]=(df[c].map(self.sumcount[c]['sum'])-self.y)/(df[c].map(self.sumcount[c]['count'])-1)
        else:
            for c in df.columns:
                df[c]=df[c].map(self.sumcount[c]['sum'])/df[c].map(self.sumcount[c]['count'])
        if self.na_action=='average':df.fillna(np.average(self.y),inplace=True)
        elif isinstance(self.na_action, numbers.Number): df.fillna(self.na_action,inplace=True)
        Xo=np.array(X)
        Xo[:,self.catLabelPos]=df.values
        return Xo

def load_raw2():
    train2path=path+'train2.csv'
    test2path=path+'test2.csv'
    if os.path.isfile(train2path) and os.path.isfile(test2path):
      train2 = pd.read_csv(train2path,index_col=0)
      test2 = pd.read_csv(test2path,index_col=0)
    else:
      train2 = pd.read_csv(path+'train.csv',usecols=['id'])
      test2 = pd.read_csv(path+'test.csv',usecols=['id'])
    return train2,test2

def load_raw():
    train = pd.read_csv(path+'train.csv')
    test = pd.read_csv(path+'test.csv')
    ntrain=train.shape[0]
    train_test= pd.concat((train,test)).reset_index(drop=True)
    train_test.reindex_axis(cat+numeric+[target],axis=1)
    labelInd = {}
    for c in cat:
        if train[c].nunique()!=test[c].nunique():
          set_train = set(train[c].unique())
          set_test = set(test[c].unique())
          remove = (set_train | set_test) - (set_train & set_test)
          train_test[c] = train_test[c].apply(lambda x: np.nan if x in remove else x,convert_dtype=True)
    for c in cat: 
        train_test[c],labelInd[c] = pd.factorize(train_test[c],sort=True) #nan are -1, labelInd does not contain nan
    return ntrain,train_test,labelInd
    
def load_raw_feat2():
    train = pd.read_csv(path+'train.csv')
    test = pd.read_csv(path+'test.csv')
    ntrain=train.shape[0]
    train_test= pd.concat((train,test)).reset_index(drop=True)
    cat = [x for x in train_test.columns if 'cat' in x]
    train_test.reindex_axis(cat+numeric+[target],axis=1)
    COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25'.split(',')
    labelInd = {}
    import itertools
    for comb in itertools.combinations(COMB_FEATURE, 2):
        feat = comb[0] + "_" + comb[1]
        train[feat] = train[comb[0]] + train[comb[1]]
        test[feat] = test[comb[0]] + test[comb[1]]
        train_test[feat] = train_test[comb[0]] + train_test[comb[1]]
    cat = [x for x in train_test.columns if 'cat' in x]
    for c in cat:
        if train[c].nunique()!=test[c].nunique():
          set_train = set(train[c].unique())
          set_test = set(test[c].unique())
          remove = (set_train | set_test) - (set_train & set_test)
          train_test[c] = train_test[c].apply(lambda x: np.nan if x in remove else x,convert_dtype=True)
    for c in cat: 
        train_test[c],labelInd[c] = pd.factorize(train_test[c],sort=True)
    return ntrain,train_test,labelInd

def load_catint_cont():
    picklefile=path+'catintcont.dat'
    if os.path.isfile(picklefile):
      import cPickle as pickle
      with open(picklefile,'rb') as infile:
        (X,y,X_test,featurelen)=pickle.load(infile)
      return X,y,X_test,featurelen
    else:
      ntrain,train_test,_=load_raw_feat2()
      train=train_test[:ntrain]
      cat = [x for x in train_test.columns if 'cat' in x]
      X=train[cat+numeric].values
      y=train[target].values
      X_test=train_test[ntrain:][cat+numeric].values
      featurelen=[1]*X_test.shape[1]
      import cPickle as pickle
      with open(picklefile,'wb') as outfile:
        pickle.dump((X,y,X_test,featurelen),outfile, pickle.HIGHEST_PROTOCOL)
      return X,y,X_test,featurelen
      
def load_cat_cont():
    picklefile=path+'catcont.dat'
    if os.path.isfile(picklefile):
      import cPickle as pickle
      with open(picklefile,'rb') as infile:
        (X,y,X_test,featurelen)=pickle.load(infile)
      return X,y,X_test,featurelen
    else:
      ntrain,train_test,_=load_raw()
      train=train_test[:ntrain]
      X=train[cat+numeric].values
      y=train[target].values
      X_test=train_test[ntrain:][cat+numeric].values
      featurelen=[1]*X_test.shape[1]
      import cPickle as pickle
      with open(picklefile,'wb') as outfile:
        pickle.dump((X,y,X_test,featurelen),outfile, pickle.HIGHEST_PROTOCOL)
      return X,y,X_test,featurelen
      
def load_cat1_cont():
    picklefile=path+'cat1cont.dat'
    import os.path
    if os.path.isfile(picklefile):
      import cPickle as pickle
      with open(picklefile,'rb') as infile:
        (X,y,X_test,featurelen)=pickle.load(infile)
      return X,y,X_test,featurelen
    else:
      cat1=[]  #one-hot encoding
      featurelen=[]
      ntrain,train_test,_=load_raw()
      for f in cat:
          tmp = scipy.sparse.csr_matrix( pd.get_dummies(train_test[f].replace(-1,np.nan))) #.ix[:,1:] first col = all other col being empty?; nan are ignored by get_dummies
          cat1.append(tmp)
          featurelen.append(tmp.shape[1])
      cat1=scipy.sparse.hstack(cat1,format='csr')
      Xdense=[]
      from scipy.stats import skew, boxcox
      from sklearn.preprocessing import StandardScaler
      scaler = StandardScaler()
      for f in numeric:
          # if abs(skew(train_test[f]))>0.25:
          #     transformed,_=boxcox(train_test[f]+1)
          #     Xdense.append(transformed.reshape(-1,1))
          # else:
              Xdense.append(train_test[[f]])
              featurelen.append(1)
      Xdense = scaler.fit_transform(np.hstack(Xdense))
      X=scipy.sparse.hstack((cat1[:ntrain],Xdense[:ntrain]),format='csr')
      X_test=scipy.sparse.hstack((cat1[ntrain:],Xdense[ntrain:]),format='csr')
      y=train_test.ix[:ntrain-1,target].values
      import cPickle as pickle
      with open(picklefile,'wb') as outfile:
        pickle.dump((X,y,X_test,featurelen),outfile, pickle.HIGHEST_PROTOCOL)
      return X,y,X_test,featurelen

def load_cat1_cont2():
    X,y,X_test,_ = load_cat1_cont()
    train2 = pd.read_csv(path+"train2.csv")
    test2 = pd.read_csv(path+"test2.csv")
    ntrain = train2.shape[0]
    train_test2 = pd.concat((train2,test2)).reset_index(drop=True)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xdense2 = scaler.fit_transform(np.log(train_test2.values+250))
    X = scipy.sparse.hstack((X,Xdense2[:ntrain]),format='csr')
    X_test = scipy.sparse.hstack((X_test,Xdense2[ntrain:]),format='csr')
    return X,y,X_test
     
def load_cat2_cont():
    picklefile=path+'cat2cont.dat'
    import os.path
    if os.path.isfile(picklefile):
      import cPickle as pickle
      with open(picklefile,'rb') as infile:
        (X,y,X_test,featurelen)=pickle.load(infile)
      return X,y,X_test,featurelen
    else:
      cat2=[]  #binary encoding
      featurelen=[]
      ntrain,train_test,labelInd=load_raw()
      for f in cat:
        if len(labelInd[f])<=2:
            cat2.append(scipy.sparse.csr_matrix(train_test[[f]]))
            featurelen.append(1)
        else:
            s = 1 if -1 in train_test[f].unique() else 0;
            ndig = int(np.ceil(np.log2(len(labelInd[f])+s))) #0,1,2,3 -> ndig=3
            tmp=train_test[f].apply(lambda x:format(x+s,"0%db"%ndig)) #nan->000, 0->001, 3->110
            for i in xrange(ndig):
                cat2.append(scipy.sparse.csr_matrix(tmp.apply(lambda x:int(x[i]))).T)
            featurelen.append(ndig)
        sys.stdout.write("\r"+f+"/116\t"); sys.stdout.flush()
      cat2=scipy.sparse.hstack(cat2,format='csr')
      Xdense=[]
      from scipy.stats import skew, boxcox
      from sklearn.preprocessing import StandardScaler
      scaler = StandardScaler()
      for f in numeric:
          # if abs(skew(train_test[f]))>0.25:
          #     transformed,_=boxcox(train_test[f]+1)
          #     Xdense.append(transformed.reshape(-1,1))
          # else:
              Xdense.append(train_test[[f]])
              featurelen.append(1)
      Xdense = scaler.fit_transform(np.hstack(Xdense))
      X=scipy.sparse.hstack((cat2[:ntrain],Xdense[:ntrain]),format='csr')
      X_test=scipy.sparse.hstack((cat2[ntrain:],Xdense[ntrain:]),format='csr')
      y=train_test.ix[:ntrain-1,target].values
      import cPickle as pickle
      with open(picklefile,'wb') as outfile:
        pickle.dump((X,y,X_test,featurelen),outfile, pickle.HIGHEST_PROTOCOL)
      return X,y,X_test,featurelen
    
def load_save_2(train2col=None,test2col=None,colname=None,delcols=[]):
    train2path=path+'train2.csv'
    test2path=path+'test2.csv'
    if os.path.isfile(train2path):
        train2=pd.read_csv(train2path,index_col=0)
        while train2.shape[0]!=188318:
          time.sleep(5)
          train2=pd.read_csv(train2path,index_col=0)
        test2=pd.read_csv(test2path,index_col=0)
        while test2.shape[0]!=125546:
          time.sleep(5)
          test2=pd.read_csv(test2path,index_col=0)
    else:
        train2 = pd.read_csv(path+"train.csv",usecols=[0],index_col=0)
        test2 = pd.read_csv(path+'test.csv',usecols=[0],index_col=0)
    
    if train2col is not None and test2col is not None and colname is not None:
      train2[colname]=train2col
      test2[colname]=test2col
      for c in delcols:
        train2=train2.drop(c,1)
        test2=test2.drop(c,1)
      train2.round(2).to_csv(train2path,index=True)
      test2.round(2).to_csv(test2path,index=True)
    
    return train2,test2

def load_save_3(traincol=None,testcol=None,colname=None,delcols=[]):
    trainpath=path+'train3.csv'
    testpath=path+'test3.csv'
    if os.path.isfile(trainpath):
      train = pd.DataFrame()
      test = pd.DataFrame()
      i=0
      while train.shape[0]!=188318 or test.shape[0]!=125546:
        time.sleep(random.random()*15*i)
        try:
          train=pd.read_csv(trainpath,index_col=0)
          test=pd.read_csv(testpath,index_col=0)
        except: pass
        i+=1
    else:
        train = pd.read_csv(path+"train.csv",usecols=[0],index_col=0)
        test = pd.read_csv(path+'test.csv',usecols=[0],index_col=0)
    
    if traincol is not None and testcol is not None and colname is not None:
      train[colname]=traincol
      test[colname]=testcol
      for c in delcols:
        train=train.drop(c,1)
        test=test.drop(c,1)
      lentrain=0
      lentest=0
      i=0
      while lentrain!=188318 or lentest!=125546:
        time.sleep(random.random()*30*i)
        train.round(2).to_csv(trainpath,index=True)
        test.round(2).to_csv(testpath,index=True)
        with open(trainpath) as f:
          for lentrain,_ in enumerate(f): pass
        with open(testpath) as f:
          for lentest,_ in enumerate(f): pass
        i+=1
      
    return train,test
    
def load_save_4(traincol=None,testcol=None,colname=None,delcols=[]):
    trainpath=path+'train4.csv'
    testpath=path+'test4.csv'
    if os.path.isfile(trainpath):
      train = pd.DataFrame()
      test = pd.DataFrame()
      i=0
      while train.shape[0]!=188318 or test.shape[0]!=125546:
        time.sleep(random.random()*15*i)
        try:
          train=pd.read_csv(trainpath,index_col=0)
          test=pd.read_csv(testpath,index_col=0)
        except: pass
        i+=1
    else:
        train = pd.read_csv(path+"train.csv",usecols=[0],index_col=0)
        test = pd.read_csv(path+'test.csv',usecols=[0],index_col=0)
    
    if traincol is not None and testcol is not None and colname is not None:
      train[colname]=traincol
      test[colname]=testcol
      for c in delcols:
        train=train.drop(c,1)
        test=test.drop(c,1)
      lentrain=0
      lentest=0
      i=0
      while lentrain!=188318 or lentest!=125546:
        time.sleep(random.random()*30*i)
        train.round(2).to_csv(trainpath,index=True)
        test.round(2).to_csv(testpath,index=True)
        with open(trainpath) as f:
          for lentrain,_ in enumerate(f): pass
        with open(testpath) as f:
          for lentest,_ in enumerate(f): pass
        i+=1
      
    return train,test


    
    
def load_save_config(p_path,default,bo=None,p_prev_before_run={},maxrows=None,remove_extremes=False):  #BayesianOptimization object
    defmaxrows = min(max(600, int(10**(len(default)/2.)+2**len(default))),2000)
    maxrows = maxrows or defmaxrows
    if os.path.isfile(p_path):
      p_prev = pd.read_csv(p_path,index_col=0).round(4)
      for k,v in default.iteritems():
        if k not in p_prev.columns:
          print 'added: ',k
          p_prev[k]=v
      p_prev = p_prev.fillna(default)
      trimmed= p_prev.tail(maxrows)
      if trimmed.shape[0]==maxrows and remove_extremes:
        for k in p_prev.columns:
          trimmed=trimmed.loc[trimmed[k]!=trimmed[k].max()]
          trimmed=trimmed.loc[trimmed[k]!=trimmed[k].min()]
        p_range= {k:(trimmed.min().to_dict()[k],trimmed.max().to_dict()[k]) for k in p_prev.columns}
        for i in xrange(p_prev.shape[0]-maxrows):
          if any([p_prev.iloc[i][k] in p_range[k] for k in p_prev.columns]):
            trimmed=trimmed.append(p_prev.iloc[i])
      elif trimmed.shape[0]==defmaxrows:
        p_range= {k:(trimmed.min().to_dict()[k],trimmed.max().to_dict()[k]) for k in p_prev.columns}
        for i in xrange(p_prev.shape[0]-maxrows):
          if any([p_prev.iloc[i][k] in p_range[k] for k in p_prev.columns]):
            trimmed=trimmed.append(p_prev.iloc[i])
      else:
        p_range=None
      p_prev = trimmed.drop_duplicates(subset=p_prev.columns,keep="last")
      p_prev = p_prev.T.to_dict()
    else:
      p_prev = {}
      p_range=None
    if bo is not None and (p_prev or not p_prev_before_run):
      # if any([e in os.uname()[1] for e in ['compute-','tbird','phys.vt']]):
        # os.system("ssh -oStrictHostKeyChecking=no hokieone.arc.vt.edu touch "+p_path)
        # time.sleep(30)
      idx=[i for i in xrange(len(bo.Y)) if bo.Y[i] not in bo.y_init]
      p_cur_df = pd.DataFrame(bo.X[idx],columns=bo.keys).set_index([bo.Y[idx]])
      p_cur = p_cur_df.round(4).T.to_dict()
      p_cur.update(p_prev)
      pd.DataFrame(p_cur).T.to_csv(p_path)
      print 'saved: ',p_path
    return p_prev,p_range


