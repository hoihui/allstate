#cannot pipeline custom preprocessor with keras model because we need keras' native fit_generator to handle sparse matrices
import os,time,argparse,random
import cPickle as pickle
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=int,  default=0)
parser.add_argument('-acq', type=str,  default=random.choice(['poi']))
parser.add_argument("--verbose", help="increase output verbosity",action="store_true")
parser.add_argument('-l',   type=int, default=random.choice([2,3]))
parser.add_argument('-t',   type=str, default=random.choice(['raw']))
parser.add_argument('-init',type=int, default=1)
parser.add_argument('-iter',type=int, default=4)
args = parser.parse_args()
if args.gpu<0: os.environ["THEANO_FLAGS"] = "base_compiledir=/tmp/hyhui/.theano,device=cpu,floatX=float32"
else: os.environ["THEANO_FLAGS"] = "base_compiledir=/tmp/hyhui/.theano,device=gpu%d,floatX=float32,force_device=True" % (args.gpu)
if args.gpu>0: time.sleep(30*args.gpu)
try:import theano
except Exception,err:
  print err
  import sys
  argv = sys.argv
  if '-gpu' not in argv:
    argv.extend(['-gpu','1'])
  else:
    argv[argv.index('-gpu')+1]=str(int(argv[argv.index('-gpu')+1])+1) if 'Bad device number' not in str(err) else '0'
  print 'restarting with ', argv
  os.execv(sys.executable,[sys.executable]+argv)
  exit()
  
from data import *
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

nlayers = args.l
targetType = args.t

(featimp,_,_,_)=pickle.load(open(path+"feat1.dat",'rb'))
cols=list(featimp.tail(1).dropna(1,how='all').columns)
train = pd.read_csv(path+"train_exp.csv",index_col=0,usecols=set(['id','loss']+cols))
y = train['loss'].values
train = train.drop(['loss'],1)
cat1=[]
for f in train.columns:
  if 'cat' in f:
    tmp = scipy.sparse.csr_matrix( pd.get_dummies(train[f].replace(-1,np.nan)))
    cat1.append(tmp)
cat1=scipy.sparse.hstack(cat1,format='csr')
Xdense=[]
from scipy.stats import skew, boxcox
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
for f in train.columns:
  if 'cont' in f:
    Xdense.append(train[[f]])
Xdense = scaler.fit_transform(np.hstack(Xdense))
X=scipy.sparse.hstack((cat1,Xdense),format='csr')
print X.shape,y.shape

maxepoch=1000  #1000
patience=10  #10 for early stopping

base= 'nn_'+str(nlayers)
p_path = opath+base+".csv"
print base, args.gpu

if True: #parameters initialization
  discreteP=['hiddenSize1','hiddenSize2','hiddenSize3','hiddenSize4']
  if nlayers==2: 
      p_init= {'dropOut1':.4,'hiddenSize1':400,
               'dropOut2':.2,'hiddenSize2':200,
              }
      p_range={'dropOut1': (.01,.99),'hiddenSize1': (200,1000),
               'dropOut2': (.01,.99),'hiddenSize2': (50,400),
              }
  elif nlayers==3:
      p_init= {'dropOut1':.4,'hiddenSize1':400,
               'dropOut2':.2,'hiddenSize2':200,
               'dropOut3':.2,'hiddenSize3':50,
              }
      p_range={'dropOut1': (.01,.99),'hiddenSize1': (300,500),
               'dropOut2': (.01,.99),'hiddenSize2': (100,300),
               'dropOut3': (.01,.99),'hiddenSize3': (10,100),
              }
  elif nlayers==4:
      p_init= {'dropOut1':.4,'hiddenSize1':400,
               'dropOut2':.3,'hiddenSize2':200,
               'dropOut3':.2,'hiddenSize3':100,
               'dropOut4':.1,'hiddenSize4':20,
              }
      p_range={'dropOut1': (.01,.99),'hiddenSize1': (350,550),
               'dropOut2': (.01,.99),'hiddenSize2': (150,350),
               'dropOut3': (.01,.99),'hiddenSize3': (100,250),
               'dropOut4': (.01,.99),'hiddenSize4': (5,100),
              }
  if targetType!='raw':
      p_init['y__shift']=np.random.uniform(100,300)
      p_range['y__shift']=(0,600)

if True: #data,y,make_model,batch_generators preparation
  if targetType=='raw':
      yforw=lambda x,d={}:x
      yback=lambda x,d={}:x
  elif targetType=='log':
      yforw=lambda x,d={'y__shift':0}:np.log(x+d['y__shift'])
      yback=lambda x,d={'y__shift':0}:np.exp(x)-d['y__shift']
  elif targetType=='sqrt':
      yforw=lambda x,d={'y__shift':0}:np.sqrt(x+d['y__shift'])
      yback=lambda x,d={'y__shift':0}:np.power(x,2)-d['y__shift']

  from keras import backend as K
  def evalerror(y_true, y_pred):
      if targetType=='log': return K.mean(K.abs(K.exp(y_pred) - K.exp(y_true)), axis=-1)
      elif targetType=='sqrt': return K.mean(K.abs(K.square(y_pred) - K.square(y_true)), axis=-1)
      return K.mean(K.abs(y_pred - y_true), axis=-1)
  def batch_generator(X, y, batch_size):
      number_of_batches = np.ceil(X.shape[0]/float(batch_size))
      counter = 0
      sample_index = np.arange(X.shape[0])
      np.random.shuffle(sample_index)
      while True:
          batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
          X_batch = X[batch_index,:]
          if scipy.sparse.issparse(X_batch): X_batch =X_batch.toarray()
          y_batch = y[batch_index]
          counter += 1
          yield X_batch, y_batch
          if (counter == number_of_batches):
              np.random.shuffle(sample_index)
              counter = 0
  def batch_generatorp(X, batch_size):
      number_of_batches = np.ceil(X.shape[0]/float(batch_size))
      counter = 0
      sample_index = np.arange(X.shape[0])
      while True:
          batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
          X_batch = X[batch_index,:]
          if scipy.sparse.issparse(X_batch): X_batch =X_batch.toarray()
          counter += 1
          yield X_batch
          if (counter == number_of_batches):
              counter = 0
  def make_model(p):
      dropOuts = filter(None,[p.get('dropOut1'),p.get('dropOut2'),p.get('dropOut3'),p.get('dropOut4')])
      hiddenSizes = filter(None,[p.get('hiddenSize1'),p.get('hiddenSize2'),p.get('hiddenSize3'),p.get('hiddenSize4')])
      model = Sequential()
      input_dim = X.shape[1]
      for hiddenSize,dropOut in zip(hiddenSizes,dropOuts):
        # model.add(Dense(hiddenSize, input_dim = input_dim,init='zero' if targetType=='log' else 'he_normal')) #if use evalerror, could explode
        model.add(Dense(hiddenSize, input_dim = input_dim,init='he_normal'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(float(dropOut)))
        input_dim = hiddenSize
      model.add(Dense(1, init = 'he_normal'))
      # model.compile(loss = evalerror, optimizer = 'adadelta')  #doesn't make sense to transform in to original spac
      model.compile(loss = 'mae', optimizer = 'adadelta')
      return model

if True: #optimization 
  weightsfile='/tmp/w'+str(time.time()).replace(".","")+'.h5'  #use time for weightsfile to prevent collision
  cat2cont=Pipeline([
              ('enc', Cat2Cont(range(len(cat)),na_action='average')),
              ('std', StandardScaler()),
          ])
  def score(**params):
      for k in params.keys():
        if k in discreteP:
          params[k]=int(params[k])
      hiddenSizes = filter(None,[params.get('hiddenSize1'),params.get('hiddenSize2'),params.get('hiddenSize3'),params.get('hiddenSize4')])
      if any(hiddenSizes[i] <= hiddenSizes[i+1] for i in xrange(len(hiddenSizes)-1)):
          return -3000+10*np.random.normal()
      p_all=dict(p_init)
      p_all.update(params)
      y_=yforw(y,params)  #underscore on y means "forwarded" with current shift
      s=[]
      modelfile='/tmp/m'+str(time.time()).replace(".","")+'.h5'
      model = make_model(p_all)
      model.save(modelfile)
      for train_idx, vales_idx in kftune.split(X): #########################
          val_idx,es_idx=np.array_split(vales_idx,2)
          X_train,y_train_=X[train_idx],y_[train_idx]
          X_val  ,y_val_  =X[val_idx]  ,y_[val_idx]
          X_es   ,y_es_   =X[es_idx]   ,y_[es_idx]
          try: model = load_model(modelfile)
          except: model = make_model(p_all)
          if scipy.sparse.issparse(X_es): X_es =X_es.toarray()
          es=EarlyStopping(monitor='val_loss', patience=patience, verbose=args.verbose)
          mc=ModelCheckpoint(weightsfile,monitor='val_loss',save_best_only=True,save_weights_only=True,verbose=0)
          # if featureType=='c':  #underscore on X means transformed according to X_train (Cat2Cont)
          #     X_train_ = cat2cont.fit_transform(X_train,y_train_)
          #     X_es_ = cat2cont.transform(X_es)
          #     X_val_ = cat2cont.transform(X_val)
          # elif featureType=='0':
          #     std = StandardScaler()
          #     X_train_ = std.fit_transform(X_train)
          #     X_es_ = std.transform(X_es)
          #     X_val_ = std.transform(X_val)
          # else:
          X_train_ = X_train
          X_es_ = X_es
          X_val_ = X_val
          hist=model.fit_generator(batch_generator(X_train_,y_train_,512),nb_epoch=maxepoch,
                                   samples_per_epoch = X_train_.shape[0],
                                   callbacks=[es,mc],validation_data=(X_es_,y_es_), verbose=args.verbose)
          time.sleep(5)
          try:
            model.load_weights(weightsfile)
            os.remove(weightsfile)
          except:pass
          predy_val_=model.predict_generator(batch_generatorp(X_val_,512),val_samples=X_val_.shape[0]).ravel()
          s.append(mean_absolute_error(yback(y_val_,params),yback(predy_val_,params)))
          if args.verbose: 
            print 'epochs:', hist.history['val_loss'].index(min(hist.history['val_loss']))
            print 'test loss:', s[-1]
      s = -np.average(s)
      try: os.remove(modelfile)
      except:pass
      return s

if True:
  p,p_range_ = load_save_config(p_path,default=p_init)
  init_points=args.init
  n_iter=args.iter
  bo = BayesianOptimization(score, p_range)
  if p: bo.initialize(p)
  else: init_points,n_iter=1,0
  bo.maximize(init_points=init_points, n_iter=n_iter, acq=args.acq)
  print "best: ", bo.res['max']
  load_save_config(p_path,default=p_init,bo=bo,p_prev_before_run=p)


import sys
os.execv(sys.executable,[sys.executable]+sys.argv)
# os.execv(sys.executable,[sys.executable]+[sys.argv[0].replace('gen','make')])
