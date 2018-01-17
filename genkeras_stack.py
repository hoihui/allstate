from data import *
thislevel=load_save_4
base= 'nn4'
import os,time,argparse,random,sys
parser = argparse.ArgumentParser()
parser.add_argument('-gpu',   type=int,  default=0)
parser.add_argument('-acq', type=str,  default=random.choice(['poi']))
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument('-init', type=int, default=1)
parser.add_argument('-iter', type=int, default=2)
args = parser.parse_args()
maxepoch=1000 if not args.debug else 1
batchSize=512 if not args.debug else 1024
patience=10  #10 for early stopping
shift=200

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
  
from keras.models import Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
X0,y,X0_test,_ = load_cat1_cont()
trainthis,testthis = thislevel()
ntrain = trainthis.shape[0]
train_testthis = pd.concat((trainthis,testthis)).reset_index(drop=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Xdense2 = scaler.fit_transform(np.log(np.clip(train_testthis.values+shift,1,np.inf)))

scores0=trainthis.apply(lambda x:mean_absolute_error(x,y))
scores=(scores0-scores0.min())/scores0.min()
corr=trainthis.corr()
idx=[]
idx.append(np.argsort(scores.values)[0])
while len(idx)<trainthis.shape[1]:
    cur_scores=[]
    for i in xrange(trainthis.shape[1]):
        if i not in idx:
            maxcorr=corr.values[idx,i].max()
            scorecorr=1-maxcorr
            cur_scores.append( (scorecorr-scores[i],i) )  #max corr and min scores wrt best one
    idx.append(max(cur_scores)[1])
Xdense2 = Xdense2[:,idx]
print trainthis.columns[idx]

p_path = opath+base+".csv"
print base, args.gpu

if True: #parameters initialization
  discreteP=['hiddenSize1','hiddenSize2','ncols']
  p_init= {'hiddenSize1':400,'dropOut1':.4,
           'hiddenSize2':200,'dropOut2':.2,                  
          }
  p_range={'hiddenSize1': (10,500),'dropOut1': (.01,.99),
           'hiddenSize2': (2,250),  'dropOut2': (.01,.99),            
          }
  p_init['ncols'] =Xdense2.shape[1]/2
  p_range['ncols']=(2,Xdense2.shape[1])

if True: #data,y,make_model,batch_generators preparation
  yforw=lambda x:np.log(x+shift)
  yback=lambda x:np.exp(x)-shift
  def make_model(p):
      dropOuts = filter(None,[p.get('dropOut1'),p.get('dropOut2'),p.get('dropOut3')])
      hiddenSizes = filter(None,[p.get('hiddenSize1'),p.get('hiddenSize2'),p.get('hiddenSize3')])
      model = Sequential()
      input_dim = p['ncols']
      for hiddenSize,dropOut in zip(hiddenSizes,dropOuts):
        model.add(Dense(hiddenSize, input_dim = input_dim,init='he_normal'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(float(dropOut)))
        input_dim = hiddenSize
      model.add(Dense(1, init = 'he_normal'))
      model.compile(loss = 'mae', optimizer = 'adadelta')
      return model

if True: #optimization 
  weightsfile='/tmp/'+str(time.time()).replace(".","")+'weights.h5'  #use time for weightsfile to prevent collision
  def score(**params):
      for k in params.keys():
        if k in discreteP:
          params[k]=int(params[k])
      if (params.get('dropOut3') and not (params['hiddenSize1']>params['hiddenSize2'] and params['hiddenSize2']>params['hiddenSize3'])) or\
         (not params['hiddenSize1']>params['hiddenSize2']):
          return -3000+10*np.random.normal()
      # X = scipy.sparse.hstack((X0,Xdense2[:ntrain,:params['ncols']]),format='csr')
      X = Xdense2[:ntrain,:params['ncols']]
      # X_test = scipy.sparse.hstack((X0_test,Xdense2[ntrain:,:params['ncols']]),format='csr')
      p_all=dict(p_init)
      p_all.update(params)
      y_=yforw(y)
      s=[]
      modelfile='/tmp/m'+str(time.time()).replace(".","")+'model.h5'
      model = make_model(p_all)
      model.save(modelfile)
      for train_idx, vales_idx in kftune.split(X): #########################
          val_idx,es_idx=np.array_split(vales_idx,2)
          X_train,y_train_=X[train_idx],y_[train_idx]
          X_val  ,y_val_  =X[val_idx]  ,y_[val_idx]
          X_es   ,y_es_   =X[es_idx]   ,y_[es_idx]
          if scipy.sparse.issparse(X_es): X_es =X_es.toarray()
          try: model = load_model(modelfile)
          except: model = make_model(p_all)
          es=EarlyStopping(monitor='val_loss', patience=patience, verbose=args.verbose)
          mc=ModelCheckpoint(weightsfile,monitor='val_loss',save_best_only=True,save_weights_only=True,verbose=0)
          X_train_ = X_train
          X_es_ = X_es
          X_val_ = X_val
          hist=model.fit(X_train_,y_train_,128,nb_epoch=maxepoch,
                                   callbacks=[es,mc],validation_data=(X_es_,y_es_), verbose=args.verbose)
          time.sleep(5)
          try:
            model.load_weights(weightsfile)
            os.remove(weightsfile)
          except:pass
          predy_val_=model.predict(X_val_).ravel()
          s.append(mean_absolute_error(yback(y_val_),yback(predy_val_)))
          if args.verbose: 
            print 'epochs:', hist.history['val_loss'].index(min(hist.history['val_loss']))
            print 'test loss:', s[-1]
      s = -np.average(s)
      return s

while True:
  init_points=args.init
  n_iter=args.iter
  p_prev,p_range_ = load_save_config(p_path,default=p_init)
  bo = BayesianOptimization(score, p_range)
  if p_range_: p_range=p_range_
  if p_prev: bo.initialize(p_prev)
  else: init_points,n_iter=1,0
  bo.maximize(init_points=init_points, n_iter=n_iter, acq=args.acq)
  print "best: ", bo.res['max']
  load_save_config(p_path,default=p_init,bo=bo,p_prev_before_run=p_prev)
  break
  
os.execv(sys.executable,[sys.executable]+sys.argv)
# os.execv(sys.executable,[sys.executable]+[sys.argv[0].replace('gen','make')])