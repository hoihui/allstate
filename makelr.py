from data import *
thislevel=load_save_3
nextlevel=load_save_4
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import RidgeCV
train,test = thislevel()
X=train.values
X_test=test.values
y=pd.read_csv(path+"train.csv",index_col=0,usecols=['id','loss']).values

alphas = (.03,.1,.3,1,3,10)
shifts=[200]
k_features=(1,15)
# alphas = (.1,1,10)
# shifts=np.linspace(100,400,7)
# k_features=(1,10)

def scorer(model,X,y):
	return -mean_absolute_error(np.exp(model.predict(X)),np.exp(y))
lr=RidgeCV(alphas=alphas,fit_intercept=False,scoring=scorer)

bestscore=-np.inf
bestsfs=None
bestshift=None
for shift in shifts:
    sfs=SFS(lr,k_features=k_features,forward=True,floating=False,scoring=scorer,cv=kftune)
    sfs.fit(np.log(X+shift),np.log(y+shift))
    print shift,sfs.k_score_,len(sfs.k_feature_idx_)
    if sfs.k_score_>bestscore:
        bestscore=sfs.k_score_
        bestsfs=sfs
        bestshift=shift

yforw=lambda x:np.log(np.clip(x+bestshift,1,np.inf))
yback=lambda x:np.exp(x)-bestshift
X=yforw(X)
X_test=yforw(X_test)
y_=yforw(y)
lr.fit(bestsfs.transform(X),y_)
testcolnext=yback(lr.predict(bestsfs.transform(X_test)))
traincolnext=yback(cross_val_predict(lr,bestsfs.transform(X),y_,cv=kftrain))
print traincolnext.shape,testcolnext.shape,mean_absolute_error(traincolnext,y)

trainnext,testnext=nextlevel(np.clip(traincolnext,0,np.inf),np.clip(testcolnext,0,np.inf),'lr')

