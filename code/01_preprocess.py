"""causal feature engineering and leakage-aware chronological baselines."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'raw'
FILES={'SP500':'SP500_1990_2026.csv','DOW':'DowJones_1990_2026.csv',
       'NASDAQ':'NASDAQ_1990_2026.csv','RUSSELL2000':'Russell2000_1990_2026.csv'}
from config import HORIZON, CRASH_THRESHOLD
THRESHOLD=CRASH_THRESHOLD

def load_prices():
    cols={}
    for name,fn in FILES.items():
        df=pd.read_csv(DATA/fn,skiprows=[1,2],parse_dates=['Price']).rename(columns={'Price':'Date'}).set_index('Date')
        cols[name]=pd.to_numeric(df['Close'],errors='coerce')
    out=pd.concat(cols,axis=1).sort_index().dropna()
    assert (out>0).all().all() and out.index.is_unique
    return out

def acf1(x):
    a=np.asarray(x,dtype=float); a=a-a.mean(); den=np.dot(a,a)
    return float(np.dot(a[1:],a[:-1])/den) if den>0 else 0.

def build():
    (ROOT/'data'/'processed').mkdir(parents=True,exist_ok=True)
    prices=load_prices()
    returns=np.log(prices/prices.shift(1))
    features=pd.DataFrame(index=prices.index)
    for name in prices.columns:
        r=returns[name]
        features[f'{name}_vol30']=r.rolling(30).std()
        features[f'{name}_acf30']=r.rolling(30).apply(acf1,raw=True)
        features[f'{name}_ewma_vol']=np.sqrt(r.pow(2).ewm(span=30,adjust=False,min_periods=30).mean())
        features[f'{name}_ret5']=r.rolling(5).sum()
    features['cross_absret']=returns.abs().mean(axis=1)
    features['cross_dispersion']=returns.std(axis=1)
    # Label at t requires all next N observations; last N are undefined.
    future=np.column_stack([prices['SP500'].shift(-h).to_numpy() for h in range(1,HORIZON+1)])
    # Forward drawdown relative to the current close.
    future_min=np.min(future,axis=1)
    drawdown=1-future_min/prices['SP500'].to_numpy()
    features['future_drawdown']=drawdown
    features['crash']=(drawdown>=THRESHOLD).astype(int)
    features=features.replace([np.inf,-np.inf],np.nan).dropna()
    features.index.name='Date'
    features.to_csv(ROOT/'data'/'processed'/'features_classical.csv')
    prices.to_csv(ROOT/'data'/'processed'/'prices_aligned.csv')
    returns.to_csv(ROOT/'data'/'processed'/'log_returns.csv')
    return features

def evaluate(df):
    (ROOT/'experiments'/'initial_classical_baselines').mkdir(parents=True,exist_ok=True)
    xcols=[c for c in df if c not in ('future_drawdown','crash')]
    n=len(df); train_end=int(n*.70); val_end=int(n*.85)
    # Training labels must mature before the following block starts.
    tr=np.arange(0,train_end-HORIZON)
    va=np.arange(train_end,val_end-HORIZON)
    te=np.arange(val_end,n)
    rows=[]; predictions=[]
    models={'LogisticRegression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42)),
            'GradientBoosting':HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=9,l2_regularization=1.0,random_state=42)}
    X=df[xcols].to_numpy(); y=df.crash.to_numpy()
    for name,model in models.items():
        model.fit(X[tr],y[tr])
        for split,ids in [('validation',va),('test',te)]:
            prob=model.predict_proba(X[ids])[:,1]
            pred=(prob>=0.5).astype(int)
            rows.append({'model':name,'split':split,'n':len(ids),'positives':int(y[ids].sum()),
                         'PR_AUC':average_precision_score(y[ids],prob),
                         'ROC_AUC':roc_auc_score(y[ids],prob) if len(np.unique(y[ids]))>1 else np.nan,
                         'precision_at_0.5':precision_score(y[ids],pred,zero_division=0),
                         'recall_at_0.5':recall_score(y[ids],pred,zero_division=0),
                         'F1_at_0.5':f1_score(y[ids],pred,zero_division=0)})
            if split=='test':
                predictions.append(pd.DataFrame({'Date':df.index[ids], 'model':name,'true_label':y[ids],'crash_probability':prob}))
    pd.DataFrame(rows).to_csv(ROOT/'experiments'/'initial_classical_baselines'/'metrics.csv',index=False)
    pd.concat(predictions).to_csv(ROOT/'experiments'/'initial_classical_baselines'/'test_predictions.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print('Rows:',n,'Range:',df.index.min(),df.index.max(),'positive rate:',round(y.mean(),4))

if __name__=='__main__':
    df=build()
    print('Preprocessing completed:',len(df),'labelled dates')


