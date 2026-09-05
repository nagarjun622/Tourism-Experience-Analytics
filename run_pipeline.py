"""Reproducible end-to-end build for Tourism Experience Analytics."""
from pathlib import Path
import json, shutil, sys, platform
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data'/'raw'/'Tourism Dataset'; PRO=ROOT/'data'/'processed'; REP=ROOT/'reports'; MOD=ROOT/'models'
for d in [PRO,REP/'data_understanding',REP/'data_quality',REP/'eda'/'figures',REP/'regression'/'figures',REP/'classification',MOD/'regression',MOD/'classification',MOD/'recommendation',ROOT/'notebooks']:
    d.mkdir(parents=True,exist_ok=True)
SEED=42

def read(name, sub=''):
    return pd.read_excel(RAW/sub/f'{name}.xlsx')
def dump(obj,path):
    with open(path,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,default=str)
def safe_name(s): return str(s).lower().replace(' ','_')

def audit(tables):
    rows=[]; details={}
    for n,df in tables.items():
        key=df.columns[0]
        info={'shape':list(df.shape),'columns':list(df.columns),'dtypes':{c:str(t) for c,t in df.dtypes.items()},'missing':df.isna().sum().to_dict(),'duplicates':int(df.duplicated().sum()),'key':key,'key_unique':bool(df[key].is_unique),'unique':df.nunique(dropna=False).to_dict()}
        details[n]=info
        rows.append({'dataset':n,'rows':len(df),'columns':len(df.columns),'duplicate_rows':info['duplicates'],'key':key,'key_unique':info['key_unique'],'missing_cells':int(df.isna().sum().sum())})
    pd.DataFrame(rows).to_csv(REP/'data_understanding'/'dataset_audit_summary.csv',index=False)
    dump(details,REP/'data_understanding'/'dataset_profiles.json')

def fk_report(t):
    checks=[('transaction','UserId','user','UserId'),('transaction','AttractionId','item','AttractionId'),('transaction','VisitMode','mode','VisitModeId'),('user','CityId','city','CityId'),('user','CountryId','country','CountryId'),('user','RegionId','region','RegionId'),('user','ContinentId','continent','ContinentId'),('city','CountryId','country','CountryId'),('country','RegionId','region','RegionId'),('region','ContinentId','continent','ContinentId'),('item','AttractionTypeId','type','AttractionTypeId'),('item','AttractionCityId','city','CityId')]
    out=[]
    for a,ak,b,bk in checks:
        missing=~t[a][ak].isin(t[b][bk])
        out.append({'relationship':f'{a}.{ak} -> {b}.{bk}','orphan_rows':int(missing.sum()),'valid':bool(not missing.any())})
    pd.DataFrame(out).to_csv(REP/'data_understanding'/'foreign_key_validation.csv',index=False)

def clean(tables):
    log=[]; cleaned={}
    for n,df in tables.items():
        before=len(df); x=df.copy(); x.columns=[c.strip() for c in x.columns]
        # Only remove exact duplicate rows: documented, lossless deduplication.
        x=x.drop_duplicates().copy()
        cleaned[n]=x
        log.append({'dataset':n,'before_rows':before,'after_rows':len(x),'rows_affected':before-len(x),'reason':'Removed exact duplicate records only' if before!=len(x) else 'No rows removed'})
        x.to_csv(PRO/f'{n}_clean.csv',index=False)
    pd.DataFrame(log).to_csv(REP/'data_quality'/'cleaning_log.csv',index=False)
    return cleaned

def integrate(t):
    tx=t['transaction'].copy()
    # Transactions store the visit-mode key; expose the human-readable label for analysis and classification.
    if pd.api.types.is_numeric_dtype(tx['VisitMode']):
        tx=tx.merge(t['mode'][['VisitModeId','VisitMode']].rename(columns={'VisitMode':'VisitModeLabel'}),left_on='VisitMode',right_on='VisitModeId',how='left')
        tx['VisitMode']=tx['VisitModeLabel']
        tx=tx.drop(columns=['VisitModeId','VisitModeLabel'])
    # User geography
    uc=t['city'].rename(columns={'CityId':'UserCityId','CityName':'UserCity','CountryId':'UserCityCountryId'})[['UserCityId','UserCity','UserCityCountryId']]
    countries=t['country'][['CountryId','Country','RegionId']].rename(columns={'Country':'UserCountry','RegionId':'UserRegionLookupId'})
    regions=t['region'][['RegionId','Region','ContinentId']].rename(columns={'RegionId':'UserRegionLookupId','Region':'UserRegion','ContinentId':'UserContinentLookupId'})
    continents=t['continent'].rename(columns={'ContinentId':'UserContinentLookupId','Continent':'UserContinent'})
    u=t['user'].merge(uc,left_on='CityId',right_on='UserCityId',how='left').merge(countries,on='CountryId',how='left').merge(regions,on='UserRegionLookupId',how='left').merge(continents,on='UserContinentLookupId',how='left')
    u=u[['UserId','UserCity','UserCountry','UserRegion','UserContinent']]
    # Attraction geography
    ac=t['city'].rename(columns={'CityId':'AttractionCityId','CityName':'AttractionCity','CountryId':'AttractionCountryId'})[['AttractionCityId','AttractionCity','AttractionCountryId']]
    i=t['item'].merge(t['type'],on='AttractionTypeId',how='left').merge(ac,on='AttractionCityId',how='left').merge(t['country'][['CountryId','Country','RegionId']],left_on='AttractionCountryId',right_on='CountryId',how='left').merge(t['region'][['RegionId','Region','ContinentId']],on='RegionId',how='left').merge(t['continent'],on='ContinentId',how='left')
    i=i.rename(columns={'Country':'AttractionCountry','Region':'AttractionRegion','Continent':'AttractionContinent'})[['AttractionId','Attraction','AttractionType','AttractionCity','AttractionCountry','AttractionRegion','AttractionContinent','AttractionAddress']]
    master=tx.merge(u,on='UserId',how='left',validate='many_to_one').merge(i,on='AttractionId',how='left',validate='many_to_one')
    master.to_csv(PRO/'master_dataset.csv',index=False)
    pd.DataFrame([{'column':c,'description':('Visit rating target (1 to 5)' if c=='Rating' else 'Visit-mode classification target' if c=='VisitMode' else 'Integrated tourism field')} for c in master.columns]).to_csv(PRO/'data_dictionary.csv',index=False)
    dump({'transaction_rows_before':len(tx),'master_rows':len(master),'row_multiplication':len(master)-len(tx),'duplicate_rows':int(master.duplicated().sum()),'missing_by_column':master.isna().sum().to_dict()},REP/'data_quality'/'master_validation.json')
    return master

def eda(m):
    sns.set_theme(style='whitegrid')
    charts=[('visits_by_year',m['VisitYear'].value_counts().sort_index(),'Visits by year'),('visit_mode_distribution',m['VisitMode'].value_counts(),'Visit mode distribution'),('rating_distribution',m['Rating'].value_counts().sort_index(),'Rating distribution'),('top_attractions',m['Attraction'].value_counts().head(15).sort_values(),'Most visited attractions'),('attraction_type_popularity',m['AttractionType'].value_counts().head(15).sort_values(),'Attraction type popularity')]
    for fn,s,title in charts:
        plt.figure(figsize=(10,5)); s.plot(kind='barh' if fn in ['top_attractions','attraction_type_popularity'] else 'bar',color='#147d8a'); plt.title(title); plt.ylabel('Visits' if 'rating' not in fn else 'Count'); plt.tight_layout(); plt.savefig(REP/'eda'/'figures'/f'{fn}.png',dpi=150); plt.close()
    summary=m.groupby('Attraction',dropna=False).agg(visits=('TransactionId','size'),average_rating=('Rating','mean')).sort_values('visits',ascending=False)
    summary.head(30).to_csv(REP/'eda'/'attraction_summary.csv')
    insights=[{'observation':'Visit demand is concentrated','evidence':'Top attractions account for a meaningful share of recorded interactions','business_implication':'Use popularity as a safe cold-start recommendation signal.'},{'observation':'Interactions are sparse across users and attractions','evidence':'User-attraction matrix sparsity is reported in recommendation artifacts','business_implication':'Use a hybrid approach with content and popularity fallback.'},{'observation':'Visit modes are modeled as a multiclass target','evidence':'Class counts and macro metrics are retained in classification reports','business_implication':'Monitor minority-class recall alongside overall accuracy.'}]
    dump(insights,REP/'eda'/'business_insights.json')

FEATURES=['VisitYear','VisitMonth','UserContinent','UserRegion','UserCountry','UserCity','AttractionType','AttractionCity','AttractionCountry','AttractionRegion','AttractionContinent']
def prep_pipeline(model):
    cats=[c for c in FEATURES if c not in ['VisitYear','VisitMonth']]
    return Pipeline([('preprocessing',ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'),cats),('numeric','passthrough',['VisitYear','VisitMonth'])])),('model',model)])
def regression(m):
    x=m[FEATURES].fillna('Unknown'); y=m['Rating']
    Xtr,Xte,ytr,yte=train_test_split(x,y,test_size=.2,random_state=SEED)
    candidates={'baseline_mean':DummyRegressor(strategy='mean'),'random_forest':RandomForestRegressor(n_estimators=60,min_samples_leaf=2,random_state=SEED,n_jobs=-1)}; results=[]; trained={}
    for n,model in candidates.items():
        p=prep_pipeline(model); p.fit(Xtr,ytr); pred=p.predict(Xte); results.append({'model':n,'MAE':mean_absolute_error(yte,pred),'MSE':mean_squared_error(yte,pred),'RMSE':mean_squared_error(yte,pred)**.5,'R2':r2_score(yte,pred)}); trained[n]=p
    res=pd.DataFrame(results).sort_values('RMSE'); res.to_csv(REP/'regression'/'model_comparison.csv',index=False); best=res.iloc[0]['model']; joblib.dump(trained[best],MOD/'regression'/'rating_model.pkl'); dump({'selected_model':best,'features':FEATURES,'target':'Rating','random_state':SEED,'test_size':.2,'metrics':res.iloc[0].to_dict(),'application_prediction_behavior':'Clip output to the valid 1 to 5 rating range.'},MOD/'regression'/'model_metadata.json'); dump(res.iloc[0].to_dict(),REP/'regression'/'metrics.json')
def classification(m):
    x=m[FEATURES].fillna('Unknown'); y=m['VisitMode'].astype(str)
    Xtr,Xte,ytr,yte=train_test_split(x,y,test_size=.2,random_state=SEED,stratify=y)
    candidates={'baseline_most_frequent':DummyClassifier(strategy='most_frequent'),'random_forest_balanced':RandomForestClassifier(n_estimators=60,min_samples_leaf=2,class_weight='balanced',random_state=SEED,n_jobs=-1)}; results=[]; trained={}
    for n,model in candidates.items():
        p=prep_pipeline(model); p.fit(Xtr,ytr); pr=p.predict(Xte); prec,rec,f1,_=precision_recall_fscore_support(yte,pr,average='macro',zero_division=0); results.append({'model':n,'accuracy':accuracy_score(yte,pr),'macro_precision':prec,'macro_recall':rec,'macro_f1':f1}); trained[n]=p
    res=pd.DataFrame(results).sort_values('macro_f1',ascending=False); res.to_csv(REP/'classification'/'model_comparison.csv',index=False); best=res.iloc[0]['model']; p=trained[best]; pr=p.predict(Xte); joblib.dump(p,MOD/'classification'/'visit_mode_model.pkl'); pd.DataFrame(classification_report(yte,pr,output_dict=True,zero_division=0)).T.to_csv(REP/'classification'/'classification_report.csv'); cm=confusion_matrix(yte,pr,labels=sorted(y.unique())); plt.figure(figsize=(7,6)); sns.heatmap(cm,annot=True,fmt='d',xticklabels=sorted(y.unique()),yticklabels=sorted(y.unique()),cmap='Blues'); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout(); plt.savefig(REP/'classification'/'confusion_matrix.png',dpi=150); plt.close(); dump({'selected_model':best,'features':FEATURES,'target':'VisitMode','random_state':SEED,'test_size':.2,'metrics':res.iloc[0].to_dict()},MOD/'classification'/'model_metadata.json'); dump(res.iloc[0].to_dict(),REP/'classification'/'metrics.json')
def recommendations(m):
    a=m.groupby(['AttractionId','Attraction','AttractionType','AttractionCity','AttractionCountry'],dropna=False).agg(interactions=('TransactionId','size'),mean_rating=('Rating','mean')).reset_index(); global_mean=m.Rating.mean(); min_int=max(3,int(a.interactions.quantile(.5))); a['recommendation_score']=(a['interactions']/(a['interactions']+min_int)*a['mean_rating']+min_int/(a['interactions']+min_int)*global_mean); a['reason']='Popularity and smoothed average rating'; a.sort_values('recommendation_score',ascending=False).to_csv(MOD/'recommendation'/'popular_recommendations.csv',index=False)
    content=m[['AttractionId','Attraction','AttractionType','AttractionCity','AttractionCountry']].drop_duplicates(); content.to_csv(MOD/'recommendation'/'attraction_catalog.csv',index=False)
    n_users=m.UserId.nunique(); n_items=m.AttractionId.nunique(); n_int=len(m); sparsity=1-n_int/(n_users*n_items); dump({'approach':'Popularity baseline with content attributes for profile matching. Collaborative filtering was not selected as the default because interaction sparsity can make user-neighbor scores unstable.','users':int(n_users),'attractions':int(n_items),'interactions':int(n_int),'sparsity':float(sparsity),'cold_start':'Return ranked popularity list and optionally filter by attraction type or location.','offline_evaluation':'Not reported because timestamps/order are insufficient for a reliable held-out next-item protocol.'},MOD/'recommendation'/'metadata.json')
def notebooks():
    try: import nbformat
    except ImportError: return
    names=['01_Data_Understanding','02_Data_Cleaning_Preprocessing','03_Data_Integration','04_EDA_Visualization','05_Regression_Rating_Prediction','06_Classification_Visit_Mode','07_Recommendation_System']
    for idx,n in enumerate(names,1):
        nb=nbformat.v4.new_notebook(); nb.cells=[nbformat.v4.new_markdown_cell(f'# {n.replace("_"," ")}\n\n**Input:** saved project data.  \n**Processing:** reproducible stage implemented in `run_pipeline.py`.  \n**Output:** shared project artifacts.  \n**Next dependency:** next numbered notebook or Streamlit app.'),nbformat.v4.new_code_cell("# Run from project root to reproduce all stages\n!python run_pipeline.py")]; nbformat.write(nb,ROOT/'notebooks'/f'{n}.ipynb')
def main():
    tables={n:read(src,sub) for n,src,sub in [('transaction','Transaction',''),('user','User',''),('city','City',''),('country','Country',''),('region','Region',''),('continent','Continent',''),('mode','Mode',''),('type','Type',''),('item','Item',''),('updated_item','Updated_Item','Additional_Data_for_Attraction_Sites')]}
    audit(tables); fk_report(tables)
    # Updated item compatibility is reported, never merged automatically.
    primary=tables['item']; updated=tables['updated_item']; dump({'item_shape':list(primary.shape),'updated_item_shape':list(updated.shape),'same_columns':list(primary.columns)==list(updated.columns),'shared_attraction_ids':int(primary.AttractionId.isin(updated.AttractionId).sum()),'decision':'Updated_Item is retained as a separate supplemental source and is not merged automatically.'},REP/'data_understanding'/'updated_item_assessment.json')
    clean_tables=clean({k:v for k,v in tables.items() if k!='updated_item'}); master=integrate(clean_tables); eda(master); regression(master); classification(master); recommendations(master); notebooks()
    dump({'python':sys.version,'platform':platform.platform(),'pandas':pd.__version__,'numpy':np.__version__,'random_state':SEED},REP/'reproducibility.json')
if __name__=='__main__': main()
