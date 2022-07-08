    #!/usr/bin/env python
# coding: utf-8



import os
import sys
import pandas as pd
import random
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression as LR
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
path = sys.argv[1]
ETC_tocsv = os.path.join(sys.argv[2],"mergebreed_FeatureImportance_ETC.csv")
chi2_tocsv = os.path.join(sys.argv[2],"mergebreed_FeatureImportance_chi2.csv")
info_tocsv = os.path.join(sys.argv[2],"mergebreed_FeatureImportance_info_classif.csv")
XGB_tocsv = os.path.join(sys.argv[2],"mergebreed_FeatureImportance_XGBoost.csv")
f_cla_tocsv = os.path.join(sys.argv[2],"mergebreed_FeatureImportance_f_classif.csv")
RF_tocsv = os.path.join(sys.argv[2],"mergebreed_FeatureImportance_RF.csv")
LR_tocsv = os.path.join(sys.argv[2],"mergebreed_FeatureImportance_LR.csv")
seednum = eval(sys.argv[3])
print(ETC_tocsv);print(chi2_tocsv);print(info_tocsv);print(XGB_tocsv);print(f_cla_tocsv);
print(RF_tocsv);print(LR_tocsv);
os.chdir(path)
all300 = pd.read_csv("./SourceDataFile/MergeBreed03/merge_breed_02_fill_04.raw",sep=" ",header=0)
all300.index = range(1,len(all300)+1)
X = all300.iloc[:,6:]
y = all300.iloc[:,0]


all300_pd = pd.read_csv("./SourceDataFile/MergeBreed03/merge_breed_02_fill_04.raw",sep=" ",header=0)
all300_pd.index = range(1,len(all300_pd)+1)
all300_pd.loc[all300_pd["FID"] == "BMX", "FID"] = 0
all300_pd.loc[all300_pd["FID"] == "CNEH", "FID"] = 1
all300_pd.loc[all300_pd["FID"] == "DHB", "FID"] = 2
all300_pd.loc[all300_pd["FID"] == "DNXE", "FID"] = 3
all300_pd.loc[all300_pd["FID"] == "DUR", "FID"] = 4
all300_pd.loc[all300_pd["FID"] == "ESCM", "FID"] = 5
all300_pd.loc[all300_pd["FID"] == "GLGS", "FID"] = 6
all300_pd.loc[all300_pd["FID"] == "ITWB", "FID"] = 7
all300_pd.loc[all300_pd["FID"] == "LDR", "FID"] = 8
all300_pd.loc[all300_pd["FID"] == "LT", "FID"] = 9
all300_pd.loc[all300_pd["FID"] == "LWT", "FID"] = 10
all300_pd.loc[all300_pd["FID"] == "MGZ", "FID"] = 11
all300_pd.loc[all300_pd["FID"] == "PIT", "FID"] = 12
# all300_pd.loc[all300_pd["FID"] == "RUWB", "FID"] = 13
all300_pd.loc[all300_pd["FID"] == "UKBS", "FID"] = 13
all300_pd.loc[all300_pd["FID"] == "WZS", "FID"] = 14
X_pd = all300_pd.iloc[:,6:]
y_pd = all300_pd.iloc[:,0]

model = ExtraTreesClassifier(random_state=seednum)
model.fit(X,y)
df_importance = pd.DataFrame(model.feature_importances_)
df_columns = pd.DataFrame(X.columns)
df_feature_scores = pd.concat([df_columns,df_importance],axis=1)
df_feature_scores.columns = ['feature_name','Score']
feature_scores = df_feature_scores.sort_values(by="Score", ascending=False)
feature_scores.index = range(1,feature_scores.shape[0]+1)
feature_scores["seed"] = seednum
feature_scores.to_csv(
    ETC_tocsv,
    sep=',',
    header=True,
    index=False
)




random.seed(seednum)
model = SelectKBest(chi2, k="all")
random.seed(seednum)
model_select = model.fit(X, y)
df_importance = pd.DataFrame(model_select.scores_)
df_columns = pd.DataFrame(X.columns)
df_feature_scores = pd.concat([df_columns,df_importance],axis=1)
df_feature_scores.columns = ['feature_name','Score']
feature_scores = df_feature_scores.sort_values(by="Score", ascending=False)
feature_scores.index = range(1,feature_scores.shape[0]+1)
feature_scores["seed"] = seednum
feature_scores.to_csv(
    chi2_tocsv,
    sep=',',
    header=True,
    index=False
)

selector = SelectKBest(score_func=partial(mutual_info_classif, random_state=0), k='all')
selector.fit(X, y)
df_scores = pd.DataFrame(selector.scores_)
df_columns = pd.DataFrame(X.columns)
df_feature_scores = pd.concat([df_columns,df_scores],axis=1)
df_feature_scores.columns = ['feature_name','Score']
feature_scores = df_feature_scores.sort_values(by="Score", ascending=False)
feature_scores.index = range(1,feature_scores.shape[0]+1)
feature_scores["seed"] = seednum
feature_scores.to_csv(
    info_tocsv,
    sep=',',
    header=True,
    index=False
)

random.seed(seednum)
model = XGBClassifier(use_label_encoder=False)
random.seed(seednum)
model.fit(X_pd, y_pd)
df_importance = pd.DataFrame(model.feature_importances_)
df_columns = pd.DataFrame(X.columns)
df_feature_scores = pd.concat([df_columns,df_importance],axis=1)
df_feature_scores.columns = ['feature_name','Score']
feature_scores = df_feature_scores.sort_values(by="Score", ascending=False)
feature_scores.index = range(1,feature_scores.shape[0]+1)
feature_scores["seed"] = seednum
feature_scores.to_csv(
    XGB_tocsv,
    sep=',',
    header=True,
    index=False
)

random.seed(seednum)
selector = SelectKBest(f_classif, k="all").fit(X, y)
random.seed(seednum)
df_scores = pd.DataFrame(selector.scores_)
df_columns = pd.DataFrame(X.columns)
df_feature_scores = pd.concat([df_columns,df_scores],axis=1)
df_feature_scores.columns = ['feature_name','Score']
feature_scores = df_feature_scores.sort_values(by="Score", ascending=False)
feature_scores.index = range(1,feature_scores.shape[0]+1)
feature_scores["seed"] = seednum
feature_scores.to_csv(
    f_cla_tocsv,
    sep=',',
    header=True,
    index=False
)




RF = RandomForestRegressor(n_estimators=100, max_features=X_pd.shape[1],max_depth=4,random_state=seednum)
fit = RF.fit(X_pd,y_pd)
df_scores = pd.DataFrame(fit.feature_importances_)
df_columns = pd.DataFrame(X.columns)
df_feature_scores = pd.concat([df_columns,df_scores],axis=1)
df_feature_scores.columns = ['feature_name','Score']
feature_scores = df_feature_scores.sort_values(by="Score", ascending=False)
feature_scores.index = range(1,feature_scores.shape[0]+1)
feature_scores["seed"] = seednum
feature_scores.to_csv(
    RF_tocsv,
    sep=',',
    header=True,
    index=False
)

random.seed(seednum)
reg = LR()
random.seed(seednum)
reg.fit(X_pd,y_pd)
df_importance = pd.DataFrame(reg.coef_)
df_columns = pd.DataFrame(X.columns)
df_feature_scores = pd.concat([df_columns,df_importance],axis=1)
df_feature_scores.columns = ['feature_name','Score']
feature_scores = df_feature_scores.sort_values(by="Score", ascending=False)
feature_scores.index = range(1,feature_scores.shape[0]+1)
feature_scores["seed"] = seednum
feature_scores.to_csv(
    LR_tocsv,
    sep=',',
    header=True,
    index=False
)

