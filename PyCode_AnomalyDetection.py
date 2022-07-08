#!/usr/bin/env python
# coding: utf-8


import os
import sys
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.knn import KNN


# In[ ]:

py_path = sys.argv[1]
FS_path = sys.argv[2]
AD_tocsv = sys.argv[3]
os.chdir(py_path)
outliers_fraction = eval(sys.argv[4])
seednum01 = eval(sys.argv[5])
seednum02 = eval(sys.argv[6])
sequ_fe = sys.argv[7]
sequ_ml = sys.argv[8]
PC = sys.argv[9]
TrainIndex = [int(x) for x in sys.argv[10].split(",",-1)]
TestIndex_IN = [int(x) for x in sys.argv[11].split(",",-1)]
TestIndex_OUT = [int(x) for x in sys.argv[12].split(",",-1)]
FeatureName_blank = [x for x in sys.argv[13].split(",",-1)]
FeatureName_chi2 = [x for x in sys.argv[14].split(",",-1)]
FeatureName_ETC = [x for x in sys.argv[15].split(",",-1)]
FeatureName_f_classif = [x for x in sys.argv[16].split(",",-1)]
FeatureName_info_classif = [x for x in sys.argv[17].split(",",-1)]
FeatureName_LR = [x for x in sys.argv[18].split(",",-1)]
FeatureName_RF = [x for x in sys.argv[19].split(",",-1)]
FeatureName_XGBoost = [x for x in sys.argv[20].split(",",-1)]
FeatureName_LD = [x for x in sys.argv[21].split(",",-1)]
Feature = eval(sys.argv[22])
Size = eval(sys.argv[23])

data = pd.read_csv("./SourceDataFile/MergeBreed03/merge_breed_02_fill_04.raw",sep=" ",header=0)
data.index = range(1, len(data)+1)
data_fam = data.iloc[:, 0:6]
data_fea = data.iloc[:, 6:]
data_fea.columns = [i[0:11] for i in data_fea.columns]
data = pd.concat([data_fam, data_fea], axis=1)

data_blank = pd.concat([data.loc[:, (data.columns[0:6])],data_fea.loc[:, FeatureName_blank]], axis=1)
data_chi2 = pd.concat([data.loc[:, (data.columns[0:6])],data_fea.loc[:, FeatureName_chi2]], axis=1)
data_ETC = pd.concat([data.loc[:, (data.columns[0:6])],data_fea.loc[:, FeatureName_ETC]], axis=1)
data_f_classif = pd.concat([data.loc[:, (data.columns[0:6])],data_fea.loc[:, FeatureName_f_classif]], axis=1)
data_info_classif = pd.concat([data.loc[:, (data.columns[0:6])],data_fea.loc[:, FeatureName_info_classif]], axis=1)
data_LR = pd.concat([data.loc[:, (data.columns[0:6])],data_fea.loc[:, FeatureName_LR]], axis=1)
data_RF = pd.concat([data.loc[:, (data.columns[0:6])],data_fea.loc[:, FeatureName_RF]], axis=1)
data_XGBoost = pd.concat([data.loc[:, (data.columns[0:6])],data_fea.loc[:, FeatureName_XGBoost]], axis=1)
data_LD = pd.concat([data.loc[:, (data.columns[0:6])],data_fea.loc[:, FeatureName_LD]], axis=1)

FS_AD_dict = {
    'AnomalyDetection_blank':data_blank,
    'AnomalyDetection_chi2': data_chi2,
    'AnomalyDetection_ETC': data_ETC,
    'AnomalyDetection_f_classif': data_f_classif,
    'AnomalyDetection_info_classif': data_info_classif,
    'AnomalyDetection_LR': data_LR,
    'AnomalyDetection_RF': data_RF,
    'AnomalyDetection_XGBoost': data_XGBoost,
    'AnomalyDetection_LD': data_LD
}

for i, (name, data) in enumerate(FS_AD_dict.items(), start=1):
    data_train = data.iloc[TrainIndex,:]
    data_test_in = data.iloc[TestIndex_IN,:]
    data_test_out = data.iloc[TestIndex_OUT,:]
    X_train = data_train.iloc[:, 6:]
    y_train = data_train.iloc[:, 1]
    X_test_in = data_test_in.iloc[:, 6:]
    y_test_ID_in = data_test_in.iloc[:, np.array([0, 1])]
    X_test_out = data_test_out.iloc[:, 6:]
    y_test_ID_out = data_test_out.iloc[:, np.array([0, 1])]
    Class_true_y_test_in = pd.DataFrame(np.repeat(1, len(y_test_ID_in), axis=0))
    Class_true_y_test_in.index = y_test_ID_in.index
    Class_true_y_test_out = pd.DataFrame(np.repeat(-1, len(y_test_ID_out), axis=0))
    Class_true_y_test_out.index = y_test_ID_out.index
    test_in = pd.concat([pd.DataFrame(Class_true_y_test_in),y_test_ID_in, X_test_in], axis=1)
    test_out = pd.concat([pd.DataFrame(Class_true_y_test_out),y_test_ID_out, X_test_out], axis=1)
    test = pd.concat([test_in, test_out],axis=0).sort_values(by=["FID"], ascending=True)
    test = test.rename(columns={0: 'Class_true'})
    X_test = test.iloc[:, 3:]
    y_test = test.iloc[:, 1]

    LOF = LocalOutlierFactor(n_neighbors=10, contamination=outliers_fraction, novelty=True)
    LOF.fit(X_train)
    pred_lof = LOF.predict(X_test)
    KNN_lar = KNN(algorithm='auto', contamination=outliers_fraction, leaf_size=30, method='largest',
                  metric='minkowski', metric_params=None, n_neighbors=5, p=2,
                  radius=1.0)
    KNN_lar.fit(X_train)
    pred_KNN_lar = KNN_lar.predict(X_test)
    pred_KNN_lar[pred_KNN_lar == 1] = -1
    pred_KNN_lar[pred_KNN_lar == 0] = 1
    y_test_scores_KNN_lar = KNN_lar.decision_function(X_test)
    IForest_fit = IForest(contamination=outliers_fraction)
    IForest_fit.fit(X_train)
    pred_IForest = IForest_fit.predict(X_test)
    pred_IForest[pred_IForest == 1] = -1
    pred_IForest[pred_IForest == 0] = 1
    y_test_scores_IForest = IForest_fit.decision_function(X_test)
    pred_vote = pred_lof + pred_KNN_lar + pred_IForest
    for i in range(0, len(pred_vote)):
        if pred_vote[i] > 0:
            pred_vote[i] = 1
        if pred_vote[i] == 0:
            pred_vote[i] = pred_lof[i]
        if pred_vote[i] < 0:
            pred_vote[i] = -1
    normal_vote = X_test[pred_vote == 1]
    abnormal_vote = X_test[pred_vote == -1]
    IndividualNum = pd.DataFrame(y_test.index, columns=["IndividualNum"])
    IndividualNum.index = y_test.index
    OutliersFraction = np.repeat(outliers_fraction, repeats=IndividualNum.shape[0], axis=0)
    seed01 = np.repeat(seednum01, repeats=IndividualNum.shape[0], axis=0)
    seed02 = np.repeat(seednum02, repeats=IndividualNum.shape[0], axis=0)
    FeatureNum = np.repeat(Feature, repeats=IndividualNum.shape[0], axis=0)
    SizeNum = np.repeat(Size, repeats=IndividualNum.shape[0], axis=0)

    resultdata = pd.concat([IndividualNum,
                            data.iloc[y_test.index-1, range(0, 6)],
                            pd.DataFrame(seed01, columns={"seed01"},index=y_test.index),
                            pd.DataFrame(seed02, columns={"seed02"},index=y_test.index),
                            pd.DataFrame(FeatureNum,columns={"FeatureNum"},index=y_test.index),
                            pd.DataFrame(SizeNum,columns={"SizeNum"},index=y_test.index),
                            pd.DataFrame(OutliersFraction, columns={"OutliersFraction"},index=y_test.index),
                            test["Class_true"],
                            pd.DataFrame(pred_lof, columns={"pred_lof"},index=y_test.index),
                            pd.DataFrame(pred_KNN_lar, columns={"pred_KNN_lar"},index=y_test.index),
                            pd.DataFrame(pred_IForest, columns={"pred_IForest"},index=y_test.index),
                            pd.DataFrame(pred_vote, columns={"pred_vote"},index=y_test.index)
                            ],axis=1)
    resultdata.to_csv(
        AD_tocsv + "/" + name + "_" + sequ_fe + "_" + sequ_ml + "_" + PC + ".csv",
        sep=',',
        header=True,
        index=False
    )
AnomalyDetection_fam = resultdata.iloc[:,range(0, 7)]
AnomalyDetection_fam = pd.concat([AnomalyDetection_fam,
                                  pd.DataFrame(seed01, columns={"seed01"}, index=y_test.index),
                                  pd.DataFrame(seed02, columns={"seed02"}, index=y_test.index),
                                  pd.DataFrame(FeatureNum, columns={"FeatureNum"}, index=y_test.index),
                                  pd.DataFrame(SizeNum, columns={"SizeNum"}, index=y_test.index),
                                  pd.DataFrame(OutliersFraction, columns={"OutliersFraction"}, index=y_test.index),
                                  test["Class_true"]
                                  ],axis=1)
AnomalyDetection_fam.to_csv(
    AD_tocsv + "/" + "AnomalyDetection_fam" + "_" + sequ_fe + "_" + sequ_ml + "_" + PC + ".csv",
    sep=',',
    header=True,
    index=False
)
