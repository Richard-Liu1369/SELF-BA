# Packages & Switch -------------------------------------------------------

# Library Packages
library("data.table")
library("stringr")
library("plyr")
library("dplyr")
library("randomForest")
library("C50")
library("e1071")
library("reticulate")
library("kknn")
library("caret")
library("forcats")
library("doParallel")
library("parallel")
library("nnet")
library("reshape")
library("class")
# Switch
LearningSwitch <- function(type){switch (type,T = "TRUE",F = "FALSE")}
# switch ------------------------------------------------------------------

FeatureSelection <- "T"
AnomalyDetection <- "T"
WorkInWindows <- "T"

# Parameter ---------------------------------------------------------------

date_num <- "20220113"
sequ <- 1
size <- c(4,8,12,16,20)
feature <- c(2,4,8,16,32,64,128,256,512)
seednum01 <- seed <- sample(1:10^5,sequ,replace = F)
TestScaleSize <- 0.2
OutliersFraction <- 0.15
path_R <- "E:/R/RCode_ResearchProject/RCode_Github"

# Stacking01
FeatureSelection_1 <- "chi2"
FeatureSelection_2 <- "ETC"
FeatureSelection_3 <- "f_classif"
FeatureSelection_4 <- "info_classif"
FeatureSelection_5 <- "LR"
FeatureSelection_6 <- "RF"
FeatureSelection_7 <- "XGBoost"
FeatureSelection_8 <- "LD"
FeatureSelection_9 <- "blank"
MachineLearning_1 <- "c50"
MachineLearning_2 <- "rf"
MachineLearning_3 <- "knn"
MachineLearning_4 <- "mlr"
MachineLearning_5 <- "svm"
MachineLearning_6 <- "nb"
MachineLearning_7 <- "nnet"
for (SMI in 1:length(ls(pattern = "FeatureSelection_[0-9]"))) {
  assign(paste0("stacking_model01_",SMI),
         list("FeatureSelection" = get(paste0("FeatureSelection_",SMI)),
              "Base_model_1" = paste(get(paste0("FeatureSelection_",SMI)),MachineLearning_1,sep = "-"),
              "Base_model_2" = paste(get(paste0("FeatureSelection_",SMI)),MachineLearning_2,sep = "-"),
              "Base_model_3" = paste(get(paste0("FeatureSelection_",SMI)),MachineLearning_3,sep = "-"),
              "Base_model_4" = paste(get(paste0("FeatureSelection_",SMI)),MachineLearning_4,sep = "-"),
              "Base_model_5" = paste(get(paste0("FeatureSelection_",SMI)),MachineLearning_5,sep = "-"),
              "Base_model_6" = paste(get(paste0("FeatureSelection_",SMI)),MachineLearning_6,sep = "-"),
              "Base_model_7" = paste(get(paste0("FeatureSelection_",SMI)),MachineLearning_7,sep = "-")))
}

# Stackng02
FeatureSelection_A <- "chi2"
FeatureSelection_B <- "ETC"
FeatureSelection_C <- "f_classif"
FeatureSelection_D <- "info_classif"
FeatureSelection_E <- "LR"
FeatureSelection_F <- "RF"
FeatureSelection_G <- "XGBoost"
FeatureSelection_H <- "LD"
FeatureSelection_I <- "blank"
MachineLearning_A <- "c50"
MachineLearning_B <- "rf"
MachineLearning_C <- "knn"
MachineLearning_D <- "mlr"
MachineLearning_E <- "svm"
MachineLearning_F <- "nb"
MachineLearning_G <- "nnet"


assign(paste0("stacking_model02"),
       list("Base_model_1" = paste(get(paste0("FeatureSelection_H")),MachineLearning_D,sep = "-"),
            "Base_model_2" = paste(get(paste0("FeatureSelection_C")),MachineLearning_B,sep = "-"),
            "Base_model_3" = paste(get(paste0("FeatureSelection_D")),MachineLearning_C,sep = "-"),
            "Base_model_4" = paste(get(paste0("FeatureSelection_E")),MachineLearning_A,sep = "-"),
            "Base_model_5" = paste(get(paste0("FeatureSelection_G")),MachineLearning_F,sep = "-"),
            "Base_model_6" = paste(get(paste0("FeatureSelection_F")),MachineLearning_F,sep = "-")
       ))


# Windows & Linux ---------------------------------------------------------


# Six machine learning classification methods, including Naïve Bayes, K-Nearest
if (LearningSwitch(WorkInWindows) == TRUE) {
  "./SourceCode_Github/PyCode_FeatureSelection.py"
  "./SourceCode_Github/PyCode_AnomalyDetection.py"
  "D:/Softwares/anaconda3/envs/r-reticulate"
  path_FS_py <- paste(path_R,"SourceCode_Github/PyCode_FeatureSelection.py",sep = "/")
  path_AD_py <- paste(path_R,"SourceCode_Github/PyCode_AnomalyDetection.py",sep = "/")
  python_env_path <- "D:/Softwares/anaconda3/envs/r-reticulate"
  python_reticulate_path <- paste(python_env_path,"python.exe",sep = "/")
  use_condaenv(python_env_path, required = TRUE)
  use_python(python_reticulate_path,required = TRUE)
  py_available(initialize=TRUE)
}else{
  path_R <- "/GFPS8p/zhangz/WORK/liurq/project_20210403_"
  path_FS_py <- paste(path_R,"SourceCode/FeatureSelection_20210531_01.py",sep = "/")
  path_AD_py <- paste(path_R,"SourceCode/AnomalyDetection_20210607_02.py",sep = "/")
}
# CreateFolder&Path -------------------------------------------------------

setwd(path_R)
sequ_folder <- sequ
date_sequ_f <- paste(date_num,sequ_folder,sep = "_")
FS_ResultPath <- paste("FS",date_sequ_f,sep = "_")
path_FS <- paste0("./FeatureSelection/",FS_ResultPath)
AD_ResultPath <- paste("AD",date_sequ_f,sep = "_")
ML_ResultPath <- paste("ML",date_sequ_f,sep = "_")
path_AD <- paste0("./AnomalyDetection_MachineLearning/",AD_ResultPath)
path_ML <- paste0("./AnomalyDetection_MachineLearning/",ML_ResultPath)
dir.create(path_AD)
dir.create(path_ML)
# FeatureSelection --------------------------------------------------------

if (LearningSwitch(FeatureSelection) == TRUE) {
  for (g in sequ:sequ) {
    seednum <- seednum01[sequ]
    dir.create(path_FS)
    path_chdir_FS <- paste0("r","'",path_R,"'")
    path_chdir_FS <- path_R
    path_FS_py <- paste(path_R,"SourceCode_Github/PyCode_FeatureSelection.py",sep = "/")

    system(paste("python",
                 path_FS_py,
                 path_chdir_FS,# 01
                 path_FS,# 02
                 seednum,# 03
                 sep = " ")) #0-success;1-fail
    
    system(paste("plink --bfile ./SourceDataFile/MergeBreed03/merge_breed_02_fill_04",
                 "--indep-pairwise",50,5,0.015,"--out",
                 paste(path_FS,"merge_breed_02_fill_04",sep = "/"),sep = " "))
    Feature_LD.in <- fread(paste(path_FS,"merge_breed_02_fill_04.prune.in",sep = "/"),header=FALSE,data.table = FALSE)
    Feature_LD.out <- fread(paste(path_FS,"merge_breed_02_fill_04.prune.out",sep = "/"),header=FALSE,data.table = FALSE)
    Feature_LD <- rbind(Feature_LD.in,Feature_LD.out)
    colnames(Feature_LD) <- "feature_name"
    fwrite(x = Feature_LD,
           file = paste0(path_FS,"/","mergebreed_FeatureImportance_LD",".csv"),
           row.names = FALSE,col.names = TRUE, sep = ",")
  }
}
# dataImporting01 ---------------------------------------------------------


raw <- fread("./SourceDataFile/MergeBreed03/merge_breed_02_fill_04.raw",header=TRUE,data.table = FALSE)
Feature_chi2 <- fread(paste(path_FS,"mergebreed_FeatureImportance_chi2.csv",sep = "/"),header=TRUE,data.table = FALSE)
Feature_ETC <- fread(paste(path_FS,"mergebreed_FeatureImportance_ETC.csv",sep = "/"),header=TRUE,data.table = FALSE)
Feature_f_classif <- fread(paste(path_FS,"mergebreed_FeatureImportance_f_classif.csv",sep = "/"),header=TRUE,data.table = FALSE)
Feature_info_classif <- fread(paste(path_FS,"mergebreed_FeatureImportance_info_classif.csv",sep = "/"),header=TRUE,data.table = FALSE)
Feature_LR <- fread(paste(path_FS,"mergebreed_FeatureImportance_LR.csv",sep = "/"),header=TRUE,data.table = FALSE)
Feature_RF <- fread(paste(path_FS,"mergebreed_FeatureImportance_RF.csv",sep = "/"),header=TRUE,data.table = FALSE)
Feature_XGBoost <- fread(paste(path_FS,"mergebreed_FeatureImportance_XGBoost.csv",sep = "/"),header=TRUE,data.table = FALSE)
Feature_LD <- fread(paste(path_FS,"mergebreed_FeatureImportance_LD.csv",sep = "/"),header=TRUE,data.table = FALSE)
data_raw <- NULL %>% cbind(.,1:nrow(raw)) %>% cbind(.,raw)
colnames(data_raw)[1] <- "NUM"
varietyname_label <- sort(unique(data_raw$FID))
colnames(data_raw)[c(8:ncol(data_raw))] <- sapply(X = colnames(data_raw)[-c(1:7)],substr,1,11)
seednum01 <- Feature_chi2$seed[1]
print(paste0("seednum01 = ",seednum01))
Feature_chi2$feature_name <- sapply(X = Feature_chi2$feature_name,substr,1,11)
Feature_ETC$feature_name <- sapply(X = Feature_ETC$feature_name,substr,1,11)
Feature_f_classif$feature_name <- sapply(X = Feature_f_classif$feature_name,substr,1,11)
Feature_info_classif$feature_name <- sapply(X = Feature_info_classif$feature_name,substr,1,11)
Feature_LR$feature_name <- sapply(X = Feature_LR$feature_name,substr,1,11)
Feature_RF$feature_name <- sapply(X = Feature_RF$feature_name,substr,1,11)
Feature_XGBoost$feature_name <- sapply(X = Feature_XGBoost$feature_name,substr,1,11)
print("pass1")

# ParallelComputing -------------------------------------------------------


times <- c(1:20)
no_cores <- detectCores()
registerDoParallel(no_cores)
foreach(t = times, .packages=c("data.table","stringr","e1071","kknn","randomForest","nnet","C50","forcats","caret","nnet","plyr","dplyr","reticulate")) %dopar% {
  sequ_ml <- TableNum <- 0
  seednum02 <- seednum01 + as.numeric(paste0(t,"0000"))
  for (e in length(size):length(size)) {
    SizeNum <- size[e]
    fold <- SizeNum
    for (f in length(feature):length(feature)) {
      FeatureNum <- feature[f]
      sequ_ml <- TableNum <- sequ_ml + 1
      data_sequ_ml <- paste(date_sequ_f,sequ_ml,sep = "_")
      data_sequ_ml_t <- paste(data_sequ_ml,t,sep = "_")
      # Division of training set and test set -----------------------------------
      
      varietyname_label_in <- c("DHB","DUR","GLGS","ITWB","LDR","LWT","MGZ","PIT","WZS")
      varietyname_label_out <- varietyname_label[-match(x = varietyname_label_in,table = varietyname_label)]
      data_raw_in <- data_raw[which(data_raw$FID %in% varietyname_label_in),]
      data_raw_out <- data_raw[which(data_raw$FID %in% varietyname_label_out),]
      #
      set.seed(seednum02)
      trainlabel <- createDataPartition(y = factor(data_raw_in$FID),times = 1,p = 0.8,list = F)
      #
      data_raw_train <- data_raw_in_train <- data_raw_in[trainlabel,]
      data_raw_in_test <- data_raw_in[-trainlabel,]
      data_raw_test <- testdata_all <- rbind(data_raw_in_test,data_raw_out)
      #
      set.seed(seed = seednum02)
      FeatureNum_blank <- sample(x = 1:(ncol(data_raw)-7),size = FeatureNum,replace = F)
      FeatureName_blank <- colnames(data_raw)[FeatureNum_blank+7]
      FeatureName_chi2 <- Feature_chi2$feature_name[c(1:FeatureNum)]
      FeatureName_ETC <- Feature_ETC$feature_name[c(1:FeatureNum)]
      FeatureName_f_classif <- Feature_f_classif$feature_name[c(1:FeatureNum)]
      FeatureName_info_classif <- Feature_info_classif$feature_name[c(1:FeatureNum)]
      FeatureName_LR <- Feature_LR$feature_name[c(1:FeatureNum)]
      FeatureName_RF <- Feature_RF$feature_name[c(1:FeatureNum)]
      FeatureName_XGBoost <- Feature_XGBoost$feature_name[c(1:FeatureNum)]
      FeatureName_LD <- Feature_LD$feature_name[c(1:FeatureNum)]
      
      print(paste0("FeatureNum=",FeatureNum));print(paste0("sequ_ml = TableNum = ",sequ_ml))
      #
      train_fam <- data_raw_train[,c(1:7)]
      train_blank <- data_raw_train[,c(1:7,match(x = FeatureName_blank,table = colnames(data_raw)))]
      train_chi2 <- data_raw_train[,c(1:7,match(x = FeatureName_chi2,table = colnames(data_raw)))]
      train_ETC <- data_raw_train[,c(1:7,match(x = FeatureName_ETC,table = colnames(data_raw)))]
      train_f_classif <- data_raw_train[,c(1:7,match(x = FeatureName_f_classif,table = colnames(data_raw)))]
      train_info_classif <- data_raw_train[,c(1:7,match(x = FeatureName_info_classif,table = colnames(data_raw)))]
      train_LR <- data_raw_train[,c(1:7,match(x = FeatureName_LR,table = colnames(data_raw)))]
      train_RF <- data_raw_train[,c(1:7,match(x = FeatureName_RF,table = colnames(data_raw)))]
      train_XGBoost <- data_raw_train[,c(1:7,match(x = FeatureName_XGBoost,table = colnames(data_raw)))]
      train_LD <- data_raw_train[,c(1:7,match(x = FeatureName_LD,table = colnames(data_raw)))]
      
      train.fam <- NULL
      train.blank <- NULL
      train_label_stacking <- NULL
      train.chi2 <- NULL
      train.ETC <- NULL
      train.f_classif <- NULL
      train.info_classif <- NULL
      train.LR <- NULL
      train.RF <- NULL
      train.XGBoost <- NULL
      train.LD <- NULL
      index_i <- list()
      for (i in 1:length(varietyname_label_in)) {
        data_i <- train_fam[train_fam$FID == varietyname_label_in[i],]
        set.seed(seednum02)
        index_i[i] <- sample(1:nrow(data_i),size=SizeNum) %>% data_i[.,1] %>% as.data.frame(.)
        train.fam <- train_label_stacking <- rbind(train.fam,train_fam[match(x = index_i[[i]],table = train_fam[,1]),])
        train.blank <- rbind(train.blank,train_blank[match(x = index_i[[i]],table = train_blank[,1]),])
        train.chi2 <- rbind(train.chi2,train_chi2[match(x = index_i[[i]],table = train_chi2[,1]),])
        train.ETC <- rbind(train.ETC,train_ETC[match(x = index_i[[i]],table = train_ETC[,1]),])
        train.f_classif <- rbind(train.f_classif,train_f_classif[match(x = index_i[[i]],table = train_f_classif[,1]),])
        train.info_classif <- rbind(train.info_classif,train_info_classif[match(x = index_i[[i]],table = train_info_classif[,1]),])
        train.LR <- rbind(train.LR,train_LR[match(x = index_i[[i]],table = train_LR[,1]),])
        train.RF <- rbind(train.RF,train_RF[match(x = index_i[[i]],table = train_RF[,1]),])
        train.XGBoost <- rbind(train.XGBoost,train_XGBoost[match(x = index_i[[i]],table = train_XGBoost[,1]),])
        train.LD <- rbind(train.LD,train_LD[match(x = index_i[[i]],table = train_LD[,1]),])
      }
      train.blank$FID <- as.factor(train.blank$FID)
      train.chi2$FID <- as.factor(train.chi2$FID)
      train.ETC$FID <- as.factor(train.ETC$FID)
      train.f_classif$FID <- as.factor(train.f_classif$FID)
      train.info_classif$FID <- as.factor(train.info_classif$FID)
      train.LR$FID <- as.factor(train.LR$FID)
      train.RF$FID <- as.factor(train.RF$FID)
      train.XGBoost$FID <- as.factor(train.XGBoost$FID)
      train.LD$FID <- as.factor(train.LD$FID)
      print(paste("pass:","data pre-processing"))
      
      # AnomalyDetection --------------------------------------------------------
      
      
      if (LearningSwitch(AnomalyDetection) == TRUE){
        #
        path_chdir_AD <- path_R
        TrainIndex_py <- (melt(index_i)[,1]-1) %>% paste(.,collapse = ",")
        TestIndex_IN_py <- (data_raw_in_test$NUM-1) %>% paste(.,collapse = ",")
        TestIndex_OUT_py <- (data_raw_out$NUM-1) %>% paste(.,collapse = ",")
        
        FeatureName_blank_py <- paste(FeatureName_blank,collapse = ",")
        FeatureName_chi2_py <- paste(FeatureName_chi2,collapse = ",")
        FeatureName_ETC_py <- paste(FeatureName_ETC,collapse = ",")
        FeatureName_f_classif_py <- paste(FeatureName_f_classif,collapse = ",")
        FeatureName_info_classif_py <- paste(FeatureName_info_classif,collapse = ",")
        FeatureName_LR_py <- paste(FeatureName_LR,collapse = ",")
        FeatureName_RF_py <- paste(FeatureName_RF,collapse = ",")
        FeatureName_XGBoost_py <- paste(FeatureName_XGBoost,collapse = ",")
        FeatureName_LD_py <- paste(FeatureName_LD,collapse = ",")
        #
        system(paste("python",
                     path_AD_py,
                     path_chdir_AD,# 01
                     path_FS,# 02
                     path_AD,# 03
                     OutliersFraction,# 04
                     seednum01,# 05
                     seednum02,# 06
                     sequ_folder,# 07
                     sequ_ml,# 08
                     t,# 09
                     TrainIndex_py,# 10
                     TestIndex_IN_py,# 11
                     TestIndex_OUT_py,# 12
                     FeatureName_blank_py,# 13 blank
                     FeatureName_chi2_py,# 14 chi2
                     FeatureName_ETC_py,# 15 ETC
                     FeatureName_f_classif_py,# 16 f_classif
                     FeatureName_info_classif_py,# 17 info_classif
                     FeatureName_LR_py,# 18 LR
                     FeatureName_RF_py,# 19 RF
                     FeatureName_XGBoost_py,# 20 XGBoost
                     FeatureName_LD_py,# 21 LD
                     FeatureNum,# 22
                     SizeNum,# 23
                     sep = " ")) #0-success;1-fail
      }
      
      AD_fam <- fread(paste0(path_AD,"/","AnomalyDetection_fam","_",sequ_folder,"_",sequ_ml,"_",t,".csv"),header=TRUE,data.table = FALSE) %>% 
        .[match(x = testdata_all$IID,table = .$IID),]
      AnomalyDetection_blank <- fread(paste0(path_AD,"/","AnomalyDetection_blank","_",sequ_folder,"_",sequ_ml,"_",t,".csv"),header=TRUE,data.table = FALSE) %>% 
        .[match(x = testdata_all$IID,table = .$IID),]
      AnomalyDetection_chi2 <- fread(paste0(path_AD,"/","AnomalyDetection_chi2","_",sequ_folder,"_",sequ_ml,"_",t,".csv"),header=TRUE,data.table = FALSE) %>% 
        .[match(x = testdata_all$IID,table = .$IID),]
      AnomalyDetection_ETC <- fread(paste0(path_AD,"/","AnomalyDetection_ETC","_",sequ_folder,"_",sequ_ml,"_",t,".csv"),header=TRUE,data.table = FALSE) %>% 
        .[match(x = testdata_all$IID,table = .$IID),]
      AnomalyDetection_f_classif <- fread(paste0(path_AD,"/","AnomalyDetection_f_classif","_",sequ_folder,"_",sequ_ml,"_",t,".csv"),header=TRUE,data.table = FALSE) %>% 
        .[match(x = testdata_all$IID,table = .$IID),]
      AnomalyDetection_info_classif <- fread(paste0(path_AD,"/","AnomalyDetection_info_classif","_",sequ_folder,"_",sequ_ml,"_",t,".csv"),header=TRUE,data.table = FALSE) %>% 
        .[match(x = testdata_all$IID,table = .$IID),]
      AnomalyDetection_LR <- fread(paste0(path_AD,"/","AnomalyDetection_LR","_",sequ_folder,"_",sequ_ml,"_",t,".csv"),header=TRUE,data.table = FALSE) %>% 
        .[match(x = testdata_all$IID,table = .$IID),]
      AnomalyDetection_RF <- fread(paste0(path_AD,"/","AnomalyDetection_RF","_",sequ_folder,"_",sequ_ml,"_",t,".csv"),header=TRUE,data.table = FALSE) %>% 
        .[match(x = testdata_all$IID,table = .$IID),]
      AnomalyDetection_XGBoost <- fread(paste0(path_AD,"/","AnomalyDetection_XGBoost","_",sequ_folder,"_",sequ_ml,"_",t,".csv"),header=TRUE,data.table = FALSE) %>% 
        .[match(x = testdata_all$IID,table = .$IID),]
      AnomalyDetection_LD <- fread(paste0(path_AD,"/","AnomalyDetection_LD","_",sequ_folder,"_",sequ_ml,"_",t,".csv"),header=TRUE,data.table = FALSE) %>% 
        .[match(x = testdata_all$IID,table = .$IID),]
      # k-Nearest Neighbor ------------------------------------------------------
      
      
      blank_knn <- function(data.train,data.test,varietyname_label){
        result_blank_knn <- list()
        StartTime <- proc.time()
        kknn <- kknn(data.train$FID~.,data.train[,-c(1:7)], data.test[,-c(1:7)], distance = 2,k= 2,kernel= "triangular")
        predictions_blank_kknn <- predict(object = kknn,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_blank_kknn <- table(predictions_blank_kknn,data.test[,2])
        ratio_blank_kknn <- sum(diag(table_blank_kknn))/sum(table_blank_kknn)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_blank_knn[1] <- as.data.frame(predictions_blank_kknn)
        result_blank_knn[2] <- as.data.frame(ratio_blank_kknn)
        result_blank_knn[3] <- RunningTime[3][[1]]
        return(result_blank_knn)
      }
      
      chi2_knn <- function(data.train,data.test,varietyname_label){
        result_chi2_knn <- list()
        StartTime <- proc.time()
        kknn <- kknn(data.train$FID~.,data.train[,-c(1:7)], data.test[,-c(1:7)], distance = 2,k= 2,kernel= "triangular")
        predictions_chi2_kknn <- predict(object = kknn,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_chi2_kknn <- table(predictions_chi2_kknn,data.test[,2])
        ratio_chi2_kknn <- sum(diag(table_chi2_kknn))/sum(table_chi2_kknn)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_chi2_knn[1] <- as.data.frame(predictions_chi2_kknn)
        result_chi2_knn[2] <- as.data.frame(ratio_chi2_kknn)
        result_chi2_knn[3] <- RunningTime[3][[1]]
        return(result_chi2_knn)
      }
      
      ETC_knn <- function(data.train,data.test,varietyname_label){
        result_ETC_knn <- list()
        StartTime <- proc.time()
        kknn <- kknn(data.train$FID~.,data.train[,-c(1:7)], data.test[,-c(1:7)], distance = 2,k= 2,kernel= "triangular")
        predictions_ETC_kknn <- predict(object = kknn,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_ETC_kknn <- table(predictions_ETC_kknn,data.test[,2])
        ratio_ETC_kknn <- sum(diag(table_ETC_kknn))/sum(table_ETC_kknn)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_ETC_knn[1] <- as.data.frame(predictions_ETC_kknn)
        result_ETC_knn[2] <- as.data.frame(ratio_ETC_kknn)
        result_ETC_knn[3] <- RunningTime[3][[1]]
        return(result_ETC_knn)
      }
      
      f_classif_knn <- function(data.train,data.test,varietyname_label){
        result_f_classif_knn <- list()
        StartTime <- proc.time()
        kknn <- kknn(data.train$FID~.,data.train[,-c(1:7)], data.test[,-c(1:7)], distance = 2,k= 2,kernel= "triangular")
        predictions_f_classif_kknn <- predict(object = kknn,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_f_classif_kknn <- table(predictions_f_classif_kknn,data.test[,2])
        ratio_f_classif_kknn <- sum(diag(table_f_classif_kknn))/sum(table_f_classif_kknn)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_f_classif_knn[1] <- as.data.frame(predictions_f_classif_kknn)
        result_f_classif_knn[2] <- as.data.frame(ratio_f_classif_kknn)
        result_f_classif_knn[3] <- RunningTime[3][[1]]
        return(result_f_classif_knn)
      }
      
      info_classif_knn <- function(data.train,data.test,varietyname_label){
        result_info_classif_knn <- list()
        StartTime <- proc.time()
        kknn <- kknn(data.train$FID~.,data.train[,-c(1:7)], data.test[,-c(1:7)], distance = 2,k= 2,kernel= "triangular")
        predictions_info_classif_kknn <- predict(object = kknn,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_info_classif_kknn <- table(predictions_info_classif_kknn,data.test[,2])
        ratio_info_classif_kknn <- sum(diag(table_info_classif_kknn))/sum(table_info_classif_kknn)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_info_classif_knn[1] <- as.data.frame(predictions_info_classif_kknn)
        result_info_classif_knn[2] <- as.data.frame(ratio_info_classif_kknn)
        result_info_classif_knn[3] <- RunningTime[3][[1]]
        return(result_info_classif_knn)
      }
      
      LR_knn <- function(data.train,data.test,varietyname_label){
        result_LR_knn <- list()
        StartTime <- proc.time()
        kknn <- kknn(data.train$FID~.,data.train[,-c(1:7)], data.test[,-c(1:7)], distance = 2,k= 2,kernel= "triangular")
        predictions_LR_kknn <- predict(object = kknn,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_LR_kknn <- table(predictions_LR_kknn,data.test[,2])
        ratio_LR_kknn <- sum(diag(table_LR_kknn))/sum(table_LR_kknn)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_LR_knn[1] <- as.data.frame(predictions_LR_kknn)
        result_LR_knn[2] <- as.data.frame(ratio_LR_kknn)
        result_LR_knn[3] <- RunningTime[3][[1]]
        return(result_LR_knn)
      }
      
      RF_knn <- function(data.train,data.test,varietyname_label){
        result_RF_knn <- list()
        StartTime <- proc.time()
        kknn <- kknn(data.train$FID~.,data.train[,-c(1:7)], data.test[,-c(1:7)], distance = 2,k= 2,kernel= "triangular")
        predictions_RF_kknn <- predict(object = kknn,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_RF_kknn <- table(predictions_RF_kknn,data.test[,2])
        ratio_RF_kknn <- sum(diag(table_RF_kknn))/sum(table_RF_kknn)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_RF_knn[1] <- as.data.frame(predictions_RF_kknn)
        result_RF_knn[2] <- as.data.frame(ratio_RF_kknn)
        result_RF_knn[3] <- RunningTime[3][[1]]
        return(result_RF_knn)
      }
      
      XGBoost_knn <- function(data.train,data.test,varietyname_label){
        result_XGBoost_knn <- list()
        StartTime <- proc.time()
        kknn <- kknn(data.train$FID~.,data.train[,-c(1:7)], data.test[,-c(1:7)], distance = 2,k= 2,kernel= "triangular")
        predictions_XGBoost_kknn <- predict(object = kknn,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_XGBoost_kknn <- table(predictions_XGBoost_kknn,data.test[,2])
        ratio_XGBoost_kknn <- sum(diag(table_XGBoost_kknn))/sum(table_XGBoost_kknn)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_XGBoost_knn[1] <- as.data.frame(predictions_XGBoost_kknn)
        result_XGBoost_knn[2] <- as.data.frame(ratio_XGBoost_kknn)
        result_XGBoost_knn[3] <- RunningTime[3][[1]]
        return(result_XGBoost_knn)
      }
      
      LD_knn <- function(data.train,data.test,varietyname_label){
        result_LD_knn <- list()
        StartTime <- proc.time()
        kknn <- kknn(data.train$FID~.,data.train[,-c(1:7)], data.test[,-c(1:7)], distance = 2,k= 2,kernel= "triangular")
        predictions_LD_kknn <- predict(object = kknn,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_LD_kknn <- table(predictions_LD_kknn,data.test[,2])
        ratio_LD_kknn <- sum(diag(table_LD_kknn))/sum(table_LD_kknn)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_LD_knn[1] <- as.data.frame(predictions_LD_kknn)
        result_LD_knn[2] <- as.data.frame(ratio_LD_kknn)
        result_LD_knn[3] <- RunningTime[3][[1]]
        return(result_LD_knn)
      }
      
      # Random Forest -----------------------------------------------------------
      
      blank_RF <- function(data.train,data.test,varietyname_label){
        result_blank_RF <- list()
        StartTime <- proc.time()
        RF <- randomForest(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_blank_RF <- predict(object = RF,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_blank_RF <- table(predictions_blank_RF,data.test[,2])
        ratio_blank_RF <- sum(diag(table_blank_RF))/sum(table_blank_RF)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_blank_RF[1] <- as.data.frame(predictions_blank_RF)
        result_blank_RF[2] <- as.data.frame(ratio_blank_RF)
        result_blank_RF[3] <- RunningTime[3][[1]]
        return(result_blank_RF)
      }
      
      chi2_RF <- function(data.train,data.test,varietyname_label){
        result_chi2_RF <- list()
        StartTime <- proc.time()
        RF <- randomForest(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_chi2_RF <- predict(object = RF,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_chi2_RF <- table(predictions_chi2_RF,data.test[,2])
        ratio_chi2_RF <- sum(diag(table_chi2_RF))/sum(table_chi2_RF)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_chi2_RF[1] <- as.data.frame(predictions_chi2_RF)
        result_chi2_RF[2] <- as.data.frame(ratio_chi2_RF)
        result_chi2_RF[3] <- RunningTime[3][[1]]
        return(result_chi2_RF)
      }
      
      ETC_RF <- function(data.train,data.test,varietyname_label){
        result_ETC_RF <- list()
        StartTime <- proc.time()
        RF <- randomForest(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_ETC_RF <- predict(object = RF,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_ETC_RF <- table(predictions_ETC_RF,data.test[,2])
        ratio_ETC_RF <- sum(diag(table_ETC_RF))/sum(table_ETC_RF)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_ETC_RF[1] <- as.data.frame(predictions_ETC_RF)
        result_ETC_RF[2] <- as.data.frame(ratio_ETC_RF)
        result_ETC_RF[3] <- RunningTime[3][[1]]
        return(result_ETC_RF)
      }
      
      f_classif_RF <- function(data.train,data.test,varietyname_label){
        result_f_classif_RF <- list()
        StartTime <- proc.time()
        RF <- randomForest(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_f_classif_RF <- predict(object = RF,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_f_classif_RF <- table(predictions_f_classif_RF,data.test[,2])
        ratio_f_classif_RF <- sum(diag(table_f_classif_RF))/sum(table_f_classif_RF)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_f_classif_RF[1] <- as.data.frame(predictions_f_classif_RF)
        result_f_classif_RF[2] <- as.data.frame(ratio_f_classif_RF)
        result_f_classif_RF[3] <- RunningTime[3][[1]]
        return(result_f_classif_RF)
      }
      
      info_classif_RF <- function(data.train,data.test,varietyname_label){
        result_info_classif_RF <- list()
        StartTime <- proc.time()
        RF <- randomForest(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_info_classif_RF <- predict(object = RF,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_info_classif_RF <- table(predictions_info_classif_RF,data.test[,2])
        ratio_info_classif_RF <- sum(diag(table_info_classif_RF))/sum(table_info_classif_RF)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_info_classif_RF[1] <- as.data.frame(predictions_info_classif_RF)
        result_info_classif_RF[2] <- as.data.frame(ratio_info_classif_RF)
        result_info_classif_RF[3] <- RunningTime[3][[1]]
        return(result_info_classif_RF)
      }
      
      LR_RF <- function(data.train,data.test,varietyname_label){
        result_LR_RF <- list()
        StartTime <- proc.time()
        RF <- randomForest(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_LR_RF <- predict(object = RF,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_LR_RF <- table(predictions_LR_RF,data.test[,2])
        ratio_LR_RF <- sum(diag(table_LR_RF))/sum(table_LR_RF)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_LR_RF[1] <- as.data.frame(predictions_LR_RF)
        result_LR_RF[2] <- as.data.frame(ratio_LR_RF)
        result_LR_RF[3] <- RunningTime[3][[1]]
        return(result_LR_RF)
      }
      
      RF_RF <- function(data.train,data.test,varietyname_label){
        result_RF_RF <- list()
        StartTime <- proc.time()
        RF <- randomForest(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_RF_RF <- predict(object = RF,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_RF_RF <- table(predictions_RF_RF,data.test[,2])
        ratio_RF_RF <- sum(diag(table_RF_RF))/sum(table_RF_RF)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_RF_RF[1] <- as.data.frame(predictions_RF_RF)
        result_RF_RF[2] <- as.data.frame(ratio_RF_RF)
        result_RF_RF[3] <- RunningTime[3][[1]]
        return(result_RF_RF)
      }
      
      XGBoost_RF <- function(data.train,data.test,varietyname_label){
        result_XGBoost_RF <- list()
        StartTime <- proc.time()
        RF <- randomForest(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_XGBoost_RF <- predict(object = RF,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_XGBoost_RF <- table(predictions_XGBoost_RF,data.test[,2])
        ratio_XGBoost_RF <- sum(diag(table_XGBoost_RF))/sum(table_XGBoost_RF)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_XGBoost_RF[1] <- as.data.frame(predictions_XGBoost_RF)
        result_XGBoost_RF[2] <- as.data.frame(ratio_XGBoost_RF)
        result_XGBoost_RF[3] <- RunningTime[3][[1]]
        return(result_XGBoost_RF)
      }
      
      LD_RF <- function(data.train,data.test,varietyname_label){
        result_LD_RF <- list()
        StartTime <- proc.time()
        RF <- randomForest(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_LD_RF <- predict(object = RF,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_LD_RF <- table(predictions_LD_RF,data.test[,2])
        ratio_LD_RF <- sum(diag(table_LD_RF))/sum(table_LD_RF)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_LD_RF[1] <- as.data.frame(predictions_LD_RF)
        result_LD_RF[2] <- as.data.frame(ratio_LD_RF)
        result_LD_RF[3] <- RunningTime[3][[1]]
        return(result_LD_RF)
      }
      
      
      # Naive Bayes -------------------------------------------------------------
      
      
      blank_NB <- function(data.train,data.test,varietyname_label){
        result_blank_NB <- list()
        StartTime <- proc.time()
        NB <- naiveBayes(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_blank_NB <- predict(object = NB,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_blank_NB <- table(predictions_blank_NB,data.test[,2])
        ratio_blank_NB <- sum(diag(table_blank_NB))/sum(table_blank_NB)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_blank_NB[1] <- as.data.frame(predictions_blank_NB)
        result_blank_NB[2] <- as.data.frame(ratio_blank_NB)
        result_blank_NB[3] <- RunningTime[3][[1]]
        return(result_blank_NB)
      }
      
      chi2_NB <- function(data.train,data.test,varietyname_label){
        result_chi2_NB <- list()
        StartTime <- proc.time()
        NB <- naiveBayes(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_chi2_NB <- predict(object = NB,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_chi2_NB <- table(predictions_chi2_NB,data.test[,2])
        ratio_chi2_NB <- sum(diag(table_chi2_NB))/sum(table_chi2_NB)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_chi2_NB[1] <- as.data.frame(predictions_chi2_NB)
        result_chi2_NB[2] <- as.data.frame(ratio_chi2_NB)
        result_chi2_NB[3] <- RunningTime[3][[1]]
        return(result_chi2_NB)
      }
      
      ETC_NB <- function(data.train,data.test,varietyname_label){
        result_ETC_NB <- list()
        StartTime <- proc.time()
        NB <- naiveBayes(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_ETC_NB <- predict(object = NB,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_ETC_NB <- table(predictions_ETC_NB,data.test[,2])
        ratio_ETC_NB <- sum(diag(table_ETC_NB))/sum(table_ETC_NB)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_ETC_NB[1] <- as.data.frame(predictions_ETC_NB)
        result_ETC_NB[2] <- as.data.frame(ratio_ETC_NB)
        result_ETC_NB[3] <- RunningTime[3][[1]]
        return(result_ETC_NB)
      }
      
      f_classif_NB <- function(data.train,data.test,varietyname_label){
        result_f_classif_NB <- list()
        StartTime <- proc.time()
        NB <- naiveBayes(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_f_classif_NB <- predict(object = NB,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_f_classif_NB <- table(predictions_f_classif_NB,data.test[,2])
        ratio_f_classif_NB <- sum(diag(table_f_classif_NB))/sum(table_f_classif_NB)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_f_classif_NB[1] <- as.data.frame(predictions_f_classif_NB)
        result_f_classif_NB[2] <- as.data.frame(ratio_f_classif_NB)
        result_f_classif_NB[3] <- RunningTime[3][[1]]
        return(result_f_classif_NB)
      }
      
      info_classif_NB <- function(data.train,data.test,varietyname_label){
        result_info_classif_NB <- list()
        StartTime <- proc.time()
        NB <- naiveBayes(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_info_classif_NB <- predict(object = NB,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_info_classif_NB <- table(predictions_info_classif_NB,data.test[,2])
        ratio_info_classif_NB <- sum(diag(table_info_classif_NB))/sum(table_info_classif_NB)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_info_classif_NB[1] <- as.data.frame(predictions_info_classif_NB)
        result_info_classif_NB[2] <- as.data.frame(ratio_info_classif_NB)
        result_info_classif_NB[3] <- RunningTime[3][[1]]
        return(result_info_classif_NB)
      }
      
      LR_NB <- function(data.train,data.test,varietyname_label){
        result_LR_NB <- list()
        StartTime <- proc.time()
        NB <- naiveBayes(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_LR_NB <- predict(object = NB,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_LR_NB <- table(predictions_LR_NB,data.test[,2])
        ratio_LR_NB <- sum(diag(table_LR_NB))/sum(table_LR_NB)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_LR_NB[1] <- as.data.frame(predictions_LR_NB)
        result_LR_NB[2] <- as.data.frame(ratio_LR_NB)
        result_LR_NB[3] <- RunningTime[3][[1]]
        return(result_LR_NB)
      }
      
      RF_NB <- function(data.train,data.test,varietyname_label){
        result_RF_NB <- list()
        StartTime <- proc.time()
        NB <- naiveBayes(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_RF_NB <- predict(object = NB,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_RF_NB <- table(predictions_RF_NB,data.test[,2])
        ratio_RF_NB <- sum(diag(table_RF_NB))/sum(table_RF_NB)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_RF_NB[1] <- as.data.frame(predictions_RF_NB)
        result_RF_NB[2] <- as.data.frame(ratio_RF_NB)
        result_RF_NB[3] <- RunningTime[3][[1]]
        return(result_RF_NB)
      }
      
      XGBoost_NB <- function(data.train,data.test,varietyname_label){
        result_XGBoost_NB <- list()
        StartTime <- proc.time()
        NB <- naiveBayes(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_XGBoost_NB <- predict(object = NB,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_XGBoost_NB <- table(predictions_XGBoost_NB,data.test[,2])
        ratio_XGBoost_NB <- sum(diag(table_XGBoost_NB))/sum(table_XGBoost_NB)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_XGBoost_NB[1] <- as.data.frame(predictions_XGBoost_NB)
        result_XGBoost_NB[2] <- as.data.frame(ratio_XGBoost_NB)
        result_XGBoost_NB[3] <- RunningTime[3][[1]]
        return(result_XGBoost_NB)
      }
      
      LD_NB <- function(data.train,data.test,varietyname_label){
        result_LD_NB <- list()
        StartTime <- proc.time()
        NB <- naiveBayes(y = data.train$FID,x = data.train[,-c(1:7)])
        predictions_LD_NB <- predict(object = NB,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_LD_NB <- table(predictions_LD_NB,data.test[,2])
        ratio_LD_NB <- sum(diag(table_LD_NB))/sum(table_LD_NB)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_LD_NB[1] <- as.data.frame(predictions_LD_NB)
        result_LD_NB[2] <- as.data.frame(ratio_LD_NB)
        result_LD_NB[3] <- RunningTime[3][[1]]
        return(result_LD_NB)
      }
      
      # Support Vector Machine --------------------------------------------------
      
      blank_SVM <- function(data.train,data.test,varietyname_label){
        result_blank_SVM <- list()
        StartTime <- proc.time()
        if (ncol(data.train)==ncol(data.test)) {
          SVM <-  svm(y = data.train$FID,x = data.train[,-c(1:7)],kernel = "linear",cost = 1,gamma = 1/ncol(data.train),scale = FALSE)
        }else{
          SVM <-  svm(data.train$FID~.,data = data.train[,-c(1:7)],kernel = "linear",cost = 1,gamma = 1/ncol(data.train),scale = FALSE)
        }
        predictions_blank_SVM <- predict(object = SVM,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_blank_SVM <- table(predictions_blank_SVM,data.test[,2])
        ratio_blank_SVM <- sum(diag(table_blank_SVM))/sum(table_blank_SVM)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_blank_SVM[1] <- as.data.frame(predictions_blank_SVM)
        result_blank_SVM[2] <- as.data.frame(ratio_blank_SVM)
        result_blank_SVM[3] <- RunningTime[3][[1]]
        return(result_blank_SVM)
      }
      
      chi2_SVM <- function(data.train,data.test,varietyname_label){
        result_chi2_SVM <- list()
        StartTime <- proc.time()
        SVM <-  svm(data.train$FID~.,data = data.train[,-c(1:7)],kernel = "linear",cost = 1,gamma = 1/ncol(data.train),scale = FALSE)
        predictions_chi2_SVM <- predict(object = SVM,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_chi2_SVM <- table(predictions_chi2_SVM,data.test[,2])
        ratio_chi2_SVM <- sum(diag(table_chi2_SVM))/sum(table_chi2_SVM)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_chi2_SVM[1] <- as.data.frame(predictions_chi2_SVM)
        result_chi2_SVM[2] <- as.data.frame(ratio_chi2_SVM)
        result_chi2_SVM[3] <- RunningTime[3][[1]]
        return(result_chi2_SVM)
      }
      
      ETC_SVM <- function(data.train,data.test,varietyname_label){
        result_ETC_SVM <- list()
        StartTime <- proc.time()
        SVM <-  svm(data.train$FID~.,data = data.train[,-c(1:7)],kernel = "linear",cost = 1,gamma = 1/ncol(data.train),scale = FALSE)
        predictions_ETC_SVM <- predict(object = SVM,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_ETC_SVM <- table(predictions_ETC_SVM,data.test[,2])
        ratio_ETC_SVM <- sum(diag(table_ETC_SVM))/sum(table_ETC_SVM)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_ETC_SVM[1] <- as.data.frame(predictions_ETC_SVM)
        result_ETC_SVM[2] <- as.data.frame(ratio_ETC_SVM)
        result_ETC_SVM[3] <- RunningTime[3][[1]]
        return(result_ETC_SVM)
      }
      
      f_classif_SVM <- function(data.train,data.test,varietyname_label){
        result_f_classif_SVM <- list()
        StartTime <- proc.time()
        SVM <-  svm(data.train$FID~.,data = data.train[,-c(1:7)],kernel = "linear",cost = 1,gamma = 1/ncol(data.train),scale = FALSE)
        predictions_f_classif_SVM <- predict(object = SVM,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_f_classif_SVM <- table(predictions_f_classif_SVM,data.test[,2])
        ratio_f_classif_SVM <- sum(diag(table_f_classif_SVM))/sum(table_f_classif_SVM)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_f_classif_SVM[1] <- as.data.frame(predictions_f_classif_SVM)
        result_f_classif_SVM[2] <- as.data.frame(ratio_f_classif_SVM)
        result_f_classif_SVM[3] <- RunningTime[3][[1]]
        return(result_f_classif_SVM)
      }
      
      info_classif_SVM <- function(data.train,data.test,varietyname_label){
        result_info_classif_SVM <- list()
        StartTime <- proc.time()
        SVM <-  svm(data.train$FID~.,data = data.train[,-c(1:7)],kernel = "linear",cost = 1,gamma = 1/ncol(data.train),scale = FALSE)
        predictions_info_classif_SVM <- predict(object = SVM,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_info_classif_SVM <- table(predictions_info_classif_SVM,data.test[,2])
        ratio_info_classif_SVM <- sum(diag(table_info_classif_SVM))/sum(table_info_classif_SVM)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_info_classif_SVM[1] <- as.data.frame(predictions_info_classif_SVM)
        result_info_classif_SVM[2] <- as.data.frame(ratio_info_classif_SVM)
        result_info_classif_SVM[3] <- RunningTime[3][[1]]
        return(result_info_classif_SVM)
      }
      
      LR_SVM <- function(data.train,data.test,varietyname_label){
        result_LR_SVM <- list()
        StartTime <- proc.time()
        SVM <-  svm(data.train$FID~.,data = data.train[,-c(1:7)],kernel = "linear",cost = 1,gamma = 1/ncol(data.train),scale = FALSE)
        predictions_LR_SVM <- predict(object = SVM,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_LR_SVM <- table(predictions_LR_SVM,data.test[,2])
        ratio_LR_SVM <- sum(diag(table_LR_SVM))/sum(table_LR_SVM)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_LR_SVM[1] <- as.data.frame(predictions_LR_SVM)
        result_LR_SVM[2] <- as.data.frame(ratio_LR_SVM)
        result_LR_SVM[3] <- RunningTime[3][[1]]
        return(result_LR_SVM)
      }
      
      RF_SVM <- function(data.train,data.test,varietyname_label){
        result_RF_SVM <- list()
        StartTime <- proc.time()
        SVM <-  svm(data.train$FID~.,data = data.train[,-c(1:7)],kernel = "linear",cost = 1,gamma = 1/ncol(data.train),scale = FALSE)
        predictions_RF_SVM <- predict(object = SVM,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_RF_SVM <- table(predictions_RF_SVM,data.test[,2])
        ratio_RF_SVM <- sum(diag(table_RF_SVM))/sum(table_RF_SVM)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_RF_SVM[1] <- as.data.frame(predictions_RF_SVM)
        result_RF_SVM[2] <- as.data.frame(ratio_RF_SVM)
        result_RF_SVM[3] <- RunningTime[3][[1]]
        return(result_RF_SVM)
      }
      
      XGBoost_SVM <- function(data.train,data.test,varietyname_label){
        result_XGBoost_SVM <- list()
        StartTime <- proc.time()
        SVM <-  svm(data.train$FID~.,data = data.train[,-c(1:7)],kernel = "linear",
                    cost = 1,gamma = 1/ncol(data.train[,-c(1:7)]),scale = FALSE)
        predictions_XGBoost_SVM <- predict(object = SVM,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_XGBoost_SVM <- table(predictions_XGBoost_SVM,data.test[,2])
        ratio_XGBoost_SVM <- sum(diag(table_XGBoost_SVM))/sum(table_XGBoost_SVM)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_XGBoost_SVM[1] <- as.data.frame(predictions_XGBoost_SVM)
        result_XGBoost_SVM[2] <- as.data.frame(ratio_XGBoost_SVM)
        result_XGBoost_SVM[3] <- RunningTime[3][[1]]
        return(result_XGBoost_SVM)
      }
      
      LD_SVM <- function(data.train,data.test,varietyname_label){
        result_LD_SVM <- list()
        StartTime <- proc.time()
        SVM <-  svm(data.train$FID~.,data = data.train[,-c(1:7)],kernel = "linear",cost = 1,gamma = 1/ncol(data.train),scale = FALSE)
        predictions_LD_SVM <- predict(object = SVM,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_LD_SVM <- table(predictions_LD_SVM,data.test[,2])
        ratio_LD_SVM <- sum(diag(table_LD_SVM))/sum(table_LD_SVM)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_LD_SVM[1] <- as.data.frame(predictions_LD_SVM)
        result_LD_SVM[2] <- as.data.frame(ratio_LD_SVM)
        result_LD_SVM[3] <- RunningTime[3][[1]]
        return(result_LD_SVM)
      }
      
      
      # C50 ---------------------------------------------------------------------
      
      
      blank_C50 <- function(data.train,data.test,varietyname_label){
        result_blank_C50 <- list()
        StartTime <- proc.time()
        C50 <- C5.0(y = data.train$FID,x = data.train[,-c(1:7)],trials=30)
        predictions_blank_C50 <- predict(object = C50,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_blank_C50 <- table(predictions_blank_C50,data.test[,2])
        ratio_blank_C50 <- sum(diag(table_blank_C50))/sum(table_blank_C50)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_blank_C50[1] <- as.data.frame(predictions_blank_C50)
        result_blank_C50[2] <- as.data.frame(ratio_blank_C50)
        result_blank_C50[3] <- RunningTime[3][[1]]
        return(result_blank_C50)
      }
      
      chi2_C50 <- function(data.train,data.test,varietyname_label){
        result_chi2_C50 <- list()
        StartTime <- proc.time()
        C50 <- C5.0(y = data.train$FID,x = data.train[,-c(1:7)],trials=30)
        predictions_chi2_C50 <- predict(object = C50,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_chi2_C50 <- table(predictions_chi2_C50,data.test[,2])
        ratio_chi2_C50 <- sum(diag(table_chi2_C50))/sum(table_chi2_C50)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_chi2_C50[1] <- as.data.frame(predictions_chi2_C50)
        result_chi2_C50[2] <- as.data.frame(ratio_chi2_C50)
        result_chi2_C50[3] <- RunningTime[3][[1]]
        return(result_chi2_C50)
      }
      
      ETC_C50 <- function(data.train,data.test,varietyname_label){
        result_ETC_C50 <- list()
        StartTime <- proc.time()
        C50 <- C5.0(y = data.train$FID,x = data.train[,-c(1:7)],trials=30)
        predictions_ETC_C50 <- predict(object = C50,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_ETC_C50 <- table(predictions_ETC_C50,data.test[,2])
        ratio_ETC_C50 <- sum(diag(table_ETC_C50))/sum(table_ETC_C50)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_ETC_C50[1] <- as.data.frame(predictions_ETC_C50)
        result_ETC_C50[2] <- as.data.frame(ratio_ETC_C50)
        result_ETC_C50[3] <- RunningTime[3][[1]]
        return(result_ETC_C50)
      }
      
      f_classif_C50 <- function(data.train,data.test,varietyname_label){
        result_f_classif_C50 <- list()
        StartTime <- proc.time()
        C50 <- C5.0(y = data.train$FID,x = data.train[,-c(1:7)],trials=30)
        predictions_f_classif_C50 <- predict(object = C50,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_f_classif_C50 <- table(predictions_f_classif_C50,data.test[,2])
        ratio_f_classif_C50 <- sum(diag(table_f_classif_C50))/sum(table_f_classif_C50)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_f_classif_C50[1] <- as.data.frame(predictions_f_classif_C50)
        result_f_classif_C50[2] <- as.data.frame(ratio_f_classif_C50)
        result_f_classif_C50[3] <- RunningTime[3][[1]]
        return(result_f_classif_C50)
      }
      
      info_classif_C50 <- function(data.train,data.test,varietyname_label){
        result_info_classif_C50 <- list()
        StartTime <- proc.time()
        C50 <- C5.0(y = data.train$FID,x = data.train[,-c(1:7)],trials=30)
        predictions_info_classif_C50 <- predict(object = C50,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_info_classif_C50 <- table(predictions_info_classif_C50,data.test[,2])
        ratio_info_classif_C50 <- sum(diag(table_info_classif_C50))/sum(table_info_classif_C50)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_info_classif_C50[1] <- as.data.frame(predictions_info_classif_C50)
        result_info_classif_C50[2] <- as.data.frame(ratio_info_classif_C50)
        result_info_classif_C50[3] <- RunningTime[3][[1]]
        return(result_info_classif_C50)
      }
      
      LR_C50 <- function(data.train,data.test,varietyname_label){
        result_LR_C50 <- list()
        StartTime <- proc.time()
        C50 <- C5.0(y = data.train$FID,x = data.train[,-c(1:7)],trials=30)
        predictions_LR_C50 <- predict(object = C50,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_LR_C50 <- table(predictions_LR_C50,data.test[,2])
        ratio_LR_C50 <- sum(diag(table_LR_C50))/sum(table_LR_C50)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_LR_C50[1] <- as.data.frame(predictions_LR_C50)
        result_LR_C50[2] <- as.data.frame(ratio_LR_C50)
        result_LR_C50[3] <- RunningTime[3][[1]]
        return(result_LR_C50)
      }
      
      RF_C50 <- function(data.train,data.test,varietyname_label){
        result_RF_C50 <- list()
        StartTime <- proc.time()
        C50 <- C5.0(y = data.train$FID,x = data.train[,-c(1:7)],trials=30)
        predictions_RF_C50 <- predict(object = C50,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_RF_C50 <- table(predictions_RF_C50,data.test[,2])
        ratio_RF_C50 <- sum(diag(table_RF_C50))/sum(table_RF_C50)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_RF_C50[1] <- as.data.frame(predictions_RF_C50)
        result_RF_C50[2] <- as.data.frame(ratio_RF_C50)
        result_RF_C50[3] <- RunningTime[3][[1]]
        return(result_RF_C50)
      }
      
      XGBoost_C50 <- function(data.train,data.test,varietyname_label){
        result_XGBoost_C50 <- list()
        StartTime <- proc.time()
        C50 <- C5.0(y = data.train$FID,x = data.train[,-c(1:7)],trials=30)
        predictions_XGBoost_C50 <- predict(object = C50,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_XGBoost_C50 <- table(predictions_XGBoost_C50,data.test[,2])
        ratio_XGBoost_C50 <- sum(diag(table_XGBoost_C50))/sum(table_XGBoost_C50)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_XGBoost_C50[1] <- as.data.frame(predictions_XGBoost_C50)
        result_XGBoost_C50[2] <- as.data.frame(ratio_XGBoost_C50)
        result_XGBoost_C50[3] <- RunningTime[3][[1]]
        return(result_XGBoost_C50)
      }
      
      LD_C50 <- function(data.train,data.test,varietyname_label){
        result_LD_C50 <- list()
        StartTime <- proc.time()
        C50 <- C5.0(y = data.train$FID,x = data.train[,-c(1:7)],trials=30)
        predictions_LD_C50 <- predict(object = C50,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_LD_C50 <- table(predictions_LD_C50,data.test[,2])
        ratio_LD_C50 <- sum(diag(table_LD_C50))/sum(table_LD_C50)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_LD_C50[1] <- as.data.frame(predictions_LD_C50)
        result_LD_C50[2] <- as.data.frame(ratio_LD_C50)
        result_LD_C50[3] <- RunningTime[3][[1]]
        return(result_LD_C50)
      }
      
      # nnet --------------------------------------------------------------------
      
      blank_nnet <- function(data.train,data.test,varietyname_label){
        result_blank_nnet <- list()
        StartTime <- proc.time()
        BP_model <- nnet(data.train$FID~.,data=data.train[,-c(1:7)],size=2,decay=5e-4,maxit=200,rang=0.1,MaxNWts=10000)
        predictions_blank_nnet <- predict(BP_model,data.test,type = "class") %>% lvls_expand(.,varietyname_label)
        table_blank_nnet <- table(data.test[,2],predictions_blank_nnet)
        ratio_blank_nnet <- sum(diag(table_blank_nnet))/sum(table_blank_nnet)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_blank_nnet[1] <- as.data.frame(predictions_blank_nnet)
        result_blank_nnet[2] <- as.data.frame(ratio_blank_nnet)
        result_blank_nnet[3] <- RunningTime[3][[1]]
        return(result_blank_nnet)
      }
      
      chi2_nnet <- function(data.train,data.test,varietyname_label){
        result_chi2_nnet <- list()
        StartTime <- proc.time()
        BP_model <- nnet(data.train$FID~.,data=data.train[,-c(1:7)],size=2,decay=5e-4,maxit=200,rang=0.1,MaxNWts=10000)
        predictions_chi2_nnet <- predict(BP_model,data.test,type = "class") %>% lvls_expand(.,varietyname_label)
        table_chi2_nnet <- table(data.test[,2],predictions_chi2_nnet)
        ratio_chi2_nnet <- sum(diag(table_chi2_nnet))/sum(table_chi2_nnet)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_chi2_nnet[1] <- as.data.frame(predictions_chi2_nnet)
        result_chi2_nnet[2] <- as.data.frame(ratio_chi2_nnet)
        result_chi2_nnet[3] <- RunningTime[3][[1]]
        return(result_chi2_nnet)
      }
      
      ETC_nnet <- function(data.train,data.test,varietyname_label){
        result_ETC_nnet <- list()
        StartTime <- proc.time()
        BP_model <- nnet(data.train$FID~.,data=data.train[,-c(1:7)],size=2,decay=5e-4,maxit=200,rang=0.1,MaxNWts=10000)
        predictions_ETC_nnet <- predict(BP_model,data.test,type = "class") %>% lvls_expand(.,varietyname_label)
        table_ETC_nnet <- table(data.test[,2],predictions_ETC_nnet)
        ratio_ETC_nnet <- sum(diag(table_ETC_nnet))/sum(table_ETC_nnet)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_ETC_nnet[1] <- as.data.frame(predictions_ETC_nnet)
        result_ETC_nnet[2] <- as.data.frame(ratio_ETC_nnet)
        result_ETC_nnet[3] <- RunningTime[3][[1]]
        return(result_ETC_nnet)
      }
      
      f_classif_nnet <- function(data.train,data.test,varietyname_label){
        result_f_classif_nnet <- list()
        StartTime <- proc.time()
        BP_model <- nnet(data.train$FID~.,data=data.train[,-c(1:7)],size=2,decay=5e-4,maxit=200,rang=0.1,MaxNWts=10000)
        predictions_f_classif_nnet <- predict(BP_model,data.test,type = "class") %>% lvls_expand(.,varietyname_label)
        table_f_classif_nnet <- table(data.test[,2],predictions_f_classif_nnet)
        ratio_f_classif_nnet <- sum(diag(table_f_classif_nnet))/sum(table_f_classif_nnet)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_f_classif_nnet[1] <- as.data.frame(predictions_f_classif_nnet)
        result_f_classif_nnet[2] <- as.data.frame(ratio_f_classif_nnet)
        result_f_classif_nnet[3] <- RunningTime[3][[1]]
        return(result_f_classif_nnet)
      }
      
      info_classif_nnet <- function(data.train,data.test,varietyname_label){
        result_info_classif_nnet <- list()
        StartTime <- proc.time()
        BP_model <- nnet(data.train$FID~.,data=data.train[,-c(1:7)],size=2,decay=5e-4,maxit=200,rang=0.1,MaxNWts=10000)
        predictions_info_classif_nnet <- predict(BP_model,data.test,type = "class") %>% lvls_expand(.,varietyname_label)
        table_info_classif_nnet <- table(data.test[,2],predictions_info_classif_nnet)
        ratio_info_classif_nnet <- sum(diag(table_info_classif_nnet))/sum(table_info_classif_nnet)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_info_classif_nnet[1] <- as.data.frame(predictions_info_classif_nnet)
        result_info_classif_nnet[2] <- as.data.frame(ratio_info_classif_nnet)
        result_info_classif_nnet[3] <- RunningTime[3][[1]]
        return(result_info_classif_nnet)
      }
      
      LR_nnet <- function(data.train,data.test,varietyname_label){
        result_LR_nnet <- list()
        StartTime <- proc.time()
        BP_model <- nnet(data.train$FID~.,data=data.train[,-c(1:7)],size=2,decay=5e-4,maxit=200,rang=0.1,MaxNWts=10000)
        predictions_LR_nnet <- predict(BP_model,data.test,type = "class") %>% lvls_expand(.,varietyname_label)
        table_LR_nnet <- table(data.test[,2],predictions_LR_nnet)
        ratio_LR_nnet <- sum(diag(table_LR_nnet))/sum(table_LR_nnet)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_LR_nnet[1] <- as.data.frame(predictions_LR_nnet)
        result_LR_nnet[2] <- as.data.frame(ratio_LR_nnet)
        result_LR_nnet[3] <- RunningTime[3][[1]]
        return(result_LR_nnet)
      }
      
      RF_nnet <- function(data.train,data.test,varietyname_label){
        result_RF_nnet <- list()
        StartTime <- proc.time()
        BP_model <- nnet(data.train$FID~.,data=data.train[,-c(1:7)],size=2,decay=5e-4,maxit=200,rang=0.1,MaxNWts=10000)
        predictions_RF_nnet <- predict(BP_model,data.test,type = "class") %>% lvls_expand(.,varietyname_label)
        table_RF_nnet <- table(data.test[,2],predictions_RF_nnet)
        ratio_RF_nnet <- sum(diag(table_RF_nnet))/sum(table_RF_nnet)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_RF_nnet[1] <- as.data.frame(predictions_RF_nnet)
        result_RF_nnet[2] <- as.data.frame(ratio_RF_nnet)
        result_RF_nnet[3] <- RunningTime[3][[1]]
        return(result_RF_nnet)
      }
      XGBoost_nnet <- function(data.train,data.test,varietyname_label){
        result_XGBoost_nnet <- list()
        StartTime <- proc.time()
        BP_model <- nnet(data.train$FID~.,data=data.train[,-c(1:7)],size=2,decay=5e-4,maxit=200,rang=0.1,MaxNWts=10000)
        predictions_XGBoost_nnet <- predict(BP_model,data.test,type = "class") %>% lvls_expand(.,varietyname_label)
        table_XGBoost_nnet <- table(data.test[,2],predictions_XGBoost_nnet)
        ratio_XGBoost_nnet <- sum(diag(table_XGBoost_nnet))/sum(table_XGBoost_nnet)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_XGBoost_nnet[1] <- as.data.frame(predictions_XGBoost_nnet)
        result_XGBoost_nnet[2] <- as.data.frame(ratio_XGBoost_nnet)
        result_XGBoost_nnet[3] <- RunningTime[3][[1]]
        return(result_XGBoost_nnet)
      }
      
      LD_nnet <- function(data.train,data.test,varietyname_label){
        result_LD_nnet <- list()
        StartTime <- proc.time()
        BP_model <- nnet(data.train$FID~.,data=data.train[,-c(1:7)],size=2,decay=5e-4,maxit=200,rang=0.1,MaxNWts=10000)
        predictions_LD_nnet <- predict(BP_model,data.test,type = "class") %>% lvls_expand(.,varietyname_label)
        table_LD_nnet <- table(data.test[,2],predictions_LD_nnet)
        ratio_LD_nnet <- sum(diag(table_LD_nnet))/sum(table_LD_nnet)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_LD_nnet[1] <- as.data.frame(predictions_LD_nnet)
        result_LD_nnet[2] <- as.data.frame(ratio_LD_nnet)
        result_LD_nnet[3] <- RunningTime[3][[1]]
        return(result_LD_nnet)
      }
      
      
      
      
      
      blank_MLR <- function(data.train,data.test,varietyname_label){
        result_blank_MLR <- list()
        StartTime <- proc.time()
        MLR <- multinom(formula = data.train$FID~.,data = data.train[,-c(1:7)],MaxNWts = 143541315)
        predictions_blank_MLR <- predict(object = MLR,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_blank_MLR <- table(predictions_blank_MLR,data.test[,2])
        ratio_blank_MLR <- sum(diag(table_blank_MLR))/sum(table_blank_MLR)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_blank_MLR[1] <- as.data.frame(predictions_blank_MLR)
        result_blank_MLR[2] <- as.data.frame(ratio_blank_MLR)
        result_blank_MLR[3] <- RunningTime[3][[1]]
        return(result_blank_MLR)
      }
      
      chi2_MLR <- function(data.train,data.test,varietyname_label){
        result_chi2_MLR <- list()
        StartTime <- proc.time()
        MLR <- multinom(formula = data.train$FID~.,data = data.train[,-c(1:7)],MaxNWts = 143541315)
        predictions_chi2_MLR <- predict(object = MLR,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_chi2_MLR <- table(predictions_chi2_MLR,data.test[,2])
        ratio_chi2_MLR <- sum(diag(table_chi2_MLR))/sum(table_chi2_MLR)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_chi2_MLR[1] <- as.data.frame(predictions_chi2_MLR)
        result_chi2_MLR[2] <- as.data.frame(ratio_chi2_MLR)
        result_chi2_MLR[3] <- RunningTime[3][[1]]
        return(result_chi2_MLR)
      }
      
      ETC_MLR <- function(data.train,data.test,varietyname_label){
        result_ETC_MLR <- list()
        StartTime <- proc.time()
        MLR <- multinom(formula = data.train$FID~.,data = data.train[,-c(1:7)],MaxNWts = 143541315)
        predictions_ETC_MLR <- predict(object = MLR,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_ETC_MLR <- table(predictions_ETC_MLR,data.test[,2])
        ratio_ETC_MLR <- sum(diag(table_ETC_MLR))/sum(table_ETC_MLR)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_ETC_MLR[1] <- as.data.frame(predictions_ETC_MLR)
        result_ETC_MLR[2] <- as.data.frame(ratio_ETC_MLR)
        result_ETC_MLR[3] <- RunningTime[3][[1]]
        return(result_ETC_MLR)
      }
      
      f_classif_MLR <- function(data.train,data.test,varietyname_label){
        result_f_classif_MLR <- list()
        StartTime <- proc.time()
        MLR <- multinom(formula = data.train$FID~.,data = data.train[,-c(1:7)],MaxNWts = 143541315)
        predictions_f_classif_MLR <- predict(object = MLR,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_f_classif_MLR <- table(predictions_f_classif_MLR,data.test[,2])
        ratio_f_classif_MLR <- sum(diag(table_f_classif_MLR))/sum(table_f_classif_MLR)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_f_classif_MLR[1] <- as.data.frame(predictions_f_classif_MLR)
        result_f_classif_MLR[2] <- as.data.frame(ratio_f_classif_MLR)
        result_f_classif_MLR[3] <- RunningTime[3][[1]]
        return(result_f_classif_MLR)
      }
      
      info_classif_MLR <- function(data.train,data.test,varietyname_label){
        result_info_classif_MLR <- list()
        StartTime <- proc.time()
        MLR <- multinom(formula = data.train$FID~.,data = data.train[,-c(1:7)],MaxNWts = 143541315)
        predictions_info_classif_MLR <- predict(object = MLR,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_info_classif_MLR <- table(predictions_info_classif_MLR,data.test[,2])
        ratio_info_classif_MLR <- sum(diag(table_info_classif_MLR))/sum(table_info_classif_MLR)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_info_classif_MLR[1] <- as.data.frame(predictions_info_classif_MLR)
        result_info_classif_MLR[2] <- as.data.frame(ratio_info_classif_MLR)
        result_info_classif_MLR[3] <- RunningTime[3][[1]]
        return(result_info_classif_MLR)
      }
      
      LR_MLR <- function(data.train,data.test,varietyname_label){
        result_LR_MLR <- list()
        StartTime <- proc.time()
        MLR <- multinom(formula = data.train$FID~.,data = data.train[,-c(1:7)],MaxNWts = 143541315)
        predictions_LR_MLR <- predict(object = MLR,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_LR_MLR <- table(predictions_LR_MLR,data.test[,2])
        ratio_LR_MLR <- sum(diag(table_LR_MLR))/sum(table_LR_MLR)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_LR_MLR[1] <- as.data.frame(predictions_LR_MLR)
        result_LR_MLR[2] <- as.data.frame(ratio_LR_MLR)
        result_LR_MLR[3] <- RunningTime[3][[1]]
        return(result_LR_MLR)
      }
      
      RF_MLR <- function(data.train,data.test,varietyname_label){
        result_RF_MLR <- list()
        StartTime <- proc.time()
        MLR <- multinom(formula = data.train$FID~.,data = data.train[,-c(1:7)],MaxNWts = 143541315)
        predictions_RF_MLR <- predict(object = MLR,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_RF_MLR <- table(predictions_RF_MLR,data.test[,2])
        ratio_RF_MLR <- sum(diag(table_RF_MLR))/sum(table_RF_MLR)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_RF_MLR[1] <- as.data.frame(predictions_RF_MLR)
        result_RF_MLR[2] <- as.data.frame(ratio_RF_MLR)
        result_RF_MLR[3] <- RunningTime[3][[1]]
        return(result_RF_MLR)
      }
      
      XGBoost_MLR <- function(data.train,data.test,varietyname_label){
        result_XGBoost_MLR <- list()
        StartTime <- proc.time()
        MLR <- multinom(formula = data.train$FID~.,data = data.train[,-c(1:7)],MaxNWts = 143541315)
        predictions_XGBoost_MLR <- predict(object = MLR,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_XGBoost_MLR <- table(predictions_XGBoost_MLR,data.test[,2])
        ratio_XGBoost_MLR <- sum(diag(table_XGBoost_MLR))/sum(table_XGBoost_MLR)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_XGBoost_MLR[1] <- as.data.frame(predictions_XGBoost_MLR)
        result_XGBoost_MLR[2] <- as.data.frame(ratio_XGBoost_MLR)
        result_XGBoost_MLR[3] <- RunningTime[3][[1]]
        return(result_XGBoost_MLR)
      }
      
      LD_MLR <- function(data.train,data.test,varietyname_label){
        result_LD_MLR <- list()
        StartTime <- proc.time()
        MLR <- multinom(formula = data.train$FID~.,data = data.train[,-c(1:7)],MaxNWts = 143541315)
        predictions_LD_MLR <- predict(object = MLR,newdata = data.test[,-c(1:7)]) %>% lvls_expand(.,varietyname_label)
        table_LD_MLR <- table(predictions_LD_MLR,data.test[,2])
        ratio_LD_MLR <- sum(diag(table_LD_MLR))/sum(table_LD_MLR)
        EndTime <- proc.time()
        RunningTime <- EndTime - StartTime
        result_LD_MLR[1] <- as.data.frame(predictions_LD_MLR)
        result_LD_MLR[2] <- as.data.frame(ratio_LD_MLR)
        result_LD_MLR[3] <- RunningTime[3][[1]]
        return(result_LD_MLR)
      }
      
      # Machine Learning --------------------------------------------------------
      
      
      Blank_knn <- blank_knn(train.blank,testdata_all,varietyname_label)
      XGBoost_knn <- XGBoost_knn(train.XGBoost,testdata_all,varietyname_label)
      RF_knn <- RF_knn(train.RF,testdata_all,varietyname_label)
      LR_knn <- LR_knn(train.LR,testdata_all,varietyname_label)
      info_classif_knn <- info_classif_knn(train.info_classif,testdata_all,varietyname_label)
      f_classif_knn <- f_classif_knn(train.f_classif,testdata_all,varietyname_label)
      ETC_knn <- ETC_knn(train.ETC,testdata_all,varietyname_label)
      chi2_knn <- chi2_knn(train.chi2,testdata_all,varietyname_label)
      LD_knn <- LD_knn(train.LD,testdata_all,varietyname_label)
      print("KNN Finish")
      Blank_RF <- blank_RF(train.blank,testdata_all,varietyname_label)
      XGBoost_RF <- XGBoost_RF(train.XGBoost,testdata_all,varietyname_label)
      RF_RF <- RF_RF(train.RF,testdata_all,varietyname_label)
      LR_RF <- LR_RF(train.LR,testdata_all,varietyname_label)
      info_classif_RF <- info_classif_RF(train.info_classif,testdata_all,varietyname_label)
      f_classif_RF <- f_classif_RF(train.f_classif,testdata_all,varietyname_label)
      ETC_RF <- ETC_RF(train.ETC,testdata_all,varietyname_label)
      chi2_RF <- chi2_RF(train.chi2,testdata_all,varietyname_label)
      LD_RF <- LD_RF(train.LD,testdata_all,varietyname_label)
      print("RF Finish")
      Blank_NB <- blank_NB(train.blank,testdata_all,varietyname_label)
      LD_NB <- LD_NB(train.LD,testdata_all,varietyname_label)
      XGBoost_NB <- XGBoost_NB(train.XGBoost,testdata_all,varietyname_label)
      RF_NB <- RF_NB(train.RF,testdata_all,varietyname_label)
      LR_NB <- LR_NB(train.LR,testdata_all,varietyname_label)
      info_classif_NB <- info_classif_NB(train.info_classif,testdata_all,varietyname_label)
      f_classif_NB <- f_classif_NB(train.f_classif,testdata_all,varietyname_label)
      ETC_NB <- ETC_NB(train.ETC,testdata_all,varietyname_label)
      chi2_NB <- chi2_NB(train.chi2,testdata_all,varietyname_label)
      print("NB Finish")
      Blank_SVM <- blank_SVM(train.blank,testdata_all,varietyname_label)
      LD_SVM <- LD_SVM(train.LD,testdata_all,varietyname_label)
      XGBoost_SVM <- XGBoost_SVM(train.XGBoost,testdata_all,varietyname_label)
      RF_SVM <- RF_SVM(train.RF,testdata_all,varietyname_label)
      LR_SVM <- LR_SVM(train.LR,testdata_all,varietyname_label)
      info_classif_SVM <- info_classif_SVM(train.info_classif,testdata_all,varietyname_label)
      f_classif_SVM <- f_classif_SVM(train.f_classif,testdata_all,varietyname_label)
      ETC_SVM <- ETC_SVM(train.ETC,testdata_all,varietyname_label)
      chi2_SVM <- chi2_SVM(train.chi2,testdata_all,varietyname_label)
      print("SVM Finish")
      Blank_C50 <- blank_C50(train.blank,testdata_all,varietyname_label)
      LD_C50 <- LD_C50(train.LD,testdata_all,varietyname_label)
      XGBoost_C50 <- XGBoost_C50(train.XGBoost,testdata_all,varietyname_label)
      RF_C50 <- RF_C50(train.RF,testdata_all,varietyname_label)
      LR_C50 <- LR_C50(train.LR,testdata_all,varietyname_label)
      info_classif_C50 <- info_classif_C50(train.info_classif,testdata_all,varietyname_label)
      f_classif_C50 <- f_classif_C50(train.f_classif,testdata_all,varietyname_label)
      ETC_C50 <- ETC_C50(train.ETC,testdata_all,varietyname_label)
      chi2_C50 <- chi2_C50(train.chi2,testdata_all,varietyname_label)
      print("C50 Finish")
      Blank_nnet <- blank_nnet(data.train = train.blank,data.test = testdata_all,varietyname_label = varietyname_label)
      LD_nnet <- LD_nnet(train.LD,testdata_all,varietyname_label)
      XGBoost_nnet <- XGBoost_nnet(train.XGBoost,testdata_all,varietyname_label)
      RF_nnet <- RF_nnet(train.RF,testdata_all,varietyname_label)
      LR_nnet <- LR_nnet(train.LR,testdata_all,varietyname_label)
      info_classif_nnet <- info_classif_nnet(train.info_classif,testdata_all,varietyname_label)
      f_classif_nnet <- f_classif_nnet(train.f_classif,testdata_all,varietyname_label)
      ETC_nnet <- ETC_nnet(train.ETC,testdata_all,varietyname_label)
      chi2_nnet <- chi2_nnet(train.chi2,testdata_all,varietyname_label)
      print("nnet Finish")
      Blank_MLR <- blank_MLR(train.blank,testdata_all,varietyname_label)
      LD_MLR <- LD_MLR(train.LD,testdata_all,varietyname_label)
      XGBoost_MLR <- XGBoost_MLR(train.XGBoost,testdata_all,varietyname_label)
      RF_MLR <- RF_MLR(train.RF,testdata_all,varietyname_label)
      LR_MLR <- LR_MLR(train.LR,testdata_all,varietyname_label)
      info_classif_MLR <- info_classif_MLR(train.info_classif,testdata_all,varietyname_label)
      f_classif_MLR <- f_classif_MLR(train.f_classif,testdata_all,varietyname_label)
      ETC_MLR <- ETC_MLR(train.ETC,testdata_all,varietyname_label)
      chi2_MLR <- chi2_MLR(train.chi2,testdata_all,varietyname_label)
      print("MLR Finish")
      # Result_MachineLearning --------------------------------------------------
      
      
      # Blank_nnet[2];LD_nnet[2];XGBoost_nnet[2];RF_nnet[2];LR_nnet[2];info_classif_nnet[2];f_classif_nnet[2];ETC_nnet[2];chi2_nnet[2];
      # 
      # Blank_C50[2];LD_C50[2];XGBoost_C50[2];RF_C50[2];LR_C50[2];info_classif_C50[2];f_classif_C50[2];ETC_C50[2];chi2_C50[2];
      # 
      # Blank_knn[2];LD_knn[2];XGBoost_knn[2];RF_knn[2];LR_knn[2];info_classif_knn[2];f_classif_knn[2];ETC_knn[2];chi2_knn[2];
      # 
      # Blank_NB[2];LD_NB[2];XGBoost_NB[2];RF_NB[2];LR_NB[2];info_classif_NB[2];f_classif_NB[2];ETC_NB[2];chi2_NB[2];
      # 
      # Blank_RF[2];LD_RF[2];XGBoost_RF[2];RF_RF[2];LR_RF[2];info_classif_RF[2];f_classif_RF[2];ETC_RF[2];chi2_RF[2];
      # 
      # Blank_SVM[2];LD_SVM[2];XGBoost_SVM[2];RF_SVM[2];LR_SVM[2];info_classif_SVM[2];f_classif_SVM[2];ETC_SVM[2];chi2_SVM[2];
      # 
      # Blank_MLR[2];LD_MLR[2];XGBoost_MLR[2];RF_MLR[2];LR_MLR[2];info_classif_MLR[2];f_classif_MLR[2];ETC_MLR[2];chi2_MLR[2];
      
      # Ensemble Learning 01 -------------------------------------------------------
      
      print(paste("begin:","Ensemble Learning 01"))
      FS <- list()
      for (i in 1:9) {
        FS[i] <- get(paste("stacking","model01",i,sep = "_"))[[1]]
        if (FS[i] == "chi2") {
          assign(paste("stacking",i,"train",sep = "_"),train.chi2)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.chi2$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",i,"train",j,sep = "_"),train.chi2[k_fold[[j]],])#
            assign(paste("stacking",i,"test",j,sep = "_"),train.chi2[-k_fold[[j]],])#
          }
        }
        if (FS[i] == "ETC") {
          # train.ETC$FID <- as.factor(train.ETC$FID)
          assign(paste("stacking",i,"train",sep = "_"),train.ETC)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.ETC$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",i,"train",j,sep = "_"),train.ETC[k_fold[[j]],])#
            assign(paste("stacking",i,"test",j,sep = "_"),train.ETC[-k_fold[[j]],])#
          }
        }
        if (FS[i] == "f_classif") {
          assign(paste("stacking",i,"train",sep = "_"),train.f_classif)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.f_classif$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",i,"train",j,sep = "_"),train.f_classif[k_fold[[j]],])#
            assign(paste("stacking",i,"test",j,sep = "_"),train.f_classif[-k_fold[[j]],])#
          }
        }
        if (FS[i] == "info_classif") {
          assign(paste("stacking",i,"train",sep = "_"),train.info_classif)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.info_classif$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",i,"train",j,sep = "_"),train.info_classif[k_fold[[j]],])#
            assign(paste("stacking",i,"test",j,sep = "_"),train.info_classif[-k_fold[[j]],])#
          }
        }
        if (FS[i] == "LR") {
          assign(paste("stacking",i,"train",sep = "_"),train.LR)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.LR$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",i,"train",j,sep = "_"),train.LR[k_fold[[j]],])#
            assign(paste("stacking",i,"test",j,sep = "_"),train.LR[-k_fold[[j]],])#
          }
        }
        if (FS[i] == "RF") {
          assign(paste("stacking",i,"train",sep = "_"),train.RF)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.RF$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",i,"train",j,sep = "_"),train.RF[k_fold[[j]],])#
            assign(paste("stacking",i,"test",j,sep = "_"),train.RF[-k_fold[[j]],])#
          }
        }
        if (FS[i] == "XGBoost") {
          assign(paste("stacking",i,"train",sep = "_"),train.XGBoost)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.XGBoost$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",i,"train",j,sep = "_"),train.XGBoost[k_fold[[j]],])#
            assign(paste("stacking",i,"test",j,sep = "_"),train.XGBoost[-k_fold[[j]],])#
          }
        }
        if (FS[i] == "LD") {
          assign(paste("stacking",i,"train",sep = "_"),train.LD)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.LD$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",i,"train",j,sep = "_"),train.LD[k_fold[[j]],])#
            assign(paste("stacking",i,"test",j,sep = "_"),train.LD[-k_fold[[j]],])#
          }
        }
        if (FS[i] == "blank") {
          assign(paste("stacking",i,"train",sep = "_"),train.blank)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.blank$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",i,"train",j,sep = "_"),train.blank[k_fold[[j]],])#
            assign(paste("stacking",i,"test",j,sep = "_"),train.blank[-k_fold[[j]],])#
          }
        }
      }
      
      for (i in 1:9) {
        print(paste0("i=",i))
        assign(paste0("StackingParameter_",FS[i]),data.frame(MetaModel = NA,
                                                             FeatureSelection = NA,
                                                             Accuracy = NA,
                                                             BaseModel = NA,
                                                             BaseModel_Num = NA,
                                                             Distance_KNN = NA,
                                                             K_KNN = NA,
                                                             stringsAsFactors = F))
        ML <- list()
        train_label_stacking <- train_label_stacking[order(train_label_stacking$NUM),]
        stacking_train <- NULL %>% rbind(.,train_label_stacking[,c(1,2)])
        stacking_train <- NULL %>% rbind(.,train_label_stacking[,c(1,2)])
        stacking_test <- NULL %>% cbind(.,testdata_all[,1])
        stacking_test <- NULL %>% cbind(.,testdata_all[,1])
        assign(paste("StartTime01",FS[i],sep = "_"),proc.time())
        for (fm in 1:7) {
          ML[fm] <- strsplit(get(paste("stacking","model01",i,sep = "_"))[[1+fm]],split = "-")[[1]][2]
          print(paste0("fm=",fm));print(paste0("ML[fm]=",ML[fm]));print(paste0("FS[i]=",FS[i]))
          if (ML[fm] == "knn") {
            assign(paste("predictions",FS[i],ML[fm],sep = "_"),NULL)
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),NULL)
            assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),NULL)
            for (j in 1:fold) {
              # print(paste("new_feature",FS[i],ML[fm],sep = "_"));print(get(paste("new_feature",FS[i],ML[fm],sep = "_")))
              assign(paste(FS[i],ML[fm],j,sep = "_"),
                     kknn(get(paste("stacking",i,"train",j,sep = "_"))$FID~.,
                          get(paste("stacking",i,"train",j,sep = "_"))[,-c(1:7)],
                          get(paste("stacking",i,"test",j,sep = "_"))[,-c(1:7)],
                          distance = 2,k= 2,kernel= "triangular"))#
              assign(paste(FS[i],ML[fm],"test",j,sep = "_"),
                     kknn(get(paste("stacking",i,"train",j,sep = "_"))$FID~.,
                          get(paste("stacking",i,"train",j,sep = "_"))[,-c(1:7)], 
                          testdata_all[,-c(1:7)], 
                          distance = 2,k= 2,kernel= "triangular"))
              assign(paste("predictions",FS[i],ML[fm],j,sep = "_"),
                     predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                             newdata = get(paste("stacking",i,"test",j,sep = "_"))[,-c(1:7)]) %>% as.character(.))
              assign(paste("new_feature",FS[i],ML[fm],j,sep = "_"),
                     cbind(get(paste("stacking",i,"test",j,sep = "_"))[,1],
                           get(paste("predictions",FS[i],ML[fm],j,sep = "_"))))
              assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                     rbind(get(paste("new_feature",FS[i],ML[fm],sep = "_")),
                           get(paste("new_feature",FS[i],ML[fm],j,sep = "_"))))
              assign(paste("predictions",FS[i],ML[fm],"test",j,sep = "_"),
                     predict(object = get(paste(FS[i],ML[fm],"test",j,sep = "_")),
                             newdata = testdata_all[,-c(1:7)],type = "prob"))
              if (j==1) {
                assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))
              }else{assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),
                           get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))+get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))}
            }
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   as.data.frame(get(paste("new_feature",FS[i],ML[fm],sep = "_"))))
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   get(paste("new_feature",FS[i],ML[fm],sep = "_"))[order(as.numeric(get(paste("new_feature",FS[i],ML[fm],sep = "_"))[,1])),])
            assign(paste("vote",FS[i],ML[fm],sep = "_"),
                   lapply(X = 1:nrow(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))),FUN = function(x){
                     max_labelnum <- which(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]==max(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]))
                     if (length(max_labelnum) != 1) {return(sample(x = max_labelnum,size = 1) %>% names(.))}
                     else{return(names(max_labelnum))}
                   }) %>% ldply(.data = .))
          }
          if (ML[fm] == "nb") {
            assign(paste("predictions",FS[i],ML[fm],sep = "_"),NULL)
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),NULL)
            assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),NULL)
            for (j in 1:fold) {
              assign(paste(FS[i],ML[fm],j,sep = "_"),
                     naiveBayes(y = get(paste("stacking",i,"train",j,sep = "_"))$FID,
                                x = get(paste("stacking",i,"train",j,sep = "_"))[,-c(1,3,4,5,6,7)]))
              assign(paste("predictions",FS[i],ML[fm],j,sep = "_"),
                     predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                             newdata = get(paste("stacking",i,"test",j,sep = "_"))[,-c(1:7)]) %>% as.character(.))
              assign(paste("new_feature",FS[i],ML[fm],j,sep = "_"),
                     cbind(get(paste("stacking",i,"test",j,sep = "_"))[,1],
                           get(paste("predictions",FS[i],ML[fm],j,sep = "_"))))
              assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                     rbind(get(paste("new_feature",FS[i],ML[fm],sep = "_")),
                           get(paste("new_feature",FS[i],ML[fm],j,sep = "_"))))
              assign(paste("predictions",FS[i],ML[fm],"test",j,sep = "_"),
                     predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                             newdata = testdata_all[,-c(1:7)],type = "raw"))
              if (j==1) {
                assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))
              }else{assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),
                           get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))+get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))}
            }
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   as.data.frame(get(paste("new_feature",FS[i],ML[fm],sep = "_"))))
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   get(paste("new_feature",FS[i],ML[fm],sep = "_"))[order(as.numeric(get(paste("new_feature",FS[i],ML[fm],sep = "_"))[,1])),])
            assign(paste("vote",FS[i],ML[fm],sep = "_"),
                   lapply(X = 1:nrow(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))),FUN = function(x){
                     max_labelnum <- which(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]==max(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]))
                     if (length(max_labelnum) != 1) {return(sample(x = max_labelnum,size = 1) %>% names(.))}
                     else{return(names(max_labelnum))}
                   }) %>% ldply(.data = .))
          }
          if (ML[fm] == "rf") {
            assign(paste("predictions",FS[i],ML[fm],sep = "_"),NULL)
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),NULL)
            assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),NULL)
            for (j in 1:fold) {
              assign(paste(FS[i],ML[fm],j,sep = "_"),
                     randomForest(y = get(paste("stacking",i,"train",j,sep = "_"))$FID,
                                  x = get(paste("stacking",i,"train",j,sep = "_"))[,-c(1:7)]))
              assign(paste("predictions",FS[i],ML[fm],j,sep = "_"),
                     predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                             newdata = get(paste("stacking",i,"test",j,sep = "_"))[,-c(1:7)]) %>% as.character(.))
              assign(paste("new_feature",FS[i],ML[fm],j,sep = "_"),
                     cbind(get(paste("stacking",i,"test",j,sep = "_"))[,1],
                           get(paste("predictions",FS[i],ML[fm],j,sep = "_"))))
              assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                     rbind(get(paste("new_feature",FS[i],ML[fm],sep = "_")),
                           get(paste("new_feature",FS[i],ML[fm],j,sep = "_"))))
              assign(paste("predictions",FS[i],ML[fm],"test",j,sep = "_"),
                     predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                             newdata = testdata_all[,-c(1:7)],type = "prob"))
              if (j==1) {
                assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))
              }else{assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),
                           get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))+get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))}
            }
            print("Finish CrossValidation")
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   as.data.frame(get(paste("new_feature",FS[i],ML[fm],sep = "_"))))
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   get(paste("new_feature",FS[i],ML[fm],sep = "_"))[order(as.numeric(get(paste("new_feature",FS[i],ML[fm],sep = "_"))[,1])),])
            print("Begin RF Vote")
            assign(paste("vote",FS[i],ML[fm],sep = "_"),
                   lapply(X = 1:nrow(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))),FUN = function(x){
                     max_labelnum <- which(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]==max(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]))
                     if (length(max_labelnum) != 1) {return(sample(x = max_labelnum,size = 1) %>% names(.))}
                     else{return(names(max_labelnum))}
                   }) %>% ldply(.data = .))
          }
          if (ML[fm] == "svm") {
            assign(paste("predictions",FS[i],ML[fm],sep = "_"),NULL)
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),NULL)
            assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),NULL)
            for (j in 1:fold) {
              assign(paste(FS[i],ML[fm],j,sep = "_"),
                     svm(get(paste("stacking",i,"train",j,sep = "_"))$FID~.,
                         get(paste("stacking",i,"train",j,sep = "_"))[,-c(1:7)],
                         kernel = "linear",cost = 1,
                         gamma = 1/ncol(get(paste("stacking",i,"train",j,sep = "_"))),probability=TRUE))
              assign(paste("predictions",FS[i],ML[fm],j,sep = "_"),
                     predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                             newdata = get(paste("stacking",i,"test",j,sep = "_"))[,-c(1:7)]) %>% as.character(.))
              assign(paste("new_feature",FS[i],ML[fm],j,sep = "_"),
                     cbind(get(paste("stacking",i,"test",j,sep = "_"))[,1],
                           get(paste("predictions",FS[i],ML[fm],j,sep = "_"))))
              assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                     rbind(get(paste("new_feature",FS[i],ML[fm],sep = "_")),
                           get(paste("new_feature",FS[i],ML[fm],j,sep = "_"))))
              assign(paste("predictions",FS[i],ML[fm],"test",j,sep = "_"),
                     predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                             newdata = testdata_all[,-c(1:7)], probability=TRUE) %>% attr(., "probabilities"))
              if (j==1) {
                assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))
              }else{assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),
                           get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))+get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))}
            }
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   as.data.frame(get(paste("new_feature",FS[i],ML[fm],sep = "_"))))
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   get(paste("new_feature",FS[i],ML[fm],sep = "_"))[order(as.numeric(get(paste("new_feature",FS[i],ML[fm],sep = "_"))[,1])),])
            assign(paste("vote",FS[i],ML[fm],sep = "_"),
                   lapply(X = 1:nrow(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))),FUN = function(x){
                     max_labelnum <- which(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]==max(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]))
                     if (length(max_labelnum) != 1) {return(sample(x = max_labelnum,size = 1) %>% names(.))}
                     else{return(names(max_labelnum))}
                   }) %>% ldply(.data = .))
          }
          if (ML[fm] == "c50") {
            assign(paste("predictions",FS[i],ML[fm],sep = "_"),NULL)
            assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),NULL)
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),NULL)
            for (j in 1:fold) {
              assign(paste(FS[i],ML[fm],j,sep = "_"),
                     C5.0(y = get(paste("stacking",i,"train",j,sep = "_"))$FID,
                          x = get(paste("stacking",i,"train",j,sep = "_"))[,-c(1:7)],
                          trials=20))
              assign(paste("predictions",FS[i],ML[fm],j,sep = "_"),
                     predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                             newdata = get(paste("stacking",i,"test",j,sep = "_"))[,-c(1:7)]) %>% as.character(.))
              assign(paste("new_feature",FS[i],ML[fm],j,sep = "_"),
                     cbind(get(paste("stacking",i,"test",j,sep = "_"))[,1],
                           get(paste("predictions",FS[i],ML[fm],j,sep = "_"))))
              assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                     rbind(get(paste("new_feature",FS[i],ML[fm],sep = "_")),
                           get(paste("new_feature",FS[i],ML[fm],j,sep = "_"))))
              assign(paste("predictions",FS[i],ML[fm],"test",j,sep = "_"),
                     predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                             newdata = testdata_all[,-c(1:7)],type = "prob"))
              if (j==1) {
                assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))
              }else{assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),
                           get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))+get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))}
            }
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   as.data.frame(get(paste("new_feature",FS[i],ML[fm],sep = "_"))))
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   get(paste("new_feature",FS[i],ML[fm],sep = "_"))[order(as.numeric(get(paste("new_feature",FS[i],ML[fm],sep = "_"))[,1])),])
            assign(paste("vote",FS[i],ML[fm],sep = "_"),
                   lapply(X = 1:nrow(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))),FUN = function(x){
                     max_labelnum <- which(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]==max(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]))
                     if (length(max_labelnum) != 1) {return(sample(x = max_labelnum,size = 1) %>% names(.))}
                     else{return(names(max_labelnum))}
                   }) %>% ldply(.data = .))
          }
          if (ML[fm] == "nnet") {
            assign(paste("predictions",FS[i],ML[fm],sep = "_"),NULL)
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),NULL)
            assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),NULL)
            for (j in 1:fold) {
              assign(paste(FS[i],ML[fm],j,sep = "_"),
                     nnet(get(paste("stacking",i,"train",j,sep = "_"))$FID~.,
                          data=get(paste("stacking",i,"train",j,sep = "_"))[,-c(1:7)],
                          size=2,decay=5e-4,maxit=200,rang=0.1,MaxNWts=10000))
              assign(paste("predictions",FS[i],ML[fm],j,sep = "_"),
                     predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                             newdata = get(paste("stacking",i,"test",j,sep = "_"))[,-c(1:7)],type = "class"))#
              assign(paste("new_feature",FS[i],ML[fm],j,sep = "_"),
                     cbind(get(paste("stacking",i,"test",j,sep = "_"))[,1],
                           get(paste("predictions",FS[i],ML[fm],j,sep = "_"))))
              assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                     rbind(get(paste("new_feature",FS[i],ML[fm],sep = "_")),
                           get(paste("new_feature",FS[i],ML[fm],j,sep = "_"))))
              assign(paste("predictions",FS[i],ML[fm],"test",j,sep = "_"),
                     predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                             newdata = testdata_all[,-c(1:7)],type = "raw"))#
              if (j==1) {
                assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))
              }else{assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),
                           get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))+get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))}
            }
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   as.data.frame(get(paste("new_feature",FS[i],ML[fm],sep = "_"))))
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   get(paste("new_feature",FS[i],ML[fm],sep = "_"))[order(as.numeric(get(paste("new_feature",FS[i],ML[fm],sep = "_"))[,1])),])
            assign(paste("vote",FS[i],ML[fm],sep = "_"),
                   lapply(X = 1:nrow(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))),FUN = function(x){
                     max_labelnum <- which(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]==max(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]))
                     if (length(max_labelnum) != 1) {return(sample(x = max_labelnum,size = 1) %>% names(.))}
                     else{return(names(max_labelnum))}
                   }) %>% ldply(.data = .))
          }
          if (ML[fm] == "mlr") {
            assign(paste("predictions",FS[i],ML[fm],sep = "_"),NULL)
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),NULL)
            assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),NULL)
            for (j in 1:fold) {
              print(paste("j=",j))
              assign(paste(FS[i],ML[fm],j,sep = "_"),
                     multinom(formula = get(paste("stacking",i,"train",j,sep = "_"))$FID~.,
                              data = get(paste("stacking",i,"train",j,sep = "_"))[,-c(1:7)],MaxNWts = 143541315))
              assign(paste("predictions",FS[i],ML[fm],j,sep = "_"),
                     predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                             newdata = get(paste("stacking",i,"test",j,sep = "_"))[,-c(1:7)]) %>% as.character(.))
              assign(paste("new_feature",FS[i],ML[fm],j,sep = "_"),
                     cbind(get(paste("stacking",i,"test",j,sep = "_"))[,1],
                           get(paste("predictions",FS[i],ML[fm],j,sep = "_"))))
              assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                     rbind(get(paste("new_feature",FS[i],ML[fm],sep = "_")),
                           get(paste("new_feature",FS[i],ML[fm],j,sep = "_"))))
              assign(paste("predictions",FS[i],ML[fm],"test",j,sep = "_"),
                     predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                             newdata = testdata_all[,-c(1:7)],type = "prob"))
              if (j==1) {
                assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))
              }else{assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),
                           get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))+get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))}
            }
            print("Finish CrossValidation")
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   as.data.frame(get(paste("new_feature",FS[i],ML[fm],sep = "_"))))
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   get(paste("new_feature",FS[i],ML[fm],sep = "_"))[order(as.numeric(get(paste("new_feature",FS[i],ML[fm],sep = "_"))[,1])),])
            assign(paste("vote",FS[i],ML[fm],sep = "_"),
                   lapply(X = 1:nrow(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))),FUN = function(x){
                     max_labelnum <- which(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]==max(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]))
                     if (length(max_labelnum) != 1) {return(sample(x = max_labelnum,size = 1) %>% names(.))}
                     else{return(names(max_labelnum))}
                   }) %>% ldply(.data = .))
          }
          stacking_train <- cbind(stacking_train,get(paste("new_feature",FS[i],ML[fm],sep = "_"))[,2]) %>% as.data.frame(.)
          colnumber_train <- ncol(stacking_train)
          colnames(stacking_train)[colnumber_train] <- ML[fm]
          stacking_test <- cbind(stacking_test,get(paste("vote",FS[i],ML[fm],sep = "_"))) %>% as.data.frame(.)
          colnumber_test <- ncol(stacking_test)
          colnames(stacking_test)[colnumber_test] <- ML[fm]
        }
        stacking_train$FID <- as.factor(stacking_train$FID)
        
        assign(paste0("stacking_train_",FS[i]),stacking_train)
        assign(paste0("stacking_test_",FS[i]),stacking_test)
        
        
        for (distance in 1:2) {
          for (k in 1:7) {
            assign(paste0("stacking_ML_KNN_",FS[i],"_",distance,"_",k),
                   tryCatch(
                     {
                       kknn(formula = get(paste0("stacking_train_",FS[i]))$FID~.,
                            train = get(paste0("stacking_train_",FS[i]))[,-c(1:2)],
                            test = get(paste0("stacking_test_",FS[i]))[,-1],
                            distance = distance,k=k,kernel= "triangular")
                     },
                     error = function(e){
                       return(NULL)
                     }
                   ))
            
            if (is.null(get(paste0("stacking_ML_KNN_",FS[i],"_",distance,"_",k)))) {
              next
            }else{
              assign(paste0("stacking_pre_KNN_",FS[i],"_",distance,"_",k),
                     predict(object = get(paste0("stacking_ML_KNN_",FS[i],"_",distance,"_",k)),
                             newdata = get(paste0("stacking_test_",FS[i]))[,-1]) %>% lvls_expand(.,varietyname_label))
              assign(paste0("table_stacking_KNN_",FS[i],"_",distance,"_",k),
                     table(get(paste0("stacking_pre_KNN_",FS[i],"_",distance,"_",k)),testdata_all$FID))
              assign(paste0("ratio_stacking_KNN_",FS[i],"_",distance,"_",k),
                     sum(diag(get(paste0("table_stacking_KNN_",FS[i],"_",distance,"_",k))))/sum(get(paste0("table_stacking_KNN_",FS[i],"_",distance,"_",k))))
              assign(paste0("StackingParameter_",FS[i]),rbind(get(paste0("StackingParameter_",FS[i])),
                                                              c("KNN",
                                                                FS[i],
                                                                get(paste0("ratio_stacking_KNN_",FS[i],"_",distance,"_",k)),
                                                                paste(colnames(get(paste0("stacking_train_",FS[i]))[,-c(1:2)]),collapse = "_"),
                                                                length(colnames(get(paste0("stacking_train_",FS[i]))[,-c(1:2)])),
                                                                distance,
                                                                k)))
            }
          }
        }
        assign(paste0("stacking_ML_RF_",FS[i]),
               randomForest(get(paste0("stacking_train_",FS[i]))$FID~.,
                            data = get(paste0("stacking_train_",FS[i]))[,-c(1:2)],importance=T))
        assign(paste0("stacking_pre_RF_",FS[i]),
               predict(object = get(paste0("stacking_ML_RF_",FS[i])),
                       newdata = get(paste0("stacking_test_",FS[i]))[,-1]) %>% lvls_expand(.,varietyname_label))
        assign(paste0("table_stacking_RF_",FS[i]),
               table(get(paste0("stacking_pre_RF_",FS[i])),testdata_all$FID))
        assign(paste0("ratio_stacking_RF_",FS[i]),
               sum(diag(get(paste0("table_stacking_RF_",FS[i]))))/sum(get(paste0("table_stacking_RF_",FS[i]))))
        
        assign(paste0("StackingParameter_",FS[i]),rbind(get(paste0("StackingParameter_",FS[i])),
                                                        c("RF",
                                                          FS[i],
                                                          get(paste0("ratio_stacking_RF_",FS[i])),
                                                          paste(colnames(get(paste0("stacking_train_",FS[i]))[,-c(1:2)]),collapse = "_"),
                                                          length(colnames(get(paste0("stacking_train_",FS[i]))[,-c(1:2)])),
                                                          "",
                                                          "")))
        
        
        assign(paste0("StackingParameter_",FS[i]),na.omit(get(paste0("StackingParameter_",FS[i]))))
        assign(paste("ratio_stacking",FS[i],sep = "_"),max(get(paste0("StackingParameter_",FS[i]))$Accuracy))
        assign(paste0("MaxLine_",FS[i]),
               get(paste0("StackingParameter_",FS[i]))[which(get(paste0("StackingParameter_",FS[i]))$Accuracy == get(paste("ratio_stacking",FS[i],sep = "_"))),] %>% .[order(.$BaseModel_Num,decreasing = T),])
        if (nrow(get(paste0("MaxLine_",FS[i]))) > 1) {
          Maxline_sample <- sample(x = nrow(get(paste0("MaxLine_",FS[i]))),size = 1)
          assign(paste0("MaxLine_02_",FS[i]),get(paste0("MaxLine_",FS[i]))[Maxline_sample,])
          if (length(unique(get(paste0("MaxLine_",FS[i]))$MetaModel))==1) {
            assign(paste("MetaModel01",FS[i],sep = "_"),unique(get(paste0("MaxLine_",FS[i]))$MetaModel))
          }
          if (length(unique(get(paste0("MaxLine_",FS[i]))$MetaModel))==2) {
            assign(paste("MetaModel01",FS[i],sep = "_"),paste(unique(get(paste0("MaxLine_",FS[i]))$MetaModel),collapse = "_"))
          }
        }else{
          assign(paste0("MaxLine_02_",FS[i]),get(paste0("MaxLine_",FS[i])))
          assign(paste("MetaModel01",FS[i],sep = "_"),unique(get(paste0("MaxLine_",FS[i]))$MetaModel))
          }
        if (get(paste0("MaxLine_02_",FS[i]))$MetaModel=="KNN") {
          # get(paste0("MaxLine_02_",FS[i]))$BaseModel_Num,
          assign(paste("stacking_pre",FS[i],sep = "_"),
                 get(paste("stacking_pre_KNN",FS[i],
                           get(paste0("MaxLine_02_",FS[i]))$Distance_KNN,
                           get(paste0("MaxLine_02_",FS[i]))$K_KNN,sep = "_")))
        }
        if (get(paste0("MaxLine_02_",FS[i]))$MetaModel=="RF") {
          assign(paste("stacking_pre",FS[i],sep = "_"),
                 get(paste("stacking_pre_RF",FS[i],sep = "_")))
        }
        
        
        print(get(paste("ratio_stacking",FS[i],sep = "_")))
        print("pass")
        assign(paste("EndTime01",FS[i],sep = "_"),proc.time())
        assign(paste("RunningTime01",FS[i],sep = "_"),get(paste("EndTime01",FS[i],sep = "_")) - get(paste("StartTime01",FS[i],sep = "_")))
      }
      # Ensemble Learning 02 ----------------------------------------------------
      
      print(paste("begin:","Ensemble Learning 02"))

      FS02 <- list()
      for (i in 1:length(stacking_model02)) {
        FS02[i] <- strsplit(stacking_model02[[i]],split = "-")[[1]][1]
        if (FS02[i] == "chi2") {
          assign(paste("stacking",LETTERS[i],"train",sep = "_"),train.chi2)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.chi2$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",LETTERS[i],"train",LETTERS[j],sep = "_"),train.chi2[k_fold[[j]],])#
            assign(paste("stacking",LETTERS[i],"test",LETTERS[j],sep = "_"),train.chi2[-k_fold[[j]],])#
          }
        }
        if (FS02[i] == "ETC") {
          assign(paste("stacking",LETTERS[i],"train",sep = "_"),train.ETC)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.ETC$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",LETTERS[i],"train",LETTERS[j],sep = "_"),train.ETC[k_fold[[j]],])#
            assign(paste("stacking",LETTERS[i],"test",LETTERS[j],sep = "_"),train.ETC[-k_fold[[j]],])#
          }
        }
        if (FS02[i] == "f_classif") {
          assign(paste("stacking",LETTERS[i],"train",sep = "_"),train.f_classif)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.f_classif$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",LETTERS[i],"train",LETTERS[j],sep = "_"),train.f_classif[k_fold[[j]],])#
            assign(paste("stacking",LETTERS[i],"test",LETTERS[j],sep = "_"),train.f_classif[-k_fold[[j]],])#
          }
        }
        if (FS02[i] == "info_classif") {
          assign(paste("stacking",LETTERS[i],"train",sep = "_"),train.info_classif)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.info_classif$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",LETTERS[i],"train",LETTERS[j],sep = "_"),train.info_classif[k_fold[[j]],])#
            assign(paste("stacking",LETTERS[i],"test",LETTERS[j],sep = "_"),train.info_classif[-k_fold[[j]],])#
          }
        }
        if (FS02[i] == "LR") {
          assign(paste("stacking",LETTERS[i],"train",sep = "_"),train.LR)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.LR$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",LETTERS[i],"train",LETTERS[j],sep = "_"),train.LR[k_fold[[j]],])#
            assign(paste("stacking",LETTERS[i],"test",LETTERS[j],sep = "_"),train.LR[-k_fold[[j]],])#
          }
        }
        if (FS02[i] == "RF") {
          assign(paste("stacking",LETTERS[i],"train",sep = "_"),train.RF)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.RF$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",LETTERS[i],"train",LETTERS[j],sep = "_"),train.RF[k_fold[[j]],])#
            assign(paste("stacking",LETTERS[i],"test",LETTERS[j],sep = "_"),train.RF[-k_fold[[j]],])#
          }
        }
        if (FS02[i] == "XGBoost") {
          assign(paste("stacking",LETTERS[i],"train",sep = "_"),train.XGBoost)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.XGBoost$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",LETTERS[i],"train",LETTERS[j],sep = "_"),train.XGBoost[k_fold[[j]],])#
            assign(paste("stacking",LETTERS[i],"test",LETTERS[j],sep = "_"),train.XGBoost[-k_fold[[j]],])#
          }
        }
        if (FS02[i] == "LD") {
          assign(paste("stacking",LETTERS[i],"train",sep = "_"),train.LD)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.LD$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",LETTERS[i],"train",LETTERS[j],sep = "_"),train.LD[k_fold[[j]],])#
            assign(paste("stacking",LETTERS[i],"test",LETTERS[j],sep = "_"),train.LD[-k_fold[[j]],])#
          }
        }
        if (FS02[i] == "blank") {
          assign(paste("stacking",LETTERS[i],"train",sep = "_"),train.blank)
          set.seed(seednum02)
          assign("k_fold",createFolds(y=train.blank$FID,k =fold,list = TRUE, returnTrain = TRUE))
          for (j in 1:fold) {
            assign(paste("stacking",LETTERS[i],"train",LETTERS[j],sep = "_"),train.blank[k_fold[[j]],])#
            assign(paste("stacking",LETTERS[i],"test",LETTERS[j],sep = "_"),train.blank[-k_fold[[j]],])#
          }
        }
      }
      
      assign(paste0("StackingParameter02"),data.frame(MetaModel = NA,
                                                      FeatureSelection = NA,
                                                      Accuracy = NA,
                                                      BaseModel = NA,
                                                      BaseModel_Num = NA,
                                                      Distance_KNN = NA,
                                                      K_KNN = NA,
                                                      stringsAsFactors = F))
      ML02 <- list()
      train_label_stacking <- train_label_stacking[order(train_label_stacking$NUM),]
      stacking_train <- NULL %>% rbind(.,train_label_stacking[,c(1,2)])
      stacking_train <- NULL %>% rbind(.,train_label_stacking[,c(1,2)])
      stacking_test <- NULL %>% cbind(.,testdata_all[,1])
      stacking_test <- NULL %>% cbind(.,testdata_all[,1])
      assign(paste("StartTime02",sep = "_"),proc.time())
      for (fm in 1:length(stacking_model02)) {
        ML02[fm] <- strsplit(stacking_model02[[fm]],split = "-")[[1]][2]
        print(paste0("fm=",fm));print(paste0("ML02[fm]=",ML02[fm]));print(paste0("FS02[fm]=",FS02[fm]))
        if (ML[fm] == "knn") {
          assign(paste("predictions",FS[i],ML[fm],sep = "_"),NULL)
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),NULL)
          assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),NULL)
          for (j in 1:fold) {
            assign(paste(FS[i],ML[fm],j,sep = "_"),
                   kknn(get(paste("stacking",i,"train",j,sep = "_"))$FID~.,
                        get(paste("stacking",i,"train",j,sep = "_"))[,-c(1:7)],
                        get(paste("stacking",i,"test",j,sep = "_"))[,-c(1:7)],
                        distance = 2,k= 2,kernel= "triangular"))#
            assign(paste(FS[i],ML[fm],"test",j,sep = "_"),
                   kknn(get(paste("stacking",i,"train",j,sep = "_"))$FID~.,
                        get(paste("stacking",i,"train",j,sep = "_"))[,-c(1:7)], 
                        testdata_all[,-c(1:7)], 
                        distance = 2,k= 2,kernel= "triangular"))
            assign(paste("predictions",FS[i],ML[fm],j,sep = "_"),
                   predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                           newdata = get(paste("stacking",i,"test",j,sep = "_"))[,-c(1:7)]) %>% as.character(.))
            assign(paste("new_feature",FS[i],ML[fm],j,sep = "_"),
                   cbind(get(paste("stacking",i,"test",j,sep = "_"))[,1],
                         get(paste("predictions",FS[i],ML[fm],j,sep = "_"))))
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   rbind(get(paste("new_feature",FS[i],ML[fm],sep = "_")),
                         get(paste("new_feature",FS[i],ML[fm],j,sep = "_"))))
            assign(paste("predictions",FS[i],ML[fm],"test",j,sep = "_"),
                   predict(object = get(paste(FS[i],ML[fm],"test",j,sep = "_")),
                           newdata = testdata_all[,-c(1:7)],type = "prob"))
            if (j==1) {
              assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))
            }else{assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),
                         get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))+get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))}
          }
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                 as.data.frame(get(paste("new_feature",FS[i],ML[fm],sep = "_"))))
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                 get(paste("new_feature",FS[i],ML[fm],sep = "_"))[order(as.numeric(get(paste("new_feature",FS[i],ML[fm],sep = "_"))[,1])),])
          assign(paste("vote",FS[i],ML[fm],sep = "_"),
                 lapply(X = 1:nrow(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))),FUN = function(x){
                   max_labelnum <- which(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]==max(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]))
                   if (length(max_labelnum) != 1) {return(sample(x = max_labelnum,size = 1) %>% names(.))}
                   else{return(names(max_labelnum))}
                 }) %>% ldply(.data = .))
        }
        if (ML[fm] == "nb") {
          assign(paste("predictions",FS[i],ML[fm],sep = "_"),NULL)
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),NULL)
          assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),NULL)
          for (j in 1:fold) {
            assign(paste(FS[i],ML[fm],j,sep = "_"),
                   naiveBayes(y = get(paste("stacking",i,"train",j,sep = "_"))$FID,
                              x = get(paste("stacking",i,"train",j,sep = "_"))[,-c(1,3,4,5,6,7)]))
            assign(paste("predictions",FS[i],ML[fm],j,sep = "_"),
                   predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                           newdata = get(paste("stacking",i,"test",j,sep = "_"))[,-c(1:7)]) %>% as.character(.))
            assign(paste("new_feature",FS[i],ML[fm],j,sep = "_"),
                   cbind(get(paste("stacking",i,"test",j,sep = "_"))[,1],
                         get(paste("predictions",FS[i],ML[fm],j,sep = "_"))))
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   rbind(get(paste("new_feature",FS[i],ML[fm],sep = "_")),
                         get(paste("new_feature",FS[i],ML[fm],j,sep = "_"))))
            assign(paste("predictions",FS[i],ML[fm],"test",j,sep = "_"),
                   predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                           newdata = testdata_all[,-c(1:7)],type = "raw"))
            if (j==1) {
              assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))
            }else{assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),
                         get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))+get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))}
          }
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                 as.data.frame(get(paste("new_feature",FS[i],ML[fm],sep = "_"))))
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                 get(paste("new_feature",FS[i],ML[fm],sep = "_"))[order(as.numeric(get(paste("new_feature",FS[i],ML[fm],sep = "_"))[,1])),])
          assign(paste("vote",FS[i],ML[fm],sep = "_"),
                 lapply(X = 1:nrow(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))),FUN = function(x){
                   max_labelnum <- which(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]==max(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]))
                   if (length(max_labelnum) != 1) {return(sample(x = max_labelnum,size = 1) %>% names(.))}
                   else{return(names(max_labelnum))}
                 }) %>% ldply(.data = .))
        }
        if (ML[fm] == "rf") {
          assign(paste("predictions",FS[i],ML[fm],sep = "_"),NULL)
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),NULL)
          assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),NULL)
          for (j in 1:fold) {
            assign(paste(FS[i],ML[fm],j,sep = "_"),
                   randomForest(y = get(paste("stacking",i,"train",j,sep = "_"))$FID,
                                x = get(paste("stacking",i,"train",j,sep = "_"))[,-c(1:7)]))
            assign(paste("predictions",FS[i],ML[fm],j,sep = "_"),
                   predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                           newdata = get(paste("stacking",i,"test",j,sep = "_"))[,-c(1:7)]) %>% as.character(.))
            assign(paste("new_feature",FS[i],ML[fm],j,sep = "_"),
                   cbind(get(paste("stacking",i,"test",j,sep = "_"))[,1],
                         get(paste("predictions",FS[i],ML[fm],j,sep = "_"))))
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   rbind(get(paste("new_feature",FS[i],ML[fm],sep = "_")),
                         get(paste("new_feature",FS[i],ML[fm],j,sep = "_"))))
            assign(paste("predictions",FS[i],ML[fm],"test",j,sep = "_"),
                   predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                           newdata = testdata_all[,-c(1:7)],type = "prob"))
            if (j==1) {
              assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))
            }else{assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),
                         get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))+get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))}
          }
          print("Finish CrossValidation")
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                 as.data.frame(get(paste("new_feature",FS[i],ML[fm],sep = "_"))))
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                 get(paste("new_feature",FS[i],ML[fm],sep = "_"))[order(as.numeric(get(paste("new_feature",FS[i],ML[fm],sep = "_"))[,1])),])
          print("Begin RF Vote")
          assign(paste("vote",FS[i],ML[fm],sep = "_"),
                 lapply(X = 1:nrow(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))),FUN = function(x){
                   max_labelnum <- which(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]==max(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]))
                   if (length(max_labelnum) != 1) {return(sample(x = max_labelnum,size = 1) %>% names(.))}
                   else{return(names(max_labelnum))}
                 }) %>% ldply(.data = .))
        }
        if (ML[fm] == "svm") {
          assign(paste("predictions",FS[i],ML[fm],sep = "_"),NULL)
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),NULL)
          assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),NULL)
          for (j in 1:fold) {
            assign(paste(FS[i],ML[fm],j,sep = "_"),
                   svm(get(paste("stacking",i,"train",j,sep = "_"))$FID~.,
                       get(paste("stacking",i,"train",j,sep = "_"))[,-c(1:7)],
                       kernel = "linear",cost = 1,
                       gamma = 1/ncol(get(paste("stacking",i,"train",j,sep = "_"))),probability=TRUE))
            assign(paste("predictions",FS[i],ML[fm],j,sep = "_"),
                   predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                           newdata = get(paste("stacking",i,"test",j,sep = "_"))[,-c(1:7)]) %>% as.character(.))
            assign(paste("new_feature",FS[i],ML[fm],j,sep = "_"),
                   cbind(get(paste("stacking",i,"test",j,sep = "_"))[,1],
                         get(paste("predictions",FS[i],ML[fm],j,sep = "_"))))
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   rbind(get(paste("new_feature",FS[i],ML[fm],sep = "_")),
                         get(paste("new_feature",FS[i],ML[fm],j,sep = "_"))))
            assign(paste("predictions",FS[i],ML[fm],"test",j,sep = "_"),
                   predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                           newdata = testdata_all[,-c(1:7)], probability=TRUE) %>% attr(., "probabilities"))
            if (j==1) {
              assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))
            }else{assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),
                         get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))+get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))}
          }
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                 as.data.frame(get(paste("new_feature",FS[i],ML[fm],sep = "_"))))
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                 get(paste("new_feature",FS[i],ML[fm],sep = "_"))[order(as.numeric(get(paste("new_feature",FS[i],ML[fm],sep = "_"))[,1])),])
          assign(paste("vote",FS[i],ML[fm],sep = "_"),
                 lapply(X = 1:nrow(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))),FUN = function(x){
                   max_labelnum <- which(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]==max(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]))
                   if (length(max_labelnum) != 1) {return(sample(x = max_labelnum,size = 1) %>% names(.))}
                   else{return(names(max_labelnum))}
                 }) %>% ldply(.data = .))
        }
        if (ML[fm] == "c50") {
          assign(paste("predictions",FS[i],ML[fm],sep = "_"),NULL)
          assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),NULL)
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),NULL)
          for (j in 1:fold) {
            assign(paste(FS[i],ML[fm],j,sep = "_"),
                   C5.0(y = get(paste("stacking",i,"train",j,sep = "_"))$FID,
                        x = get(paste("stacking",i,"train",j,sep = "_"))[,-c(1:7)],
                        trials=20))
            assign(paste("predictions",FS[i],ML[fm],j,sep = "_"),
                   predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                           newdata = get(paste("stacking",i,"test",j,sep = "_"))[,-c(1:7)]) %>% as.character(.))
            assign(paste("new_feature",FS[i],ML[fm],j,sep = "_"),
                   cbind(get(paste("stacking",i,"test",j,sep = "_"))[,1],
                         get(paste("predictions",FS[i],ML[fm],j,sep = "_"))))
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   rbind(get(paste("new_feature",FS[i],ML[fm],sep = "_")),
                         get(paste("new_feature",FS[i],ML[fm],j,sep = "_"))))
            assign(paste("predictions",FS[i],ML[fm],"test",j,sep = "_"),
                   predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                           newdata = testdata_all[,-c(1:7)],type = "prob"))
            if (j==1) {
              assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))
            }else{assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),
                         get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))+get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))}
          }
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                 as.data.frame(get(paste("new_feature",FS[i],ML[fm],sep = "_"))))
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                 get(paste("new_feature",FS[i],ML[fm],sep = "_"))[order(as.numeric(get(paste("new_feature",FS[i],ML[fm],sep = "_"))[,1])),])
          assign(paste("vote",FS[i],ML[fm],sep = "_"),
                 lapply(X = 1:nrow(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))),FUN = function(x){
                   max_labelnum <- which(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]==max(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]))
                   if (length(max_labelnum) != 1) {return(sample(x = max_labelnum,size = 1) %>% names(.))}
                   else{return(names(max_labelnum))}
                 }) %>% ldply(.data = .))
        }
        if (ML[fm] == "nnet") {
          assign(paste("predictions",FS[i],ML[fm],sep = "_"),NULL)
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),NULL)
          assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),NULL)
          for (j in 1:fold) {
            # print(paste("new_feature",FS[i],ML[fm],sep = "_"));print(get(paste("new_feature",FS[i],ML[fm],sep = "_")))
            assign(paste(FS[i],ML[fm],j,sep = "_"),
                   nnet(get(paste("stacking",i,"train",j,sep = "_"))$FID~.,
                        data=get(paste("stacking",i,"train",j,sep = "_"))[,-c(1:7)],
                        size=2,decay=5e-4,maxit=200,rang=0.1,MaxNWts=10000))
            assign(paste("predictions",FS[i],ML[fm],j,sep = "_"),
                   predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                           newdata = get(paste("stacking",i,"test",j,sep = "_"))[,-c(1:7)],type = "class"))#
            assign(paste("new_feature",FS[i],ML[fm],j,sep = "_"),
                   cbind(get(paste("stacking",i,"test",j,sep = "_"))[,1],
                         get(paste("predictions",FS[i],ML[fm],j,sep = "_"))))
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   rbind(get(paste("new_feature",FS[i],ML[fm],sep = "_")),
                         get(paste("new_feature",FS[i],ML[fm],j,sep = "_"))))
            assign(paste("predictions",FS[i],ML[fm],"test",j,sep = "_"),
                   predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                           newdata = testdata_all[,-c(1:7)],type = "raw"))#
            if (j==1) {
              assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))
            }else{assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),
                         get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))+get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))}
          }
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                 as.data.frame(get(paste("new_feature",FS[i],ML[fm],sep = "_"))))
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                 get(paste("new_feature",FS[i],ML[fm],sep = "_"))[order(as.numeric(get(paste("new_feature",FS[i],ML[fm],sep = "_"))[,1])),])
          assign(paste("vote",FS[i],ML[fm],sep = "_"),
                 lapply(X = 1:nrow(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))),FUN = function(x){
                   max_labelnum <- which(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]==max(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]))
                   if (length(max_labelnum) != 1) {return(sample(x = max_labelnum,size = 1) %>% names(.))}
                   else{return(names(max_labelnum))}
                 }) %>% ldply(.data = .))
        }
        if (ML[fm] == "mlr") {
          assign(paste("predictions",FS[i],ML[fm],sep = "_"),NULL)
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),NULL)
          assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),NULL)
          for (j in 1:fold) {
            print(paste("j=",j))
            assign(paste(FS[i],ML[fm],j,sep = "_"),
                   multinom(formula = get(paste("stacking",i,"train",j,sep = "_"))$FID~.,
                            data = get(paste("stacking",i,"train",j,sep = "_"))[,-c(1:7)],MaxNWts = 143541315))
            assign(paste("predictions",FS[i],ML[fm],j,sep = "_"),
                   predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                           newdata = get(paste("stacking",i,"test",j,sep = "_"))[,-c(1:7)]) %>% as.character(.))
            assign(paste("new_feature",FS[i],ML[fm],j,sep = "_"),
                   cbind(get(paste("stacking",i,"test",j,sep = "_"))[,1],
                         get(paste("predictions",FS[i],ML[fm],j,sep = "_"))))
            assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                   rbind(get(paste("new_feature",FS[i],ML[fm],sep = "_")),
                         get(paste("new_feature",FS[i],ML[fm],j,sep = "_"))))
            assign(paste("predictions",FS[i],ML[fm],"test",j,sep = "_"),
                   predict(object = get(paste(FS[i],ML[fm],j,sep = "_")),
                           newdata = testdata_all[,-c(1:7)],type = "prob"))
            if (j==1) {
              assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))
            }else{assign(paste("predictions",FS[i],ML[fm],"test",sep = "_"),
                         get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))+get(paste("predictions",FS[i],ML[fm],"test",j,sep = "_")))}
          }
          print("Finish CrossValidation")
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                 as.data.frame(get(paste("new_feature",FS[i],ML[fm],sep = "_"))))
          assign(paste("new_feature",FS[i],ML[fm],sep = "_"),
                 get(paste("new_feature",FS[i],ML[fm],sep = "_"))[order(as.numeric(get(paste("new_feature",FS[i],ML[fm],sep = "_"))[,1])),])
          assign(paste("vote",FS[i],ML[fm],sep = "_"),
                 lapply(X = 1:nrow(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))),FUN = function(x){
                   max_labelnum <- which(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]==max(get(paste("predictions",FS[i],ML[fm],"test",sep = "_"))[x,]))
                   if (length(max_labelnum) != 1) {return(sample(x = max_labelnum,size = 1) %>% names(.))}
                   else{return(names(max_labelnum))}
                 }) %>% ldply(.data = .))
        }
        stacking_train <- cbind(stacking_train,get(paste("new_feature",FS02[fm],ML02[fm],sep = "_"))[,2]) %>% as.data.frame(.)
        colnumber_train <- ncol(stacking_train)
        colnames(stacking_train)[colnumber_train] <- ML02[fm]
        stacking_test <- cbind(stacking_test,get(paste("vote",FS02[fm],ML02[fm],sep = "_"))) %>% as.data.frame(.)
        colnumber_test <- ncol(stacking_test)
        colnames(stacking_test)[colnumber_test] <- ML02[fm]
      }
      stacking_train$FID <- as.factor(stacking_train$FID)
      
      for (distance in 1:2) {
        for (k in 1:7) {
          assign(paste0("stacking_ML02_KNN_",distance,"_",k),
                 tryCatch(
                   {
                     kknn(formula = stacking_train$FID~.,
                          train = stacking_train[,-c(1:2)],
                          test = stacking_test[,-1],
                          distance = distance,k=k,kernel= "triangular")
                   },
                   error = function(e){
                     return(NULL)
                   }
                 ))
          
          if (is.null(get(paste0("stacking_ML02_KNN_",distance,"_",k)))) {
            next
          }else{
            assign(paste0("stacking_pre02_KNN_",distance,"_",k),
                   predict(object = get(paste0("stacking_ML02_KNN_",distance,"_",k)),
                           newdata = stacking_test[,-1]) %>% lvls_expand(.,varietyname_label))
            assign(paste0("table_stacking02_KNN_",distance,"_",k),
                   table(get(paste0("stacking_pre02_KNN_",distance,"_",k)),testdata_all$FID))
            assign(paste0("ratio_stacking02_KNN_",distance,"_",k),
                   sum(diag(get(paste0("table_stacking02_KNN_",distance,"_",k))))/sum(get(paste0("table_stacking02_KNN_",distance,"_",k))))
            assign(paste0("StackingParameter02"),rbind(get(paste0("StackingParameter02")),
                                                       c("KNN",
                                                         FS02[fm],
                                                         get(paste0("ratio_stacking02_KNN_",distance,"_",k)),
                                                         paste(colnames(stacking_train[,-c(1:2)]),collapse = "_"),
                                                         length(colnames(stacking_train[,-c(1:2)])),
                                                         distance,
                                                         k)))
          }
        }
      }
      assign(paste0("stacking_ML02_RF"),
             randomForest(stacking_train$FID~.,
                          data = stacking_train[,-c(1:2)],importance=T))
      assign(paste0("stacking_pre02_RF"),
             predict(object = get(paste0("stacking_ML02_RF")),
                     newdata = stacking_test[,-1]) %>% lvls_expand(.,varietyname_label))
      assign(paste0("table_stacking02_RF"),
             table(get(paste0("stacking_pre02_RF")),testdata_all$FID))
      assign(paste0("ratio_stacking02_RF"),
             sum(diag(get(paste0("table_stacking02_RF"))))/sum(get(paste0("table_stacking02_RF"))))
      assign(paste0("StackingParameter02"),rbind(get(paste0("StackingParameter02")),
                                                 c("RF",
                                                   FS02[fm],
                                                   get(paste0("ratio_stacking02_RF")),
                                                   paste(colnames(stacking_train[,-c(1:2)]),collapse = "_"),
                                                   length(colnames(stacking_train[,-c(1:2)])),
                                                   "",
                                                   "")))
      
      assign(paste0("StackingParameter02"),na.omit(get(paste0("StackingParameter02"))))
      assign(paste("ratio_stacking02"),max(get(paste0("StackingParameter02"))$Accuracy))
      assign(paste0("MaxLine02"),
             get(paste0("StackingParameter02"))[which(get(paste0("StackingParameter02"))$Accuracy == get(paste("ratio_stacking02"))),] %>% .[order(.$BaseModel_Num,decreasing = T),])
      if (nrow(get(paste0("MaxLine02"))) > 1) {
        Maxline_sample <- sample(x = nrow(get(paste0("MaxLine02"))),size = 1)
        assign(paste0("MaxLine02_02"),get(paste0("MaxLine02"))[Maxline_sample,])
        if (length(unique(MaxLine02$MetaModel))==1) {MetaModel02 <- unique(MaxLine02$MetaModel)}
        if (length(unique(MaxLine02$MetaModel))==2) {MetaModel02 <- paste(unique(MaxLine02$MetaModel),collapse = "_")}
      }else{
        assign(paste0("MaxLine02_02"),get(paste0("MaxLine02")))
        MetaModel02 <- unique(MaxLine02$MetaModel)
      }
      if (get(paste0("MaxLine02_02"))$MetaModel=="KNN") {
        assign(paste("stacking_pre02"),
               get(paste("stacking_pre02_KNN",
                         get(paste0("MaxLine02_02"))$Distance_KNN,
                         get(paste0("MaxLine02_02"))$K_KNN,sep = "_")))
      }
      if (get(paste0("MaxLine02_02"))$MetaModel=="RF") {
        assign(paste("stacking_pre02"),
               get(paste("stacking_pre02_RF")))
      }
      assign(paste("EndTime02",sep = "_"),proc.time())
      assign(paste("RunningTime02",sep = "_"),get(paste("EndTime02",sep = "_")) - get(paste("StartTime02",sep = "_")))
      
      # Original Discriminant Result --------------------------------------------
      TestID <- testdata_all[,"IID"]
      TestLabel <- testdata_all[,"FID"]
      
      Table <- data.frame(TableNum = rep(TableNum,nrow(testdata_all)),
                          stringsAsFactors=FALSE)
      Table["TestNum"] <- testdata_all[,"NUM"]
      Table["TestLabel"] <- TestLabel
      Table["TestID"] <- TestID
      Table["Seed01"] <- seednum01
      Table["Seed02"] <- seednum02
      Table["TestScale"] <- TestScaleSize
      Table["OutliersFraction"] <- AD_fam$OutliersFraction
      Table["KFold"] <- NA
      Table["FeatureMethod"] <- NA
      Table["SizeNum"] <- SizeNum
      Table["FeatureNum"] <- FeatureNum
      Table["AD_True"] <- AD_fam$Class_true
      Table["AD_LOF"] <- NA
      Table["AD_KNN_lar"] <- NA
      Table["AD_IForest"] <- NA
      Table["AD_Vote"] <- NA
      Table["KNN"] <- NA
      Table["RF"] <- NA
      Table["SVM"] <- NA
      Table["nnet"] <- NA
      Table["NB"] <- NA
      Table["C50"] <- NA
      Table["MLR"] <- NA
      Table["Stacking01"] <- NA
      Table["Stacking02"] <- NA
      Table["MetaModel01"] <- NA
      Table["MetaModel02"] <- NA
      Table["KNN_Time"] <- NA
      Table["NB_Time"] <- NA
      Table["RF_Time"] <- NA
      Table["SVM_Time"] <- NA
      Table["C50_Time"] <- NA
      Table["nnet_Time"] <- NA
      Table["MLR_Time"] <- NA
      Table["Stacking01_Time"] <- NA
      Table["Stacking02_Time"] <- NA
      
      Table_blank <- Table
      Table_blank["FeatureNum"] <- ncol(train.blank)-7
      Table_blank["KFold"] <- fold
      Table_blank["FeatureMethod"] <- "blank"
      Table_blank["AD_LOF"] <- AnomalyDetection_blank$pred_lof
      Table_blank["AD_KNN_lar"] <- AnomalyDetection_blank$pred_KNN_lar
      Table_blank["AD_IForest"] <- AnomalyDetection_blank$pred_IForest
      Table_blank["AD_Vote"] <- AnomalyDetection_blank$pred_vote
      Table_blank["KNN"] <- Blank_knn[1]
      Table_blank["RF"] <- Blank_RF[1]
      Table_blank["SVM"] <- Blank_SVM[1]
      Table_blank["NB"] <- Blank_NB[1]
      Table_blank["C50"] <- Blank_C50[1]
      Table_blank["nnet"] <- Blank_nnet[1]
      Table_blank["MLR"] <- Blank_MLR[1]
      Table_blank["Stacking01"] <- stacking_pre_blank
      Table_blank["Stacking02"] <- stacking_pre02
      Table_blank["MetaModel01"] <- MetaModel01_blank
      Table_blank["MetaModel02"] <- MetaModel02
      Table_blank["KNN_Time"] <- Blank_knn[3]
      Table_blank["NB_Time"] <- Blank_NB[3]
      Table_blank["RF_Time"] <- Blank_RF[3]
      Table_blank["SVM_Time"] <- Blank_SVM[3]
      Table_blank["C50_Time"] <- Blank_C50[3]
      Table_blank["nnet_Time"] <- Blank_nnet[3]
      Table_blank["MLR_Time"] <- Blank_MLR[3]
      Table_blank["Stacking01_Time"] <- get(paste("RunningTime01","blank",sep = "_"))[3][[1]]
      Table_blank["Stacking02_Time"] <- get(paste("RunningTime02",sep = "_"))[3][[1]]
      
      Table_chi2 <- Table
      Table_chi2["KFold"] <- fold
      Table_chi2["FeatureMethod"] <- "chi2"
      Table_chi2["AD_LOF"] <- AnomalyDetection_chi2$pred_lof
      Table_chi2["AD_KNN_lar"] <- AnomalyDetection_chi2$pred_KNN_lar
      Table_chi2["AD_IForest"] <- AnomalyDetection_chi2$pred_IForest
      Table_chi2["AD_Vote"] <- AnomalyDetection_chi2$pred_vote
      Table_chi2["KNN"] <- chi2_knn[1]
      Table_chi2["RF"] <- chi2_RF[1]
      Table_chi2["SVM"] <- chi2_SVM[1]
      Table_chi2["NB"] <- chi2_NB[1]
      Table_chi2["C50"] <- chi2_C50[1]
      Table_chi2["nnet"] <- chi2_nnet[1]
      Table_chi2["MLR"] <- chi2_MLR[1]
      Table_chi2["Stacking01"] <- stacking_pre_chi2
      Table_chi2["Stacking02"] <- stacking_pre02
      Table_chi2["MetaModel01"] <- MetaModel01_chi2
      Table_chi2["MetaModel02"] <- MetaModel02
      Table_chi2["KNN_Time"] <- chi2_knn[3]
      Table_chi2["NB_Time"] <- chi2_NB[3]
      Table_chi2["RF_Time"] <- chi2_RF[3]
      Table_chi2["SVM_Time"] <- chi2_SVM[3]
      Table_chi2["C50_Time"] <- chi2_C50[3]
      Table_chi2["nnet_Time"] <- chi2_nnet[3]
      Table_chi2["MLR_Time"] <- chi2_MLR[3]
      Table_chi2["Stacking01_Time"] <- get(paste("RunningTime01","chi2",sep = "_"))[3][[1]]
      Table_chi2["Stacking02_Time"] <- get(paste("RunningTime02",sep = "_"))[3][[1]]
      
      Table_ETC <- Table
      Table_ETC["KFold"] <- fold
      Table_ETC["FeatureMethod"] <- "ETC"
      Table_ETC["AD_LOF"] <- AnomalyDetection_ETC$pred_lof
      Table_ETC["AD_KNN_lar"] <- AnomalyDetection_ETC$pred_KNN_lar
      Table_ETC["AD_IForest"] <- AnomalyDetection_ETC$pred_IForest
      Table_ETC["AD_Vote"] <- AnomalyDetection_ETC$pred_vote
      Table_ETC["KNN"] <- ETC_knn[1]
      Table_ETC["RF"] <- ETC_RF[1]
      Table_ETC["SVM"] <- ETC_SVM[1]
      Table_ETC["NB"] <- ETC_NB[1]
      Table_ETC["C50"] <- ETC_C50[1]
      Table_ETC["nnet"] <- ETC_nnet[1]
      Table_ETC["MLR"] <- ETC_MLR[1]
      Table_ETC["Stacking01"] <- stacking_pre_ETC
      Table_ETC["Stacking02"] <- stacking_pre02
      Table_ETC["MetaModel01"] <- MetaModel01_ETC
      Table_ETC["MetaModel02"] <- MetaModel02
      Table_ETC["KNN_Time"] <- ETC_knn[3]
      Table_ETC["NB_Time"] <- ETC_NB[3]
      Table_ETC["RF_Time"] <- ETC_RF[3]
      Table_ETC["SVM_Time"] <- ETC_SVM[3]
      Table_ETC["C50_Time"] <- ETC_C50[3]
      Table_ETC["nnet_Time"] <- ETC_nnet[3]
      Table_ETC["MLR_Time"] <- ETC_MLR[3]
      Table_ETC["Stacking01_Time"] <- get(paste("RunningTime01","ETC",sep = "_"))[3][[1]]
      Table_ETC["Stacking02_Time"] <- get(paste("RunningTime02",sep = "_"))[3][[1]]
      
      Table_f_classif <- Table
      Table_f_classif["KFold"] <- fold
      Table_f_classif["FeatureMethod"] <- "f_classif"
      Table_f_classif["AD_LOF"] <- AnomalyDetection_f_classif$pred_lof
      Table_f_classif["AD_KNN_lar"] <- AnomalyDetection_f_classif$pred_KNN_lar
      Table_f_classif["AD_IForest"] <- AnomalyDetection_f_classif$pred_IForest
      Table_f_classif["AD_Vote"] <- AnomalyDetection_f_classif$pred_vote
      Table_f_classif["KNN"] <- f_classif_knn[1]
      Table_f_classif["RF"] <- f_classif_RF[1]
      Table_f_classif["SVM"] <- f_classif_SVM[1]
      Table_f_classif["NB"] <- f_classif_NB[1]
      Table_f_classif["C50"] <- f_classif_C50[1]
      Table_f_classif["nnet"] <- f_classif_nnet[1]
      Table_f_classif["MLR"] <- f_classif_MLR[1]
      Table_f_classif["Stacking01"] <- stacking_pre_f_classif
      Table_f_classif["Stacking02"] <- stacking_pre02
      Table_f_classif["MetaModel01"] <- MetaModel01_f_classif
      Table_f_classif["MetaModel02"] <- MetaModel02
      Table_f_classif["KNN_Time"] <- f_classif_knn[3]
      Table_f_classif["NB_Time"] <- f_classif_NB[3]
      Table_f_classif["RF_Time"] <- f_classif_RF[3]
      Table_f_classif["SVM_Time"] <- f_classif_SVM[3]
      Table_f_classif["C50_Time"] <- f_classif_C50[3]
      Table_f_classif["nnet_Time"] <- f_classif_nnet[3]
      Table_f_classif["MLR_Time"] <- f_classif_MLR[3]
      Table_f_classif["Stacking01_Time"] <- get(paste("RunningTime01","f_classif",sep = "_"))[3][[1]]
      Table_f_classif["Stacking02_Time"] <- get(paste("RunningTime02",sep = "_"))[3][[1]]
      
      Table_info_classif <- Table
      Table_info_classif["KFold"] <- fold
      Table_info_classif["FeatureMethod"] <- "info_classif"
      Table_info_classif["AD_LOF"] <- AnomalyDetection_info_classif$pred_lof
      Table_info_classif["AD_KNN_lar"] <- AnomalyDetection_info_classif$pred_KNN_lar
      Table_info_classif["AD_IForest"] <- AnomalyDetection_info_classif$pred_IForest
      Table_info_classif["AD_Vote"] <- AnomalyDetection_info_classif$pred_vote
      Table_info_classif["KNN"] <- info_classif_knn[1]
      Table_info_classif["RF"] <- info_classif_RF[1]
      Table_info_classif["SVM"] <- info_classif_SVM[1]
      Table_info_classif["NB"] <- info_classif_NB[1]
      Table_info_classif["C50"] <- info_classif_C50[1]
      Table_info_classif["nnet"] <- info_classif_nnet[1]
      Table_info_classif["MLR"] <- info_classif_MLR[1]
      Table_info_classif["Stacking01"] <- stacking_pre_info_classif
      Table_info_classif["Stacking02"] <- stacking_pre02
      Table_info_classif["MetaModel01"] <- MetaModel01_info_classif
      Table_info_classif["MetaModel02"] <- MetaModel02
      Table_info_classif["KNN_Time"] <- info_classif_knn[3]
      Table_info_classif["NB_Time"] <- info_classif_NB[3]
      Table_info_classif["RF_Time"] <- info_classif_RF[3]
      Table_info_classif["SVM_Time"] <- info_classif_SVM[3]
      Table_info_classif["C50_Time"] <- info_classif_C50[3]
      Table_info_classif["nnet_Time"] <- info_classif_nnet[3]
      Table_info_classif["MLR_Time"] <- info_classif_MLR[3]
      Table_info_classif["Stacking01_Time"] <- get(paste("RunningTime01","info_classif",sep = "_"))[3][[1]]
      Table_info_classif["Stacking02_Time"] <- get(paste("RunningTime02",sep = "_"))[3][[1]]
      
      
      Table_LD <- Table
      Table_LD["KFold"] <- fold
      Table_LD["FeatureMethod"] <- "LD"
      Table_LD["AD_LOF"] <- AnomalyDetection_LD$pred_lof
      Table_LD["AD_KNN_lar"] <- AnomalyDetection_LD$pred_KNN_lar
      Table_LD["AD_IForest"] <- AnomalyDetection_LD$pred_IForest
      Table_LD["AD_Vote"] <- AnomalyDetection_LD$pred_vote
      Table_LD["KNN"] <- LD_knn[1]
      Table_LD["RF"] <- LD_RF[1]
      Table_LD["SVM"] <- LD_SVM[1]
      Table_LD["NB"] <- LD_NB[1]
      Table_LD["C50"] <- LD_C50[1]
      Table_LD["nnet"] <- LD_nnet[1]
      Table_LD["MLR"] <- LD_MLR[1]
      Table_LD["Stacking01"] <- stacking_pre_LD
      Table_LD["Stacking02"] <- stacking_pre02
      Table_LD["MetaModel01"] <- MetaModel01_LD
      Table_LD["MetaModel02"] <- MetaModel02
      Table_LD["KNN_Time"] <- LD_knn[3]
      Table_LD["NB_Time"] <- LD_NB[3]
      Table_LD["RF_Time"] <- LD_RF[3]
      Table_LD["SVM_Time"] <- LD_SVM[3]
      Table_LD["C50_Time"] <- LD_C50[3]
      Table_LD["nnet_Time"] <- LD_nnet[3]
      Table_LD["MLR_Time"] <- LD_MLR[3]
      Table_LD["Stacking01_Time"] <- get(paste("RunningTime01","LD",sep = "_"))[3][[1]]
      Table_LD["Stacking02_Time"] <- get(paste("RunningTime02",sep = "_"))[3][[1]]
      
      
      
      Table_LR <- Table
      Table_LR["KFold"] <- fold
      Table_LR["FeatureMethod"] <- "LR"
      Table_LR["AD_LOF"] <- AnomalyDetection_LR$pred_lof
      Table_LR["AD_KNN_lar"] <- AnomalyDetection_LR$pred_KNN_lar
      Table_LR["AD_IForest"] <- AnomalyDetection_LR$pred_IForest
      Table_LR["AD_Vote"] <- AnomalyDetection_LR$pred_vote
      Table_LR["KNN"] <- LR_knn[1]
      Table_LR["RF"] <- LR_RF[1]
      Table_LR["SVM"] <- LR_SVM[1]
      Table_LR["NB"] <- LR_NB[1]
      Table_LR["C50"] <- LR_C50[1]
      Table_LR["nnet"] <- LR_nnet[1]
      Table_LR["MLR"] <- LR_MLR[1]
      Table_LR["Stacking01"] <- stacking_pre_LR
      Table_LR["Stacking02"] <- stacking_pre02
      Table_LR["MetaModel01"] <- MetaModel01_LR
      Table_LR["MetaModel02"] <- MetaModel02
      Table_LR["KNN_Time"] <- LR_knn[3]
      Table_LR["NB_Time"] <- LR_NB[3]
      Table_LR["RF_Time"] <- LR_RF[3]
      Table_LR["SVM_Time"] <- LR_SVM[3]
      Table_LR["C50_Time"] <- LR_C50[3]
      Table_LR["nnet_Time"] <- LR_nnet[3]
      Table_LR["MLR_Time"] <- LR_MLR[3]
      Table_LR["Stacking01_Time"] <- get(paste("RunningTime01","LR",sep = "_"))[3][[1]]
      Table_LR["Stacking02_Time"] <- get(paste("RunningTime02",sep = "_"))[3][[1]]
      
      
      Table_RF <- Table
      Table_RF["KFold"] <- fold
      Table_RF["FeatureMethod"] <- "RF"
      Table_RF["AD_LOF"] <- AnomalyDetection_RF$pred_lof
      Table_RF["AD_KNN_lar"] <- AnomalyDetection_RF$pred_KNN_lar
      Table_RF["AD_IForest"] <- AnomalyDetection_RF$pred_IForest
      Table_RF["AD_Vote"] <- AnomalyDetection_RF$pred_vote
      Table_RF["KNN"] <- RF_knn[1]
      Table_RF["RF"] <- RF_RF[1]
      Table_RF["SVM"] <- RF_SVM[1]
      Table_RF["NB"] <- RF_NB[1]
      Table_RF["C50"] <- RF_C50[1]
      Table_RF["nnet"] <- RF_nnet[1]
      Table_RF["MLR"] <- RF_MLR[1]
      Table_RF["Stacking01"] <- stacking_pre_RF
      Table_RF["Stacking02"] <- stacking_pre02
      Table_RF["MetaModel01"] <- MetaModel01_RF
      Table_RF["MetaModel02"] <- MetaModel02
      Table_RF["KNN_Time"] <- RF_knn[3]
      Table_RF["NB_Time"] <- RF_NB[3]
      Table_RF["RF_Time"] <- RF_RF[3]
      Table_RF["SVM_Time"] <- RF_SVM[3]
      Table_RF["C50_Time"] <- RF_C50[3]
      Table_RF["nnet_Time"] <- RF_nnet[3]
      Table_RF["MLR_Time"] <- RF_MLR[3]
      Table_RF["Stacking01_Time"] <- get(paste("RunningTime01","RF",sep = "_"))[3][[1]]
      Table_RF["Stacking02_Time"] <- get(paste("RunningTime02",sep = "_"))[3][[1]]
      
      Table_XGBoost <- Table
      Table_XGBoost["KFold"] <- fold
      Table_XGBoost["FeatureMethod"] <- "XGBoost"
      Table_XGBoost["AD_LOF"] <- AnomalyDetection_XGBoost$pred_lof
      Table_XGBoost["AD_KNN_lar"] <- AnomalyDetection_XGBoost$pred_KNN_lar
      Table_XGBoost["AD_IForest"] <- AnomalyDetection_XGBoost$pred_IForest
      Table_XGBoost["AD_Vote"] <- AnomalyDetection_XGBoost$pred_vote
      Table_XGBoost["KNN"] <- XGBoost_knn[1]
      Table_XGBoost["RF"] <- XGBoost_RF[1]
      Table_XGBoost["SVM"] <- XGBoost_SVM[1]
      Table_XGBoost["NB"] <- XGBoost_NB[1]
      Table_XGBoost["C50"] <- XGBoost_C50[1]
      Table_XGBoost["nnet"] <- XGBoost_nnet[1]
      Table_XGBoost["MLR"] <- XGBoost_MLR[1]
      Table_XGBoost["Stacking01"] <- stacking_pre_XGBoost
      Table_XGBoost["Stacking02"] <- stacking_pre02
      Table_XGBoost["MetaModel01"] <- MetaModel01_XGBoost
      Table_XGBoost["MetaModel02"] <- MetaModel02
      Table_XGBoost["KNN_Time"] <- XGBoost_knn[3]
      Table_XGBoost["NB_Time"] <- XGBoost_NB[3]
      Table_XGBoost["RF_Time"] <- XGBoost_RF[3]
      Table_XGBoost["SVM_Time"] <- XGBoost_SVM[3]
      Table_XGBoost["C50_Time"] <- XGBoost_C50[3]
      Table_XGBoost["nnet_Time"] <- XGBoost_nnet[3]
      Table_XGBoost["MLR_Time"] <- XGBoost_MLR[3]
      Table_XGBoost["Stacking01_Time"] <- get(paste("RunningTime01","XGBoost",sep = "_"))[3][[1]]
      Table_XGBoost["Stacking02_Time"] <- get(paste("RunningTime02",sep = "_"))[3][[1]]
      
      Table_Result <- NULL %>% rbind.fill(.,Table_blank) %>% rbind.fill(.,Table_chi2) %>% rbind.fill(.,Table_ETC) %>% 
        rbind.fill(.,Table_f_classif) %>% rbind.fill(.,Table_info_classif) %>% rbind.fill(.,Table_LD) %>% 
        rbind.fill(.,Table_RF) %>% rbind.fill(.,Table_XGBoost) %>% rbind.fill(.,Table_LR)# %>% rbind.fill(.,Table_SD)
      
      fwrite(x = Table_Result,
             file = paste0(path_ML,"/","Table","_",data_sequ_ml_t,".csv"),
             row.names = FALSE,col.names = TRUE, sep = ",")
      
    }
  }
}
stopImplicitCluster()


# Output logs ------------------------------------------------------------------

stacking_model02_df <- melt(stacking_model02) %>% .[,c(2,1)]
stacking_model02_df_02 <- as.data.frame(str_split_fixed(stacking_model02_df, "-", 2))
stacking_model02_df_03 <- cbind(stacking_model02_df$L1,stacking_model02_df_02)
write.table(x = stacking_model02_df_03,file = paste0(path_ML,"/","Stacking02_BaseModel.csv"),row.names = F,col.names = T,quote = F,sep = ",")
