#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#### USER INPUT SECTION ###

# Global Variables #
global_source_name = "C:/Users/debas/Downloads/Python Code Library/PySpark Model Codes/Datasets/Classification_Model_Dataset.csv" 
global_source_format = "csv"
global_dep_var = 'Churn'
global_id_var = 'Phone_Number'
global_train_split = 0.8
global_seed =1234

# Model Configurations (Logistic Regression, Random Forest, Gradient Boosting, Support Vector)
model_param_max_iter = 100
model_param_max_depth = 10
model_param_max_bins = 5
model_param_n_trees = 1000
model_param_fit_intercept = True
model_param_lr_standardize = False
model_param_svm_standardize = True
model_param_elasticnet_param = 0.8
model_param_reg_param = 0.4


# In[ ]:


### ENVIORNMENT SET UP ###

# Initialize PySpark Engine #
import findspark
findspark.init()
    
# Initiate A Spark Session On Local Machine With 4 Physical Cores #
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ML_Classification_Pyspark_V1').master('local[4]').getOrCreate()


# In[ ]:


### RAW DATA IMPORT ###

from pyspark.sql.types import *
from time import *

def data_import(source_name, source_format):
    
    import_start_time = time()
    
    print("\nSpark Session Initiated Successfully. Kindly Follow The Log For Further Output\n")
    
    df = spark.read.format(source_format).option("header","true").option("inferSchema","true").load(source_name)
    
    import_end_time = time()
    import_elapsed_time = (import_end_time - import_start_time)/60
    print("\nTime To Perform Data Import: %.3f Minutes\n" % import_elapsed_time)
    
    return(df)


# In[ ]:


### USER DEFINED FUNCTION: EXPLORATORY DATA ANALYSIS (EDA) ###

from time import *
import pandas as pd
from pyspark.sql.functions import isnan, when, count, col
  
def basic_eda(df,dependent_var,id_var):
    
    eda_start_time = time()
  
  # Extracting Data Types of All Columns
    print("\n++++++ Printing Data Types of All Columns ++++++\n")
    df.printSchema()
  
  # Duplicate Observation Checking
    print("\n++++++ Printing Duplicate Removal Summary ++++++\n")
    print("Total No of Obs Before Duplicate Removal: "+str(df.count()))
    print("Unique No of Obs Before Duplicate Removal: "+str(df.distinct().count()))
  
  # Removing Duplicate Observations
    df = df.dropDuplicates()
    df = df.na.drop('all')
    print("Total No of Obs After Duplicate Removal: "+str(df.count()))
    print("Unique No of Obs After Duplicate Removal: "+str(df.distinct().count()))
  
  # Extracting Dependent and Independent Variables
    column_names = [item[0] for item in df.dtypes]
    categorical_var = [item[0] for item in df.dtypes if item[1].startswith('string')]
    independent_catgorical_var=[x for x in categorical_var if x not in [id_var,dependent_var]]
    independent_continuous_var=[x for x in column_names if x not in independent_catgorical_var+[id_var,dependent_var]]
 
  # Descriptive Summary of Numeric Variables
    temp_df_1 = pd.DataFrame()
    desc_summary_1 = pd.DataFrame()
    
    for col_name in df[independent_continuous_var].columns:
        temp_df_1.loc[0,"Column_Name"] = col_name
        temp_df_1.loc[0,"Total_Obs"] = df.agg({col_name: "count"}).collect()[0][0]
        temp_df_1.loc[0,"Unique_No_Obs"] = df.select(col_name).distinct().count()
        temp_df_1.loc[0,"Missing_No_Obs"] = df.select(count(when(isnan(col_name)
                                                             |col(col_name).isNull(), col_name))).toPandas().iloc[0,0]
        temp_df_1.loc[0,"Min"] = df.agg({col_name: "min"}).collect()[0][0]
        temp_var = df.approxQuantile(col_name,[0.01,0.05,0.1,0.25,0.5,0.75,0.85,0.95,0.99,],0)
        temp_df_1.loc[0,"Pct_1"] = temp_var[0]
        temp_df_1.loc[0,"Pct_5"] = temp_var[1]
        temp_df_1.loc[0,"Pct_10"] = temp_var[2]
        temp_df_1.loc[0,"Pct_25"] = temp_var[3]
        temp_df_1.loc[0,"Median"] = temp_var[4]
        temp_df_1.loc[0,"Average"] = df.agg({col_name: "avg"}).collect()[0][0]
        temp_df_1.loc[0,"Pct_75"] = temp_var[5]
        temp_df_1.loc[0,"Pct_85"] = temp_var[6]
        temp_df_1.loc[0,"Pct_95"] = temp_var[7]
        temp_df_1.loc[0,"Pct_99"] = temp_var[8]
        temp_df_1.loc[0,"Max"] = df.agg({col_name: "max"}).collect()[0][0]
        desc_summary_1 = desc_summary_1.append(temp_df_1)
        desc_summary_1.reset_index(inplace = True, drop = True)       

    print("\n++++++ Printing Summary Statistics For Numeric Variables ++++++\n")
    display(desc_summary_1)
    
    # Target Variables V/s Numeric Variables
    temp_df_2 = pd.DataFrame()
    desc_summary_2 = pd.DataFrame({dependent_var : 
                                   list(df.select(dependent_var).distinct().select(dependent_var).toPandas()[dependent_var])})
    
    for x in independent_continuous_var:
        temp_df_2 = df.groupby(dependent_var).agg({x: "avg"}).toPandas()
        desc_summary_2 = desc_summary_2.merge(temp_df_2,on=dependent_var, how="left")
    
    desc_summary_2 = desc_summary_2.transpose() 
    desc_summary_2.columns = desc_summary_2.iloc[0]
    desc_summary_2 = desc_summary_2[1:]
    desc_summary_2["Column_Name"] = desc_summary_2.index
    desc_summary_2.reset_index(drop=True, inplace=True)
    desc_summary_2 = desc_summary_2.iloc[:,[2,0,1]]
    desc_summary_2.index.name = None

    print("\n++++++ Printing Averages of All Numeric Variables Grouped By Target Variable ++++++\n")
    display(desc_summary_2)
    
    # Inspecting All Categorical Variables
    temp_df_3 = pd.DataFrame()
    desc_summary_3 = pd.DataFrame()
    
    for x in independent_catgorical_var:
        temp_df_3 = df.crosstab(dependent_var, x).toPandas().transpose()
        temp_df_3.columns = temp_df_3.iloc[0]
        temp_df_3 = temp_df_3[1:]
        temp_df_3["Column_Name"] = x
        temp_df_3["Value"] = temp_df_3.index
        temp_df_3 = temp_df_3.iloc[:,[2,3,0,1]]
        temp_df_3.reset_index(drop=True, inplace=True)
        desc_summary_3 = desc_summary_3.append(temp_df_3)
  
    print("\n++++++ Printing Cross Tabulation of All Categorical Variables With Target Variable ++++++\n")
    display(desc_summary_3)
    
  # Target Variable Response Rate
    desc_summary_4 = df.groupBy(dependent_var).count().sort(dependent_var,ascending=False).toPandas()
    desc_summary_4["count_pct"] = desc_summary_4["count"].div(df.agg({id_var: "count"}).collect()[0][0])
    
    print("\n++++++ Printing Target Variable Hit Rate ++++++\n")
    display(desc_summary_4)
    
  # Returning Final Output
    desc_summary = [desc_summary_1,desc_summary_2,desc_summary_3, desc_summary_4]
    final_list = (df,independent_catgorical_var,independent_continuous_var, desc_summary)
    
    eda_end_time = time()
    eda_elapsed_time = (eda_end_time - eda_start_time)/60
    print("\nTime To Perform EDA: %.3f Minutes\n" % eda_elapsed_time)
  
    return(final_list)


# In[ ]:


### USER DEFINED FUNCTION: FEATURE ENGINEERING ###

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from time import *

def feature_engineering(df,independent_catgorical_var,independent_continuous_var,dependent_var,id_var):
    
    fe_start_time = time()
  
  # Initiating Pipeline 
    stages = []
    for categoricalCol in independent_catgorical_var:
        # Convert Categorical Variables In To Numeric Indices
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + '_Index')
        # Perform One Hot Encoding
        onehotEncoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "_classVec"])
        stages += [stringIndexer, onehotEncoder]

  # Index The Target Variable
    label_stringIdx = StringIndexer(inputCol = dependent_var, outputCol = 'label')
    stages += [label_stringIdx]
    
  # Assembling All Features
    assemblerInputs = [c + "_classVec" for c in independent_catgorical_var] + independent_continuous_var

  # Creating Feature Vector
    vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [vecAssembler]

  # Finalizing Pipeline
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(df)
    df_transformed = pipelineModel.transform(df)
    selectedCols = [id_var, 'label', 'features'] 
    final_df = df_transformed.select(selectedCols)
    print("\n++++++ Printing Structure of Final Dataset ++++++\n")
    final_df.printSchema()
    
    fe_end_time = time()
    fe_elapsed_time = (fe_end_time - fe_start_time)/60
    print("\nTime To Perform Feature Engineering: %.3f Minutes\n" % fe_elapsed_time)
    
    return(final_df)


# In[ ]:


### USER DEFINED FUNCTION: TRAIN & TEST SAMPLE CREATION USING RANDOM SAMPLING ###

from time import *

def random_sampling(final_df,train_prop, seed):
  
    sampling_start_time = time()
    
    print("\n++++++ Printing Development & Validation Sample Details ++++++\n")
    train, test = final_df.randomSplit([train_prop, 1-train_prop], seed = seed)
    
    print("Training Dataset Count: " + str(train.count()))
    train.groupby('label').agg({'label': 'count'}).show()
    
    print("Test Dataset Count: " + str(test.count()))
    test.groupby('label').agg({'label': 'count'}).show()
    
    final_list = (train,test)
    
    sampling_end_time = time()
    sampling_elapsed_time = (sampling_end_time - sampling_start_time)/60
    print("\nTime To Perform Data Split: %.3f Minutes\n" % sampling_elapsed_time)  
    
    return (final_list)


# In[ ]:


### USER DEFINED FUNCTION: LOGISTIC REGRESSION MODEL ###

from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics
import pandas as pd
from time import *

def model_dev_lr(df_train, df_test, max_iter, max_depth, fit_intercept, reg_param, elasticnet_param, lr_standardize):
    
    lr_start_time = time()
    
    # Create an Initial Model Instance
    mod_lr = LogisticRegression(labelCol='label',
                                featuresCol='features',
                                aggregationDepth=max_depth,
                                elasticNetParam=elasticnet_param,
                                fitIntercept=fit_intercept,
                                maxIter=max_iter,
                                regParam=reg_param,
                                standardization=lr_standardize)
    
    # Training The Model
    lr_final_model = mod_lr.fit(df_train)
    
    # Scoring The Model On Test Sample
    lr_transformed = lr_final_model.transform(df_test)
    lr_test_results = lr_transformed.select(['prediction', 'label'])
    lr_predictionAndLabels= lr_test_results.rdd
    lr_test_metrics = MulticlassMetrics(lr_predictionAndLabels)
    
    # Collecting The Model Statistics
    lr_cm=lr_test_metrics.confusionMatrix().toArray()
    lr_accuracy=round(float((lr_cm[0][0]+lr_cm[1][1])/lr_cm.sum())*100,2)
    lr_precision=round(float((lr_cm[0][0])/(lr_cm[0][0]+lr_cm[1][0]))*100,2)
    lr_recall=round(float((lr_cm[0][0])/(lr_cm[0][0]+lr_cm[0][1]))*100,2)
    lr_auc = round(float(BinaryClassificationMetrics(lr_predictionAndLabels).areaUnderROC)*100,2)
    
    # Printing The Model Statitics
    print("\n++++++ Printing Logistic Regression Model Accuracy ++++++\n")
    print("Accuracy: "+str(lr_accuracy)+"%")
    print("AUC: "+str(lr_auc)+"%")
    print("Precision: "+str(lr_precision)+"%")
    print("Recall: "+str(lr_recall)+"%")
    
    lr_end_time = time()
    lr_elapsed_time = (lr_end_time - lr_start_time)/60
    lr_model_stat = pd.DataFrame({"Model Name" : ["Logistic Regression"],
                                  "Accuracy" : lr_accuracy,
                                  "AUC": lr_auc, 
                                  "Precision": lr_precision,
                                  "Recall": lr_recall, 
                                  "Time (Min.)": round(lr_elapsed_time,3)})
    lr_output = (lr_final_model,lr_model_stat,lr_cm)
    print("Time To Build Logistic Regression Model: %.3f Minutes" % lr_elapsed_time)
    
    return (lr_output)


# In[ ]:


### USER DEFINED FUNCTION: RANDOM FOREST MODEL ###

from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics
import pandas as pd
from time import *

def model_dev_rf(df_train, df_test, max_depth, max_bins, n_trees):
    
    rf_start_time = time()
    
    # Create an Initial Model Instance
    mod_rf = RandomForestClassifier(labelCol='label',
                                    featuresCol='features',
                                    maxDepth=max_depth,
                                    maxBins=max_bins,
                                    numTrees=n_trees)
    
    # Training The Model
    rf_final_model = mod_rf.fit(df_train)
    
    # Scoring The Model On Test Sample
    rf_transformed = rf_final_model.transform(df_test)
    rf_test_results = rf_transformed.select(['prediction', 'label'])
    rf_predictionAndLabels = rf_test_results.rdd
    rf_test_metrics = MulticlassMetrics(rf_predictionAndLabels)
    
    # Collecting The Model Statistics
    rf_cm=rf_test_metrics.confusionMatrix().toArray()
    rf_accuracy=round(float((rf_cm[0][0]+rf_cm[1][1])/rf_cm.sum())*100,2)
    rf_precision=round(float((rf_cm[0][0])/(rf_cm[0][0]+rf_cm[1][0]))*100,2)
    rf_recall=round(float((rf_cm[0][0])/(rf_cm[0][0]+rf_cm[0][1]))*100,2)
    rf_auc = round(float(BinaryClassificationMetrics(rf_predictionAndLabels).areaUnderROC)*100,2)
    
    # Printing The Model Statitics
    print("\n++++++ Printing Random Forest Model Accuracy ++++++\n")
    print("Accuracy: "+str(rf_accuracy)+"%")
    print("AUC: "+str(rf_auc)+"%")
    print("Precision: "+str(rf_precision)+"%")
    print("Recall: "+str(rf_recall)+"%")
    
    rf_end_time = time()
    rf_elapsed_time = (rf_end_time - rf_start_time)/60
    rf_model_stat = pd.DataFrame({"Model Name" : ["Random Forest"],
                              "Accuracy" : rf_accuracy,
                              "AUC": rf_auc, 
                              "Precision": rf_precision,
                              "Recall": rf_recall, 
                              "Time (Min.)": round(rf_elapsed_time,3)})
    rf_output = (rf_final_model,rf_model_stat,rf_cm)
    print("Time To Build Random Forest Model: %.3f Minutes" % rf_elapsed_time)
    
    return (rf_output)


# In[ ]:


### USER DEFINED FUNCTION: SUPPORT VECTOR MACHINE MODEL ###

from pyspark.ml.classification import LinearSVC
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics
import pandas as pd
from time import *

def model_dev_svm(df_train, df_test, max_depth, fit_intercept, max_iter, reg_param, svm_standardize):
    
    svm_start_time = time()
    
    # Create an Initial Model Instance
    mod_svm = LinearSVC(labelCol='label',
                        featuresCol='features',
                        aggregationDepth=max_depth,
                        fitIntercept=fit_intercept,
                        maxIter=max_iter,
                        regParam=reg_param,
                        standardization=svm_standardize)
    
    # Training The Model
    svm_final_model = mod_svm.fit(df_train)
    
    # Scoring The Model On Test Sample
    svm_transformed = svm_final_model.transform(df_test)
    svm_test_results = svm_transformed.select(['prediction', 'label'])
    svm_predictionAndLabels= svm_test_results.rdd
    svm_test_metrics = MulticlassMetrics(svm_predictionAndLabels)
    
    # Collecting The Model Statistics
    svm_cm=svm_test_metrics.confusionMatrix().toArray()
    svm_accuracy=round(float((svm_cm[0][0]+svm_cm[1][1])/svm_cm.sum())*100,2)
    svm_precision=round(float((svm_cm[0][0])/(svm_cm[0][0]+svm_cm[1][0]))*100,2)
    svm_recall=round(float((svm_cm[0][0])/(svm_cm[0][0]+svm_cm[0][1]))*100,2)
    svm_auc = round(float(BinaryClassificationMetrics(svm_predictionAndLabels).areaUnderROC)*100,2)

    # Printing The Model Statitics
    print("\n++++++ Printing SVM Model Accuracy ++++++\n")
    print("Accuracy: "+str(svm_accuracy)+"%")
    print("AUC: "+str(svm_auc)+"%")
    print("Precision: "+str(svm_precision)+"%")
    print("Recall: "+str(svm_recall)+"%")

    svm_end_time = time()
    svm_elapsed_time = (svm_end_time - svm_start_time)/60
    svm_model_stat = pd.DataFrame({"Model Name" : ["Support Vector Machine"],
                                  "Accuracy" : svm_accuracy,
                                  "AUC": svm_auc, 
                                  "Precision": svm_precision,
                                  "Recall": svm_recall, 
                                  "Time (Min.)": round(svm_elapsed_time,3)})
    svm_output = (svm_final_model,svm_model_stat,svm_cm)
    print("Time To Build SVM Model: %.3f Minutes" % svm_elapsed_time)
    
    return(svm_output)


# In[ ]:


### USER DEFINED FUNCTION: GRADIENT BOOSTING MACHINE MODEL ###

from pyspark.ml.classification import GBTClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics
import pandas as pd
from time import *

def model_dev_gbm(df_train, df_test, max_depth, max_bins, max_iter):
    
    gbm_start_time = time()
    
    # Create an Initial Model Instance
    mod_gbm= GBTClassifier(labelCol='label',
                           featuresCol='features',
                           maxDepth=max_depth,
                           maxBins=max_bins,
                           maxIter=max_iter)
    
    # Training The Model
    gbm_final_model = mod_gbm.fit(df_train)
    
    # Scoring The Model On Test Sample
    gbm_transformed = gbm_final_model.transform(df_test)
    gbm_test_results = gbm_transformed.select(['prediction', 'label'])
    gbm_predictionAndLabels= gbm_test_results.rdd
    gbm_test_metrics = MulticlassMetrics(gbm_predictionAndLabels)
    
    # Collecting The Model Statistics
    gbm_cm=gbm_test_metrics.confusionMatrix().toArray()
    gbm_accuracy=round(float((gbm_cm[0][0]+gbm_cm[1][1])/gbm_cm.sum())*100,2)
    gbm_precision=round(float((gbm_cm[0][0])/(gbm_cm[0][0]+gbm_cm[1][0]))*100,2)
    gbm_recall=round(float((gbm_cm[0][0])/(gbm_cm[0][0]+gbm_cm[0][1]))*100,2)
    gbm_auc = round(float(BinaryClassificationMetrics(gbm_predictionAndLabels).areaUnderROC)*100,2)
    
    # Printing The Model Statitics
    print("\n++++++ Printing GBM Model Accuracy ++++++\n")
    print("Accuracy: "+str(gbm_accuracy)+"%")
    print("AUC: "+str(gbm_auc)+"%")
    print("Precision: "+str(gbm_precision)+"%")
    print("Recall: "+str(gbm_recall)+"%")
    gbm_end_time = time()
    
    gbm_elapsed_time = (gbm_end_time - gbm_start_time)/60
    gbm_model_stat = pd.DataFrame({"Model Name" : ["Gradient Boosting Machine"],
                                  "Accuracy" : gbm_accuracy,
                                  "AUC": gbm_auc, 
                                  "Precision": gbm_precision,
                                  "Recall": gbm_recall, 
                                  "Time (Min.)": round(gbm_elapsed_time,3)})
    gbm_output = (gbm_final_model,gbm_model_stat,gbm_cm)
    print("Time To Build GBM Model: %.3f Minutes" % gbm_elapsed_time)
    
    return(gbm_output)


# In[ ]:


### SCRIPT EXECUTION ###

# Data Import #
raw_data = data_import(global_source_name, global_source_format)

# Exploratory Data Analysis #
eda_output = basic_eda(raw_data,global_dep_var,global_id_var)

# Feature Engineering #
transformed_data = feature_engineering(eda_output[0],eda_output[1],eda_output[2],global_dep_var,global_id_var)

# Random Sampling of Test & Train Data #
sampling  = random_sampling(transformed_data,global_train_split,global_seed)

# Logistic Regression Model #
model_logistic_regression = model_dev_lr(sampling[0],
                                         sampling[1],
                                         model_param_max_iter,
                                         model_param_max_depth,
                                         model_param_fit_intercept,
                                         model_param_reg_param,
                                         model_param_elasticnet_param,
                                         model_param_lr_standardize)

# Random Forest Model #
model_random_forest = model_dev_rf(sampling[0],
                                   sampling[1],
                                   model_param_max_depth,
                                   model_param_max_bins,
                                   model_param_n_trees)

# Support Vector Machine Model #
model_support_vector = model_dev_svm(sampling[0],
                                    sampling[1],
                                    model_param_max_depth,
                                    model_param_fit_intercept,
                                    model_param_max_iter,
                                    model_param_reg_param,
                                    model_param_svm_standardize)

# Gradient Boosting Machine Model #
model_gradient_boosting = model_dev_gbm(sampling[0],
                                    sampling[1],
                                    model_param_max_depth,
                                    model_param_max_bins,
                                    model_param_max_iter)

# Collecting All Model Output #
print("\n++++++ Overall Model Summary ++++++\n")
all_model_summary = pd.DataFrame()
all_model_summary = all_model_summary.append(model_logistic_regression[1],ignore_index=True).append(model_random_forest[1],ignore_index=True).append(model_support_vector[1],ignore_index=True).append(model_gradient_boosting[1],ignore_index=True)
display(all_model_summary)

print("\n++++++ Process Completed ++++++\n")


# In[ ]:




