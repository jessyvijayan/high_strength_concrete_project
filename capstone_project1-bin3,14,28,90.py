#!/usr/bin/env python
# coding: utf-8

# ## Importing necessary libraries

# In[77]:


# Data manipulation & handling libraries
import pandas as pd 
import numpy as np 

# Data visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sci

# VIF library
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Data preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Model selection libraries
from sklearn.model_selection import train_test_split,cross_val_score

# Model evaluation libraries
from sklearn.metrics import r2_score,mean_squared_error

# Machine learning libraries
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
import xgboost 
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Hyperparameter tuning parameters
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

# Clustering
from sklearn.cluster import KMeans

# Learning curve analysis
from sklearn.model_selection import learning_curve

import warnings
warnings.filterwarnings('ignore')


# ## Loading the dataset

# In[2]:


df = pd.read_excel('Capstone Project.xlsx')


# In[3]:


df.head()


# In[4]:


df.tail()


# Problem statement: Build a predictive model to understand the strength of concrete based on the ingredients used and the age.

# ## Studying the dataset

# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


for i in df.columns:
    print(f'{i} has {df[i].nunique()} unique values')


# ### Analysis:
#     - We can conclude that age can be a categorical variable

# In[8]:


df['age'].unique()


# ## Encoding the age column

# In[9]:


df['age'].value_counts()


# In[10]:


df['age_bins'] = pd.cut(x=df['age'],bins=[1,3,14,28,90,365],include_lowest=True,
                        labels=['1 to 3','4 to 14','15 to 28','29 to 90','91 to 365'])


# In[11]:


df.head()


# In[12]:


df['age_bins'].value_counts()


# In[13]:


df['age_bins'] = df['age_bins'].replace({'1 to 3':1,'4 to 14':2,'15 to 28':3,'29 to 90':4,'91 to 365':5})


# In[14]:


df.head()


# In[15]:


df.drop('age',axis=1,inplace=True)


# ## Exploratory Data Analysis

# In[16]:


X1 = df.drop('age_bins',axis=1)


# In[17]:


X1.describe()


# ### Analysis from the describe function:
#     1.There are no missing values in the dataset
#     2.There is a chance that cement,slag,ash could be platykurtic since the standard deviation is high
#     3.Cement is positively skewed since mean is greater than median
#     4.Slag,ash is highly positively skewed which gives a hint that there could be outliers in slag
#     5.Water,coarseagg could be normally distributed since mean and median are close and standard deviation is small when compared to mean which means either it could be leptokurtic or mesokurtic
#     6.Superplasticisers,fineagg,strength could be normally distributed and there is a chance there are outliers in the upper wisker region

# ## Custom descriptive statistics function

# In[18]:


def custom_summary(dataframe):
    from collections import OrderedDict
    result = []
    
    for col in list(dataframe.columns):
        if(dataframe[col].dtype != object):
            stats = OrderedDict({'column_name':col,
                                'count':dataframe[col].count(),
                                 'no. of non-null values':dataframe[col].notnull().count(),
                                 'data_type':dataframe[col].dtype,
                                 'minimum':dataframe[col].min(),
                                 'Q1':dataframe[col].quantile(0.25),
                                 'mean':dataframe[col].mean(),
                                 'Q2':dataframe[col].quantile(0.5),
                                'Q3':dataframe[col].quantile(0.75),
                                 'maximum':dataframe[col].max(),
                                 'std dev':dataframe[col].std(),
                                 'kurtosis':dataframe[col].kurt(),
                                 'skewness':dataframe[col].skew(),
                                 'IQR':dataframe[col].quantile(0.75) - dataframe[col].quantile(0.25)
                                }) 
            result.append(stats)
            # labels for skewness
            if dataframe[col].skew() <= -1:
                sklabel = 'Highly Negatively Skewed'
            elif -1 <= dataframe[col].skew() < -0.5:
                sklabel = 'Moderately Negatively Skewed'
            elif -0.5 <= dataframe[col].skew() < 0:
                sklabel = 'Fairly Symmetric(negative)'
            elif 0 <= dataframe[col].skew() < 0.5:
                sklabel = 'Fairly Symmetric(positive)'
            elif 0.5 <= dataframe[col].skew() < 1:
                sklabel = 'Moderately Positively Skewed'
            elif dataframe[col].skew() > 1:
                sklabel = 'Highly Positively Skewed'
            else:
                sklabel = 'Error'
            stats['Skewness Comments'] = sklabel 
        
            # labels for outliers
            upper_limit = stats['Q3']+ 1.5*stats['IQR']
            lower_limit = stats['Q1']- 1.5*stats['IQR']
            if len([x for x in dataframe[col] if x < lower_limit or x > upper_limit]) > 0:
               outlier_label = 'Has Outliers'
            else:
               outlier_label = 'No Outliers'
        
            stats['Outliers Comments'] = outlier_label
            stats['No.of outliers'] = len((dataframe.loc[(dataframe[col]< lower_limit) | (dataframe[col]> upper_limit)]))
           
        resultdf = pd.DataFrame(data = result)
    return resultdf.T
        


# In[19]:


custom_summary(dataframe=X1)


# ### Analysis:
# 1. Slag,water,superplastic,fineagg and strength have outliers

# ## Multivariate analysis using regression

# In[20]:


for col in df.columns:
    if col!= 'strength':
        fig,ax1 = plt.subplots(figsize = (10,5))
        sns.regplot(x=df[col],y=df['strength'],ax=ax1).set_title(f'Relationship between {col} and strength')


# #### Analysis:
#     1. Strength and cement are highly positively correlated
#     2. Strength and slag are slightly positively correlated
#     3. Strength and ash are slightly negatively correlated
#     4. Strength and water are highly negatively correlated
#     5. Strength and superplastic are highly positively correlated
#     6. Strength and coarseagg are slightly negatively correlated
#     7. Strength and fineagg are slightly negatively correlated
#     8. Strength and age are highly positively correlated

# ## Multicollinearity test

# ### Stage 1: Correlation heatmap

# In[21]:


corr = df.corr()
plt.subplots(figsize = (8,8))
sns.heatmap(corr,annot=True)


# ### Analysis:
#     1. Cement and ash has 40% correlation
#     2. Superplastic and ash has 45% correlation
#     3. Superplastic and water has 66% correlation
#     4. Fineagg and water has 43% correlation
#     5. Ash and slag has 32% correlation
# -Conclusion: Many features have correlation greater than 30%. So we can conclude collinearity exists as stage 1 results

# ## Stage 2: Variance Inflation Factor(VIF)
#     - Formula for VIF is : 1/(1-r2)
#     - Steps:
#         1. Regress every independent variable with each other and find the R square
#         2. Find out the VIF using the above formula
#         3. If VIF is more than 5 then we can say multicollinearity exists(threshold can be 5 or 10)

# In[22]:


def VIF(features):
    vif = pd.DataFrame()
    vif['VIF Score'] = [variance_inflation_factor(features.values,i) for i in range(features.shape[1])]
    vif['Features']  = features.columns
    vif.sort_values(by=['VIF Score'],ascending= False,inplace=True)
    return vif


# In[23]:


VIF(df.drop('strength',axis=1))


# ### Analysis:
#     Many features have VIF more than 5 so we can conclude multicollinearity exists as Stage 2 results

# ### Multicollinearity with target feature

# In[24]:


def cwt(data,t_col):
    independent_variables = data.drop(t_col,axis = 1).columns
    corr_result =[]
    for col in independent_variables:
        corr_result.append(data[t_col].corr(data[col]))
    result = pd.DataFrame([independent_variables,corr_result],index=['Independent_variables','Correlation']).T
    return result.sort_values(by= ['Correlation'])


# In[25]:


cwt(df,'strength')


# ### Analysis 
#     1. Age_bins and cement has 51% and 49% correlation respectively with strength(highly correlated)
#     2. Superplastic has 34% correlation with strength
#     3. Water has -30% correlation with strength i.e., it is negatively correlated

# ## Applying PCA to treat multicollinearity

# In[26]:


def pca_func(X):
    n_comp = len(X.columns)
    
    # Feature scaling
    X = StandardScaler().fit_transform(X)
    
    # Applying PCA
    for i in range(1,n_comp):
        pca = PCA(n_components=i)
        p_comp = pca.fit_transform(X)
        evr = np.cumsum(pca.explained_variance_ratio_)
        if(evr[i-1]>0.9):
            pcs = i
            break
            
    print('Explained variance ratio after PCA is:',evr)
#creating dataframe using principal components
    col = []
    for j in range(1,pcs+1):
        col.append('PC_'+str(j))
    pca_df = pd.DataFrame(data=p_comp,columns=col)   
    return pca_df


# In[27]:


pca_df = pca_func(df.drop('strength',axis=1))


# In[28]:


pca_df.head()


# In[29]:


pca_df1 = pca_func(X1.drop('strength',axis=1))


# In[30]:


pca_df1.head()


# ## Joining PCA with age_bins and target variable

# In[31]:


transformed_df = pca_df.join(df['strength'])


# In[32]:


transformed_df.head()


# In[33]:


transformed_df1 = pca_df1.join([df['strength'],df['age_bins']])


# In[34]:


transformed_df1.head()


# ## Model Building
#     - Train test split
#     - Cross validation
#     - Hyperparameter tuning

# ### Train test split

# In[35]:


def train_and_test_split(data,y,test_size=0.3,random_state=10):
    X = data.drop(y,1)
    return train_test_split(X,data[y],test_size=test_size,random_state=random_state)


# In[36]:


def model_builder(model_name,estimator,data,t_col):
    X_train,X_test,y_train,y_test = train_and_test_split(data,t_col)
    estimator.fit(X_train,y_train)
    y_pred = estimator.predict(X_test)
    r2score = r2_score(y_test,y_pred)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    return model_name,r2score,rmse


# In[37]:


model_builder('Linear regression',LinearRegression(),transformed_df,'strength')


# In[38]:


model_builder('Linear regression',LinearRegression(),transformed_df1,'strength')


# ### Building multiple models

# In[39]:


def multiple_models(data,data1,t_col):
    col = ['Model_Name','R2 Score','RMSE']
    result = pd.DataFrame(columns=col)
    
    # Adding values to result dataframe
    result.loc[len(result)] = model_builder('Linear regression',LinearRegression(),data1,t_col)
    result.loc[len(result)] = model_builder('Lasso regression',Lasso(),data1,t_col)
    result.loc[len(result)] = model_builder('Ridge regression',Ridge(),data1,t_col)
    result.loc[len(result)] = model_builder('Decision Tree regressor',DecisionTreeRegressor(),data,t_col)
    result.loc[len(result)] = model_builder('Support Vector regressor',SVR(),data1,t_col)
    result.loc[len(result)] = model_builder('K Neighbors regressor',KNeighborsRegressor(),data1,t_col)
    result.loc[len(result)] = model_builder('Random Forest regressor',RandomForestRegressor(),data,t_col)
    result.loc[len(result)] = model_builder('Adaboost regressor',AdaBoostRegressor(),data,t_col)
    result.loc[len(result)] = model_builder('Gradient Boost regressor',GradientBoostingRegressor(),data,t_col)
    result.loc[len(result)] = model_builder('XGBoost regressor',XGBRegressor(),data,t_col)
    
    return result.sort_values(by=['R2 Score'],ascending=False,ignore_index=True)


# In[40]:


multiple_models(df,transformed_df,'strength')


# ### The top 5 models in the dataset are:
#      1. Random Forest
#      2. XGBoost
#      3. Gradient Boost
#      4. KNeighbors 
#      5. Decision Tree

# In[41]:


multiple_models(df,transformed_df1,'strength')


# ### The top 5 models in the dataset are:
#      1. XGBoost
#      2. Random Forest
#      3. Gradient Boost
#      4. Decision Tree
#      5. KNeighbors

# ## Cross Validation

# In[42]:


def kfold_cv(data,data1,t_col,cv=10):
    model_names = [LinearRegression(),Lasso(),Ridge(),SVR(),KNeighborsRegressor(),DecisionTreeRegressor(),
                   RandomForestRegressor(),AdaBoostRegressor(),GradientBoostingRegressor(),XGBRegressor()]
    scores = ['Score_LR','Score_LS','Score_RD','Score_SVR','Score_KNR','Score_DTR','Score_RFR',
             'Score_ABR','Score_GBR','Score_XGBR']
    
    for model,i in zip(model_names,range(len(scores))):
        if(i<=4):
            scores[i] = (cross_val_score(estimator=model,X=data1.drop(t_col,1),y=data1[t_col],cv=cv))
        else:
            scores[i] = (cross_val_score(estimator=model,X=data.drop(t_col,1),y=data[t_col],cv=cv))
    
    result = []
    for i in range(len(model_names)):
        score_mean = np.mean(scores[i])
        score_std = np.std(scores[i])
        model_name = type(model_names[i]).__name__
        temp = [model_name,score_mean,score_std]
        result.append(temp)
    
    result_df = pd.DataFrame(result,columns=['Model Name','R2 Score','Score Std Deviation'])
    return result_df.sort_values(by=['R2 Score'],ascending=False,ignore_index=True)


# In[43]:


kfold_cv(df,transformed_df,'strength')


# ### The top 5 models in the dataset are:
#      1. Random Forest
#      2. XGBoost
#      3. Gradient Boost
#      4. KNeighbors 
#      5. Decision Tree

# In[44]:


kfold_cv(df,transformed_df1,'strength')


# ### The top 5 models in the dataset are:
#      1. XGBoost
#      2. Random Forest
#      3. Gradient Boost
#      4. Decision Tree
#      5. KNeighbors

# ### Hyperparameter tuning

# In[79]:


def tuning(X,y,cv=10):
    # Creating the parameter grid
    param_las = {'alpha':[1e-15,1e-13,1e-11,1e-9,1e-7,1e-5,1e-5,1e-3,1e-1,0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,
                          60,70,80,90,100,200,300,400,500]}
    param_rds = {'alpha':[1e-15,1e-13,1e-11,1e-9,1e-7,1e-5,1e-5,1e-3,1e-1,0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,
                          60,70,80,90,100,200,300,400,500]} 
    param_knn = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]}
    param_dtr = {'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                'max_features':['auto', 'sqrt', 'log2','none']}
    param_adb = {'learning_rate':[0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}
    param_gboost = {'alpha':[0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}
    param_xgb = {'alpha':[0.1,0.3,0.5,0.7,0.9,1],'gamma':[0,10,20,30,40,50,60,70,80,90,100],
                'eta':[0.01,0.1,0.2]}
    param_rf = {'n_estimators':[50,100,150,200,250,300],
               'criterion':['squared_error', 'absolute_error', 'poisson']}
    
    # Hyperparameter tuning
    tune_las = RandomizedSearchCV(Lasso(),param_las,cv=cv)
    tune_rds = RandomizedSearchCV(Ridge(),param_rds,cv=cv)
    tune_knn = RandomizedSearchCV(KNeighborsRegressor(),param_knn,cv=cv)  
    tune_dtr = RandomizedSearchCV(DecisionTreeRegressor(),param_dtr,cv=cv)    
    tune_adb = RandomizedSearchCV(AdaBoostRegressor(),param_adb,cv=cv)    
    tune_gboost = RandomizedSearchCV(GradientBoostingRegressor(),param_gboost,cv=cv)    
    tune_xgb = RandomizedSearchCV(XGBRegressor(),param_xgb,cv=cv)    
    tune_rf = RandomizedSearchCV(RandomForestRegressor(),param_rf,cv=cv) 
    
    # Model fitting
    tune_models = [tune_las,tune_rds,tune_knn,tune_dtr,tune_adb,tune_gboost,tune_xgb,tune_rf]
    models = ['Lasso','Ridge','KNN','DTR','ADB','GBoost','XGB','RF']
    for i in range(len(tune_models)):
        tune_models[i].fit(X,y)
        
    for i in range(len(tune_models)):
        print('Model: ',models[i])
        print('Best parameters: ',tune_models[i].best_params_)


# In[80]:


tuning(transformed_df1.drop('strength',axis=1),transformed_df1['strength'])


# In[81]:


def CV_post_hpt(X,y,cv=10):
    score_lr = cross_val_score(LinearRegression(),X,y,cv=cv)
    score_las = cross_val_score(Lasso(alpha=0.1),X,y,cv=cv)
    score_rd = cross_val_score(Ridge(alpha=20),X,y,cv=cv)
    score_knn = cross_val_score(KNeighborsRegressor(n_neighbors=2),X,y,cv=cv)
    score_dtr = cross_val_score(DecisionTreeRegressor(criterion='friedman_mse',max_features='auto'),X,y,cv=cv)
    score_svr = cross_val_score(SVR(),X,y,cv=cv)
    score_rf = cross_val_score(RandomForestRegressor(n_estimators=300,criterion='absolute_error'),X,y,cv=cv)
    score_adb = cross_val_score(AdaBoostRegressor(learning_rate=0.9),X,y,cv=cv)
    score_gboost = cross_val_score(GradientBoostingRegressor(alpha=0.001),X,y,cv=cv)
    score_xgb = cross_val_score(XGBRegressor(alpha=0.9,gamma=0,eta=0.2),X,y,cv=cv)
    
    model_names = ['Linear Regression','Lasso','Ridge','Decision Tree Regressor','SVR','KNeighbors Regressor',
                   'Random Forest Regressor','AdaBoost Regressor','Gradient Boosting Regressor','XGB Regressor']
    scores = [score_lr,score_las,score_rd,score_dtr,score_svr,score_knn,score_rf,score_adb,score_gboost,score_xgb]
    
    result = []
    for i in range(len(model_names)):
        score_mean = np.mean(scores[i])
        score_std = np.std(scores[i])
        m_name = model_names[i]
        temp = [m_name,score_mean,score_std]
        result.append(temp)
    result_df = pd.DataFrame(result,columns=['model names','R2 Mean','R2 Std'])
    return result_df.sort_values(by='R2 Mean',ascending=False,ignore_index=True)


# In[82]:


CV_post_hpt(transformed_df1.drop('strength',axis=1),transformed_df1['strength'])


# ## Clustering

# In[50]:


labels = KMeans(n_clusters = 2,random_state = 10)
cluster = labels.fit_predict(df.drop('strength',axis=1))
sns.scatterplot(x = df['cement'],y = df['strength'],hue = cluster)


# In[55]:


def clustering(x,t_col,cluster):
    column = list(set(list(x.columns)) - set(x[t_col]))
    r = int(len(column)/2)
    if(r%2==0):
        r=r
    else:
        r+=1
    f,ax = plt.subplots(r,2,figsize=(20,18))
    a=0
    for row in range(r):
        for col in range(2):
            if(a!=len(column)):
                ax[row][col].scatter(x[t_col],x[column[a]],c=cluster)
                ax[row][col].set_xlabel(t_col)               
                ax[row][col].set_ylabel(column[a])               
                a+=1


# In[56]:


X = df.drop('strength',axis=1)


# In[57]:


for col in X.columns:
    clustering(X,col,cluster)


# ### Analysis
#     - Cement is forming clusters with all features

# In[59]:


new_df = df.join(pd.DataFrame(cluster,columns=['clusters']),how='left')
new_df.head()


# In[60]:


temp_df = new_df.groupby('clusters')['cement'].agg(['mean','median'])
temp_df.head()


# In[61]:


cluster_df = new_df.merge(temp_df,on= 'clusters',how='left')
cluster_df.head()


# In[62]:


X = cluster_df.drop(['strength','clusters'],axis=1)
y = cluster_df['strength']


# In[64]:


multiple_models(cluster_df,cluster_df,'strength')


# In[65]:


kfold_cv(cluster_df,cluster_df,'strength')


# In[66]:


CV_post_hpt(X,y)


# ## Feature importance using XGBoost

# In[67]:


X_train,X_test,y_train,y_test = train_and_test_split(cluster_df.drop('clusters',axis=1),'strength')


# In[68]:


xgb = XGBRegressor()


# In[69]:


xgb.fit(X_train,y_train)


# In[72]:


xgboost.plot_importance(xgb)


# In[73]:


f_df = cluster_df[['age_bins','cement','coarseagg','water','fineagg','slag','superplastic','strength']]
f_df.head()


# In[74]:


CV_post_hpt(f_df.drop('strength',axis=1),f_df['strength'])


# ## Learning curve analysis

# In[75]:


def generate_learning_curve(model_name,estimator,X,y,cv=10):
    train_size,train_score,test_score = learning_curve(estimator=estimator,X=X,y=y,cv=cv)
    train_score_mean = np.mean(train_score,axis=1)
    test_score_mean = np.mean(test_score,axis=1)
    plt.plot(train_size,train_score_mean,c='blue')
    plt.plot(train_size,test_score_mean,c='red')   
    plt.xlabel('Samples')
    plt.ylabel('Accuracy')
    plt.title('Learning curve for '+model_name)
    plt.legend(('Training_accuracy','Testing_accuracy'))


# In[ ]:


generate_learning_curve('Linear Regression',LinearRegression(),f_df.drop('strength',axis=1),f_df['strength'])


# In[78]:


model_names = [LinearRegression(),Lasso(),Ridge(),DecisionTreeRegressor(),SVR(),KNeighborsRegressor(),
               RandomForestRegressor(),AdaBoostRegressor(),GradientBoostingRegressor(),XGBRegressor()]
for a ,model in enumerate(model_names):
    fg = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(5,2,a+1)
    generate_learning_curve(type(model_names[a]).__name__,model,f_df.drop('strength',1),f_df['strength'])


# In[ ]:





# In[ ]:




