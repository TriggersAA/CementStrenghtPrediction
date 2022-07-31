#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#The-Why" data-toc-modified-id="The-Why-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>The Why</a></span></li><li><span><a href="#Importing-Libraries-" data-toc-modified-id="Importing-Libraries--2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Importing Libraries <a name="anc1" rel="nofollow"></a></a></span></li><li><span><a href="#Dealing-with-Nan-Values" data-toc-modified-id="Dealing-with-Nan-Values-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Dealing with Nan Values</a></span></li><li><span><a href="#Data-cleaning" data-toc-modified-id="Data-cleaning-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Data cleaning</a></span></li><li><span><a href="#Removing-outliers" data-toc-modified-id="Removing-outliers-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Removing outliers</a></span><ul class="toc-item"><li><span><a href="#Feature-Transformation" data-toc-modified-id="Feature-Transformation-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Feature Transformation</a></span></li></ul></li><li><span><a href="#Initial-Data-Visualization" data-toc-modified-id="Initial-Data-Visualization-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Initial Data Visualization</a></span><ul class="toc-item"><li><span><a href="#Heat-Mapping" data-toc-modified-id="Heat-Mapping-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Heat Mapping</a></span></li><li><span><a href="#PairPlotting" data-toc-modified-id="PairPlotting-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>PairPlotting</a></span></li></ul></li><li><span><a href="#Feature-Engineering" data-toc-modified-id="Feature-Engineering-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Feature Engineering</a></span><ul class="toc-item"><li><span><a href="#Feature-Selection" data-toc-modified-id="Feature-Selection-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Feature Selection</a></span></li><li><span><a href="#Feature-scalling" data-toc-modified-id="Feature-scalling-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Feature scalling</a></span></li></ul></li><li><span><a href="#Model-Training" data-toc-modified-id="Model-Training-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Model Training</a></span><ul class="toc-item"><li><span><a href="#Spliting-our-Data" data-toc-modified-id="Spliting-our-Data-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Spliting our Data</a></span></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Conclusion</a></span></li></ul></div>

# # The Why
# Knowing the mix ratio for concrete to obtain a specific strength in MPa is a strenuous task that requires lots of trails especially for high strength concrete. Based on common materials and methods, we thus present an ML approach to limit the mix trails.
# The materials are Blast,Cement,Coarse Aggregate, Fine Aggregate,Fly ash, Superplasticizers and Water all in  m^3 . 
# The method is Age in days
# The result is Compressive Strength in MPa

# # Importing Libraries <a name="anc1"></a>

# In[309]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import statsmodels.api as sm

import scipy.stats as scp
get_ipython().run_line_magic('matplotlib', 'inline')
from math import floor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")


# In[310]:


#Reading the CSV file from system
All_concrete = pd.read_csv("./concrete.csv")


# In[311]:


All_concrete.columns


# In[312]:


All_concrete.shape  #gives the shape of the dataset array


# In[313]:


types = All_concrete.dtypes
nonFloat = All_concrete.dtypes[types == str]
print(f"Non-integer types are {nonFloat}")


# In[314]:


All_concrete.head() #sampling to get a glimpse of the data


# The dataset is of different units of float

# In[315]:


All_concrete.describe() 


# # Dealing with Nan Values

# In[316]:


All_concrete.info(verbose=True)


# There are non-null values in the whole dataset. This makes the whole process smoother

# # Data cleaning

# In[317]:


duplicates = All_concrete[All_concrete.duplicated()]
duplicates.shape


# In[318]:


All_concrete = All_concrete[~All_concrete.duplicated(keep='first')] #removing duplicates from the dataset
All_concrete.shape
cols = All_concrete.columns


# In[319]:


#creating a boxplot function for easy visualization and cleaning for every column in the dataset
def boxPlot():
    f = plt.figure(figsize=(15,15))
    cols = All_concrete.columns
    flr = floor(len(cols)/3) +1
    for col in cols:
        plt.subplot(flr,3,list(cols).index(col)+1)
        sns.boxplot(All_concrete[col])
    plt.show()
boxPlot()


# Fine aggregate really has much outliers below the first quartile. A testimony that the removed outlier has greatly affected the mean before. We might adjust this by other measures. Age is also normally tested within the first 21 days. Thus we see the density skewed to those region. We might also 'engineer' it to achieve better normal distribution.

# # Removing outliers

# Testing of concrete at high number of days is not really the ideal and based on the normal distribution of concrete strength, the values won't differ much from third deviation from mean.
# Fine aggregate has one point outlier which is conspicous.
# Excessive water too is not really the ideal.

# In[320]:



#dropping outliers using standard deviation

# As there is one extreme outlier...
maxfineAgg = All_concrete['fineagg'].max()
All_concrete = All_concrete[(All_concrete['fineagg'] < maxfineAgg)]


def outlierRemover(column,data):
    upperLimit = data[column].mean() + 3*data[column].std()
    lowerLimit = data[column].mean() - 3*data[column].std()
    data = data[(data[column] < upperLimit) & (data[column] > lowerLimit)]
    return data
def outlierRemoverQuatile(column,data):
    q3 = data[column].quantile(0.95)
    data = data[(data[column] < q3)]
    return data

All_concrete = outlierRemover('age',All_concrete)
#experimentally, removing only the upperbound for age returns thesame shape
All_concrete = outlierRemoverQuatile("fineagg",All_concrete)


All_concrete = outlierRemoverQuatile("water",All_concrete)
All_concrete.shape

#box-plots again....
def ageWaterFine():
    f = plt.figure(figsize=(15,3))
    plt.subplot(1,3,1)
    sns.boxplot(All_concrete['age'])
    plt.subplot(1,3,2)
    sns.boxplot(All_concrete['fineagg'])
    plt.subplot(1,3,3)
    sns.boxplot(All_concrete['water'])
ageWaterFine()
plt.show()


# ## Feature Transformation

# In[321]:



minfineAgg = All_concrete['fineagg'].min()
minfy =  All_concrete[(All_concrete['fineagg'] == minfineAgg)].shape
print(minfy)
All_concrete['age'] = All_concrete['age']**0.3
All_concrete["fineagg"] = (All_concrete['fineagg']**2)/1000


# we have reduced the outliers by modifying the features.

# In[322]:


#getting a glimpse of zero valued entries 
shapeOfAll = All_concrete.shape
zeroValues = map(lambda x: len(All_concrete[x][All_concrete[x] ==0]),All_concrete.columns)
zeroValues = np.array(list(zeroValues))
zeroValuesPercentage = zeroValues/shapeOfAll[0]
print(zeroValues/shapeOfAll[0])


# In[323]:


zeroDataFrame


# Flyash has the most zero values and is least represented followed by Slag. Ash can be used to reduce water requirement and more workability thus its usage is secondary. Slag substitutes cement which can also be secondary in usage too.

# # Initial Data Visualization

# In[324]:


f = plt.figure(figsize=(20,15))
for col in cols:
    flr = floor(len(cols)/3) +1
    plt.subplot(flr,3,list(cols).index(col)+1)
    sns.distplot(All_concrete[col])
plt.show()


# It can be seen that cement is skewed to the left. Water,fineagg, coarseagg and strength are approximately normally distributed. Superplastic, ash and slag are conspicuously positively skewed. Age is also positively skewed

# In[325]:


#unique values in age
uniqueAges = (All_concrete['age'].unique())
uniqueAges.reshape(1,-1)


# In[326]:


f = plt.figure(figsize=(15,6))
roundedAge = np.round(All_concrete['age']**3.33,2)
sns.barplot(x=roundedAge, data=All_concrete, y='strength')
plt.legend([],[], frameon=True)
plt.show()


# 91 days has the most value thus we see that most tests in this sample are done on the 91st day followed by the 56th day. Thus we might have had a false impression that outlier days like days greater than 180days doesn't have much tests for strength. For example, tests on day 180 are more than those of third day. We should endeavor not to remove those seeming outliers.

# ## Heat Mapping

# In[327]:


plt.figure(figsize=(15,5))
cor_matrix = All_concrete.corr()
sns.heatmap(cor_matrix,annot=True, cmap="YlGn")
plt.show()


# From the heatMap, it can be inferred that:
# . Quantity of cement,amount of superplastic and Age greatly influnce the strength of concrete.
# . Slag has small positive correlation.
# . Water quantity also has high influnce but a negative correlation which is reasonable.
# . Ash, coarse and fine aggregate all have small negative correlation with ash having the least.
# 
#     It can also be observed that superplastic and water have strong negative corelation. Superplasticizers work to decrease the water cement ratio hence also increasing workability of concrete.

# ## PairPlotting

# In[328]:


sns.pairplot(All_concrete, corner=True)
plt.show()


# It can be observed that no two features have exempting strenght have reasonable correlation. 
# 
# ON STRENGTH
#     
# Cement has a good positive correlation pattern relative to others assuming homoscedasticity.
# Superplastic plot with strength also show some correlation in the scatter plot.
# Age as can be counted has finite groups of point segments.
# Others have weak correlation representations in the plots as noticed earlier in the heatmap.
# 

# # Feature Engineering
#     
# Based on component of cementitious component = cement, cement + Slag
# Based on water-cemnt ratio = water/cement
# Based on ratio of water to cementitious component = water/cement+ Slag
# Based on component for increasing workability = water + superplastic, flyash + superplastic
# Based on volume of aggregate = coarse + fine
# 

# In[329]:


#so we engineer ...
All_concrete['cement_slag'] =  All_concrete['cement'] + All_concrete['slag']
All_concrete['water_cement'] =  All_concrete['water']/ All_concrete['cement']
All_concrete['water_cementitious'] = All_concrete['water'] / All_concrete['cement_slag']
All_concrete['water_superplastic'] =  All_concrete['water'] + All_concrete['superplastic']
All_concrete['flyash_superplastic'] =  All_concrete['ash'] + All_concrete['superplastic']
All_concrete['allAggregates'] =  All_concrete['fineagg'] + All_concrete['coarseagg']

#cementitious['str']= All_concrete['strength']
print(All_concrete.shape)


# Let us perform P-test on all features to determine their significance at 95% confidence level wrt to significant impact on the dependent variable STRENGTH.
#     
# We may end up removing some features with high P values
#         
# We should at this point create our dependent and independent variables from All_concrete and also split our data.

# ## Feature Selection

# In[330]:


#we perform a statistical test to know the properties of our dataframe like skewness, feature significance etc
X = All_concrete.drop(['strength'], axis=1)
y = All_concrete['strength']
t,p_test = scp.ttest_ind(a = y.to_numpy(), b = X['ash'].to_numpy())
print("{:.6f}".format(t))

#X_ols = X.iloc[:,:X.shape[1]].values
#X_ols = np.append(arr = np.ones((X.shape[0],1)).astype(int), values=X_ols, axis=1)
#performing  OLS
X_ols2 = sm.add_constant(X)
result = sm.OLS(y, X_ols2).fit()
print(result.summary())


# We have very high F-statistic probabablity which means that our variables explain our independent variables and we can reject the null hypothesis that they do not relate at 95% confidence level.
#     
# Age has high coefficient of 11.98 relatively due to the fact that we have modified it to adjust normally.
#     
# By Prob(Omnibus) , the residuals are normally distributed.
#  The homoscedasticity is also less than 2 which is ideal.
#  By Prob(JB), we also accept that data is normally distributed.
#  The condition number  is also very high indicating high multicollinearity.
#  
#  
#  We should also drop some engineered variables with P values > 0.05

# In[331]:


X = X.drop(['water_superplastic'], axis= 1)


# ## Feature scalling

# Some features like coarseagg, fineagg are 10 folds of the others. To ensure quick convergence , we should rescale

# In[332]:



scaler = StandardScaler()
anX = scaler.fit_transform(X)
anX[1,:]


# # Model Training

# ## Spliting our Data 

# In[333]:


#splitting at 70% train size
X_train, X_test, y_train, y_test = train_test_split(anX,y, train_size=0.7, random_state=2)


# In[334]:


#fiting simple linear regression model
linMod = LinearRegression()
linMod.fit(X_train,y_train)

trainScore = linMod.score(X_train ,y_train)
testScore = linMod.score(X_test ,y_test)
r2Score = metrics.mean_squared_error(y_train, linMod.predict(X_train), squared= False)
print(f'train score is: {trainScore}')
print(f'test score is: {testScore}')
print(r2Score)

#store the results in a dataframe
resultDF = pd.DataFrame()
resultDF = pd.DataFrame({'model':'linearReg','r2Score':r2Score,'Train Performance':trainScore,                         'Test performance':testScore},index=[1])


# This, we have slight overfitting with it. 

# In[335]:


# fitting with Lasso and Ridge with polynomials 
def linearmethods(name, method,ind):
    apipe = Pipeline([('polynomial',PolynomialFeatures(degree=2,interaction_only=True)),(name,method)])
    #fitting model
    apipe.fit(X_train,y_train)
    trainScore =apipe.score(X_train,y_train)
    testScore = apipe.score(X_test,y_test)
    r2score =metrics.mean_squared_error(y_train, apipe.predict(X_train), squared= False)
    data_temp = pd.DataFrame({'model':name,'r2Score':r2Score,'Train Performance':trainScore,                             'Test performance':testScore},index=[ind])
    return data_temp
    
ridgeReg = RidgeCV(alphas = [0.01,0.1,1,3],cv=5)
lassoReg = LassoCV(alphas = [0.01,0.1,1,3],cv=5)
lmRidge = linearmethods('RidgeCV',ridgeReg,2)
resultDF = pd.concat([resultDF,lmRidge])
lmLasso = linearmethods('LassoCV',lassoReg,3)
resultDF = pd.concat([resultDF,lmLasso])

resultDF


# In[336]:


#for easy addition into dataframe, we create a result updater 

def resultUpdater(name,method):
    if not (name in resultDF['model'].unique()):
        r2score =metrics.mean_squared_error(y_train, method.predict(X_train), squared= False)
        resultDF.loc[len(resultDF.index)+1] = [name,r2score,method.score(X_train, y_train),method.score(X_test, y_test)]


# In[337]:


svc = SVR()
svc_params = {'kernel':('rbf', 'linear', 'quadratic'), 'C':[30,90,100,150,300]}
grid_svr_cv = GridSearchCV(svc, svc_params, cv=5)
grid_svr_cv.fit(X_train, y_train)
grid_svr = grid_svr_cv.best_estimator_
grid_svr.fit(X_train, y_train)
print(grid_svr_cv.best_params_)


# In[338]:



# r2score =metrics.mean_squared_error(y_train, grid_svr.predict(X_train), squared= False)
# resultDF.loc[len(resultDF.index)+1] = ['SVR',r2score,grid_svr.score(X_train, y_train),grid_svr.score(X_test, y_test)]
# resultDF
resultUpdater('SVR',grid_svr)
resultDF


# We notice an overfitting in support vector regression as the test score is considerably lesser than the train score

# We thus try lazy regressor; KNN regressor

# In[339]:


neighs = KNeighborsRegressor(n_neighbors=3)
neighs.fit(X_train,y_train)
resultUpdater('KNN',neighs)
# r2score =metrics.mean_squared_error(y_train, neighs.predict(X_train), squared= False)
# resultDF.loc[len(resultDF.index)+1] = ['KNN',r2score,neighs.score(X_train, y_train),neighs.score(X_test, y_test)]
resultDF


# The K Neighbours method performed worse on the test and train datasets than SVR and the linear methods above. SVR is thus the best so far.

# We thus try ensemble methods with bagging

# In[340]:


RFR = RandomForestRegressor(n_estimators=150,max_depth=15, random_state=0)
#fit the model
RFR.fit(X_train,y_train)
#plot the feature importance with sns and update the dataframe results
plt.figure(figsize=(15,4))         
ax = sns.barplot(x =X.columns ,y = RFR.feature_importances_)
ax.tick_params(axis='x', rotation=45)
plt.ylabel('Importance Factor')
plt.show()
resultUpdater('RandomForest',RFR)


# In[341]:


ETR = ExtraTreesRegressor(max_depth=15, random_state=1)
ETR.fit(X_train,y_train)
#plot the feature importance with sns and update the dataframe results
plt.figure(figsize=(15,4))
ax = sns.barplot(x =X.columns ,y = ETR.feature_importances_)
ax.tick_params(axis='x', rotation=45)
plt.ylabel('Feature Importance')
plt.show()
resultUpdater('ExtraTrees',ETR)
resultDF


# The RandomForestRegressor(RFR) performed better with a significantly reduced rmse error but the overfitting factor is still noticed with almost 10% difference in performance 
# The results from ExtraTreesRegressor(ETR) is the best so far with over 99.9% trainset performance and almost 90% perfrmance on test set.
# 
#     Juxtaposing ETR and RFR shows that they both have almost thesame consideration for all features except for Age and Water_cementitious, cement_slag and water_cement. Indeed ETR spread its consideration for cement related features while RFR greatly considered water_cementitious above other feeatures with cement.

# # Conclusion
# 

# 1. We were able to come up with addtional features that are germane and helped with our model.
# 
# 2. WE were able to predict the compressive strength by ordinary LSM with almost thesame test and train accuracy. This shows a nice model with accuracy of 82.7%.
# 
# 3. Using regularization and higher order polynomials, we were able to improve our accuracy but at the expense of a bit overfitting.
# 
# 4. Using Support Vector Mechanics, we got better. This was thanks to the Rbf kernel that helped us with domain transformation for a bettter result. The Standardization of features also helped to smoothen the SVR method.
# 
# 5. We thus finally were able to atrain an accuracy of 89.7% on testing using ExtraTreesRegressor which is our highest accuracy.
# 
