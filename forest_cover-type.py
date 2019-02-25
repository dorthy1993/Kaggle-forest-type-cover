#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:


train = pd.read_csv("/Users/DoryChen/Downloads/forest-cover-type-prediction/train.csv")
Test=pd.read_csv("/Users/DoryChen/Downloads/forest-cover-type-prediction/test.csv")
test=Test


# In[24]:


train.head(10)


# In[25]:


pd.set_option('display.max_columns', None) 
train.describe()


# In[26]:


train = train.drop(['Soil_Type7', 'Soil_Type15'], axis = 1)
test = test.drop(['Soil_Type7', 'Soil_Type15'], axis = 1)

#Drop 'id'  iloc[row,col]
train=train.iloc[:,1:]
test=test.iloc[:,1:]


# In[27]:


corrmat = train.iloc[:,:10].corr()
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(corrmat,vmax=0.8,square=True);


# In[28]:


size=10
data=train.iloc[:,:size]
cols = data.columns
data_corr=data.corr()
threshold=0.5
corr_list=[]
for i in range(0, 10):
    for j in range(i+1, 10):
        if data_corr.iloc[i,j]>= threshold and data_corr.iloc[i,j]<1        or data_corr.iloc[i,j] <0 and data_corr.iloc[i,j]<=-threshold:
            corr_list.append([data_corr.iloc[i,j],i,j])


# In[29]:


s_corr_list = sorted(corr_list,key= lambda x: -abs(x[0]))

# print the higher values
for v,i,j in s_corr_list:
    print("%s and %s = %.2f" % (cols[i], cols[j], v))         


# In[30]:


train.Wilderness_Area2.value_counts()


# In[32]:


# Group one-hot encoded variables of a category into one single variable
cols = train.columns
r,c = train.shape

# Create a new dataframe with r rows, one column for each encoded category, and target in the end
new_data = pd.DataFrame(index= np.arange(0,r), columns=['Wilderness_Area', 'Soil_Type', 'Cover_Type'])

# Make an entry in data for each r for category_id, target_value
for i in range(0,r):
    p = 0;
    q = 0;
    # Category1_range
    for j in range(10,14):
        if (train.iloc[i,j] == 1):
            p = j-9 # category_class
            break
    # Category2_range
    for k in range(14,54):
        if (train.iloc[i,k] == 1):
            q = k-13 # category_class
            break
    # Make an entry in data for each r
    new_data.iloc[i] = [p,q,train.iloc[i, c-1]]
    
# plot for category1
sns.countplot(x = 'Wilderness_Area', hue = 'Cover_Type', data = new_data)
plt.show()

# Plot for category2
plt.rc("figure", figsize = (25,10))
sns.countplot(x='Soil_Type', hue = 'Cover_Type', data= new_data)
plt.show()


# In[33]:


#check normality of non-binary variables
train.iloc[:,:10].skew()


# In[35]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

r,c = train.shape
X_train = train.iloc[:,:c-1]
y_train = train["Cover_Type"]


# Setting parameters
x_data, x_test_data, y_data, y_test_data = train_test_split(train, y_train, test_size = 0.3)
rf_para = [{'n_estimators':[50, 100], 'max_depth':[5,10,15], 'max_features':[0.1, 0.3],            'min_samples_leaf':[1,3], 'bootstrap':[True, False]}]


# In[36]:


from sklearn.model_selection  import GridSearchCV, RandomizedSearchCV
rfc = GridSearchCV(RandomForestClassifier(), param_grid=rf_para, cv = 10, n_jobs=-1)
rfc.fit(x_data, y_data)
rfc.best_params_


# In[37]:


print ('Best accuracy obtained: {}'.format(rfc.best_score_))
print ('Parameters:')
for key, value in rfc.best_params_.items():
    print('\t{}:{}'.format(key,value))


# In[38]:


RFC = RandomForestClassifier(n_estimators=100, max_depth=15, max_features=0.3, bootstrap=False, min_samples_leaf=1,                             n_jobs=-1)
RFC.fit(X_train, y_train)
rfc_pred=RFC.predict(test)


# In[39]:


solution = pd.DataFrame({'Id':Test.Id, 'Cover_Type':rfc_pred}, columns = ['Id','Cover_Type'])
solution.to_csv('rfc_sol.csv', index=False)


# In[ ]:




