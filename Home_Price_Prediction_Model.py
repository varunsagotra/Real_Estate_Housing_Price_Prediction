#!/usr/bin/env python
# coding: utf-8

# ## Home_Price_Prediction_Model
# - Regression Project
# - Location : Banglore

# * Import libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np


# * Load dataset into a dataframe

# # Pipeline - df1

# In[2]:


df1 = pd.read_csv("/Users/varunsagotra/Desktop/Home_Prices.csv")
df1.head()


# In[3]:


df1.info()
# Observation : data is missing in most of the features  


# In[4]:


df1.describe()
# Observation : 
#1. bath - Max value is @extreme
#2. price - std is @extreme
#2. Need to convert categorical feature to check it's description


# In[5]:


df1.shape


# In[6]:


df1.columns


# In[7]:


df1['area_type'].unique()


# In[8]:


df1['area_type'].value_counts()


# * Based on Domain Knowledge - Drop unnecessary features

# # Pipeline - df2

# In[9]:


df2 = df1.drop(['availability','society','balcony','area_type'],axis='columns')
df2.shape


# `Data Cleaning Process` :: 
# - Handle missing values in features : Find & then do required missing value treatment

# In[10]:


df2.isnull().sum()


# In[11]:


df2.shape


# # Pipeline - df3

# In[12]:


df3 = df2.dropna()
df3.isnull().sum()
# Feature : Missing 'Location' rows we can remove as the count is too low 
# Feature : Missing 'Size' rows we can remove as the count is too low 
# Feature : Missing count is low, 
#     here we have two options either fill it as median or drop it, i have chosen the later 


# In[13]:


df3.shape


# **`Feature Engineering Process`**

# ` Addding new feature bhk as a case of data formatting ` 
# - feature added (int)bhk > (Bedrooms Hall Kitchen)

# In[14]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()


# `Explore Feature :: total_sqft`

# In[15]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[16]:


df3[~df3['total_sqft'].apply(is_float)].head(10)
# Observation : 
#1.Values are present in range 2100 - 2850, 
#2.Diffeent units are present


# **Above shows that total_sqft can be a range (e.g. 2100-2850). For such case we can just take average of min and max value in the range. There are other cases such as 34.46Sq. Meter which one can convert to square ft using unit conversion. I am going to just drop such corner cases to keep things simple**

# In[17]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None   


# # Pipeline - df4

# In[18]:


df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
df4.head(2)


# **For below row, it shows total_sqft as 2475 which is an average of the range 2100-2850**

# In[19]:


df4.loc[30]


# In[20]:


(2100+2850)/2


# ### Feature Engineering

# **Add new feature called price per square feet**

# # Pipeline - df5

# In[21]:


df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()


# In[22]:


df5_stats = df5['price_per_sqft'].describe()
df5_stats


# In[23]:


df5.to_csv("bhp.csv",index=False)


# **Examine locations which is a categorical variable. We need to apply dimensionality reduction technique here to reduce number of locations**

# In[24]:


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
location_stats


# In[25]:


location_stats.values.sum()


# In[26]:


len(location_stats[location_stats>10])


# In[27]:


len(location_stats)


# In[28]:


len(location_stats[location_stats<=10])


# ### Dimensionality Reduction

# **Any location having less than 10 data points should be tagged as "other" location. This way number of categories can be reduced by huge amount. Later on when we do one hot encoding, it will help us with having fewer dummy columns**

# In[29]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[30]:


len(df5.location.unique())


# In[31]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[32]:


df5.head(10)


# ### Outlier Removal
# - Using Business Logic

# **As a data scientist when you have a conversation with your business manager (who has expertise in real estate), he will tell you that normally square ft per bedroom is 300 (i.e. 2 bhk apartment is minimum 600 sqft. If you have for example 400 sqft apartment with 2 bhk than that seems suspicious and can be removed as an outlier. We will remove such outliers by keeping our minimum thresold per bhk to be 300 sqft**

# In[33]:


df5[df5.total_sqft/df5.bhk<300].head()


# **Check above data points. We have 6 bhk apartment with 1020 sqft. Another one is 8 bhk and total sqft is 600. These are clear data errors that can be removed safely**

# In[34]:


df5.shape


# # Pipeline - df6

# In[35]:


df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# ### Outlier Removal
# - Using Standard Deviation & Mean

# In[36]:


df6.price_per_sqft.describe()


# **Here we find that min price per sqft is 267 rs/sqft whereas max is 12000000, this shows a wide variation in property prices. We should remove outliers per location using mean and one standard deviation**

# # Pipeline - df7

# In[37]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# **Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like**

# In[38]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    plt.rcParams['figure.figsize'] = (10,8)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK')
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK')
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")


# In[39]:


plot_scatter_chart(df7,"Hebbal")


# **We should also remove properties where for same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom apartment (with same square ft area). What we will do is for a given location, we will build a dictionary of stats per bhk, i.e.**
# ```
# {
#     '1' : {
#         'mean': 4000,
#         'std: 2000,
#         'count': 34
#     },
#     '2' : {
#         'mean': 4300,
#         'std: 2300,
#         'count': 22
#     },    
# }
# ```
# **Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment**

# # Pipeline - df8

# In[40]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# **Plot same scatter chart again to visualize price_per_sqft for 2 BHK and 3 BHK properties**

# In[41]:


plot_scatter_chart(df8,"Rajaji Nagar")


# In[42]:


plot_scatter_chart(df8,"Hebbal")


# **Based on above charts we can see that data points highlighted in red below are outliers and they are being removed due to remove_bhk_outliers function**

# In[43]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (15,10)
plt.hist(df8.price_per_sqft,rwidth=0.8, color= 'green')
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# ### Outlier Removal
# - Using bathroom feature as per business rules

# In[44]:


df8.bath.unique()


# In[45]:


plt.hist(df8.bath,rwidth=0.8,color = 'green')
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")
plt.show()


# In[46]:


df8[df8.bath>10]


# **It is unusual to have 2 more bathrooms than number of bedrooms in a home**

# In[47]:


df8[df8.bath>df8.bhk+2]


# **Again the business manager has a conversation with you (i.e. a data scientist) that if you have 4 bedroom home and even if you have bathroom in all 4 rooms plus one guest bathroom, you will have total bath = total bed + 1 max. Anything above that is an outlier or a data error and can be removed**

# # Pipeline - df9

# In[48]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[49]:


df9.head(2)


# # Pipeline - df10

# In[50]:


df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)


# ### Encoding Performed - To convert categorical feature to numeric feature
# - One Hot Encoding technique used :: On feature "Location"

# In[51]:


dummies = pd.get_dummies(df10.location)
dummies.head(3)


# # Pipeline - df11

# In[52]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# # Pipeline - df12

# In[53]:


df12 = df11.drop('location',axis='columns')
df12.head(2)


# ### Model Building

# In[54]:


df12.shape


# In[55]:


X = df12.drop(['price'],axis='columns')
X.head(3)


# In[56]:


X.shape


# In[57]:


y = df12.price
y.head(3)


# In[58]:


len(y)


# In[59]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[60]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# ### Using Cross Validation :: To measure accuracy of Linear Regression Model
# - K fold cross validation

# In[61]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# **We can see that in 5 iterations we get a score above 80% all the time. This is pretty good but we want to test few other algorithms for regression to see if we can get even better score. We will use GridSearchCV for this purpose**

# ### Time to find : Best Model using GridSearchCV Method
# - Linear Regression
# - Lasso REgression
# - Decision Tree

# In[62]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# **Based on above results we can say that LinearRegression gives the best score. Hence we will use that.**

# ### Model Testing for few properties
# - Verify if build model is performing as expected

# In[63]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[64]:


predict_price('1st Phase JP Nagar',1000, 2, 2)


# In[65]:


predict_price('1st Phase JP Nagar',1000, 3, 3)


# In[66]:


predict_price('Indira Nagar',1000, 2, 2)


# In[67]:


predict_price('Indira Nagar',1000, 3, 3)


# ### Export the Model - To pickle file

# In[68]:


import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# ### Export features info : To json file
# - Features such as location and so on...

# In[69]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# In[ ]:




