#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas_datareader as pdr
import datetime
import bs4 as bs
import os
import pandas_datareader.data as web
import pickle
import requests


# In[2]:


# get sp500 tickers
def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers


# In[3]:


tickers = sorted(save_sp500_tickers())
tickers[0]


# In[4]:


# get first stock ('A') from yahoo

start_date = datetime.date(2018,1,1)
end_date = datetime.date.today()
current_ticker = pdr.get_data_yahoo(symbols='A', start = start_date, end = end_date)
current_ticker['Ticker'] = len(current_ticker) * ['A']
current_ticker.reset_index('Date',inplace = True)
sp500 = current_ticker
sp500.head()


# In[5]:


# get all S&P 500 stocks from yahoo

# successful stocks
i = 1  # already add stock 'A'

for ticker in tickers[1:]:
    try:
        current_ticker = pdr.get_data_yahoo(symbols=ticker, start = start_date, end = end_date)
    except:
        print("Error : skipping",ticker)
        continue
    current_ticker['Ticker'] = len(current_ticker) * [ticker]
    current_ticker.reset_index('Date',inplace = True)
    sp500 = sp500.append(current_ticker)
    print(ticker,"finished")
    i += 1


# In[6]:


i


# In[7]:


sp500


# In[8]:


# export to csv as original

sp500_original = sp500.reset_index(drop = True) # reset index
sp500_original.rename(columns={'Date':'Timestamp'}, inplace=True) # change column name

# reorder the columns
cols = sp500_original.columns.tolist() 
cols = cols[:1] + cols[-1:] + cols[1:-1]
sp500_original = sp500_original[cols]
sp500_original.head()


# In[9]:


sp500_original.to_csv(r'/Users/panzichen/Global_AI/Zichen Pan - S&P500(Original).csv', index = None, header=True)


# In[10]:


# export to csv as requested

sp500_df = sp500_original
sp500_df.rename(columns={'Close':'Daily Closing Price', 'Volume':'Daily Volume'}, inplace = True)
sp500_df.drop(['High', 'Low', 'Open', 'Adj Close'], axis = 1,inplace = True)
sp500_df.head()


# In[11]:


sp500_df.to_csv(r'/Users/panzichen/Global_AI/Zichen Pan - S&P500(Requested).csv', index = None, header=True)


# ## Data Cleaning

# In[12]:


sp500_df.isnull().sum()


# ## Create SQL Table

# In[13]:


import sqlite3


# In[14]:


# connecting to the database  
connection = sqlite3.connect("sp500.db")


# In[15]:


# cursor  
crsr = connection.cursor()


# In[16]:


# drop table if necessary (to rerun the create command, or error: table already exists)
crsr.execute("""DROP TABLE sp500_df""")


# In[17]:


crsr.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = crsr.fetchall()
print(tables)


# In[18]:


# SQL command to create a table in the database 
sql_command_create = """CREATE TABLE sp500_df (
NUM INTEGER PRIMARY KEY,
Ticker VARCHAR(20),
Timestamp DATE,  
Close FLOAT(8,6),  
Volume INTEGER);"""


# In[19]:


crsr.execute(sql_command_create)


# In[20]:


crsr.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = crsr.fetchall()
print(tables)


# In[21]:


# Insert data
for index, row in sp500_df.iterrows():
    format_str = """INSERT INTO sp500_df (NUM, Ticker, Timestamp, Close, Volume)
    VALUES ("{num}", "{ticker}", "{timestamp}", "{close}", "{volume}");"""

    sql_command = format_str.format(num = index, ticker = row['Ticker'], timestamp = row['Timestamp'],
                                    close = row['Daily Closing Price'], volume = int(row['Daily Volume']))
    crsr.execute(sql_command)


# In[22]:


# print data
import pprint
sql_select = """SELECT * from sp500_df;"""
crsr.execute(sql_select)
pprint.pprint(crsr.fetchall())


# In[23]:


# To save the changes in the files. Never skip this.  
# If we skip this, nothing will be saved in the database. 
connection.commit()


# In[24]:


# close the connection 
connection.close()


# ## EDA

# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[26]:


sp500_df.head()


# In[27]:


sp500_plot = sp500_df.set_index("Timestamp")
sp500_plot.head()


# In[28]:


# Take Stock A as an exmple


# In[29]:


# data visualization of stock A Daily Closing Price by day
ax = sp500_plot.loc[sp500_plot['Ticker'] == 'A', ['Daily Closing Price']].plot(figsize = (20,15))
#ax.lines[0].set_alpha(0.3)
_ = plt.title('Stock A Daily Closing Price', fontdict = {'fontsize':30})


# In[30]:


# data visualization of stock A Daily Volume by day
ax = sp500_plot.loc[sp500_plot['Ticker'] == 'A', ['Daily Volume']].plot(figsize = (20,15))
#ax.lines[0].set_alpha(0.3)
_ = plt.title('Stock A Daily Volume', fontdict = {'fontsize':30})


# In[31]:


# Two features in one plot after Normalization

A_df = sp500_plot.loc[sp500_plot['Ticker'] == 'A', ['Daily Closing Price', 'Daily Volume']]

from sklearn import preprocessing
import numpy as np
import pandas as pd
close_array = np.array(A_df['Daily Closing Price'])
volume_array = np.array(A_df['Daily Volume'])
normalized_close = preprocessing.normalize([close_array])
normalized_volume = preprocessing.normalize([volume_array])
A_df['Daily Closing Price'] = normalized_close.flatten()
A_df['Daily Volume'] = normalized_volume.flatten()
A_df.head()


# In[32]:


# data visualization of stock A Daily Volume by day
ax = A_df.plot(figsize = (20,15))
_ = plt.title('Stock A Daily Data After Normalization', fontdict = {'fontsize':30})


# In[33]:


# data visualization of stock AA Closing Price by rolling of a week
ax = sp500_plot.loc[sp500_plot['Ticker'] == 'A', ['Daily Closing Price']].rolling(7)                                                    .mean().plot(style=['-',':'], figsize=(20,20))
_ = plt.title('Stock AA Daily Closing Price By Rolling of a Week', fontdict = {'fontsize':30})


# In[34]:


# data visualization of stock AA Volume by rolling of a week
ax = sp500_plot.loc[sp500_plot['Ticker'] == 'A', ['Daily Volume']].rolling(7)                                                    .mean().plot(style=['-',':'], figsize=(20,20))
_ = plt.title('Stock AA Daily Volume By Rolling of a Week', fontdict = {'fontsize':30})


# In[35]:


# Correlation 
sp500_plot.corr()


# In[36]:


df_close = sp500_plot.loc[sp500_plot['Ticker'] == 'A', ['Daily Closing Price']]
pd.plotting.autocorrelation_plot(df_close);


# In[37]:


df_volume = sp500_plot.loc[sp500_plot['Ticker'] == 'A', ['Daily Volume']]
pd.plotting.autocorrelation_plot(df_volume);


# ## Clustering

# In[38]:


sp500_df.head()


# In[39]:


sp500_mean = sp500_df.groupby('Ticker').mean()
sp500_mean.reset_index('Ticker', inplace = True)
sp500_mean


# In[40]:


X_train = sp500_mean.loc[:, ['Daily Closing Price', 'Daily Volume']]
X_train


# ### Kmeans

# In[41]:


from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist


# In[42]:


# Elbow method to determine K
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X_train)
    kmeanModel.fit(X_train)
    distortions.append(sum(np.min(cdist(X_train, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X_train.shape[0])


# In[43]:


# Plot the elbow
_ = plt.plot(K, distortions, 'bx-')
_ = plt.xlabel('k')
_ = plt.ylabel('Distortion')
_ = plt.title('The Elbow Method showing the optimal k')


# In[44]:


# K = 5


# In[45]:


# create a KMeans object which will generate 5 clusters
km = KMeans(n_clusters=5)


# In[46]:


# use fit_predict() to both fit our k-means model and generate cluster assignments 
cluster_assignments = km.fit_predict(X_train)
cluster_assignments[:10]


# In[47]:


# Visualization of Clustering
fig = plt.figure(figsize=(10,10))
for i in range(5):
    X_subset = X_train[cluster_assignments == i]
    plt.scatter(X_subset.iloc[:,0], X_subset.iloc[:,1],s = 80,alpha = 0.8, label = 'cluster '+str(i))
    plt.plot(km.cluster_centers_[i][0],km.cluster_centers_[i][1],marker='x',c='k', ms=20, mew=5, label=None);
_ = plt.legend();
_ = plt.xlabel('Daily Closing Price');
_ = plt.ylabel('Daily Volume')
_ = plt.title('Clustering', fontdict = {'fontsize':30})


# In[48]:


# Add the cluster assignment to sp500_mean
sp500_mean['Cluster'] = cluster_assignments
sp500_mean


# ## Descriptive Statistics

# In[49]:


from scipy.stats import trim_mean, kurtosis
from scipy.stats.mstats import mode, gmean, hmean


# In[50]:


sp500_plot.describe()


# In[51]:


df_group_daily_close = sp500_mean.groupby('Cluster')['Daily Closing Price']
df_group_daily_close.describe().sort_values('mean')


# In[52]:


df_group_daily_volume = sp500_mean.groupby('Cluster')['Daily Volume']
df_group_daily_volume.describe().sort_values('mean')


# In[53]:


# trim mean for Daily Closing Price sorted
df_group_daily_close.apply(trim_mean, .1).sort_values().reset_index()


# In[54]:


# trim mean for Daily Volume sorted
df_group_daily_volume.apply(trim_mean, .1).sort_values().reset_index()


# ## Prediction 

# In[55]:


import warnings
import itertools
plt.style.use('fivethirtyeight')
import statsmodels.api as sm


# In[56]:


# We predict the Daily Closing Price of Stock A
df_A = sp500_plot.loc[sp500_plot['Ticker'] == 'A', ['Daily Closing Price']]
df_A.head()


# ### Check Stationarity

# In[57]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(7).mean()
    rolstd = timeseries.rolling(7).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)


# In[58]:


test_stationarity(df_A)


# In[59]:


# We can tell that the dataset is approximately stationary


# In[60]:


df_A_weekly = df_A.resample('B').mean()
df_A_weekly.fillna(method='ffill', inplace = True)
df_A_weekly


# In[61]:


_ = df_A_weekly.plot(figsize = (20,6))


# ### ARIMA

# In[62]:


# ARIMA
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(df_A_weekly,                                            order=param,                                            seasonal_order=param_seasonal,                                            enforce_stationarity=False,                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[63]:


# SARIMAX(1, 1, 1)x(0, 1, 1, 12) yields the lowest AIC value of 810.1769504153424


# In[64]:


mod = sm.tsa.statespace.SARIMAX(df_A_weekly,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[65]:


_ = results.plot_diagnostics(figsize=(20, 8))


# In[66]:


pred = results.get_prediction(start=pd.to_datetime('2019-01-01'), dynamic=False)
pred_confidence_interval = pred.conf_int()
ax = df_A_weekly['2018':].plot(label='observed')
_ = pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(20, 8))
ax.fill_between(pred_confidence_interval.index,
                pred_confidence_interval.iloc[:, 0],
                pred_confidence_interval.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Timestamp')
ax.set_ylabel('Weekly Mean Closing Price')
_ = plt.title('Stock A Weekly Closing Price', fontdict = {'fontsize':30})
_ = plt.legend()


# In[67]:


# RMSE
A_forecasted = pred.predicted_mean
A_truth = df_A_weekly['2019-01-01':].loc[:,'Daily Closing Price']
mse = ((A_forecasted - A_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


# In[68]:


# Prediction Visualization
pred_uc = results.get_forecast(steps=240)
pred_ci = pred_uc.conf_int()
ax = df_A_weekly.plot(label='observed', figsize=(20, 7))
_ = pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Timestamp')
ax.set_ylabel('Weely Closing Price')
_ = plt.title('Stock A Weekly Closing Price Prediction In 2019', fontdict = {'fontsize':30})
_ = plt.legend()


# ### Prophet

# In[69]:


from fbprophet import Prophet


# In[70]:


# Dataframe must have columns 'ds' and 'y' with the dates and values respectively
df_A_prophet = df_A_weekly.reset_index('Timestamp')
df_A_prophet = df_A_prophet.rename(columns={'Timestamp': 'ds',
                        'Daily Closing Price': 'y'})
df_A_prophet.head()


# In[71]:


# set the uncertainty interval to 95%
my_model = Prophet(interval_width=0.95, daily_seasonality=True, weekly_seasonality=True)


# In[72]:


my_model.fit(df_A_prophet)


# In[73]:


future_dates = my_model.make_future_dataframe(periods=240, freq='B')
future_dates.head()


# In[74]:


forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()


# In[75]:


_ = my_model.plot(forecast,uncertainty=True)


# In[76]:


_ = my_model.plot_components(forecast)


# In[77]:


forecast.set_index('ds', inplace = True)
forecast.head()


# In[78]:


A_forecasted = forecast['2019-01'].yhat
A_forecasted.head()


# In[79]:


A_truth = df_A_weekly['2019-01-01':'2019-02-01'].loc[:,'Daily Closing Price']
A_truth.head()


# In[80]:


# RMSE
mse = ((A_forecasted - A_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


# ## Classification

# In[81]:


sp500_mean.head()


# In[82]:


# Data
X = sp500_mean.iloc[:,1:3]
X.head()


# In[83]:


# Label
y = sp500_mean.loc[:, 'Cluster']
y.head()


# In[84]:


assert len(X) == len(y)


# In[85]:


# num of each cluster
sp500_mean.groupby('Cluster').agg(['count']).loc[:, 'Ticker']


# In[86]:


# incorporate cluster 2 to cluster 3
y = y.replace(2,3)


# In[87]:


from sklearn.model_selection import train_test_split


# In[88]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)


# In[89]:


from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


# In[90]:


# dummy
dummy_model = DummyClassifier(strategy='most_frequent').fit(X_train,y_train)
dummy_pred = dummy_model.predict(X_test)


# In[91]:


# logistic 
logreg_model = LogisticRegression(solver = 'liblinear', multi_class = 'auto').fit(X_train,y_train)
logreg_pred = logreg_model.predict(X_test)


# In[92]:


# Decision Tree
dtc_model = DecisionTreeClassifier().fit(X_train,y_train)
dtc_pred = dtc_model.predict(X_test)


# In[93]:


# Gradient Boosting
gbc_model = GradientBoostingClassifier().fit(X_train,y_train)
gbc_pred = gbc_model.predict(X_test)


# In[94]:


# Random Forest
from sklearn.model_selection import GridSearchCV

params = {'n_estimators':[5,50,100]}
gs = GridSearchCV(RandomForestClassifier(),params,cv=3,n_jobs=-1).fit(X_train,y_train)


# In[95]:


print('gs best accuracy: {:.3f}'.format(gs.best_estimator_.score(X_train,y_train)))
print('gs best params  : {}'.format(gs.best_params_))


# In[96]:


rf_model = RandomForestClassifier(n_estimators=50).fit(X_train,y_train)
rf_pred = rf_model.predict(X_test)


# In[97]:


# comparison of cross validation accuracy on training data

from sklearn.model_selection import cross_val_score

# dummy
scores = cross_val_score(dummy_model,X_train,y_train,cv=5)
print('Dummy: {:.2f}'.format(np.mean(scores)))

# logistic 
scores = cross_val_score(logreg_model,X_train,y_train,cv=5)
print('Logistic Regression: {:.2f}'.format(np.mean(scores)))

# Decision Tree
scores = cross_val_score(dtc_model,X_train,y_train,cv=5)
print('Decision Tree: {:.2f}'.format(np.mean(scores)))

# Gradient Boosting
scores = cross_val_score(gbc_model,X_train,y_train,cv=5)
print('Gradient Boosting: {:.2f}'.format(np.mean(scores)))

# Random Forest 
scores = cross_val_score(rf_model,X_train,y_train,cv=5)
print('Random Forest: {:.2f}'.format(np.mean(scores)))


# In[98]:


# comparison of mean square error on test data

from sklearn.metrics import mean_squared_error

print('Dummy: {:.2f}'.format(mean_squared_error(y_test, dummy_pred)))
print('Logistic Regression: {:.2f}'.format(mean_squared_error(y_test, logreg_pred)))
print('Decision Tree: {:.2f}'.format(mean_squared_error(y_test, dtc_pred)))
print('Gradient Boosting: {:.2f}'.format(mean_squared_error(y_test, gbc_pred)))
print('Random Forest: {:.2f}'.format(mean_squared_error(y_test, rf_pred)))

