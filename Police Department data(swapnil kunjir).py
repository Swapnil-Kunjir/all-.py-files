#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime as dt
import numpy as np
import seaborn as sns
from sklearn import metrics
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy as sp
import os
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
import folium
from folium.plugins import MarkerCluster


# In[3]:


df=pd.read_csv(r'C:\Users\DELL\Downloads\police_department_data.csv')


# In[4]:


df


# In[5]:


df.describe()


# In[6]:


df.count()


# In[7]:


df.shape


# In[8]:


df.isnull().sum()


# In[9]:


df=pd.read_csv(r'C:\Users\DELL\Downloads\police_department_data.csv',parse_dates=['crime_date'])


# In[10]:


df['month']=df['crime_date'].dt.month


# In[11]:


df['Hour']=df['crime_date'].dt.hour
df.head()


# In[12]:


x=[]
y=[]
for row in df['location']:
        try:
            x.append(row.split(',')[0])
            y.append(row.split(',')[1])
        except:
            x.append(np.NaN)
            y.append(np.NaN)
df['latitude']=x
df['longitude']=y

df.head()


# In[13]:


df.latitude=df.latitude.apply(lambda x:x.strip('('))
df.longitude=df.longitude.apply(lambda x:x.strip(')'))

df.head()


# In[14]:


df['month'].value_counts()


# In[15]:


month_count=df['month'].value_counts().reset_index().sort_values(by='index')
month_count.columns=['month','Count']
print(month_count)


# In[16]:


trace = go.Scatter(
    x = month_count.month,
    y = month_count.Count
)

data=[trace]
py.iplot(data,filename='basic-line')


# In[17]:


df['Hour'].value_counts()


# In[18]:


hour_count=df['Hour'].value_counts().reset_index().sort_values(by='index')
hour_count.columns=['Hour','Count']
print(hour_count)


# In[19]:


trace = go.Scatter(
    x = hour_count.Hour,
    y = hour_count.Count
)

data=[trace]
py.iplot(data,filename='basic-line')


# In[20]:


m = folium.Map(
    location=[37.7749, -122.4194],
    tiles='Stamen Toner',
    zoom_start=13
)

marker_cluster = MarkerCluster(
    name='Crime Locations',
    overlay=True,
    control=False,
    icon_create_function=None
)
for k in range(1000):
    location = df.latitude.values[k], df.longitude.values[k]
    marker = folium.Marker(location=location,icon=folium.Icon(color='green'))
    popup = df.address.values[k]
    folium.Popup(popup).add_to(marker)
    marker_cluster.add_child(marker)

marker_cluster.add_to(m)

folium.LayerControl().add_to(m)

m.save("online.html")

m


# In[23]:


data = df.groupby('category').count().iloc[:, 0].sort_values(
 ascending=False)
data = data.reindex(np.append(np.delete(data.index, 1), 'OTHER OFFENSES'))

plt.figure(figsize=(10, 10))
with sns.axes_style("whitegrid"):
    ax = sns.barplot(
        (data.values / data.values.sum()) * 100,
        data.index,
        orient='h',
        palette="vlag")

plt.title('Incidents per Crime Category', fontdict={'fontsize': 16})
plt.xlabel('Incidents (%)')


# In[24]:


def street_addr(x):
    street=x.split(' ')
    return (' '.join(street[-2:]))
df['addr']=df['address'].apply(lambda x:street_addr(x))
df['addr'].head()
x=df['addr'].value_counts()
x


# In[25]:


d=df['addr'].value_counts().sort_values(ascending=False).reset_index().head(10)
d.columns=['addr','Count']
data = [go.Bar(
            y=d.addr,
            x=d.Count,
             opacity=0.6,
             orientation = 'h'
    )]

py.iplot(data, filename='basic-bar')


# In[26]:


s=df['department_district'].value_counts().reset_index().sort_values(by='index').head(10)
s.columns=['department_district','Count']
# Create a trace
tag = (np.array(s.department_district))
sizes = (np.array((s['Count'] / s['Count'].sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Top department disticts with Most Crimes')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Activity Distribution")


# In[ ]:




