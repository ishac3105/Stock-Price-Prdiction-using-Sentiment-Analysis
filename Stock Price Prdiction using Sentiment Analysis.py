#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


# In[2]:


pip install yfinance


# In[3]:


import yfinance as yf


# In[4]:


msft = yf.Ticker("AAPL")


# In[5]:


msft.info


# In[6]:


hist = msft.history(period='max')


# In[7]:


hist


# In[8]:


hist['Open'].plot(figsize=(15,5))


# In[9]:



import plotly.graph_objects as go
import pandas as pd
from datetime import datetime


# In[ ]:


stocks=['MSFT','GOOGL','AAPL','AMZN']


# In[ ]:


hists = {}
for s in stocks:
    tkr = yf.Ticker(s)
    history = tkr.history(period="3y")
    hists[s] = history


# In[12]:


for stock in stocks:
    Temp_df = hists[stock].copy()

    fig = go.Figure(data=[go.Candlestick(x=Temp_df.index,
                    open=Temp_df['Open'],
                    high=Temp_df['High'],
                    low=Temp_df['Low'],
                    close=Temp_df['Close'])])
    fig.update_layout(
    margin=dict(l=20, r=20, t=60, b=20),
    height=300,
    paper_bgcolor="LightSteelBlue",  # Corrected property name
    title=stock,
    )
    

    fig.show()


# # pull twitter data

# In[13]:


from ntscraper import Nitter


# In[14]:


scraper =Nitter()


# In[15]:


tweets= scraper.get_tweets('Apple',mode='hashtag',number=1000)


# In[16]:


final_tweets=[]
for  tweet in tweets['tweets']:
    data=[tweet['date'],tweet['user']['profile_id'],tweet['text'],tweet['user']['username']]
    final_tweets.append(data)
data = pd.DataFrame(final_tweets,columns=['Datetime','Tweet_ID','Text','Username'])


# In[17]:


data.shape


# In[18]:


data.head()


# In[19]:


from transformers import pipeline


# In[20]:


get_ipython().system('pip install transformers')


# In[21]:


from transformers import pipeline
sentiment_task =pipeline('sentiment-analysis')
sentiment_task("Covid cases are incresing fast!")


# In[22]:



model = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_task = pipeline("sentiment-analysis", model=model)
sentiment_task("You are good boy")


# In[23]:


sentiment_task(data['Text'][2])


# In[24]:


max_sequence_length = 512  # Replace with the actual maximum sequence length of your RoBERTa model

sent_results = {}
count = 0

for i, d in data.iterrows():
    # Truncate or split the input text to fit within the model's maximum sequence length
    input_text = d["Text"][:max_sequence_length]

    # Perform sentiment analysis on the modified input text
    sent = sentiment_task(input_text)
    
    sent_results[d["Tweet_ID"]] = sent
    count += 1
    
    if count == 600:
        break


# In[25]:


len(sent_results)


# In[26]:


# [{'label': 'Negative', 'score': 0.7235767245292664}]


# In[27]:


sent_results


# In[27]:


sent_df = pd.DataFrame(sent_results).T
sent_df["label"] = sent_df[0].apply(lambda x: x["label"])
sent_df["score"] = sent_df[0].apply(lambda x: x["score"])
sent_df = sent_df.merge(data.set_index("Tweet_ID"), left_index=True, right_index=True
)


# In[28]:


sent_df


# In[29]:


sent_df.groupby("label")["score"].plot(kind="hist", bins=50)
plt.legend()
plt.show()


# In[31]:


sent_df["score_"] = sent_df["score"]

sent_df.loc[sent_df["label"] == "Negative", "score_"] = (
    sent_df.loc[sent_df["label"] == "Negative"]["score"] * -1
)

sent_df.loc[sent_df["label"] == "Neutral", "score_"] = 0


# In[43]:


sent_df["score"].plot(kind="hist", bins=50)


# In[ ]:





# In[44]:



# Assuming "Datetime" is a string representing datetime
sent_df["Datetime"] = pd.to_datetime(sent_df["Datetime"], format="%b %d, %Y · %I:%M %p UTC")

# Now you can use .dt accessor on the "Datetime" column
sent_df["Date"] = sent_df["Datetime"].dt.date


# In[45]:


sent_df.groupby("Date")["score"].mean().plot(figsize=(15,5))
hists['AAPL']["Close"].plot()


# In[46]:


sent_daily =sent_df.groupby("Date")["score"].mean()


# In[47]:


AAPL_df =hists["AAPL"].copy()


# In[48]:


AAPL_df =AAPL_df.reset_index()


# In[49]:


AAPL_df['Date']=AAPL_df['Date'].dt.date


# In[50]:


AAPL_df =AAPL_df.set_index('Date')


# In[51]:


sent_and_stock=sent_daily.to_frame('sentiment').merge(AAPL_df,left_index=True,right_index=True)


# In[52]:


sent_and_stock["sentiment"].plot()


# In[53]:


sent_and_stock["Close"].plot()


# In[54]:


ax=sent_and_stock["sentiment"].plot(legend="Sentiment")
ax2 =ax.twinx()
data=hists["AAPL"].copy()
sent_and_stock["Close"].plot(ax=ax2, color ="orange",legend="Closeing price")


# In[55]:


data['Close'].shift(-4)


# In[56]:


data['newclose']=data['Close'].shift(-4)


# In[57]:


imp=data.drop(['Close','newclose'],axis=1)
op =data['newclose'].dropna()


# In[58]:


imp.shape,op.shape


# In[59]:


train_input=imp[:-4]
pridict_input=imp[-4:]


# In[60]:


from sklearn.model_selection import train_test_split


# In[61]:


train_test_split(train_input,op,test_size=0.2)


# In[62]:


x_tr,x_ts,y_tr,y_ts=train_test_split(train_input,op,test_size=0.2)


# In[63]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM,Dense,Dropout
import os
import tensorflow as tf


# In[64]:


df=data['Open'].values
df=df.reshape(-1,1)
df[:7]


# In[65]:


dataset_train =np.array(df[:int(df.shape[0]*0.8)])
dataset_test =np.array(df[int(df.shape[0]*0.8)-50:])


# In[66]:


dataset_test.shape


# In[67]:


scaler =MinMaxScaler(feature_range=(0,1))
dataset_train=scaler.fit_transform(dataset_train)
dataset_train[:7]


# In[68]:


dataset_test=scaler.transform(dataset_test)
dataset_test[:7]


# In[69]:


def create_my_dataset(df):
    x= []
    y=[]
    for i in range(50,df.shape[0]):
        x.append(df[i-50:i,0])
        y.append(df[i,0])
    x=np.array(x)
    y=np.array(y)
    return x,y


# In[70]:


x_tr,y_tr = create_my_dataset(dataset_train)
x_tr[:1]


# In[71]:


x_tr[:1].shape


# In[72]:


x_ts,y_ts =create_my_dataset(dataset_test)
x_ts[:1]


# In[73]:


x_tr =np.reshape(x_tr,(x_tr.shape[0],x_tr.shape[1],1))
x_ts= np.reshape(x_ts,(x_ts.shape[0],x_ts.shape[1],1))
print(x_tr.shape)
print(x_ts.shape)


# In[74]:


model =Sequential()
model.add(LSTM(units=96,return_sequences=True,input_shape=(x_tr.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))


# In[75]:


model.summary()


# In[76]:


model.compile(loss='mean_squared_error',optimizer='adam')


# In[77]:


model.fit(x_tr,y_tr,epochs=50,batch_size=32)


# In[78]:


model.save(r"C:\Users\123de\OneDrive\Desktop\sentiment ai\stock_perdiction1.h5")


# In[79]:


model = load_model(r"C:\Users\123de\OneDrive\Desktop\sentiment ai\stock_perdiction1.h5")


# In[80]:


predictions = model.predict(x_ts)
predictions =scaler.inverse_transform(predictions)


# In[81]:


predictions


# In[82]:


fig, ax = plt.subplots(figsize=(8, 4))
plt.plot(df,color ='red',label ='original Stockprice')
ax.plot(range(len(y_tr)+50,len(y_tr)+50+len(predictions)),predictions,color='blue',label='predictions')
plt.legend()
print(range(len(y_tr)+50,len(y_tr)+50+len(predictions)))
plt.show()


# In[83]:


y_test_scaler =scaler.inverse_transform(y_ts.reshape(-1,1))
fig , ax =plt.subplots(figsize=(8,4))
ax.plot(y_test_scaler,color='red',label='True Price')
plt.plot(predictions,color='blue',label='Predictive price')
plt.legend()


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[ ]:





# In[ ]:




