# Stock-Price-Prdiction-using-Sentiment-Analysis


The purpose of this exercise is to forecast upcoming stock prices using the Machine Learning and  other Artificial Intelligence. Determining stock price is an detailed undertaking that is subjected to various influences such as markets tendencies, economic figures, company results and worldwide matters. Sentiment analyses may help in establishing what the common people think or feel about the price of a specific stock but it is crucial to note that such information could only be utilized alongside other strategies for correct estimations prediction. The practice began with a full review of existing literature on this topic. Online sources and research papers. Some of the tactics to tackle this problem are highlighted in the references below.


## Acknowledgements

 - [I would like to express my gratitude to Dr.Mala Sarashwat Associate Professor of Bennett University who gave me opportunity to do the project. ](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [We also acknowledge the following insightful sources 1.)Picasso, A., Merello, S., Ma, Y., Oneto, L., & Cambria, E. Technical analysis and sentiment embeddings for market trend prediction. Expert Systems with Applications, 135, 60–70. (2019).   2.)Tekin, S., & Canakoglu, E. Prediction of stock returns in Istanbul stock exchange using machine learning methods. 2018 26th Signal Processing and Communications Applications Conference (SIU). IEEE. (2018). 3.)Li, X., Wu, P., & Wang, W. Incorporating stock prices and news sentiments for stock market prediction: A case of Hong Kong. Information Processing & Management, 57(5), 102212. (2020).](https://github.com/matiassingers/awesome-readme)
 - [I am deeply grateful to the open-source community for developing and maintaining the libraries and frameworks that formed the backbone of our sentiment analysis and machine learning models.](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## Documentation

The analysis used publicly available data on stocks Market .The Information on stock is from Apple Company . The data gathered includes the standard data points including during Stock Analysis such as open, low, high .The data for close prices, Adjacent Close, Volume of Trade etc. Comes from January, 2007 among other things part of the EDA exercise

Here are some key features of the stock price prediction project described in the project are as follows:

1. Data Collection
Historical Stock Data: Utilizes the yfinance library to download historical stock data for various companies (e.g., Microsoft, Google, Apple, Amazon).

Tweet Scraping: Uses the Nitter module to scrape tweets related to specific keywords or hashtags, gathering data for sentiment analysis.

2. Data Preprocessing
Stock Data Preparation: Prepares the historical stock data for analysis and visualization, including handling missing values and formatting.

Tweet Data Preparation: Processes the scraped tweets by extracting relevant information (date, user profile ID, tweet text, username) and structuring it into a DataFrame.

3. Visualization
Candlestick Charts: Creates interactive candlestick charts using the plotly library to visualize historical stock prices.

Comparison Plots: Uses matplotlib to create line plots comparing actual stock prices to predicted prices, helping visualize the model's performance.

4. Sentiment Analysis
Pipeline Setup: Implements a sentiment analysis pipeline using the transformers library from Hugging Face.

entiment Classification: Analyzes the sentiment of collected tweets to determine their positive, negative, or neutral sentiment, contributing to the overall stock price prediction model.

5. Machine Learning Model
LSTM Network: Constructs a Long Short-Term Memory (LSTM) neural network using Keras, which is well-suited for time series prediction tasks.

Model Training: Trains the LSTM model on sequences of historical stock prices, utilizing dropout layers to prevent overfitting.
Model Evaluation: Evaluates the model's performance using the mean squared error (MSE) loss function and visual comparison of predicted versus actual prices.

6. Prediction and Analysis
Data Transformation: Prepares the data for model input by reshaping it into the required format for LSTM networks.

Prediction: Uses the trained LSTM model to predict future stock prices based on the test dataset.
Inverse Transformation: Converts the predicted values back to their original scale for meaningful comparison with actual stock prices.]
## Libraries and the general structure of the code:

Here is the content related to importing libraries and the general structure of the code:

Imports necessary libraries such as :-

pandas

numpy

matplotlib

yfinance

Downloads historical stock data using the yfinance library.

Visualizes stock price data using plotly.

Scrapes tweets using a module named Nitter.

Analyzes tweet sentiments using the transformers library from Hugging Face.

Builds and trains a Long Short-Term Memory (LSTM) neural network using Keras for predicting stock prices.


## Authors

- [Isha Kumari](https://www.github.com/octokatherine)


## Deployment

To deploy this project run

```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

```
```bash
pip install yfinance

```
```bash
msft = yf.Ticker("AAPL")

```
```bash
hist = msft.history(period='max')

```
```bash
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
```
```bash
stocks=['AAPL']

```
```bash
hists = {}
for s in stocks:
    tkr = yf.Ticker(s)
    history = tkr.history(period="3y")
    hists[s] = history

```
```bash
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

```
```bash
from ntscraper import Nitter

```
```bash
scraper =Nitter()

```
```bash
tweets= scraper.get_tweets('Apple',mode='hashtag',number=1000)

```
```bash
final_tweets=[]
for  tweet in tweets['tweets']:
    data=[tweet['date'],tweet['user']['profile_id'],tweet['text'],tweet['user']['username']]
    final_tweets.append(data)
data = pd.DataFrame(final_tweets,columns=['Datetime','Tweet_ID','Text','Username'])

```
```bash
from transformers import pipeline

```
```bash
from transformers import pipeline
sentiment_task =pipeline('sentiment-analysis')
sentiment_task("Covid cases are incresing fast!")

```
```bash
model = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_task = pipeline("sentiment-analysis", model=model)
sentiment_task("You are good boy")

```
```bash
sentiment_task(data['Text'][2])

```
```bash
len(sent_results)

```
```bash
sent_results

```
```bash
sent_df = pd.DataFrame(sent_results).T
sent_df["label"] = sent_df[0].apply(lambda x: x["label"])
sent_df["score"] = sent_df[0].apply(lambda x: x["score"])
sent_df = sent_df.merge(data.set_index("Tweet_ID"), left_index=True, right_index=True
)

```
```bash
sent_df.groupby("label")["score"].plot(kind="hist", bins=50)
plt.legend()
plt.show()

```
```bash
sent_df["score_"] = sent_df["score"]

sent_df.loc[sent_df["label"] == "Negative", "score_"] = (
    sent_df.loc[sent_df["label"] == "Negative"]["score"] * -1
)

sent_df.loc[sent_df["label"] == "Neutral", "score_"] = 0

```
```bash
sent_df["score"].plot(kind="hist", bins=50)

```
```bash

sent_df["Datetime"] = pd.to_datetime(sent_df["Datetime"], format="%b %d, %Y · %I:%M %p UTC")
sent_df["Date"] = sent_df["Datetime"].dt.date

```
```bash
sent_df.groupby("Date")["score"].mean().plot(figsize=(15,5))
hists['AAPL']["Close"].plot()

```
```bash
ax=sent_and_stock["sentiment"].plot(legend="Sentiment")
ax2 =ax.twinx()
data=hists["AAPL"].copy()
sent_and_stock["Close"].plot(ax=ax2, color ="orange",legend="Closeing price")

```
```bash
data['Close'].shift(-4)

```
```bash
data['newclose']=data['Close'].shift(-4)

```
```bash
imp=data.drop(['Close','newclose'],axis=1)
op =data['newclose'].dropna()

```
```bash
train_input=imp[:-4]
pridict_input=imp[-4:]

```
```bash
from sklearn.model_selection import train_test_split

```
```bash
train_test_split(train_input,op,test_size=0.2)

```
```bash
x_tr,x_ts,y_tr,y_ts=train_test_split(train_input,op,test_size=0.2)

```
```bash
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM,Dense,Dropout
import os
import tensorflow as tf

```
```bash
df=data['Open'].values
df=df.reshape(-1,1)
df[:7]

```
```bash
dataset_train =np.array(df[:int(df.shape[0]*0.8)])
dataset_test =np.array(df[int(df.shape[0]*0.8)-50:])

```
```bash
dataset_test.shape
```
```bash
scaler =MinMaxScaler(feature_range=(0,1))
dataset_train=scaler.fit_transform(dataset_train)
dataset_train[:7]

```
```bash
dataset_test=scaler.transform(dataset_test)
dataset_test[:7]
```
```bash
def create_my_dataset(df):
    x= []
    y=[]
    for i in range(50,df.shape[0]):
        x.append(df[i-50:i,0])
        y.append(df[i,0])
    x=np.array(x)
    y=np.array(y)
    return x,y
```
```bash
x_tr,y_tr = create_my_dataset(dataset_train)
x_tr[:1]
```
```bash
x_ts,y_ts =create_my_dataset(dataset_test)
x_ts[:1]
```
```bash
model =Sequential()
model.add(LSTM(units=96,return_sequences=True,input_shape=(x_tr.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))
```
```bash
model.summary()
```
```bash
model.compile(loss='mean_squared_error',optimizer='adam')
```
```bash
model.fit(x_tr,y_tr,epochs=50,batch_size=32)
```
```bash
model.save(r"C:\Users\123de\OneDrive\Desktop\sentiment ai\stock_perdiction1.h5")
```
```bash
predictions = model.predict(x_ts)
predictions =scaler.inverse_transform(predictions)
```
```bash
fig, ax = plt.subplots(figsize=(8, 4))
plt.plot(df,color ='red',label ='original Stockprice')
ax.plot(range(len(y_tr)+50,len(y_tr)+50+len(predictions)),predictions,color='blue',label='predictions')
plt.legend()
print(range(len(y_tr)+50,len(y_tr)+50+len(predictions)))
plt.show()
```
```bash
y_test_scaler =scaler.inverse_transform(y_ts.reshape(-1,1))
fig , ax =plt.subplots(figsize=(8,4))
ax.plot(y_test_scaler,color='red',label='True Price')
plt.plot(predictions,color='blue',label='Predictive price')
plt.legend()
```










## FAQ

#### Question 1. What is the primary objective of the "Stock Price Prediction using Sentiment Analysis" ?

Ans) The primary objective of the script is to predict stock prices using historical stock data and sentiment analysis of tweets. It combines financial data with social media sentiment to enhance the accuracy of stock price predictions.

#### Question 2. Which libraries are used for data collection and analysis in the script?

Ans) The script uses several libraries for data collection and analysis, including:

pandas and numpy for data manipulation and analysis.
yfinance for downloading historical stock data.
plotly and matplotlib for data visualization.
Nitter for scraping tweets.
transformers from Hugging Face for sentiment analysis.

#### Question 3. How does the script obtain historical stock data?

Ans) The script uses the yfinance library to download historical stock data. It creates a Ticker object for each stock and retrieves the historical price data using the history method.

#### Question 4. What method is used for sentiment analysis in the script?
Ans) The script uses the transformers library from Hugging Face to perform sentiment analysis. It creates a sentiment analysis pipeline to process the scraped tweets and classify their sentiment.



#### Question 5. Describe the neural network model used for stock price prediction in the script.
Ans) The script uses a Long Short-Term Memory (LSTM) neural network model implemented with Keras. The model consists of three LSTM layers with 96 units each, followed by dropout layers to prevent overfitting. The final layer is a dense layer with one unit to predict the stock price. The model is compiled with the mean squared error loss function and the Adam optimizer.


#### Question 6. How are the tweets collected for sentiment analysis?
Ans) The script uses the Nitter module to scrape tweets. It creates a scraper object and retrieves tweets related to a specific keyword (e.g., "Apple") using the get_tweets method, specifying the mode (e.g., 'hashtag') and the number of tweets to collect.


#### Question 7. What preprocessing steps are performed on the tweet data before sentiment analysis?
Ans) The script processes the scraped tweets by extracting relevant information such as the tweet date, user profile ID, tweet text, and username. This data is then structured into a DataFrame for further analysis.


#### Question 8. How does the script visualize the historical stock data?
Ans) The script uses the plotly library to create candlestick charts for visualizing historical stock data. It constructs a Figure object with go.Candlestick data, specifying the open, high, low, and close prices. The charts are customized with titles and layout adjustments.


#### Question 9. What metrics are used to evaluate the performance of the LSTM model?
Ans) The script evaluates the performance of the LSTM model using the mean squared error (MSE) loss function during training. For visualization, it compares the predicted stock prices against the actual prices using line plots to assess how well the model's predictions align with the true values.


#### Question 10. How does the script handle the training and testing datasets for the LSTM model?
Ans) The script splits the data into training and testing sets, then uses the create_my_dataset function to generate the sequences. It reshapes these sequences to be compatible with the LSTM input requirements. The training dataset (x_tr, y_tr) is used to train the model, and the testing dataset (x_ts, y_ts) is used to evaluate its performance.
