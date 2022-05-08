import pandas as pd
import streamlit as st
from datetime import date
import yfinance as yf
from yahoofinancials import YahooFinancials
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

start = "2010-01-01"
end = date.today().strftime("%Y-%m-%d")

st.title("Stock Predictor App")
stocks = ("FB", "AMZN", "AAPL", "NFLX","GOOGL", "MSFT", "AMD", "NVDA", "TSLA",  "UBER")
ticker_symbol = st.selectbox("Select the ticker symbol of a company for prediction", stocks)

years = st.slider("Number of years to predict in the future", 1, 20)
period = years * 365

@st.cache
def get_data(t):
    data = yf.download(t, start, end)
    data.reset_index(inplace=True)
    return data

symbol = yf.Ticker(ticker_symbol)
full_name = symbol.info['longName']
holders = symbol.institutional_holders
stock = get_data(ticker_symbol)
load_state = st.text("Data is fully downloaded!")

st.subheader("Raw data of " + full_name)
st.write(stock.tail())

def plot_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series graph of stock', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
plot_data()

st.subheader("Institutional holders of " + full_name)
st.write(holders)

train = stock[['Date', 'Close']]
train = train.rename(columns={"Date": "ds", "Close": "y"})
pred = Prophet()
pred.fit(train)
future_df = pred.make_future_dataframe(periods=period)
forecast = pred.predict(future_df)

st.subheader("Foracsted price predictions for " + full_name)
st.write(forecast.tail())

st.write("Plot of forecasted data")
fig_data = plot_plotly(pred, forecast)
st.plotly_chart(fig_data)
st.write("Important Note: These price predictions were recognized via time series analysis")