import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import warnings
import json
import time
import re
warnings.filterwarnings('ignore')

# Statistical and ML imports
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import ta

# Page configuration (only when run standalone)
if __name__ == "__main__":
    st.set_page_config(
        page_title="Stock Analysis Tool",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# NSE Headers and Session Management
def create_nse_session():
    """Create a session with NSE-compatible headers"""
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Referer': 'https://www.nseindia.com/'
    }
    session.headers.update(headers)
    # Get cookies by visiting the homepage
    try:
        session.get('https://www.nseindia.com', timeout=10)
        time.sleep(1)  # Small delay to let cookies set
    except:
        pass
    return session

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stAlert {
        background-color: #e8f4f8;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def calculate_beta_vs_nifty(stock_data, ticker):
    """Calculate beta of stock vs Nifty 50"""
    try:
        # Get Nifty 50 data
        nifty = yf.Ticker("^NSEI")
        nifty_data = nifty.history(period="1y")
        
        if len(nifty_data) == 0:
            return None
        
        # Align dates
        stock_returns = stock_data['Close'].pct_change().dropna()
        nifty_returns = nifty_data['Close'].pct_change().dropna()
        
        # Get common dates
        common_dates = stock_returns.index.intersection(nifty_returns.index)
        
        if len(common_dates) < 30:  # Need at least 30 data points
            return None
        
        stock_returns = stock_returns.loc[common_dates]
        nifty_returns = nifty_returns.loc[common_dates]
        
        # Calculate beta using covariance method
        covariance = np.cov(stock_returns, nifty_returns)[0][1]
        variance = np.var(nifty_returns)
        
        beta = covariance / variance if variance != 0 else None
        
        return beta
    except Exception as e:
        st.warning(f"Could not calculate beta: {e}")
        return None

def calculate_beta_vs_sensex(stock_data, ticker):
    """Calculate beta of stock vs Sensex (for BSE stocks)"""
    try:
        # Get Sensex data
        sensex = yf.Ticker("^BSESN")
        sensex_data = sensex.history(period="1y")
        
        if len(sensex_data) == 0:
            return None
        
        # Align dates
        stock_returns = stock_data['Close'].pct_change().dropna()
        sensex_returns = sensex_data['Close'].pct_change().dropna()
        
        # Get common dates
        common_dates = stock_returns.index.intersection(sensex_returns.index)
        
        if len(common_dates) < 30:  # Need at least 30 data points
            return None
        
        stock_returns = stock_returns.loc[common_dates]
        sensex_returns = sensex_returns.loc[common_dates]
        
        # Calculate beta using covariance method
        covariance = np.cov(stock_returns, sensex_returns)[0][1]
        variance = np.var(sensex_returns)
        
        beta = covariance / variance if variance != 0 else None
        
        return beta
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def scrape_nse_stock_info(symbol):
    """Scrape detailed stock information from NSE"""
    try:
        session = create_nse_session()
        
        # Quote API
        quote_url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        response = session.get(quote_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data
        return None
    except Exception as e:
        st.warning(f"NSE data fetch error: {e}")
        return None

@st.cache_data(ttl=3600)
def get_nse_corporate_actions(symbol):
    """Get corporate actions from NSE"""
    try:
        session = create_nse_session()
        
        # Corporate actions API
        url = f"https://www.nseindia.com/api/corporates-corporateActions?index=equities&symbol={symbol}"
        response = session.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data
        return None
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def get_nse_shareholding(symbol):
    """Get shareholding pattern from NSE"""
    try:
        session = create_nse_session()
        
        # Shareholding pattern API
        url = f"https://www.nseindia.com/api/corporates-shareholding-pattern?symbol={symbol}"
        response = session.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data
        return None
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    """Fetch stock data from Yahoo Finance - supports both NSE (.NS) and BSE (.BO)"""
    try:
        # Determine exchange from ticker suffix
        if ticker.endswith('.NS'):
            exchange = 'NSE'
            nse_symbol = ticker.replace('.NS', '').upper()
        elif ticker.endswith('.BO'):
            exchange = 'BSE'
            nse_symbol = None  # BSE stocks don't have NSE data
        else:
            # If no suffix provided, default to NSE
            exchange = 'NSE'
            ticker = ticker + '.NS'
            nse_symbol = ticker.replace('.NS', '').upper()
        
        # Fetch data from yfinance
        stock = yf.Ticker(ticker)
        data = stock.history(period="max")
        info = stock.info
        
        # Convert data to dict for serialization
        data_dict = {
            'data': data.to_dict(),
            'index': data.index.tolist()
        }
        
        # Calculate beta vs appropriate index
        if data is not None and len(data) > 0:
            if exchange == 'NSE':
                # Compare with Nifty 50
                calculated_beta = calculate_beta_vs_nifty(data, ticker)
            else:
                # Compare with Sensex for BSE
                calculated_beta = calculate_beta_vs_sensex(data, ticker)
            
            if calculated_beta is not None:
                info['calculatedBeta'] = calculated_beta
        
        # Get NSE data only for NSE stocks
        nse_data = None
        if exchange == 'NSE' and nse_symbol:
            nse_data = scrape_nse_stock_info(nse_symbol)
        
        return data_dict, info, nse_data, nse_symbol
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None, None

def reconstruct_dataframe(data_dict):
    """Reconstruct pandas DataFrame from cached dict"""
    if data_dict is None:
        return None
    df = pd.DataFrame(data_dict['data'])
    df.index = pd.DatetimeIndex(data_dict['index'])
    return df

def identify_major_fluctuations(data, threshold=5):
    """Identify major price fluctuations (peaks and troughs)"""
    if data is None or len(data) == 0:
        return []
    
    fluctuations = []
    prices = data['Close'].values
    dates = data.index
    
    # Find listing price
    listing_price = prices[0]
    listing_date = dates[0]
    fluctuations.append({
        'date': listing_date,
        'price': listing_price,
        'type': 'Listing',
        'change_pct': 0
    })
    
    # Find local minima and maxima
    for i in range(1, len(prices) - 1):
        # Local minimum
        if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
            change_pct = ((prices[i] - listing_price) / listing_price) * 100
            if abs(change_pct) > threshold:
                fluctuations.append({
                    'date': dates[i],
                    'price': prices[i],
                    'type': 'Trough',
                    'change_pct': change_pct
                })
        
        # Local maximum
        if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
            change_pct = ((prices[i] - listing_price) / listing_price) * 100
            if abs(change_pct) > threshold:
                fluctuations.append({
                    'date': dates[i],
                    'price': prices[i],
                    'type': 'Peak',
                    'change_pct': change_pct
                })
    
    # Add current price
    current_price = prices[-1]
    current_date = dates[-1]
    change_pct = ((current_price - listing_price) / listing_price) * 100
    fluctuations.append({
        'date': current_date,
        'price': current_price,
        'type': 'Current',
        'change_pct': change_pct
    })
    
    return fluctuations

def calculate_statistics(data):
    """Calculate comprehensive statistical measures"""
    if data is None or len(data) == 0:
        return {}
    
    returns = data['Close'].pct_change().dropna()
    prices = data['Close']
    
    stats_dict = {
        # Descriptive Statistics
        'Mean Price': prices.mean(),
        'Median Price': prices.median(),
        'Std Deviation': prices.std(),
        'Variance': prices.var(),
        'Min Price': prices.min(),
        'Max Price': prices.max(),
        'Range': prices.max() - prices.min(),
        'Coefficient of Variation': (prices.std() / prices.mean()) * 100,
        
        # Return Statistics
        'Mean Daily Return': returns.mean() * 100,
        'Median Daily Return': returns.median() * 100,
        'Return Std Dev': returns.std() * 100,
        'Annualized Return': returns.mean() * 252 * 100,
        'Annualized Volatility': returns.std() * np.sqrt(252) * 100,
        
        # Risk Metrics
        'Sharpe Ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0,
        'Sortino Ratio': (returns.mean() / returns[returns < 0].std()) * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0,
        'Max Drawdown': ((prices / prices.cummax()) - 1).min() * 100,
        'Value at Risk (95%)': np.percentile(returns, 5) * 100,
        'Conditional VaR (95%)': returns[returns <= np.percentile(returns, 5)].mean() * 100,
        
        # Distribution Statistics
        'Skewness': stats.skew(returns.dropna()),
        'Kurtosis': stats.kurtosis(returns.dropna()),
        'Jarque-Bera Test p-value': stats.jarque_bera(returns.dropna())[1],
        
        # Trend Indicators
        '50-Day MA': prices.tail(50).mean() if len(prices) >= 50 else prices.mean(),
        '200-Day MA': prices.tail(200).mean() if len(prices) >= 200 else prices.mean(),
    }
    
    # Add technical indicators
    if len(data) > 20:
        stats_dict['RSI (14)'] = ta.momentum.RSIIndicator(prices, window=14).rsi().iloc[-1]
        stats_dict['MACD'] = ta.trend.MACD(prices).macd().iloc[-1]
        stats_dict['MACD Signal'] = ta.trend.MACD(prices).macd_signal().iloc[-1]
    
    return stats_dict

def interpret_statistics(stats_dict, current_price):
    """Provide interpretations for statistical measures"""
    interpretations = []
    
    # Volatility interpretation
    cv = stats_dict.get('Coefficient of Variation', 0)
    if cv < 10:
        interpretations.append("📊 **Low Volatility**: Stock shows stable price movements.")
    elif cv < 30:
        interpretations.append("📊 **Moderate Volatility**: Stock has average price fluctuations.")
    else:
        interpretations.append("📊 **High Volatility**: Stock experiences significant price swings.")
    
    # Return interpretation
    annual_return = stats_dict.get('Annualized Return', 0)
    if annual_return > 15:
        interpretations.append("📈 **Strong Returns**: Historical returns exceed 15% annually.")
    elif annual_return > 0:
        interpretations.append("📈 **Positive Returns**: Stock has generated positive returns.")
    else:
        interpretations.append("📉 **Negative Returns**: Stock has declined over the period.")
    
    # Risk interpretation
    sharpe = stats_dict.get('Sharpe Ratio', 0)
    if sharpe > 1:
        interpretations.append("✅ **Good Risk-Adjusted Return**: Sharpe ratio > 1 indicates favorable returns for the risk taken.")
    elif sharpe > 0:
        interpretations.append("⚠️ **Moderate Risk-Adjusted Return**: Returns barely compensate for the risk.")
    else:
        interpretations.append("❌ **Poor Risk-Adjusted Return**: Returns don't justify the risk.")
    
    # Drawdown interpretation
    max_dd = stats_dict.get('Max Drawdown', 0)
    if max_dd > -20:
        interpretations.append("💪 **Resilient**: Maximum drawdown is moderate, showing relative strength.")
    elif max_dd > -40:
        interpretations.append("⚠️ **Significant Decline**: Stock has experienced substantial drops.")
    else:
        interpretations.append("🚨 **Severe Drawdown**: Stock has fallen significantly from its peak.")
    
    # Distribution interpretation
    skew = stats_dict.get('Skewness', 0)
    if skew > 0.5:
        interpretations.append("📐 **Positive Skew**: More frequent small losses and occasional large gains.")
    elif skew < -0.5:
        interpretations.append("📐 **Negative Skew**: More frequent small gains but occasional large losses.")
    else:
        interpretations.append("📐 **Symmetric Distribution**: Returns are fairly balanced.")
    
    # Trend interpretation
    ma50 = stats_dict.get('50-Day MA', 0)
    ma200 = stats_dict.get('200-Day MA', 0)
    if current_price > ma50 and current_price > ma200:
        interpretations.append("🚀 **Bullish Trend**: Price above both 50 and 200-day moving averages.")
    elif current_price < ma50 and current_price < ma200:
        interpretations.append("🐻 **Bearish Trend**: Price below both moving averages.")
    else:
        interpretations.append("➡️ **Neutral Trend**: Mixed signals from moving averages.")
    
    # RSI interpretation
    rsi = stats_dict.get('RSI (14)', 50)
    if rsi > 70:
        interpretations.append("🔴 **Overbought**: RSI indicates stock may be overvalued.")
    elif rsi < 30:
        interpretations.append("🟢 **Oversold**: RSI indicates stock may be undervalued.")
    else:
        interpretations.append("🟡 **Neutral RSI**: Stock is neither overbought nor oversold.")
    
    return interpretations

def forecast_prices(data, periods=30):
    """Forecast future prices using multiple improved methods"""
    if data is None or len(data) < 50:
        return None, None
    
    prices = data['Close'].values
    returns = np.diff(prices) / prices[:-1]
    
    # Use more recent data for better predictions (last 252 trading days = 1 year)
    recent_prices = prices[-min(252, len(prices)):]
    
    try:
        # Method 1: Linear Regression with Recent Trend
        lookback = min(60, len(recent_prices))  # Last 60 days for trend
        X = np.arange(lookback).reshape(-1, 1)
        y = recent_prices[-lookback:]
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        
        future_X = np.arange(lookback, lookback + periods).reshape(-1, 1)
        lr_forecast = lr_model.predict(future_X)
        
        # Constrain linear regression to reasonable bounds
        lr_forecast = np.clip(lr_forecast, recent_prices[-1] * 0.5, recent_prices[-1] * 2.0)
        
    except Exception as e:
        st.warning(f"Linear Regression failed: {str(e)[:50]}")
        lr_forecast = np.array([recent_prices[-1]] * periods)
    
    try:
        # Method 2: Improved ARIMA
        # Use log prices for better stability
        log_prices = np.log(recent_prices)
        
        # Fit ARIMA with conservative parameters
        arima_model = ARIMA(log_prices, order=(1, 1, 1))
        arima_fit = arima_model.fit()
        
        # Forecast in log space
        log_forecast = arima_fit.forecast(steps=periods)
        
        # Convert back to price space
        arima_forecast = np.exp(log_forecast)
        
        # Constrain to reasonable bounds
        arima_forecast = np.clip(arima_forecast, recent_prices[-1] * 0.5, recent_prices[-1] * 2.0)
        
    except Exception as e:
        st.warning(f"ARIMA failed, using momentum model: {str(e)[:50]}")
        # Fallback: Simple momentum model
        momentum = (recent_prices[-1] - recent_prices[-min(20, len(recent_prices))]) / min(20, len(recent_prices))
        arima_forecast = np.array([recent_prices[-1] + momentum * i for i in range(1, periods + 1)])
        arima_forecast = np.clip(arima_forecast, recent_prices[-1] * 0.5, recent_prices[-1] * 2.0)
    
    try:
        # Method 3: Exponential Smoothing (Holt's method - trend only, no seasonality)
        from statsmodels.tsa.holtwinters import Holt
        
        # Use Holt's linear method (simpler, more stable)
        holt_model = Holt(recent_prices, initialization_method="estimated")
        holt_fit = holt_model.fit(smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
        es_forecast = holt_fit.forecast(steps=periods)
        
        # Constrain to reasonable bounds
        es_forecast = np.clip(es_forecast, recent_prices[-1] * 0.5, recent_prices[-1] * 2.0)
        
    except Exception as e:
        st.warning(f"Exponential Smoothing failed, using weighted MA: {str(e)[:50]}")
        # Fallback: Exponentially weighted moving average
        weights = np.exp(np.linspace(-1, 0, min(20, len(recent_prices))))
        weights /= weights.sum()
        weighted_avg = np.average(recent_prices[-len(weights):], weights=weights)
        trend = (recent_prices[-1] - recent_prices[-min(10, len(recent_prices))]) / min(10, len(recent_prices))
        es_forecast = np.array([weighted_avg + trend * i for i in range(1, periods + 1)])
        es_forecast = np.clip(es_forecast, recent_prices[-1] * 0.5, recent_prices[-1] * 2.0)
    
    try:
        # Method 4: Adaptive Moving Average with Mean Reversion
        ma_short = np.mean(recent_prices[-min(10, len(recent_prices)):])
        ma_long = np.mean(recent_prices[-min(50, len(recent_prices)):])
        
        # Calculate momentum
        momentum = ma_short - ma_long
        
        # Mean reversion factor (decay momentum over time)
        decay_factor = 0.95
        ma_forecast = []
        current_price = recent_prices[-1]
        
        for i in range(periods):
            # Apply decaying momentum and mean revert to long-term average
            reversion = (ma_long - current_price) * 0.01  # Small mean reversion
            next_price = current_price + momentum * (decay_factor ** i) + reversion
            ma_forecast.append(next_price)
            current_price = next_price
        
        ma_forecast = np.array(ma_forecast)
        ma_forecast = np.clip(ma_forecast, recent_prices[-1] * 0.5, recent_prices[-1] * 2.0)
        
    except Exception as e:
        st.warning(f"Moving Average failed: {str(e)[:50]}")
        ma_forecast = lr_forecast.copy()
    
    # Ensemble forecast with adaptive weights
    # Give more weight to models that are closer to current price
    try:
        # Calculate model weights based on their starting predictions
        lr_weight = 1.0 / (1.0 + abs(lr_forecast[0] - recent_prices[-1]) / recent_prices[-1])
        arima_weight = 1.0 / (1.0 + abs(arima_forecast[0] - recent_prices[-1]) / recent_prices[-1])
        es_weight = 1.0 / (1.0 + abs(es_forecast[0] - recent_prices[-1]) / recent_prices[-1])
        ma_weight = 1.0 / (1.0 + abs(ma_forecast[0] - recent_prices[-1]) / recent_prices[-1])
        
        # Normalize weights
        total_weight = lr_weight + arima_weight + es_weight + ma_weight
        lr_weight /= total_weight
        arima_weight /= total_weight
        es_weight /= total_weight
        ma_weight /= total_weight
        
        # Weighted ensemble
        ensemble_forecast = (
            lr_weight * lr_forecast +
            arima_weight * arima_forecast +
            es_weight * es_forecast +
            ma_weight * ma_forecast
        )
        
        # Final safety constraint
        ensemble_forecast = np.clip(ensemble_forecast, recent_prices[-1] * 0.3, recent_prices[-1] * 3.0)
        
    except Exception as e:
        st.warning(f"Ensemble calculation failed: {str(e)[:50]}")
        ensemble_forecast = lr_forecast.copy()
    
    # Calculate realistic confidence intervals
    try:
        # Use historical volatility
        volatility = np.std(returns) if len(returns) > 0 else 0.02
        
        # Confidence intervals that widen over time (square root of time rule)
        time_factors = np.sqrt(np.arange(1, periods + 1))
        
        # 95% confidence interval (1.96 standard deviations)
        # Scale by current price level and time
        std_dev_prices = volatility * ensemble_forecast
        
        upper_bound = ensemble_forecast * (1 + 1.96 * volatility * time_factors / np.sqrt(252))
        lower_bound = ensemble_forecast * (1 - 1.96 * volatility * time_factors / np.sqrt(252))
        
        # Ensure bounds stay positive and reasonable
        lower_bound = np.maximum(lower_bound, ensemble_forecast * 0.5)
        lower_bound = np.maximum(lower_bound, recent_prices[-1] * 0.2)  # Never go below 20% of current
        upper_bound = np.minimum(upper_bound, ensemble_forecast * 2.0)
        
    except Exception as e:
        st.warning(f"Confidence interval calculation failed: {str(e)[:50]}")
        # Simple symmetric bounds if calculation fails
        bound_range = ensemble_forecast * 0.15  # +/- 15%
        upper_bound = ensemble_forecast + bound_range
        lower_bound = np.maximum(ensemble_forecast - bound_range, ensemble_forecast * 0.5)
    
    # Create forecast DataFrame
    forecast_dates = pd.date_range(
        start=data.index[-1] + timedelta(days=1), 
        periods=periods, 
        freq='D'
    )
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': ensemble_forecast,
        'Upper Bound': upper_bound,
        'Lower Bound': lower_bound,
        'Linear Regression': lr_forecast,
        'ARIMA': arima_forecast,
        'Exponential Smoothing': es_forecast,
        'Moving Average': ma_forecast
    })
    
    return forecast_df, ensemble_forecast

@st.cache_data(ttl=3600)
def scrape_fii_dii_data():
    """Scrape FII/DII data from NSE"""
    try:
        session = create_nse_session()
        
        # FII DII Trading data
        url = "https://www.nseindia.com/api/fiidiiTrading?index=equities"
        response = session.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data
        return None
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def get_company_news(company_name, ticker):
    """Fetch latest news for the company"""
    news_items = []
    
    try:
        # Yahoo Finance news - create fresh ticker object
        stock = yf.Ticker(ticker)
        yf_news = stock.news
        
        for item in yf_news[:5]:
            news_items.append({
                'title': item.get('title', 'No title'),
                'link': item.get('link', '#'),
                'publisher': item.get('publisher', 'Yahoo Finance'),
                'date': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d')
            })
    except Exception as e:
        pass  # Silently handle news fetch errors
    
    return news_items

@st.cache_data(ttl=3600)
def get_peer_comparison(ticker, info):
    """Get peer companies data for comparison"""
    try:
        industry = info.get('industry', '')
        sector = info.get('sector', '')
        
        # This is a simplified version - in production, you'd need a proper peer mapping
        peers_data = []
        
        # Get some basic peer info
        # You would need to implement proper peer discovery logic here
        
        return peers_data
    except Exception as e:
        return []

def get_valuation_metrics(info, current_price):
    """Calculate and interpret valuation metrics"""
    metrics = {}
    
    # Price ratios
    metrics['P/E Ratio'] = info.get('trailingPE', 'N/A')
    metrics['Forward P/E'] = info.get('forwardPE', 'N/A')
    metrics['P/B Ratio'] = info.get('priceToBook', 'N/A')
    metrics['P/S Ratio'] = info.get('priceToSalesTrailing12Months', 'N/A')
    metrics['PEG Ratio'] = info.get('pegRatio', 'N/A')
    
    # Profitability
    metrics['ROE (%)'] = info.get('returnOnEquity', 'N/A')
    if metrics['ROE (%)'] != 'N/A':
        metrics['ROE (%)'] = round(metrics['ROE (%)'] * 100, 2)
    
    metrics['ROA (%)'] = info.get('returnOnAssets', 'N/A')
    if metrics['ROA (%)'] != 'N/A':
        metrics['ROA (%)'] = round(metrics['ROA (%)'] * 100, 2)
    
    metrics['Profit Margin (%)'] = info.get('profitMargins', 'N/A')
    if metrics['Profit Margin (%)'] != 'N/A':
        metrics['Profit Margin (%)'] = round(metrics['Profit Margin (%)'] * 100, 2)
    
    # Dividend
    metrics['Dividend Yield (%)'] = info.get('dividendYield', 'N/A')
    if metrics['Dividend Yield (%)'] != 'N/A':
        metrics['Dividend Yield (%)'] = round(metrics['Dividend Yield (%)'] * 100, 2)
    
    # Market metrics
    metrics['Market Cap'] = info.get('marketCap', 'N/A')
    metrics['Enterprise Value'] = info.get('enterpriseValue', 'N/A')
    
    # Use calculated beta vs Nifty 50 if available
    metrics['Beta (vs Nifty 50)'] = info.get('calculatedBeta', info.get('beta', 'N/A'))
    if metrics['Beta (vs Nifty 50)'] != 'N/A' and metrics['Beta (vs Nifty 50)'] is not None:
        metrics['Beta (vs Nifty 50)'] = round(metrics['Beta (vs Nifty 50)'], 3)
    
    return metrics

def interpret_valuation(metrics):
    """Interpret valuation metrics"""
    interpretations = []
    
    # P/E interpretation
    pe = metrics.get('P/E Ratio', 'N/A')
    if pe != 'N/A' and pe is not None:
        if pe < 15:
            interpretations.append("💰 **Undervalued P/E**: P/E ratio below 15 suggests potential undervaluation.")
        elif pe < 25:
            interpretations.append("💰 **Fair P/E**: P/E ratio indicates fair valuation.")
        else:
            interpretations.append("💰 **High P/E**: P/E ratio above 25 suggests premium valuation or growth expectations.")
    
    # P/B interpretation
    pb = metrics.get('P/B Ratio', 'N/A')
    if pb != 'N/A' and pb is not None:
        if pb < 1:
            interpretations.append("📚 **Below Book Value**: Trading below book value may indicate undervaluation.")
        elif pb < 3:
            interpretations.append("📚 **Reasonable P/B**: P/B ratio suggests fair valuation relative to assets.")
        else:
            interpretations.append("📚 **High P/B**: Premium to book value, justified by intangibles or growth.")
    
    # ROE interpretation
    roe = metrics.get('ROE (%)', 'N/A')
    if roe != 'N/A' and roe is not None:
        if roe > 15:
            interpretations.append("⭐ **Strong ROE**: Company efficiently generates returns on equity.")
        elif roe > 10:
            interpretations.append("⭐ **Decent ROE**: Moderate return on equity.")
        else:
            interpretations.append("⭐ **Low ROE**: Company may struggle with profitability.")
    
    # Beta interpretation
    beta = metrics.get('Beta (vs Nifty 50)', 'N/A')
    if beta != 'N/A' and beta is not None:
        if beta > 1.2:
            interpretations.append("📊 **High Beta**: Stock is more volatile than Nifty 50, higher risk but potential for higher returns.")
        elif beta > 0.8:
            interpretations.append("📊 **Moderate Beta**: Stock moves roughly in line with Nifty 50.")
        else:
            interpretations.append("📊 **Low Beta**: Stock is less volatile than Nifty 50, more defensive.")
    
    # Dividend interpretation
    div_yield = metrics.get('Dividend Yield (%)', 'N/A')
    if div_yield != 'N/A' and div_yield is not None and div_yield > 0:
        if div_yield > 4:
            interpretations.append("💵 **High Dividend Yield**: Attractive for income investors.")
        elif div_yield > 2:
            interpretations.append("💵 **Moderate Dividend**: Provides some income.")
        else:
            interpretations.append("💵 **Low Dividend**: Focus on capital appreciation.")
    
    return interpretations

# Main App
def main():
    st.markdown('<p class="main-header">📈 Advanced Stock Analysis Tool</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔍 Stock Selection")
        
        # Exchange selection (MUTUALLY EXCLUSIVE)
        exchange_mode = st.radio("📈 Select Exchange", ["NSE", "BSE"], horizontal=True, 
            help="Choose NSE or BSE stocks (mutually exclusive)",
            key="exchange_mode_radio")
        
        # Save to session state
        st.session_state.exchange_mode = exchange_mode
        
        # Load stocks based on selection
        if exchange_mode == "NSE":
            try:
                with open('nse.txt', 'r') as f:
                    available_stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
                st.success(f"✅ Loaded {len(available_stocks)} NSE stocks")
                exchange_suffix = ".NS"
                st.session_state.exchange_suffix = ".NS"
            except FileNotFoundError:
                st.error("⚠️ nse.txt not found")
                available_stocks = []
                exchange_suffix = ".NS"
        else:  # BSE
            try:
                with open('bse.txt', 'r') as f:
                    available_stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
                st.success(f"✅ Loaded {len(available_stocks)} BSE stocks")
                exchange_suffix = ".BO"
                st.session_state.exchange_suffix = ".BO"
            except FileNotFoundError:
                st.error("⚠️ bse.txt not found")
                available_stocks = []
                exchange_suffix = ".BO"
        
        st.markdown("---")
        
        # Analysis mode
        analysis_mode = st.radio("🔬 Analysis Mode", 
            ["Single Stock", "Slot-wise Analysis"], 
            help="Analyze one stock or scan multiple stocks in slots")
        
        if analysis_mode == "Single Stock":
            ticker_input = st.text_input(f"Enter {exchange_mode} Ticker Symbol", 
                value="RELIANCE", 
                help=f"Enter the {exchange_mode} ticker (without {exchange_suffix} suffix)")
            suffix_to_use = st.session_state.get("exchange_suffix", ".NS")
            stocks_to_analyze = [ticker_input.strip().upper() + suffix_to_use] if ticker_input.strip() else []
        
        else:  # Slot-wise Analysis
            st.subheader("📦 Select Slots to Analyze")
            
            if not available_stocks:
                st.warning("No stocks loaded. Please check your files.")
                stocks_to_analyze = []
            else:
                total_stocks = len(available_stocks)
                slot_size = 1000
                num_slots = (total_stocks + slot_size - 1) // slot_size
                
                st.info(f"📊 Total: {total_stocks} stocks\n💼 Slot size: 1000\n📦 Slots: {num_slots}")
                
                # Quick select buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ All", use_container_width=True):
                        for slot_num in range(num_slots):
                            st.session_state[f"analyzer_slot_{slot_num}"] = True
                        st.rerun()
                with col2:
                    if st.button("❌ None", use_container_width=True):
                        for slot_num in range(num_slots):
                            st.session_state[f"analyzer_slot_{slot_num}"] = False
                        st.rerun()
                
                st.markdown("---")
                
                # Slot checkboxes
                selected_slots = []
                for slot_num in range(num_slots):
                    start_idx = slot_num * slot_size
                    end_idx = min((slot_num + 1) * slot_size, total_stocks)
                    slot_count = end_idx - start_idx
                    
                    slot_label = f"Slot {slot_num + 1}: {start_idx + 1}-{end_idx} ({slot_count})"
                    
                    if st.checkbox(slot_label, key=f"analyzer_slot_{slot_num}"):
                        selected_slots.append(slot_num)
                
                # Build stock list from selected slots
                stocks_to_analyze = []
                for slot_num in selected_slots:
                    start_idx = slot_num * slot_size
                    end_idx = min((slot_num + 1) * slot_size, total_stocks)
                    # Add exchange suffix to each stock
                    suffix_to_use = st.session_state.get("exchange_suffix", ".NS")
                    stocks_to_analyze.extend([s + suffix_to_use for s in available_stocks[start_idx:end_idx]])
                
                if not selected_slots:
                    st.warning("⚠️ Select at least one slot")
                else:
                    suffix_to_use = st.session_state.get("exchange_suffix", ".NS")
                    st.success(f"✅ {len(selected_slots)} slot(s) = {len(stocks_to_analyze)} stocks")
                    st.info(f"🔍 Exchange suffix: **{suffix_to_use}** | Sample: {stocks_to_analyze[0] if stocks_to_analyze else 'N/A'}")
        
        st.markdown("---")
        st.header("⚙️ Analysis Options")
        show_fluctuations = st.checkbox("Show Major Fluctuations", value=True)
        show_statistics = st.checkbox("Show Statistical Analysis", value=True)
        show_forecast = st.checkbox("Show Price Forecast", value=False)
        show_news = st.checkbox("Show News & Events", value=False)
        show_valuation = st.checkbox("Show Valuation Metrics", value=True)
        
        forecast_days = st.slider("Forecast Period (Days)", 7, 90, 30)
        
        st.markdown("---")
        analyze_button = st.button("🚀 Analyze Stock", type="primary")
    
    if analyze_button and stocks_to_analyze:
        if len(stocks_to_analyze) == 1:
            # Single stock analysis (original flow)
            # DON'T strip suffix - get_stock_data needs it to determine exchange
            ticker_with_suffix = stocks_to_analyze[0]
            ticker = ticker_with_suffix.replace('.NS', '').replace('.BO', '')
            
            with st.spinner(f"Fetching data for {ticker}..."):
                data_dict, info, nse_data, nse_symbol = get_stock_data(ticker_with_suffix)
        
        if data_dict is not None and info:
            # Reconstruct DataFrame from cached dict
            data = reconstruct_dataframe(data_dict)
            
            if data is None or len(data) == 0:
                st.error(f"❌ No historical data available for ticker '{ticker}'.")
                return
            
            # Company Header
            company_name = info.get('longName', ticker)
            current_price = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            # Show NSE data if available
            if nse_data:
                st.success("✅ NSE Data Retrieved Successfully")
                with st.expander("📊 View NSE Live Data"):
                    try:
                        nse_info = nse_data.get('priceInfo', {})
                        st.write(f"**Last Price (NSE):** ₹{nse_info.get('lastPrice', 'N/A')}")
                        st.write(f"**Change:** ₹{nse_info.get('change', 'N/A')} ({nse_info.get('pChange', 'N/A')}%)")
                        st.write(f"**52W High:** ₹{nse_info.get('weekHighLow', {}).get('max', 'N/A')}")
                        st.write(f"**52W Low:** ₹{nse_info.get('weekHighLow', {}).get('min', 'N/A')}")
                        st.write(f"**Total Traded Volume:** {nse_data.get('securityWiseDP', {}).get('quantityTraded', 'N/A'):,}")
                    except:
                        st.json(nse_data)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Company", company_name)
            with col2:
                st.metric("Current Price", f"₹{current_price:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
            with col3:
                st.metric("52W High", f"₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
            with col4:
                st.metric("52W Low", f"₹{info.get('fiftyTwoWeekLow', 'N/A')}")
            
            # Key Information
            st.markdown("---")
            st.subheader("📋 Key Information")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write("**Sector:**", info.get('sector', 'N/A'))
                st.write("**Industry:**", info.get('industry', 'N/A'))
            with col2:
                market_cap = info.get('marketCap', 'N/A')
                if market_cap != 'N/A':
                    st.write("**Market Cap:**", f"₹{market_cap/10000000:.2f} Cr")
                else:
                    st.write("**Market Cap:**", 'N/A')
            with col3:
                # Show calculated beta vs appropriate index
                calc_beta = info.get('calculatedBeta', info.get('beta', 'N/A'))
                
                # Determine index based on ticker
                index_name = "Sensex" if '.BO' in ticker_with_suffix else "Nifty 50"
                
                if calc_beta != 'N/A' and calc_beta is not None:
                    st.write(f"**Beta (vs {index_name}):**", f"{calc_beta:.3f}")
                else:
                    st.write("**Beta:**", 'N/A')
                st.write("**Volume:**", f"{info.get('volume', 'N/A'):,}")
            with col4:
                st.write("**Avg Volume:**", f"{info.get('averageVolume', 'N/A'):,}")
                st.write("**Listing Date:**", data.index[0].strftime('%Y-%m-%d'))
            
            # Major Fluctuations
            if show_fluctuations:
                st.markdown("---")
                st.subheader("📊 Major Price Fluctuations Analysis")
                
                fluctuations = identify_major_fluctuations(data, threshold=5)
                
                if fluctuations:
                    # Create visualization
                    fig = go.Figure()
                    
                    # Price line
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Mark fluctuations
                    for fluct in fluctuations:
                        color = 'red' if fluct['type'] == 'Trough' else 'green' if fluct['type'] == 'Peak' else 'blue'
                        fig.add_trace(go.Scatter(
                            x=[fluct['date']],
                            y=[fluct['price']],
                            mode='markers+text',
                            name=fluct['type'],
                            marker=dict(size=12, color=color),
                            text=[f"{fluct['type']}<br>₹{fluct['price']:.2f}"],
                            textposition="top center",
                            showlegend=False
                        ))
                    
                    fig.update_layout(
                        title=f"{company_name} - Price History with Major Fluctuations",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Fluctuations table
                    st.subheader("📌 Identified Major Events")
                    fluct_df = pd.DataFrame(fluctuations)
                    fluct_df['date'] = fluct_df['date'].dt.strftime('%Y-%m-%d')
                    fluct_df['price'] = fluct_df['price'].apply(lambda x: f"₹{x:.2f}")
                    fluct_df['change_pct'] = fluct_df['change_pct'].apply(lambda x: f"{x:.2f}%")
                    fluct_df.columns = ['Date', 'Price', 'Event Type', 'Change from Listing (%)']
                    
                    st.dataframe(fluct_df, use_container_width=True, hide_index=True)
            
            # Statistical Analysis
            if show_statistics:
                st.markdown("---")
                st.subheader("📊 Comprehensive Statistical Analysis")
                
                stats_dict = calculate_statistics(data)
                interpretations = interpret_statistics(stats_dict, current_price)
                
                # Display statistics in organized sections
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Descriptive Stats", "📉 Return Metrics", "⚠️ Risk Metrics", "📐 Technical Indicators"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean Price", f"₹{stats_dict['Mean Price']:.2f}")
                        st.metric("Median Price", f"₹{stats_dict['Median Price']:.2f}")
                        st.metric("Std Deviation", f"₹{stats_dict['Std Deviation']:.2f}")
                        st.metric("Min Price", f"₹{stats_dict['Min Price']:.2f}")
                    with col2:
                        st.metric("Max Price", f"₹{stats_dict['Max Price']:.2f}")
                        st.metric("Price Range", f"₹{stats_dict['Range']:.2f}")
                        st.metric("Coefficient of Variation", f"{stats_dict['Coefficient of Variation']:.2f}%")
                        st.metric("Variance", f"{stats_dict['Variance']:.2f}")
                
                with tab2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean Daily Return", f"{stats_dict['Mean Daily Return']:.4f}%")
                        st.metric("Median Daily Return", f"{stats_dict['Median Daily Return']:.4f}%")
                        st.metric("Annualized Return", f"{stats_dict['Annualized Return']:.2f}%")
                    with col2:
                        st.metric("Return Std Dev", f"{stats_dict['Return Std Dev']:.4f}%")
                        st.metric("Annualized Volatility", f"{stats_dict['Annualized Volatility']:.2f}%")
                        st.metric("Skewness", f"{stats_dict['Skewness']:.4f}")
                
                with tab3:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sharpe Ratio", f"{stats_dict['Sharpe Ratio']:.4f}")
                        st.metric("Sortino Ratio", f"{stats_dict['Sortino Ratio']:.4f}")
                        st.metric("Max Drawdown", f"{stats_dict['Max Drawdown']:.2f}%")
                    with col2:
                        st.metric("Value at Risk (95%)", f"{stats_dict['Value at Risk (95%)']:.4f}%")
                        st.metric("Conditional VaR", f"{stats_dict['Conditional VaR (95%)']:.4f}%")
                        st.metric("Kurtosis", f"{stats_dict['Kurtosis']:.4f}")
                
                with tab4:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("50-Day MA", f"₹{stats_dict['50-Day MA']:.2f}")
                        st.metric("200-Day MA", f"₹{stats_dict['200-Day MA']:.2f}")
                        if 'RSI (14)' in stats_dict:
                            st.metric("RSI (14)", f"{stats_dict['RSI (14)']:.2f}")
                    with col2:
                        if 'MACD' in stats_dict:
                            st.metric("MACD", f"{stats_dict['MACD']:.4f}")
                        if 'MACD Signal' in stats_dict:
                            st.metric("MACD Signal", f"{stats_dict['MACD Signal']:.4f}")
                
                # Interpretations
                st.markdown("---")
                st.subheader("🔍 Statistical Interpretations")
                for interp in interpretations:
                    st.info(interp)
            
            # Valuation Metrics
            if show_valuation:
                st.markdown("---")
                st.subheader("💰 Valuation Metrics & Analysis")
                
                valuation_metrics = get_valuation_metrics(info, current_price)
                valuation_interp = interpret_valuation(valuation_metrics)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Price Ratios**")
                    st.write(f"**P/E Ratio:** {valuation_metrics['P/E Ratio']}")
                    st.write(f"**Forward P/E:** {valuation_metrics['Forward P/E']}")
                    st.write(f"**P/B Ratio:** {valuation_metrics['P/B Ratio']}")
                    st.write(f"**P/S Ratio:** {valuation_metrics['P/S Ratio']}")
                    st.write(f"**PEG Ratio:** {valuation_metrics['PEG Ratio']}")
                
                with col2:
                    st.markdown("**Profitability**")
                    st.write(f"**ROE:** {valuation_metrics['ROE (%)']}")
                    st.write(f"**ROA:** {valuation_metrics['ROA (%)']}")
                    st.write(f"**Profit Margin:** {valuation_metrics['Profit Margin (%)']}")
                    st.write(f"**Dividend Yield:** {valuation_metrics['Dividend Yield (%)']}")
                
                with col3:
                    st.markdown("**Market Metrics**")
                    market_cap = valuation_metrics['Market Cap']
                    if market_cap != 'N/A':
                        st.write(f"**Market Cap:** ₹{market_cap/10000000:.2f} Cr")
                    else:
                        st.write(f"**Market Cap:** N/A")
                    
                    ent_val = valuation_metrics['Enterprise Value']
                    if ent_val != 'N/A':
                        st.write(f"**Enterprise Value:** ₹{ent_val/10000000:.2f} Cr")
                    else:
                        st.write(f"**Enterprise Value:** N/A")
                    
                    st.write(f"**Beta (vs Nifty 50):** {valuation_metrics['Beta (vs Nifty 50)']}")
                
                st.markdown("---")
                st.subheader("📊 Valuation Interpretations")
                for interp in valuation_interp:
                    st.success(interp)
            
            # Price Forecast
            if show_forecast:
                st.markdown("---")
                st.subheader("🔮 Price Forecast")
                
                with st.spinner("Generating forecasts..."):
                    forecast_df, ensemble = forecast_prices(data, periods=forecast_days)
                
                if forecast_df is not None:
                    # Forecast chart
                    fig = go.Figure()
                    
                    # Historical prices
                    fig.add_trace(go.Scatter(
                        x=data.index[-90:],
                        y=data['Close'].tail(90),
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Forecast'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='#ff7f0e', width=2, dash='dash')
                    ))
                    
                    # Confidence interval
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Upper Bound'],
                        mode='lines',
                        name='Upper Bound (95%)',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Lower Bound'],
                        mode='lines',
                        name='Lower Bound (95%)',
                        line=dict(width=0),
                        fillcolor='rgba(255, 127, 14, 0.2)',
                        fill='tonexty',
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title=f"{company_name} - {forecast_days} Day Price Forecast",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Price", f"₹{current_price:.2f}")
                    with col2:
                        forecast_price = forecast_df['Forecast'].iloc[-1]
                        forecast_change = ((forecast_price - current_price) / current_price) * 100
                        st.metric(f"{forecast_days}-Day Forecast", f"₹{forecast_price:.2f}", f"{forecast_change:.2f}%")
                    with col3:
                        st.metric("Upper Bound", f"₹{forecast_df['Upper Bound'].iloc[-1]:.2f}")
                    with col4:
                        st.metric("Lower Bound", f"₹{forecast_df['Lower Bound'].iloc[-1]:.2f}")
                    
                    # Forecast interpretation
                    st.markdown("---")
                    st.subheader("📝 Forecast Interpretation")
                    
                    if forecast_change > 10:
                        st.success(f"🚀 **Bullish Outlook**: Models predict a potential {forecast_change:.2f}% increase over the next {forecast_days} days.")
                    elif forecast_change > 0:
                        st.info(f"📈 **Positive Outlook**: Models suggest a modest {forecast_change:.2f}% gain over the next {forecast_days} days.")
                    elif forecast_change > -10:
                        st.warning(f"📉 **Slight Decline**: Models predict a {abs(forecast_change):.2f}% decrease over the next {forecast_days} days.")
                    else:
                        st.error(f"🔻 **Bearish Outlook**: Models suggest a significant {abs(forecast_change):.2f}% decline over the next {forecast_days} days.")
                    
                    st.info("⚠️ **Disclaimer**: Forecasts are based on historical data and statistical models. Actual prices may vary significantly due to market conditions, news events, and other factors. Past performance does not guarantee future results.")
                    
                    # Show forecast data
                    with st.expander("📋 View Detailed Forecast Data"):
                        display_df = forecast_df.copy()
                        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                        for col in ['Forecast', 'Upper Bound', 'Lower Bound', 'Linear Regression', 'ARIMA', 'Exponential Smoothing', 'Moving Average']:
                            if col in display_df.columns:
                                display_df[col] = display_df[col].apply(lambda x: f"₹{x:.2f}")
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Model comparison
                    with st.expander("🔬 View Model Comparison"):
                        st.markdown("### Individual Model Forecasts")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            lr_final = forecast_df['Linear Regression'].iloc[-1]
                            lr_change = ((lr_final - current_price) / current_price) * 100
                            st.metric("Linear Regression", f"₹{lr_final:.2f}", f"{lr_change:.2f}%")
                        
                        with col2:
                            arima_final = forecast_df['ARIMA'].iloc[-1]
                            arima_change = ((arima_final - current_price) / current_price) * 100
                            st.metric("ARIMA", f"₹{arima_final:.2f}", f"{arima_change:.2f}%")
                        
                        with col3:
                            es_final = forecast_df['Exponential Smoothing'].iloc[-1]
                            es_change = ((es_final - current_price) / current_price) * 100
                            st.metric("Exp. Smoothing", f"₹{es_final:.2f}", f"{es_change:.2f}%")
                        
                        with col4:
                            ma_final = forecast_df['Moving Average'].iloc[-1]
                            ma_change = ((ma_final - current_price) / current_price) * 100
                            st.metric("Moving Average", f"₹{ma_final:.2f}", f"{ma_change:.2f}%")
                        
                        st.info("**Ensemble Forecast** combines all models with weighted averaging for better accuracy.")
            
            # News & Events Section
            if show_news:
                st.markdown("---")
                st.subheader("📰 News & Market Events")
                
                tab1, tab2, tab3, tab4 = st.tabs(["📰 Latest News", "🏦 FII/DII Data", "📊 Corporate Actions", "👥 Shareholding Pattern"])
                
                with tab1:
                    st.markdown("### Latest News Articles")
                    news_items = get_company_news(company_name, ticker + '.NS')
                    
                    if news_items:
                        for item in news_items:
                            with st.container():
                                st.markdown(f"**{item['title']}**")
                                st.caption(f"📅 {item['date']} | 📰 {item['publisher']}")
                                st.markdown(f"[Read more]({item['link']})")
                                st.markdown("---")
                    else:
                        st.info("No recent news available.")
                
                with tab2:
                    st.markdown("### FII/DII Inflows")
                    fii_data = scrape_fii_dii_data()
                    
                    if fii_data:
                        st.success("✅ FII/DII data fetched from NSE")
                        
                        # Try to parse and display in a nice format
                        try:
                            if 'data' in fii_data:
                                df = pd.DataFrame(fii_data['data'])
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.json(fii_data)
                        except:
                            st.json(fii_data)
                    else:
                        st.info("FII/DII data not available. This data requires NSE API access with proper authentication.")
                        st.markdown("""
                        **Note**: Real-time FII/DII data requires:
                        - NSE data subscription
                        - Proper API authentication
                        - Session management with NSE servers
                        
                        You can manually check this data at: https://www.nseindia.com/reports-indices-institutional-investor-wise
                        """)
                
                with tab3:
                    st.markdown("### Corporate Actions & Announcements")
                    
                    # Get corporate actions from NSE
                    corp_actions = get_nse_corporate_actions(nse_symbol)
                    
                    if corp_actions:
                        st.success("✅ Corporate actions data fetched from NSE")
                        try:
                            if isinstance(corp_actions, list):
                                df = pd.DataFrame(corp_actions)
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.json(corp_actions)
                        except:
                            st.json(corp_actions)
                    
                    # Dividend info from Yahoo Finance
                    st.markdown("---")
                    st.markdown("### Dividend Information")
                    if info.get('dividendRate'):
                        st.success(f"💵 **Dividend**: ₹{info.get('dividendRate', 'N/A')} per share")
                        st.write(f"**Dividend Yield**: {info.get('dividendYield', 'N/A')}%")
                        st.write(f"**Ex-Dividend Date**: {info.get('exDividendDate', 'N/A')}")
                    else:
                        st.info("No dividend information available")
                    
                    # Earnings date
                    if info.get('earningsDate'):
                        st.info(f"📅 **Next Earnings Date**: {info.get('earningsDate', 'N/A')}")
                    
                    if not corp_actions:
                        st.info("For detailed corporate action history, visit: https://www.bseindia.com/corporates/corporates_act.aspx")
                
                with tab4:
                    st.markdown("### Shareholding Pattern")
                    
                    shareholding = get_nse_shareholding(nse_symbol)
                    
                    if shareholding:
                        st.success("✅ Shareholding data fetched from NSE")
                        try:
                            # Display shareholding data
                            if 'data' in shareholding:
                                for category, details in shareholding['data'].items():
                                    st.subheader(category.replace('_', ' ').title())
                                    if isinstance(details, list) and len(details) > 0:
                                        df = pd.DataFrame(details)
                                        st.dataframe(df, use_container_width=True)
                                    elif isinstance(details, dict):
                                        st.json(details)
                            else:
                                st.json(shareholding)
                        except Exception as e:
                            st.json(shareholding)
                    else:
                        st.info("Shareholding pattern data not available from NSE API")
                        st.markdown("""
                        **Note**: You can check shareholding patterns at:
                        - NSE: https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern
                        - BSE: https://www.bseindia.com/corporates/shpPromoterNGroup.aspx
                        """)
        
        else:
            # BATCH ANALYSIS MODE - Multiple stocks
            st.header(f"📊 Batch Analysis: {len(stocks_to_analyze)} Stocks")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            failed = 0
            
            for idx, ticker_with_suffix in enumerate(stocks_to_analyze):
                ticker = ticker_with_suffix.replace('.NS', '').replace('.BO', '')
                status_text.info(f"Analyzing {ticker}... ({idx+1}/{len(stocks_to_analyze)})")
                
                try:
                    # Pass ticker WITH suffix so get_stock_data knows exchange
                    data_dict, info, nse_data, nse_symbol = get_stock_data(ticker_with_suffix)
                    
                    if data_dict and info:
                        data = reconstruct_dataframe(data_dict)
                        
                        if data is not None and len(data) > 0:
                            # Extract key metrics for batch display
                            current_price = data['Close'].iloc[-1]
                            prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                            change_pct = ((current_price - prev_close) / prev_close) * 100
                            
                            results.append({
                                'Symbol': ticker_with_suffix,
                                'Company': info.get('longName', ticker),
                                'Price': current_price,
                                'Change %': change_pct,
                                'Market Cap (Cr)': info.get('marketCap', 0) / 10000000 if info.get('marketCap') else 0,
                                'PE Ratio': info.get('trailingPE', 'N/A'),
                                'Sector': info.get('sector', 'N/A'),
                                '52W High': info.get('fiftyTwoWeekHigh', 'N/A'),
                                '52W Low': info.get('fiftyTwoWeekLow', 'N/A')
                            })
                        else:
                            failed += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                
                progress_bar.progress((idx + 1) / len(stocks_to_analyze))
                time.sleep(0.3)  # Delay to avoid rate limiting
            
            status_text.success(f"✅ Analysis complete! Valid: {len(results)} | Failed: {failed}")
            
            if results:
                st.subheader("📈 Batch Results")
                
                df_results = pd.DataFrame(results)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Analyzed", len(results))
                with col2:
                    avg_change = df_results['Change %'].mean()
                    st.metric("Avg Change %", f"{avg_change:.2f}%")
                with col3:
                    gainers = len(df_results[df_results['Change %'] > 0])
                    st.metric("Gainers", gainers)
                with col4:
                    losers = len(df_results[df_results['Change %'] < 0])
                    st.metric("Losers", losers)
                
                st.markdown("---")
                
                # Display results table
                st.dataframe(
                    df_results.style.format({
                        'Price': '₹{:.2f}',
                        'Change %': '{:.2f}%',
                        'Market Cap (Cr)': '{:.2f}',
                        'PE Ratio': '{:.2f}',
                        '52W High': '₹{:.2f}',
                        '52W Low': '₹{:.2f}'
                    }).background_gradient(subset=['Change %'], cmap='RdYlGn', vmin=-5, vmax=5),
                    use_container_width=True,
                    height=600
                )
                
                # Download button
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=df_results.to_csv(index=False).encode('utf-8'),
                    file_name=f"batch_analysis_{exchange_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No valid results. All stocks failed to fetch data.")
    
    elif not stocks_to_analyze:
        st.info("👈 Please enter a ticker symbol or select slots in the sidebar and click 'Analyze Stock'")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 Advanced Stock Analysis Tool | Data powered by Yahoo Finance & NSE</p>
        <p><small>Disclaimer: This tool is for informational purposes only. Not financial advice. Please consult a financial advisor before making investment decisions.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
