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
warnings.filterwarnings('ignore')

# Statistical and ML imports
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import ta

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    """Fetch stock data from Yahoo Finance"""
    try:
        # Add .NS suffix for NSE stocks
        if not ticker.endswith('.NS'):
            ticker = ticker + '.NS'
        
        stock = yf.Ticker(ticker)
        data = stock.history(period="max")
        info = stock.info
        
        # Convert to serializable format
        return data, info, ticker
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None

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
    """Forecast future prices using multiple methods"""
    if data is None or len(data) < 50:
        return None, None
    
    prices = data['Close'].values
    
    # Method 1: Linear Regression
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    
    future_X = np.arange(len(prices), len(prices) + periods).reshape(-1, 1)
    lr_forecast = lr_model.predict(future_X)
    
    # Method 2: ARIMA (simplified)
    try:
        arima_model = ARIMA(prices, order=(5, 1, 0))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=periods)
    except:
        arima_forecast = lr_forecast
    
    # Method 3: Exponential Smoothing
    try:
        es_model = ExponentialSmoothing(prices, trend='add', seasonal=None)
        es_fit = es_model.fit()
        es_forecast = es_fit.forecast(steps=periods)
    except:
        es_forecast = lr_forecast
    
    # Ensemble forecast (average of methods)
    ensemble_forecast = (lr_forecast + arima_forecast + es_forecast) / 3
    
    # Calculate confidence intervals
    std_dev = np.std(prices[-30:])
    upper_bound = ensemble_forecast + 1.96 * std_dev
    lower_bound = ensemble_forecast - 1.96 * std_dev
    
    forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=periods, freq='D')
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': ensemble_forecast,
        'Upper Bound': upper_bound,
        'Lower Bound': lower_bound,
        'Linear Regression': lr_forecast,
        'ARIMA': arima_forecast,
        'Exponential Smoothing': es_forecast
    })
    
    return forecast_df, ensemble_forecast

@st.cache_data(ttl=3600)
def scrape_fii_dii_data():
    """Scrape FII/DII data from NSE"""
    try:
        url = "https://www.nseindia.com/api/fiidiiTrading?index=fii"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers)
        
        response = session.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return None
    except Exception as e:
        st.warning(f"Could not fetch FII/DII data: {e}")
        return None

@st.cache_data(ttl=3600)
def get_company_news(company_name, ticker):
    """Fetch latest news for the company"""
    news_items = []
    
    try:
        # Recreate Yahoo Finance ticker object
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
        st.warning(f"Could not fetch news: {e}")
    
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
    metrics['Beta'] = info.get('beta', 'N/A')
    
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
        ticker_input = st.text_input("Enter NSE Ticker Symbol", value="RELIANCE", help="Enter the NSE ticker (e.g., RELIANCE, TCS, INFY)")
        
        st.markdown("---")
        st.header("⚙️ Analysis Options")
        show_fluctuations = st.checkbox("Show Major Fluctuations", value=True)
        show_statistics = st.checkbox("Show Statistical Analysis", value=True)
        show_forecast = st.checkbox("Show Price Forecast", value=True)
        show_news = st.checkbox("Show News & Events", value=True)
        show_valuation = st.checkbox("Show Valuation Metrics", value=True)
        
        forecast_days = st.slider("Forecast Period (Days)", 7, 90, 30)
        
        st.markdown("---")
        analyze_button = st.button("🚀 Analyze Stock", type="primary")
    
    if analyze_button and ticker_input:
        ticker = ticker_input.strip().upper()
        
        with st.spinner(f"Fetching data for {ticker}..."):
            data, info, ticker_full = get_stock_data(ticker)
        
        if data is not None and len(data) > 0 and info:
            # Company Header
            company_name = info.get('longName', ticker)
            current_price = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
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
                st.write("**Beta:**", round(info.get('beta', 0), 2))
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
                        mode='lin
