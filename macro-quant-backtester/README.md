# Macro Quant Backtester

An interactive EUR/USD, GBP/USD and USD/JPY backtesting dashboard built with Python, pandas and Streamlit.

**Live demo:** _add your Streamlit link here_

## Features
- Moving-average crossover and Wilder RSI strategies, with adjustable parameters
- Vectorised engine with next-bar execution (no look-ahead bias), commission and slippage
- Tear sheet vs buy & hold: CAGR, volatility, Sharpe, max drawdown and duration, win rate, profit factor
- Interactive equity, drawdown and signal charts; downloadable trade log
- LLM (Cohere) plain-English explanations of why each mechanical signal fired
- Unit-tested engine (`pytest`)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Optional: set `COHERE_API_KEY` in a local `.env` (never commit it) to enable AI explanations.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (`.env` is git-ignored).
2. Go to share.streamlit.io, choose the repo, and set the main file to `app.py`.
3. Under Advanced settings > Secrets, add: `COHERE_API_KEY = "your-key"`.

## Structure
```
app.py            Streamlit interface
data/             Yahoo Finance loader + CSV cache
engine/           signals, backtest engine, performance metrics
ai/               Cohere trade explainer
tests/            pytest suite
```
*Educational research tool. Not financial advice.*
