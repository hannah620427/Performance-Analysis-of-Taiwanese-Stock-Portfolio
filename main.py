import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


def load_portfolio(filepath):
    """
    自input提供的csv檔案路徑，整理並回傳
    stocks: 包含所有持有股票之代碼的list
    shares: 一個index為持股代碼，values為持股數的Series
    """

    filepath = filepath.strip().strip('"')
    try:
        df = pd.read_csv(filepath, dtype={"stock": str})
    except FileNotFoundError:
        print("找不到檔案，請確認路徑是否正確")
        return None, None
    
    required_cols = {"stock", "shares"}
    if not required_cols.issubset(df.columns):
        print("CSV 欄位需包含 stock 和 shares")
        return None, None

    stocks = df["stock"].astype(str).apply(
        lambda x: x if x.endswith(".TW") else x + ".TW"
    ).to_list()
    shares = pd.Series(df["shares"].values, index=stocks)    # df["shares"].values是nd.array
    
    return stocks, shares


def compute_stocks_daily_return(stocks, starting_date):
    """
    使用yahoo finance的資料，回傳
    stocks_return: 一個index為日期，記錄各持股的日報酬率(用收盤價計算)的DataFrame
    """
    data = yf.download(
        stocks, 
        start=str(starting_date))
    stocks_return = data["Close"].pct_change(fill_method=None).dropna(how="all")

    return stocks_return


def compute_portfolio_daily_return(shares, stocks_return):
    """
    依照持有的各檔股票在投資組合中的權重，計算出投資組合的日報酬率並回傳
    portfolio_return: 一個index為日期，values為報酬率
    """
    weights = shares / shares.sum()    # index: 持股代碼
    portfolio_return = stocks_return.dot(weights)
    return portfolio_return


def compute_risk_free_discount_rate(portfolio_return):
    """
    使用臺灣央行公布的91天期國庫券利率資料作為無風險報酬率的指標，
    把91天期國庫券發行日公布的利率向前填補至下一個發行日之前並回傳
    rf: 一個index為日期，values為91天期國庫券利率的Series
    """
    rf_data = pd.read_csv("data/Taiwan_treasury_bills_final.csv")
    rf = rf_data[rf_data["Term_Days"] == 91]    

    rf = rf.copy()

    rf["Issue_Date"] = pd.to_datetime(rf["Issue_Date"])
    rf.index = rf["Issue_Date"]

    # 把rf的日期對齊成portfolio_return的日期
    # 用method="ffill"跨index外補值
    rf = rf.reindex(portfolio_return.index, method="ffill")   

    rf = rf.rename(columns={"Weighted_Avg_Discount_Rate_Pct": "discount_rate"})
    rf = rf["discount_rate"] / 100     # 因discount_rate以百分比(省略%)呈現，故需 / 100
    return rf


def load_market_return(starting_date):
    """
    使用yahoo finance的資料，以台灣加權股價指數（^TWII）做為市場利率指標，
    以收盤價計算日報酬率並回傳
    market_return:　一個index是日期，values是日報酬率的Series
    """

    # 完成合併市場報酬率與portfolio組合的報酬率的df
    market = yf.download(
        "^TWII",
        start = str(starting_date)
    )

    market_return = market["Close"].pct_change(fill_method=None).dropna()
    market_return = market_return.squeeze()     # df轉series

    return market_return


def get_concat_data(portfolio_return, market_return, rf):
    """
    使用前面整理出的三個Series，整併成DataFrame以便於後續分析與計算，回傳
    combined_RpRm: 整併投資組合日報酬率與市場日報酬率(為了後續計算Beta時拿來使用)
    combined_all: 整併投資組合日報酬率、市場日報酬率與無風險報酬率
    """
    combined_RpRm = pd.concat([portfolio_return, market_return], axis=1)
    combined_RpRm.columns = ["portfolio", "market"]
    combined_RpRm = combined_RpRm.dropna(how="any")
    
    combined_all = combined_RpRm.join(rf.rename("risk_free"))

    return combined_RpRm, combined_all


def compute_annual_portfolio_return_and_volatility(combined_all):
    """
    以Effective Annual Rate (EAR)與一年實質開盤日約252天計算並回傳
    annual_return: 投資組合的年均成長率，為一個float
    volatility: 投資組合的年波動率(標準差)，為一個float
    """
    annual_return = (1 + combined_all["portfolio"].mean()) ** 252 - 1
    volatility = combined_all["portfolio"].std() * (252 ** 0.5)

    return annual_return, volatility


def compute_annual_Sharpe_ratio(combined_all):
    """
    以Effective Annual Rate (EAR)日化無風險利率與一年實質開盤日約252天計算並回傳
    excess_return: 每日的日超額報酬率(投資組合報酬率 - 無風險利率)，為一個Series
    Sharpe_annual: 年均夏普率，為一個float
    """
    excess_return = combined_all["portfolio"] - (( 1 + combined_all["risk_free"]) ** (1 / 252) - 1)    
    Sharpe_annual = excess_return.mean() * 252 / (excess_return.std() * (252 ** 0.5))
    
    return excess_return, Sharpe_annual


def plot_rolling_annual_Sharpe_ratio(excess_return):
    """
    以120天作為一滾動期間，計算每日的滾動年均夏普率，並且繪製成折線圖
    """
    window = 120
    rolling_mean = excess_return.rolling(window).mean()
    rolling_std = excess_return.rolling(window).std()
    rolling_annual_Sharpe = rolling_mean * 252 / (rolling_std * (252 ** 0.5))

    plt.figure(figsize=(15,5))
    rolling_annual_Sharpe.plot()
    plt.title(f"Rolling Sharpe Ratio ({window} days)")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.grid()
    plt.show()


def compute_Beta(combined_RpRm):
    """
    藉由投資組合日報酬率以及市場日報酬率的DataFrame計算出共變數矩陣，計算並回傳
    Beta: 投資組合相對市場報酬率的變動比例，為一個float
    """
    combined_cov = combined_RpRm.cov()
    Beta = combined_cov.loc["portfolio", "market"] / combined_cov.loc["market", "market"]

    return Beta


def plot_rolling_Beta(combined_RpRm):
    """
    以120天作為一滾動期間，計算每日的滾動Beta，並且繪製成折線圖
    """
    window = 120
    rolling_cov = combined_RpRm["portfolio"].rolling(window).cov(combined_RpRm["market"])
    rolling_var = combined_RpRm["market"].rolling(window).var()
    rolling_Beta = rolling_cov / rolling_var

    plt.figure(figsize=(15,5))
    rolling_Beta.plot()
    plt.title(f"Rolling Beta ({window} days)")
    plt.xlabel("Date")
    plt.ylabel("Beta")
    plt.grid()
    plt.show()


def compute_Alpha(combined_all, Beta):
    Rp = combined_all["portfolio"].mean()
    Rm = combined_all["market"].mean()
    rf = combined_all["risk_free"].mean()
    Alpha = Rp - Beta * (Rm - rf)- rf
    return Alpha


def plot_cumulative_portfolio_and_market_return(combined_RpRm):
    """
    計算自輸入的分析起始日期開始，每日的累積報酬率，並且繪製成折線圖
    """
    cum_portfolio = (1 + combined_RpRm["portfolio"]).cumprod() - 1
    cum_market = (1 + combined_RpRm["market"]).cumprod() - 1

    plt.figure(figsize=(15,5))
    plt.plot(cum_portfolio, label="Portfolio")
    plt.plot(cum_market, label="Market (^TWII)")
    plt.legend()
    plt.title("Cumulative Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth Rate")
    plt.grid()
    plt.show()


def produce_summary(annual_return, volatility, Sharpe_annual, Beta, Alpha):
    """
    將年均報酬率、年均波動率、年均夏普率、Beta與Alpha的計算四捨五入至小數點後第四位，回傳
    summary_df: 一個Series
    """
    summary = pd.Series({
        "Annual Return": annual_return,
        "Volatility": volatility,
        "Sharpe Ratio": Sharpe_annual,
        "Beta": Beta,
        "Alpha": Alpha
    })
    summary_df = summary.to_frame(name="Value")
    summary_df["Value"] = summary_df["Value"].round(4)

    return summary_df


if __name__ == "__main__":
      print("""請準備一個CSV檔案
            第一欄標題為 stock ，紀錄您持有的股票之代碼
            第二欄標題為 shares ，紀錄您各檔股票之持股數""")

      filepath = input("請輸入投資組合的CSV檔案路徑")

      print("""接下來請輸入分析投資組合時，使用的歷史資料起始日
            請注意若投資組合中有任一檔股票之首次發行日期晚於輸入的起始日，
            則所有計算數值將以最晚首次發行之股票的發行日期作為起始日""")
      
      starting_date = input("請以西元年-月-日的方式輸入欲取得的歷史資料起始日(例如：2018-01-01))")

      
      stocks, shares = load_portfolio(filepath)
      stocks_return = compute_stocks_daily_return(stocks, starting_date)
      portfolio_return = compute_portfolio_daily_return(shares, stocks_return)
      rf = compute_risk_free_discount_rate(portfolio_return)
      market_return = load_market_return(starting_date)
      combined_RpRm, combined_all = get_concat_data(portfolio_return, market_return, rf)
      annual_return, volatility = compute_annual_portfolio_return_and_volatility(combined_all)
      excess_return, Sharpe_annual = compute_annual_Sharpe_ratio(combined_all)
      Beta = compute_Beta(combined_RpRm)
      Alpha = compute_Alpha(combined_all, Beta)
      summary_df = produce_summary(annual_return, volatility, Sharpe_annual, Beta, Alpha)

      print("\n=== Portfolio Summary ===")
      print(summary_df)

      print("\n=== cumulative_portfolio_and_market_return ===")
      plot_cumulative_portfolio_and_market_return(combined_RpRm)

      print("\n=== rolling_annual_Sharpe_ratio ===")
      plot_rolling_annual_Sharpe_ratio(excess_return)

      print("\n=== rolling_Beta ===")
      plot_rolling_Beta(combined_RpRm)
