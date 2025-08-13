import pickle                             #保存和加載 Python 物件
from datetime import datetime
import os                                 
from matplotlib import pyplot as plt
import h5py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
import matplotlib.dates as mdates

#未改1

""" 獲取數據，返回值 N(股數) x n_step 的數組 """
def get_data(stock_name, stock_tabel):
    
    industry = pd.read_csv('data/{}.csv'.format(stock_tabel))["code"].astype("str")#讀取股票代碼
    data = pd.read_csv('data/{}.csv'.format(stock_name)).drop(columns="DateTime")#讀取股票數據，並刪除 "DateTime"
    data = data[industry].astype("float")
    data = np.array(data.T)
    return data

#數據標準化(股票數、股價、現金餘額)，取得環境將 low 和 high 作為數據範圍進行標準化
def get_scaler(env):
    """ Takes a env and returns a scaler for its observation space """
    low = [0] * (env.n_stock * 2 + 1)#表示股票數、股價、現金餘額的最低值
    low.append(env.CLI_history.min())
    low.append(env.CPI_history.min())
    low.append(env.Initial_history.min())
    low.append(env.IPI_history.min())
    low.append(env.Manufacturing_history.min())
    low.append(env.Unemployment_history.min())

    high = []#表示股票數、股價、現金餘額的最高值
    max_price = env.stock_price_history.max(axis=1)
    min_price = env.stock_price_history.min(axis=1)
    max_cash = env.init_invest * 3  #3倍的初始投資?
    max_stock_owned = max_cash // min_price
    for i in max_stock_owned:
        high.append(i)
    for i in max_price:
        high.append(i)
    high.append(max_cash)
    high.append(env.CLI_history.max())  # 加入 CLI 的最大值
    high.append(env.CPI_history.max())
    high.append(env.Initial_history.max())
    high.append(env.IPI_history.max())
    high.append(env.Manufacturing_history.max())
    high.append(env.Unemployment_history.max())

    scaler = StandardScaler()
    scaler.fit([low, high])
    return scaler

#如果指定目錄不存在，則創建該目錄
def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

#模擬“買入並持有”策略的投資表現，作為基準比較
def buy_and_hold_benchmark(stock_name, init_invest, test):
    # 讀取股票數據
    df = pd.read_csv('./data/{}.csv'.format(stock_name)).tail(test)

    # 日期
    dates = df['DateTime'].astype("str")
    
    # 平分初始資金
    per_num_holding = init_invest // 3 
    
    # 計算每支股票的可購買數量（初始資金平分並考慮交易成本）
    num_holding = (per_num_holding // df.iloc[0, 1:]).astype(int)  # 根據第一天的價格計算每支股票可購買數量
    
    # 計算剩餘現金餘額
    balance_left = init_invest % 3 + sum([per_num_holding for _ in range(3)]) - sum(num_holding * df.iloc[0, 1:])
    
    # 計算投資組合的價值，考慮到每次的交易成本
    buy_and_hold_portfolio_values = []
    
    for i, date in enumerate(dates):
        # 股票的當日價值
        portfolio_value = (df.iloc[i, 1:] * num_holding).sum() + balance_left
        
        # 計算每次交易時的成本
        transaction_cost = 0
        
        if i > 0:
            # 計算賣出成本
            transaction_cost += sum((df.iloc[i-1, 1:] * num_holding) * 0.0025)
            # 計算買入成本
            transaction_cost += sum((df.iloc[i, 1:] * num_holding) * 0.0025)
        
        # 將交易成本從每個交易日的投資組合價值中扣除
        portfolio_value -= transaction_cost
        
        buy_and_hold_portfolio_values.append(portfolio_value)
    
    # 計算最終的投資回報
    buy_and_hold_return = buy_and_hold_portfolio_values[-1] - init_invest

    return dates, buy_and_hold_portfolio_values, buy_and_hold_return



#繪圖: 策略表現 vs Buy and Hold
def plot_all(stock_name, daily_portfolio_value, env, test):
    """combined plots of plot_portfolio_transaction_history and plot_portfolio_performance_comparison, using cumulative return"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 4), dpi=100)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())

    # 計算 Buy and Hold 的資料
    dates, buy_and_hold_portfolio_values, buy_and_hold_return = buy_and_hold_benchmark(
        stock_name, env.init_invest, test)
    
    # 將日期轉為 datetime 格式
    dates = [datetime.strptime(d, '%Y%m%d').date() for d in dates]

    # 計算累積報酬率
    initial_value = env.init_invest
    daily_cum_return = [(val - initial_value) / initial_value for val in daily_portfolio_value]
    buy_and_hold_cum_return = [(val - initial_value) / initial_value for val in buy_and_hold_portfolio_values]

    agent_return = daily_cum_return[-1]
    
    # 繪圖
    ax.set_title('{} vs. Buy and Hold (Cumulative Return)'.format("DDQN"))
    ax.plot(dates, daily_cum_return, color='green',
            label='{} Total Return: {:.2%}'.format("DDQN", agent_return))
    ax.plot(dates, buy_and_hold_cum_return, color='blue',
            label='{} Buy and Hold Total Return: {:.2%}'.format(stock_name, buy_and_hold_return / initial_value))
    
    ax.set_ylabel('Cumulative Return')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))
    plt.xticks(pd.date_range(min(dates), max(dates), freq='1m'))
    ax.legend()
    plt.gcf().autofmt_xdate()
    plt.subplots_adjust(hspace=0.5)
    plt.show()

#可視化投資組合價值文件
def visualize_portfolio_val():
    """ visualize the portfolio_val file """
    with open('portfolio_val/201912141307-train.p', 'rb') as f:
        data = pickle.load(f)
    with open('portfolio_val/201912042043-train.p', 'rb') as f:
        data0 = pickle.load(f)
    print(sum(data) / 4000)#計算平均投資組合價值 
    print('data>>>', len(data))
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), dpi=100)

    ax[0].plot(data0, linewidth=1)
    ax[0].set_title('DQN Training Performance: 2000 episodes', fontsize=24)
    ax[0].set_xlabel('episode', fontsize=24)
    ax[0].set_ylabel('final portfolio value', fontsize=24)
    ax[0].tick_params(axis='both', labelsize=12)

    ax[1].plot(data, linewidth=1)
    ax[1].set_title('DQN Training Performance: 4000 episodes', fontsize=24)
    ax[1].set_xlabel('episode', fontsize=24)
    ax[1].set_ylabel('final portfolio value', fontsize=24)
    ax[1].tick_params(axis='both', labelsize=12)

    plt.show()


# visualize_portfolio_val()