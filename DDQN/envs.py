import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
import itertools
import csv
import os
import time

class TradingEnv(gym.Env):
    """
 state：[持有股票數量、當前股票價格、庫存現金]
 - 長度為 n_stock * 2 + 1 的陣列
 - 將價格離散化（為整數）以減少狀態空間
 - 使用每檔股票的收盤價
 - 根據所採取的行動，在每個步驟中評估手頭上的現金

 action：賣出（0）、持有（1）、買入（2）
 - 出售時，出售所有股份
 - 購買時，根據手頭現金允許的數量購買(每次只能進行一筆交易200股)
  """

    #初始化
    def __init__(self, train_data, CLI_train, CPI_train, Initial_train, IPI_train, Manufacturing_train, Unemployment_train, init_invest=20000):
        # data
        self.n_industry = 3
        #交易執行時記錄具體的交易時間點（第幾步驟）
        #self.buy_date[0] 可能會用來記錄第 0 號產業的買入時間點
        # self.buy_date = [[] for _ in range(self.n_industry)]
        # self.sell_date = [[] for _ in range(self.n_industry)]

        self.c_minus = 0.0025       #交易成本 買入和賣出的交易成本均為 0.25%
        self.c_plus = 0.0025

        self.stock_price_history = train_data #  四捨五入為整數以減少狀態空間
        self.n_stock, self.n_step = self.stock_price_history.shape#股票數量、交易天數
        self.CLI_history = np.array(CLI_train).flatten()
        self.CPI_history = np.array(CPI_train).flatten()
        self.Initial_history = np.array(Initial_train).flatten()
        self.IPI_history= np.array(IPI_train).flatten()
        self.Manufacturing_history = np.array(Manufacturing_train).flatten()
        self.Unemployment_history = np.array(Unemployment_train).flatten()

        # instance attributes
         # 檢查是否有初始化 self.episode
        self.episode = 0
        self.init_invest = init_invest
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        # sharpe raio 紀錄 @@@@@@@@@
        self.freerate = 0
        self.sharpe_ratio_data = pd.read_csv('data/sharpe_ratio_data.csv', index_col=0)
        # 切分 Return 跟 STD，假設前半是 Return，後半是 STD
        self.returns = self.sharpe_ratio_data.iloc[:,:3].values         
        self.stds = self.sharpe_ratio_data.iloc[:,3:6].values     
        self.cov = self.sharpe_ratio_data.iloc[:, 6:].values  

        # action space(以行業為基礎)
        self.action_space = spaces.Discrete(3 ** self.n_industry)
        # observation space:給出估計值以便進行採樣並建立縮放器
        stock_max_price = self.stock_price_history.max(axis=1)
        stock_range = [[0, init_invest * 2 // mx] for mx in stock_max_price]#股票數量的範圍(最大購買數量)
        price_range = [[0, mx] for mx in stock_max_price]
        cash_in_hand_range = [[0, init_invest * 2]]#現金持有範圍 (最高為初始金額的 2 倍)
        # 新增觀察範圍
        CLI_min, CLI_max = self.CLI_history.min(), self.CLI_history.max()
        CLI_range = [[CLI_min, CLI_max]]
        CPI_min, CPI_max = self.CPI_history.min(), self.CPI_history.max()
        CPI_range = [[CPI_min, CPI_max]]
        Initial_min, Initial_max = self.Initial_history.min(), self.Initial_history.max()
        Initial_range = [[Initial_min, Initial_max]]
        IPI_min, IPI_max = self.IPI_history.min(), self.IPI_history.max()
        IPI_range = [[IPI_min, IPI_max]]
        Manufacturing_min, Manufacturing_max = self.Manufacturing_history.min(), self.Manufacturing_history.max()
        Manufacturing_range = [[Manufacturing_min, Manufacturing_max]]
        Unemployment_min, Unemployment_max = self.Unemployment_history.min(), self.Unemployment_history.max()
        Unemployment_range = [[Unemployment_min, Unemployment_max]]
        self.observation_space = spaces.MultiDiscrete(stock_range + price_range + cash_in_hand_range + CLI_range + CPI_range + Initial_range + IPI_range + Manufacturing_range + Unemployment_range)
        # seed and start
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #重置
    def _reset(self):
        self.episode += 1
        self.cur_step = 0
        self.stock_owned = [0] * self.n_stock
        self.stock_price = self.stock_price_history[:, self.cur_step]
        self.cash_in_hand = self.init_invest
        # 記錄初始權重
        self._save_portfolio_weights()
        return self._get_obs()

    #根據動作 (action) 更新環境狀態並計算回報 (reward)
    def _step(self, action):
        assert self.action_space.contains(action)#確保行動在 action_space
        prev_val = self._get_val()               #獲取上一步的投資組合價值
        self.cur_step += 1
        self.stock_price = self.stock_price_history[:, self.cur_step]  #更新股票價格
        self._trade(action)       #根據action進行交易
        cur_val = self._get_val() #獲取當前投資組合價值

        # 報酬率 @@@@@@@@@
        sharpe_ratio = self._calculate_sharpe_ratio()
        print('sharpe_ratio',sharpe_ratio)
        reward = sharpe_ratio

        done = self.cur_step == self.n_step - 1
        info = {'cur_val': cur_val} #提供當前投資組合的價值作為額外的資訊
        # 每次 step 之後記錄一次權重
        self._save_portfolio_weights()

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = []
        obs.extend(self.stock_owned)
        obs.extend(list(self.stock_price))
        obs.append(self.cash_in_hand)
        # 確保 CLI 是 float，避免變成 numpy 陣列
        CLI_value = float(self.CLI_history[self.cur_step])
        obs.append(CLI_value)
        CPI_value = float(self.CPI_history[self.cur_step])
        obs.append(CPI_value)
        Initial_value = float(self.Initial_history[self.cur_step])
        obs.append(Initial_value)
        IPI_value = float(self.IPI_history[self.cur_step])
        obs.append(IPI_value)
        Manufacturing_value = float(self.Manufacturing_history[self.cur_step])
        obs.append(Manufacturing_value)
        Unemployment_value = float(self.Unemployment_history[self.cur_step])
        obs.append(Unemployment_value)

        return np.array(obs, dtype=np.float32)

    def _get_val(self):
        return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand

    def _calculate_sharpe_ratio(self):
        """ 計算 Sharpe Ratio 作為獎勵 """
        # 取出當期資產報酬
        returns = self.returns[self.cur_step]
        # 計算變異數矩陣（假設 STD 間獨立，先用對角矩陣）
        stds = self.stds[self.cur_step]
        self.stds2 = np.diag(stds ** 2)
        cov = self.cov[self.cur_step]

        # 共變異數矩陣重塑（假設是上三角或全展開）
        cov_matrix = np.zeros((3, 3))
        # 先設置對角（std^2）
        np.fill_diagonal(cov_matrix, stds**2)
        # 填入對角線以外的共變異數（你需要確認 cov 的排列方式）
        # 填入對角線以外的共變異數
        cov_matrix[0, 1] = cov_matrix[1, 0] = cov[0]  # cov12
        cov_matrix[0, 2] = cov_matrix[2, 0] = cov[1]  # cov13
        cov_matrix[1, 2] = cov_matrix[2, 1] = cov[2] 
    
        portfolio_return = np.dot(self.weights, returns)
        portfolio_variance = np.dot(self.weights, np.dot(cov_matrix, self.weights))
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe_ratio =  (portfolio_return - self.freerate) / portfolio_std
    
        return sharpe_ratio
    
    #權重微調：每一支股票依目標權重進行調整；現金最小化：最後剩下的現金會用來多買一支（權重最大）股票。
    def _trade(self, action):
        # 設定動作空間的權重組合，權重分段：0.05 ~ 0.95，每格0.05，共19段
        weight_levels = np.round(np.arange(0.05, 1.0, 0.05), 2)
        action_combo = list(itertools.product(weight_levels, repeat=self.n_industry))
        action_vec = action_combo[action]

        # 正規化權重組合
        total_weight = sum(action_vec)
        target_weights = [w / total_weight for w in action_vec]

        total_value = self._get_val()
        current_weights = self.weights  # 前一步已經儲存的實際權重
        stock_prices = self.stock_price

        for i in range(self.n_industry):
            delta_weight = target_weights[i] - current_weights[i]
            if abs(delta_weight) < 1e-4:
                continue  # 差異很小就略過

            target_value = target_weights[i] * total_value
            current_value = current_weights[i] * total_value
            trade_value = abs(target_value - current_value)
            stock_price = stock_prices[i]

            shares = trade_value / stock_price

            if delta_weight > 0:  # 欲增加部位 → 買入
                cost = trade_value * self.c_plus
                total_cost = trade_value + cost

                if self.cash_in_hand >= total_cost:
                    self.stock_owned[i] += shares
                    self.cash_in_hand -= total_cost
                else:
                    max_shares = self.cash_in_hand / (stock_price * (1 + self.c_plus))
                    self.stock_owned[i] += max_shares
                    self.cash_in_hand -= max_shares * stock_price * (1 + self.c_plus)

            elif delta_weight < 0:  # 欲減少部位 → 賣出
                revenue = trade_value
                cost = revenue * self.c_minus
                self.stock_owned[i] -= shares
                self.cash_in_hand += revenue #先不考慮 -cost

        # 最後把剩餘現金盡量投入目標權重最大的股票
        max_weight_idx = target_weights.index(max(target_weights))
        stock_price = stock_prices[max_weight_idx]
        cost_per_share = stock_price * (1 + self.c_plus)

        # 避免現金餘額不足(手續費影響)
        if self.cash_in_hand >= cost_per_share:
            max_shares = self.cash_in_hand / cost_per_share
            self.stock_owned[max_weight_idx] += max_shares
            self.cash_in_hand -= max_shares * cost_per_share

    def _save_portfolio_weights(self):
        total_value = self._get_val()

        # 計算每支股票的權重
        self.weights = [(self.stock_owned[i] * self.stock_price[i]) / total_value if total_value > 0 else 0 for i in range(self.n_industry)]
        self.cash_weight = self.cash_in_hand / total_value if total_value > 0 else 0

        # 計算累積報酬率
        initial_value = self.init_invest  # 確保你有在 __init__ 時設定 self.init_invest = env.init_invest
        cum_return = (total_value - initial_value) / initial_value if initial_value > 0 else 0

        folder_path = 'portfolio_weights'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = os.path.join(folder_path, 'weights.csv')
        write_header = not os.path.exists(file_path)

        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(
                    ['episode', 'step', 'total_value', 'cumulative_return']
                    + [f'stock_{i}_weight' for i in range(self.n_industry)]
                    + ['cash_weight']
                )

            writer.writerow(
                [self.episode, self.cur_step, total_value, cum_return]
                + self.weights
                + [self.cash_weight]
            )