import pickle
import time
import os
import numpy as np
import pandas as pd
import argparse    #解析命令行參數
import re          #正則表達式，這裡用於從模型檔案名提取時間戳
import itertools

from envs import TradingEnv
from agent import DDQNAgent
from utils import get_data, get_scaler, maybe_make_dir, plot_all

#讀取特定資料集
stock_name = "PFdata"
stock_table = "PFtable"
CLI = pd.read_csv('data/CLI.csv'.format(stock_name)).drop(columns="DateTime")
CPI = pd.read_csv('data/CPI.csv'.format(stock_name)).drop(columns="DateTime")
Initial = pd.read_csv('data/Initial.csv'.format(stock_name)).drop(columns="DateTime")
IPI = pd.read_csv('data/IPI.csv'.format(stock_name)).drop(columns="DateTime")
Manufacturing = pd.read_csv('data/Manufacturing.csv'.format(stock_name)).drop(columns="DateTime")
Unemployment = pd.read_csv('data/Unemployment.csv'.format(stock_name)).drop(columns="DateTime")


#定義需要的命令列參數
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episode', type=int, default=2,
                        help='number of episode to run')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch size for experience replay')
    parser.add_argument('-i', '--initial_invest', type=int, default=20000,
                        help='initial investment amount')
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"')
    #已訓練好的模型權重（用於測試模式）
    parser.add_argument('-w', '--weights', type=str, help='a trained model weights')
    args = parser.parse_args()

    print(args)

    maybe_make_dir('weights')
    maybe_make_dir('portfolio_val')

    """讀取股票資料"""
    timestamp = time.strftime('%Y%m%d%H%M')

    # 把 data 轉成 np 所以景氣也要跟著轉
    data = get_data(stock_name, stock_table)
    CLI = np.array(CLI.T)
    CPI = np.array(CPI.T)
    Initial = np.array(Initial.T)
    IPI = np.array(IPI.T)
    Manufacturing = np.array(Manufacturing.T)
    Unemployment = np.array(Unemployment.T)
    train = 228-34

    # train = 228-34
    test = 34
    # print("train:{}, test:{}".format(data[:, train-1], data[:, test]))
    train_data = data[:, :-test]
    test_data = data[:, -test:]
    CLI_train = CLI[:, :-test]
    CLI_test = CLI[:, -test:]
    CPI_train = CPI[:, :-test]
    CPI_test = CPI[:, -test:]
    Initial_train = Initial[:, :-test]
    Initial_test = Initial[:, -test:]
    IPI_train = IPI[:, :-test]
    IPI_test = IPI[:, -test:]
    Manufacturing_train = Manufacturing[:, :-test]
    Manufacturing_test = Manufacturing[:, -test:]
    Unemployment_train = Unemployment[:, :-test]
    Unemployment_test = Unemployment[:, -test:]

    # Grid Search 模式
    if args.mode == "grid-search":
        param_grid = {
            "gamma": [0.99, 0.95, 0.90],  # 折扣因子
            "learning_rate": [0.001, 0.01, 0.1],  # 學習率
            "replay_memory_size": [1000, 2000, 5000],  # 回放記憶體大小
            "epsilon_decay": [0.995, 0.99, 0.98],  # 探索率衰減
            "batch_size": [32, 64, 128]  # 增加 batch_size
        }

        param_combinations = list(itertools.product(*param_grid.values()))
        total_tests = len(param_combinations)  # 計算總測試數量
        best_reward = -np.inf
        best_params = None
        results = []

        for idx, params in enumerate(param_combinations, start=1):
            gamma,learning_rate, replay_memory_size, epsilon_decay, batch_size = params
            print(f"\nGrid Search {idx}/{total_tests} 測試中...")
            print(f"gamma={gamma}, learning_rate={learning_rate}, replay_memory_size={replay_memory_size}, epsilon_decay={epsilon_decay}, batch_size={batch_size}")

            env = TradingEnv(train_data, CLI_train, CPI_train, Initial_train, IPI_train, Manufacturing_train, Unemployment_train, args.initial_invest)
            state_size = env.observation_space.shape
            action_size = env.action_space.n

            agent = DDQNAgent(
                state_size, action_size,
                gamma=gamma,
                replay_memory_size = replay_memory_size,
                epsilon_decay=epsilon_decay)

            scaler = get_scaler(env)
            total_rewards = []

            for e in range(args.episode):  # 未減少回合數，加快 Grid Search
                state = env.reset()
                state = scaler.transform([state])
                episode_reward = 0

                for _ in range(env.n_step):
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    next_state = scaler.transform([next_state])
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    # 檢查 reward 是否為 NaN
                    if np.isnan(reward):
                        reward = 0  
                    episode_reward += reward

                    if done:
                        break

                agent.replay()  # batch_size 內部處理
                total_rewards.append(episode_reward)

            avg_reward = np.mean(total_rewards)
            results.append((gamma, learning_rate, replay_memory_size, epsilon_decay, batch_size, avg_reward))  # 修正變數數量

            print(f"測試完成！平均獎勵: {avg_reward}")

            if avg_reward > best_reward:
                best_reward = avg_reward
                best_params = params

        df_results = pd.DataFrame(results, columns=["gamma", "learning_rate", "epsilon_min", "epsilon_decay", "batch_size", "avg_reward"])
        

        # 建立 'results' 資料夾
        save_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(save_dir, exist_ok=True)
        # 儲存 CSV 文件於新資料夾
        file_path = os.path.join(save_dir, "grid_search_results.csv")
        df_results.to_csv(file_path, index=False)
        
        print("\nGrid Search 完成！")
        print(f"最佳超參數組合: {best_params}")
        print(f"最高平均獎勵: {best_reward}")
        exit()

    # 進 env 定義 business_cycle
    env = TradingEnv(train_data, CLI_train, CPI_train, Initial_train, IPI_train, Manufacturing_train, Unemployment_train, args.initial_invest)
    #初始化環境與代理
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DDQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    #訓練或測試模式
    portfolio_value = []

    if args.mode == 'test':
        # 使用測試數據重新製作環境
        # 加入景氣函數?
        env = TradingEnv(test_data, CLI_test, CPI_test, Initial_test, IPI_test, Manufacturing_test, Unemployment_test, args.initial_invest)        # 載入訓練好的權重
        agent.load(args.weights)
        # 測試時，時間戳與訓練權重的時間相同
        timestamp = re.findall(r'\d{12}', args.weights)[0]
        # daily_portfolio_value = [env.init_invest]
        daily_portfolio_value = []

    #開始訓練或測試
    """
每個步驟中，代理選擇一個動作，環境返回下一個狀態、回報、完成標誌和額外資訊
在訓練模式下，代理會記住經驗並進行經驗回放
在測試模式下，記錄每日的投資組合價值
當回合結束時，若是測試模式且每100回合執行一次圖表繪製，並打印每個回合的最終結果
    """
    for e in range(args.episode):
        state = env.reset()
        state = scaler.transform([state])
        for time in range(env.n_step):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = scaler.transform([next_state])
            if args.mode == 'train':
                agent.remember(state, action, reward, next_state, done)
            if args.mode == "test":
                daily_portfolio_value.append(info['cur_val'])
            state = next_state
            if done:

                if args.mode == "test" and e % 100 == 0:
                    plot_all(stock_name, daily_portfolio_value, env, test-1)#-1，step 1後才開始記錄cur_val

                daily_portfolio_value = []
                print("episode: {}/{}, episode end value: {}".format(
                    e + 1, args.episode, info['cur_val']))
                portfolio_value.append(info['cur_val']) # append episode end portfolio value

                break
            if args.mode == 'train' and len(agent.memory) > args.batch_size:
                agent.replay(args.batch_size)
        #if args.mode == 'train' and (e + 1) % 10 == 0:  # checkpoint weights
        if args.mode == 'train' and (e + 1) % 2 == 0:  # checkpoint weights
            #agent.save('weights/{}-dqn.h5'.format(timestamp))
            agent.save('weights/{}-dqn.weights.h5'.format(timestamp))

    #最後輸出結果與保存投資組合價值歷史記錄
    print("mean portfolio_val:", np.mean(portfolio_value))
    print("median portfolio_val:", np.median(portfolio_value))

    # save portfolio value history to disk
    with open('portfolio_val/{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
        pickle.dump(portfolio_value, fp)
