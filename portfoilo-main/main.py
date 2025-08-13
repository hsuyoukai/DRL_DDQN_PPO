import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from algorithms.ppo.PPO import PPO
import plot
import torch.multiprocessing as mp
import os
from pathlib import Path

def main():
    plot.initialize()
    mp.set_start_method('spawn')
    np.set_printoptions(suppress=True)
    all_logs=[]


    for i in range(10):
        print(f"---------- round {i} ----------")
        plot_path = Path(f'plots/ppo/{i}2_testing.png')

        # 保險絲#1：就算圖檔已存在，也至少跑一次 test() 以確保產生日誌
        
        ppo = PPO(state_type='indicators', djia_year=2019, repeat=i)
        if not plot_path.is_file():
            ppo.train()
        ppo.test()  # 一定會跑

        print(f"📊 ppo.env.history_log 筆數：{len(ppo.env.history_log)}")
        all_logs.extend(ppo.env.history_log)

   
    if not all_logs:
        raise RuntimeError("history_log 為空：請確認 train()/test() 是否有執行成功。")

    # 用 json_normalize 展開 dict，比 pd.DataFrame(list) 穩定
    df = pd.json_normalize(all_logs)
    print("df columns:", list(df.columns))

    needed = {"returns", "weights"}
    missing = needed - set(df.columns)
    if missing:
        sample_keys = list(all_logs[0].keys()) if isinstance(all_logs[0], dict) else "非 dict"
        raise KeyError(f"缺少欄位: {missing}；請確認 env.history_log 的鍵名。樣本鍵：{sample_keys}")

    # 檢查 returns / weights 是否為長度=3 的序列，並安全展開
    def _explode_vec_col(series, prefix):
        ok = series.map(lambda x: hasattr(x, "__len__") and len(x) == 3)
        if not ok.all():
            bad_idx = ok[~ok].index[:5].tolist()
            raise ValueError(f"{prefix} 欄位出現非長度3的值；問題索引：{bad_idx}")
        return pd.DataFrame(series.tolist(), columns=[f"DBC_{prefix}", f"SHY_{prefix}", f"SPY_{prefix}"])

    returns_df = _explode_vec_col(df["returns"], "ret")
    weights_df = _explode_vec_col(df["weights"], "weight")

    # 合併輸出
    df = df.drop(columns=["returns", "weights"]).reset_index(drop=True)
    out = pd.concat([df, returns_df, weights_df], axis=1)
    
    os.makedirs("output", exist_ok=True)
    out.to_csv("output/ppo_rounds_history.csv", index=False)
    print("所有紀錄已儲存為 output/ppo_rounds_history.csv")

if __name__ == '__main__':
    main()