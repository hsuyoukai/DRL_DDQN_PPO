# Portfolio PPO Training & Backtesting

本專案使用 PPO (Proximal Policy Optimization) 演算法進行投資組合配置訓練，並透過回測分析策略績效。

---
## 環境需求
- Python 3.12.2
- 建議使用 Anaconda / venv 建立虛擬環境
- 主要套件：
  - pandas
  - numpy
  - matplotlib
  - quantstats
  - (以及程式碼中 `requirements.txt` 列出的其他套件)

安裝範例 bash：
pip install -r requirements.txt

step1:
python main.py

main.py 預設測試集日期範圍：2022-06-30 ~ 2025-03-31

step2:
執行 backtesting.ipynb — 回測分析

關鍵設定與假設
	資料切分
	訓練/驗證/測試可由比例（70/15/15）或固定日期決定。
    目前的測試集區間是：2022-06-30 ~ 2025-03-31。

