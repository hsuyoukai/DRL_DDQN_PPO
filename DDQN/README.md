1. 下載所需套件

pip install -r requirements.txt

檢查已安裝套件

pip list

2. 訓練模型

python run.py --mode train

* 更改參數

python run.py --mode train -e 10

3. 測試模型 -改模型參數
python run.py --mode test --weights ./weights/202508121555-dqn.weights.h5 -e 500

4. 回測
backtesting.ipynb 