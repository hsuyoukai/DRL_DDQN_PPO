# import tensorflow as tf
from keras.models import Sequential #引入序貫模型
from keras.layers import Dense      #引入全連接層
from keras.optimizers import Adam   #引入優化器

'''輸出：對應各個動作的 Q 值（表示每個動作的預期回報）
選擇動作：選擇 Q 值最高的動作（或使用 ε-貪婪策略探索新動作）'''

def mlp(n_obs, n_action, n_hidden_layer=1, n_neuron_per_layer=32,
        activation='relu', loss='mse',learning_rate=0.007):
    #n_obs：輸入層的神經元數量，對應於每個觀察的特徵數量
    #n_action：輸出層的神經元數量，對應於每個行動的數量
    """ DNN (多層感知器, MLP) """
    print(n_action)
    model = Sequential()

    #第一層：輸入層
    model.add(Dense(n_neuron_per_layer, input_dim=n_obs, activation=activation))
    #隱藏層
    #for _ in range(n_hidden_layer):
    
    for _ in range(int(n_hidden_layer)):
        model.add(Dense(n_neuron_per_layer, activation=activation))
    #輸出層
    model.add(Dense(n_action, activation='linear'))
    #編譯模型
    model.compile(loss=loss, optimizer=Adam(learning_rate=learning_rate))
    #模型結構
    print(model.summary())
    return model
    

    # model.add(Dense(units=64, input_dim=n_obs, activation="relu"))
    # model.add(Dense(units=64, input_dim=n_obs, activation="relu"))
    # model.add(Dense(units=32, activation="relu"))
    # model.add(Dense(units=8, activation="relu"))
    # model.add(Dense(n_action, activation="linear"))
    # model.compile(loss="mse", optimizer=Adam(lr=0.001))
    # print(model.summary())
    # return model
