from collections import deque
import random
import numpy as np
from model import mlp

class DDQNAgent(object):
    """ 一個簡單的 Double Deep Q 代理 """
    
    def __init__(self, state_size, action_size, 
                 gamma=0.9, 
                 replay_memory_size =5000,
                 epsilon_decay =0.98,
                 learning_rate =0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0  
        self.epsilon_min = 0.01

        self.gamma = gamma  # 折扣率
        self.replay_memory_size = replay_memory_size
        self.memory = deque(maxlen=replay_memory_size)
        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay
        self.model = mlp(state_size, action_size,learning_rate)  # 主網路
        self.target_model = mlp(state_size, action_size,learning_rate)  # 目標網路
        self.update_target_model()  # 初始化目標網路

    def update_target_model(self):
        """ 將主網路的權重複製到目標網路 """
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size=32, step=0):
        minibatch = random.sample(self.memory, batch_size)

        states = np.array([tup[0][0] for tup in minibatch])
        actions = np.array([tup[1] for tup in minibatch])
        rewards = np.array([tup[2] for tup in minibatch])
        next_states = np.array([tup[3][0] for tup in minibatch])
        done = np.array([tup[4] for tup in minibatch])

        # 主網絡選擇動作
        next_action = np.argmax(self.model.predict(next_states), axis=1)
        # 目標網絡計算 Q 值
        target_Q_values = self.target_model.predict(next_states)
        target = rewards + self.gamma * target_Q_values[range(batch_size), next_action]
        target[done] = rewards[done]  # 如果是終止狀態，目標值只等於即時獎勵

        target_f = self.model.predict(states)
        target_f[range(batch_size), actions] = target

        self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 週期性更新目標網絡
        if step % 10 == 0:
            self.update_target_model()

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
