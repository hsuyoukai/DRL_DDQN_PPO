
from env.environment import PortfolioEnv
from algorithms.ppo.agent import Agent
from plot import add_curve, add_hline, save_plot
import os
import pandas as pd
import matplotlib.pyplot as plt
from pyfolio import timeseries

class PPO:
    def __init__(self, load=False, alpha=0.0005, n_epochs=30,
                 batch_size=64, layer1_size=1024, layer2_size=1024,
                 policy_clip=0.3, t_max=256, gamma=0.95, gae_lambda=0.99,
                 state_type='only prices', djia_year=2019, repeat=0, entropy=0):
        self.figure_dir = 'plots/ppo'
        self.checkpoint_dir = 'checkpoints/ppo'
        os.makedirs(self.figure_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.t_max = t_max
        self.repeat = repeat

        self.env = PortfolioEnv(action_scale=1000, state_type=state_type,
                                djia_year=djia_year, repeat=repeat)
        if djia_year == 2019:
            self.intervals = self.env.get_intervals(train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15)

        self.agent = Agent(action_dims=self.env.action_shape(), batch_size=batch_size, alpha=alpha,
                           n_epochs=n_epochs, input_dims=self.env.state_shape(),
                           fc1_dims=layer1_size, fc2_dims=layer2_size, entropy=entropy)
        if load:
            self.agent.load_models(self.checkpoint_dir)

    def train(self, verbose=False, max_iterations=200, patience=10, kl_threshold=0.10):
        import math
        training_history, validation_history = [], []
        best_val, best_iter = -float('inf'), 0
        iteration = 1

        while iteration <= max_iterations:
            n_steps = 0
            observation = self.env.reset(*self.intervals['training'])
            done = False
            wealth = self.env.get_wealth()

            while not done:
                action, prob, val = self.agent.choose_action(observation)
                observation_, reward, done, info, wealth = self.env.step(action)

                # 保險絲：wealth 非有限值
                if not math.isfinite(wealth):
                    print(f"wealth 非有限值 (iter {iteration})，停止訓練")
                    return self._finalize_and_plot(training_history, validation_history)

                n_steps += 1
                self.env.wealth_history.append(self.env.get_wealth())
                self.agent.remember(observation, action, prob, val, reward, done)

                if n_steps % self.t_max == 0:
                    out = self.agent.learn()  # 建議回傳 dict（見下方註記）
                    approx_kl = out.get("approx_kl") if isinstance(out, dict) else None
                    total_loss = out.get("loss") if isinstance(out, dict) else None

                    if total_loss is not None:
                        try:
                            if not math.isfinite(float(total_loss)):
                                print(f"loss 非有限值 (iter {iteration})，停止訓練")
                                return self._finalize_and_plot(training_history, validation_history)
                        except Exception:
                            pass

                    try:
                        import torch
                        if any(torch.isnan(p).any().item() for p in self.agent.actor.parameters()):
                            print(f"actor 權重 NaN (iter {iteration})，停止訓練")
                            return self._finalize_and_plot(training_history, validation_history)
                        if hasattr(self.agent, "critic"):
                            if any(torch.isnan(p).any().item() for p in self.agent.critic.parameters()):
                                print(f"critic 權重 NaN (iter {iteration})，停止訓練")
                                return self._finalize_and_plot(training_history, validation_history)
                    except Exception:
                        pass

                    if approx_kl is not None and approx_kl > kl_threshold and verbose:
                        print(f"迭代 {iteration} KL={approx_kl:.4f} > {kl_threshold}，提前結束當前更新")

                observation = observation_

            self.agent.memory.clear_memory()
            print(f"PPO training - Iteration: {iteration},\tCumulative Return: {int(wealth) - 1000000}")
            training_history.append(wealth - 1000000)

            validation_wealth = self.validate(verbose)
            print(f"PPO validating - Iteration: {iteration},\tCumulative Return: {int(validation_wealth) - 1000000}")
            validation_history.append(validation_wealth - 1000000)

            if validation_wealth > best_val:
                best_val, best_iter = validation_wealth, iteration
                if hasattr(self.agent, "save_models"):
                    self.agent.save_models(self.checkpoint_dir)

            if iteration - best_iter >= patience:
                print(f"早停：已 {patience} 輪未創新高（最佳在 iter {best_iter}）")
                break

            iteration += 1

        print(f"總共訓練了 {iteration} 次迭代")
        return self._finalize_and_plot(training_history, validation_history)

    def _finalize_and_plot(self, training_history, validation_history):
        # 載回最佳
        if hasattr(self.agent, "load_models"):
            try: self.agent.load_models(self.checkpoint_dir)
            except Exception: pass

        add_curve(training_history, 'PPO')
        save_plot(filename=self.figure_dir + f'/{self.repeat}0_training.png',
                  title=f"Training - {self.intervals['training'][0].date()} to {self.intervals['training'][1].date()}",
                  x_label='Iteration', y_label='Cumulative Return (Dollars)')

        add_curve(validation_history, 'PPO')
        save_plot(filename=self.figure_dir + f'/{self.repeat}1_validation.png',
                  title=f"Validation - {self.intervals['validation'][0].date()} to {self.intervals['validation'][1].date()}",
                  x_label='Iteration', y_label='Cumulative Return (Dollars)')

    def validate(self, verbose=False):
        observation = self.env.reset(*self.intervals['validation'])
        done = False
        while not done:
            action, prob, val = self.agent.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action)
            observation = observation_
        return wealth

    def test(self):
        return_history = [0]
        n_steps = 0
        observation = self.env.reset(*self.intervals['testing'])
        wealth_history = [self.env.get_wealth()]
        done = False
        while not done:
            action, prob, val = self.agent.choose_action(observation)
            observation_, reward, done, info, wealth = self.env.step(action, reward_mode="sharpe")
            n_steps += 1
            self.agent.remember(observation, action, prob, val, reward, done)
            if n_steps % self.t_max == 0:
                self.agent.learn()
            observation = observation_
            return_history.append(wealth - 1000000)
            wealth_history.append(wealth)

        self.agent.memory.clear_memory()
        add_curve(return_history, 'PPO')
        save_plot(self.figure_dir + f'/{self.repeat}2_testing.png',
                  title=f"Testing - {self.intervals['testing'][0].date()} to {self.intervals['testing'][1].date()}",
                  x_label='Days', y_label='Cumulative Return (Dollars)')

        returns = pd.Series(wealth_history).pct_change().dropna()
        stats = timeseries.perf_stats(returns)
        stats.to_csv(self.figure_dir + f'/{self.repeat}3_perf.csv')