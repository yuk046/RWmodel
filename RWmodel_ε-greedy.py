import numpy as np
import random

# このプログラムは各アームの報酬確率を予測するモデル

# バンディット環境を定義
class BanditEnvironment:
    def __init__(self, reward_probs):
        """
        :param reward_probs: 各アームの報酬確率（例: [0.8, 0.5]）
        """
        self.reward_probs = reward_probs

    def pull_arm(self, arm):
        """
        指定されたアームを引き、報酬を返す
        """
        return 1 if random.random() < self.reward_probs[arm] else 0

# RWモデルを実装
class RWModel:
    def __init__(self, num_arms, learning_rate=0.1):
        """
        :param num_arms: アーム（選べるスロットマシン）の数
        :param learning_rate: 学習率 (a)
        """
        self.num_arms = num_arms
        self.learning_rate = learning_rate
        self.values = np.zeros(num_arms)  # 各アームの価値予測する配列用意

    def update(self, arm, reward):
        """
        スロットの価値予測を更新 prediction_error = 報酬予測誤差
        """
        prediction_error = reward - self.values[arm]
        self.values[arm] += self.learning_rate * prediction_error

    def select_action(self, epsilon=0.1):
        """
        ε-greedyポリシーで行動を選択
        """
        if random.random() < epsilon:
            return random.randint(0, self.num_arms - 1)  # ランダム選択
        else:
            # argmax: 最大値のインデックスを返す
            return np.argmax(self.values)  # 最大価値選択

# 実験の設定　num_trials: 試行回数, learning_rate: 学習率, epsilon: ε-greedyのε
def run_experiment(reward_probs, num_trials=100000, learning_rate=0.1, epsilon=0.1):
    env = BanditEnvironment(reward_probs)
    agent = RWModel(len(reward_probs), learning_rate)

    rewards = []
    for trial in range(num_trials):
        # action: 選択されたアーム, reward: 報酬
        action = agent.select_action(epsilon)
        reward = env.pull_arm(action)
        agent.update(action, reward)

        rewards.append(reward)
    return rewards, agent.values

# 実行
reward_probs = [0.7, 0.3]  # 各アームの報酬確率
rewards, final_values = run_experiment(reward_probs)

# 結果の表示
print("累積報酬:", sum(rewards))
print("最終的な価値予測:", final_values)
