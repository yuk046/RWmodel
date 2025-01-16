import numpy as np
import random
import matplotlib.pyplot as plt

# このプログラムは各アームの報酬確率を予測するモデル

# バンディット環境を定義
class BanditEnvironment:
    def __init__(self, reward_probs):
        """
        :param reward_probs: 各アームの報酬確率（例: [0.7, 0.3]）
        薬物依存の環境ではどちらも１
        """
        self.reward_probs = reward_probs

    def pull_arm(self, arm):
        """
        0は自然報酬
        1は薬物報酬

        """
        if arm == 0:
            return 1.0
        else:
            return 0.8

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
        self.Dopamine = [0,0.2] #ここでDopamineの値を設定することができる。（第一項：自然報酬　第二項：薬物報酬）

    def update(self, arm, reward):
        """
        スロットの価値予測を更新 prediction_error = 報酬予測誤差
        薬物が選択された場合にはDopamineに0.2の値が入る。
        """
        prediction_error = max(reward - self.values[arm]+self.Dopamine[arm],self.Dopamine[arm])
        self.values[arm] += self.learning_rate * prediction_error

        return self.values

    def select_action(self, temperature):
        """
        ソフトマックスポリシーで行動を選択
        """
        probabilities_A = 1 / (1 + np.exp(-temperature * (self.values[0] - self.values[1])))
        probabilities_B = 1 - probabilities_A
        probabilities = np.array([probabilities_A, probabilities_B])

        # 確率分布に基づいてアームを選択
        return np.random.choice(len(self.values), p = probabilities), probabilities

# 実験の実行関数　num_trials: 試行回数, learning_rate: 学習率, temperature: ソフトマックス関数における温度パラメータ
def run_experiment(reward_probs, num_trials=80, learning_rate=0.3, temperature=3.0):
    env = BanditEnvironment(reward_probs)
    agent = RWModel(len(reward_probs), learning_rate)

    probabilities = []
    value = []
    rewards = []
    actions = []
    timevalues = []

    for trial in range(num_trials):
        # action: 選択されたアーム, reward: 報酬
        action,probabilities = agent.select_action(temperature)
        reward = env.pull_arm(action)
        value = agent.update(action, reward)

        rewards.append(reward)
        actions.append(probabilities)
        timevalues.append(value.copy())

    return rewards,actions,agent.values,timevalues

# 実行
# 依存症を考慮したモデルではどちらも１
reward_probs = [1.0, 1.0]  # 各アームの報酬確率
rewards,actions,final_values,Tvalues = run_experiment(reward_probs)


# 結果の表示
print("累積報酬:", sum(rewards))
print("最終的な価値予測:", final_values)

# グラフの表示
Tvalues = np.array(Tvalues)  # numpy配列に変換
actions = np.array(actions)  # numpy配列に変換
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))

# アームAの予測値を1つ目のグラフにプロット
ax1.plot(Tvalues[:, 0], label='Arm A', color='blue')
ax1.set_title('Arm A Value Prediction')  # タイトルを設定
ax1.set_ylim(-0.2, 1.2)  # y軸の範囲を設定
ax1.set_xlabel('試行')  # x軸のラベル
ax1.set_ylabel('Q(A)')  # y軸のラベル

# アームBの予測値を2つ目のグラフにプロット
ax2.plot(Tvalues[:, 1], label='Arm B', color='red')
ax2.set_title('Arm B Value Prediction')  # タイトルを設定
ax2.set_ylim(-0.2, 1.2)  # y軸の範囲を設定
ax2.set_xlabel('試行')  # x軸のラベル
ax2.set_ylabel('Q(B)')  # y軸のラベル

# アームA選択確率をグラフにプロット
ax3.plot(actions[:, 0], label='Arm A', color='green')
ax3.set_title('Arm A probabilities')  # タイトルを設定
ax3.set_ylim(-0.2, 1.2)  # y軸の範囲を設定
ax3.set_xlabel('試行')  # x軸のラベル
ax3.set_ylabel('P(a=A)')  # y軸のラベル

# グラフを表示
plt.tight_layout()  # グラフ間の間隔を調整
plt.show()