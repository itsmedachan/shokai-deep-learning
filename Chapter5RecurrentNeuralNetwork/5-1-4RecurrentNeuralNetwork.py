import numpy as np
import tensorflow as tf

# 予備に用いるノイズ入りのsin波
def sin(x, T=100):
  return np.sin(2.0 * np.pi * x / T)


def toy_problem(T=100, ampl=0.05):
  x = np.arange(0, 2 * T + 1)
  noise = ampl * np.random.uniform(low=-0.1, high=1.0, size=len(x))
  return sin(x) + noise

# 例えば
T = 100
f = toy_problem(T)

# とすると、t=0, ..., 200におけるデータが得られるので、得られたfを全データセットとして実験する
# 全データfに対して、このτごとにデータを分割していく実装
length_of_sequences = 2 * T # 全時系列の長さ
maxlen = 25 # １つの時系列データの長さ

data = []
target = []

for i in range(0, length_of_sequences - maxlen + 1): # データセットは t - τ + 1
  data.append(f[i: i + maxlen]) # 予測に用いる長さτの時系列データ群
  target.append(f[i + maxlen]) # 予測によって得られるべきデータ群

# データ数をN、入力の次元数をI(=1)とすると、モデルに用いる全入力Xは次元が(N,τ,I)となる
# これを表現すると
X = np.array(data).reshape(len(data), maxlen, 1)

# 同様にtargetについてもモデルの出力の次元数(=1)に対応できるように変形する必要がある
Y = np.array(target).reshape(len(data), 1)

# 上記2式は下記と等価
X = np.zeros((len(data), maxlen, 1), dtype=float)
Y = np.zeros((len(data), 1), dtype=float)

for i, seq in enumerate(data):
  for t, value in enumerate(seq):
    X[i, t, 0] = value
  Y[i, 0] = target[i]

# 以上で時系列データの用意完了
# 実験のために、訓練データと検証データに分割しておく
N_train = int(len(data) * 0.9)
N_validation = len(data) - N_train

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=N_validation)

# 以下、TensorFlowによるリカレントニューラルネットワークの実装
state = initial_state
outputs = [] # 過去の隠れ層の出力を保存
with tf.variable_scope('RNN'): # 過去の値にアクセスできるように
  for t in range(maxlen):
    if t > 0:
      tf.get_varivable_scope().reuse_varivables()
    (cell_output, state) = cell(x[:, t, :], state) # 基本的には各時刻tにおける出力cell(x[:, t, :], state)を計算しているだけ
    outputs.append(cell_output)
    output = outputs[-1]

print(outputs)