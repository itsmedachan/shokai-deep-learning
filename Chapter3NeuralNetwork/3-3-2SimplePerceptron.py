import numpy as np

rng = np.random.RandomState(123)

d = 2 # データの次元
N = 10 # 各パターンのデータ数
mean = 5 # ニューロンが発火するデータの平均値

x1 = rng.randn(N, d) + np.array([0, 0]) # ニューロンが発火しないデータN個
x2 = rng.randn(N, d) + np.array([mean, mean]) # ニューロンが発火するデータN個

x = np.concatenate((x1, x2), axis=0) # x1, x2の2種類のデータをまとめて処理するためにxにまとめる

# 重みベクトルwとバイアスbの初期化
w = np.zeros(d)
b = 0

def y(x):
  return step(np.dot(w, x) + b)

# STEP関数
def step(x):
  return 1 * (x > 0) # x>0がtrueのとき1、falseのとき0

# 正しい出力値t
def t(i):
  if i < N:
    return 0
  else:
    return 1

# 以上で学習に必要な関数が揃った
# 以下、謝り訂正学習法を実装

while True:
  classified = True
  for i in range(N * 2):
    delta_w = ( t(i) - y(x[i]) ) * x[i]
    delta_b = ( t(i) - y(x[i]) )
    w += delta_w
    b += delta_b
    classified *= all(delta_w == 0) * (delta_b == 0)
    print("\t delta_w:{} delta_b:{}".format(delta_w, delta_b))
    if classified:
      break

print(y([0, 0]))
# 0 ←発火しない
print(y([5, 5]))
# 1 ←発火する