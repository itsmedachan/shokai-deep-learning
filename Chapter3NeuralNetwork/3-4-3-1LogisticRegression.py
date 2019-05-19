# implement logistic regression using tensorflow
import numpy as np
import tensorflow as tf

# 確率的降下勾配法はデータをランダムに選ぶ
tf.set_random_seed(0) # 乱数シード

# 重みベクトルwとバイアスbの初期化
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1])) # tf.zerosはnp.zerosと同様、要素が0の(多次元)配列を生成

# 続いてモデルの実装
# TensorFlowを使わずに実装すると以下

# def y(x):
#   return sigmoid(np.dot(w, x) + b)

# def sigmoid(x):
#   return 1 / (1 + np.exp(-x))

# TensorFlowを用いると以下

x = tf.placeholder(tf.float32, shape=[None, 2]) # 入力x
t = tf.placeholder(tf.float32, shape=[None, 1]) # 正解の入力t
y = tf.nn.sigmoid(tf.matmul(x, w) + b) # モデルの出力
# placeholderでデータの次元だけ定義
# shape=[None, 2]の2は2次元ベクトルの2、Noneはデータを格納する場所を表す(データ数が可変でもok)

# 続いて交差エントロピー誤差関数の実装
# ---------------------------------------------------
# 交差エントロピー誤差関数の確認
# E(w, b) = - Σ[n=1 → N]{t_n * log(y_n)  +  (1 - t_n) * log(1 - y_n)}
# ---------------------------------------------------
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y)) # tf.reduce_sum()はnp.sum()に対応

# 続いて、cross_entropyを各パラメータで偏微分し、勾配を求め(確率的)降下勾配法を適用
# TensorFlowで実装すると
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
# だけでok
# GradientDescentOptimizer()の引数0.1は学習率

# y >= 0.5でニューロンが発火することを定義
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

# 学習用のデータを用意(ORゲートで試す)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

# TensorFlowでは必ずセッションというデータのやり取りの流れの中で計算が行われるため、モデルの定義で宣言した変数・式の初期化を以下で行う
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 学習は以下
for epoch in range(200):
  sess.run(train_step, feed_dict={ # feed_dictでplaceholderであるxとtに実際に値を代入(feed)している
    x: X,
    t: Y
  })
# 今回はデータXを全て一度に渡しているため「バッチ勾配降下法」を適用している

# .eval()を用いてニューロンの発火条件を正しく分類できているか確認してみる
classified = correct_prediction.eval(session=sess, feed_dict={
  x: X,
  t: Y
})

print(classified)
# [[ True]
#  [ True]
#  [ True]
#  [ True]]

# できた :clap:
# 続いて、各入力に対する出力確率を見てみる
prob = y.eval(session=sess, feed_dict={
  x: X,
  t: Y
})
print(prob)
# [[0.22355042]
#  [0.9142595 ]
#  [0.9142595 ]
#  [0.99747413]]

# となり、確かにうまく確率を出力できていそう

# cf: w = tf.Variable(tf.zeros([2, 1]))で定義した重みwを出力してみる
print(w)
# <tf.Variable 'Variable:0' shape=(2, 1) dtype=float32_ref>
# となり、TensorFlowのデータ型が出力されてしまう(中身が確認できない)
print('w:', sess.run(w))
# w: [[3.6118839]
#  [3.6118839]]
# で中身の確認ができる

# bも然り
print('b:', sess.run(b))
# b: [-1.2450948]