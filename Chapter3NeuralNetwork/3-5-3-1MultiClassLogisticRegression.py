# implement multi-class logistic regression using tensorflow
# two iuputs, three outputs classified into three classes
# data for each class create sample datasets whose mean is μ ≠ 0, which follow normal distribution
# 100 data for each class, which means 300 data in total
# なんでここだけ英語にしたんだろう

import numpy as np
import tensorflow as tf

# (ミニバッチ)確率的勾配降下法の学習で必要な「入力データをランダムに選択する」を実現するため、shuffle機能をsklearnというライブラリから拝借
from sklearn.utils import shuffle

# 変数定義
M = 2 # 入力データの次元
K = 3 # クラス数
n = 100 # クラスごとのデータ数
N = n * K # 全データ数(300)

# サンプルデータ群の生成
X1 = np.random.randn(n, M) + np.array([0, 10])
X2 = np.random.randn(n, M) + np.array([5, 5])
X3 = np.random.randn(n, M) + np.array([10, 0])
Y1 = np.array([[1, 0, 0] for i in range(n)])
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])

X = np.concatenate((X1, X2, X3), axis=0)
Y = np.concatenate((Y1, Y2, Y3), axis=0)

# モデルの定義
W = tf.Variable(tf.zeros([M, K]))
b = tf.Variable(tf.zeros([K]))

x = tf.placeholder(tf.float32, shape=[None, M])
t = tf.placeholder(tf.float32, shape=[None, K])
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 2値分類ではsigmoidだったが、今回は多クラス(3クラス)分類なのでsoftmax

# 交差エントロピー誤差関数(ミニバッチごとの平均を求めるためにtf.reduce_mean()を利用)
cross_entropy = tf.reduce_mean( - tf.reduce_sum( t * tf.log(y), reduction_indices=[1] ) )
# reduction_indicesは行列のどの方向に向かって和を取るかを表している

# 上で定義した交差エントロピー誤差関数を確率的勾配降下法により最小化したい
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# 正しく分類されているか確認
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))

# モデルの学習
batch_size = 50 # ミニバッチサイズ
n_batches = N // batch_size

# セッションの用意
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 確率的勾配降下法では各エポックごとにデータをシャッフルするので、以下のようになる
for epoch in range(20):
  X_, Y_ = shuffle(X, Y)

  for i in range(n_batches):
    start = i * batch_size
    end = start + batch_size

    sess.run(train_step, feed_dict={
      x: X_[start:end],
      t: Y_[start:end]  # startとendにより、各ミニバッチが全体のデータのどこに位置するかを表している
    })

# 結果の確認
X_, Y_ = shuffle(X, Y)

classified = correct_prediction.eval(session=sess, feed_dict={
  x: X_[0:10],
  t: Y_[0:10]
})

prob = y.eval(session=sess, feed_dict={
  x: X_[0:10]
})

print('classified: {}'.format(classified))
# classified: [ True  True  True  True  True  True  True  True  True  True]
print('-------------------------------------')
print('output probability:')
print(prob)
# output probability:
# [[6.3360199e-02 9.3362278e-01 3.0169787e-03]
#  [1.7040487e-11 6.5947784e-04 9.9934047e-01]
#  [9.7496337e-01 2.5036419e-02 1.8700985e-07]
#  [9.7517869e-09 4.6620380e-02 9.5337963e-01]
#  [9.8402065e-01 1.5979346e-02 8.6830880e-09]
#  [9.9787092e-01 2.1290113e-03 1.5841697e-09]
#  [3.4207278e-03 9.6156156e-01 3.5017788e-02]
#  [9.9399310e-01 6.0069491e-03 6.1772005e-09]
#  [4.8423945e-03 9.6568030e-01 2.9477211e-02]
#  [9.9339032e-01 6.6097109e-03 2.4094586e-08]]

# また、クラス1とクラス2の分類直線は
# w11*x1 + w12*x2 + b1 = w21*x1 + w22*x2 + b2
# を満たす直線である
print(sess.run(W))
# [[-1.1002041   0.29816976  0.8020341 ]
#  [ 0.8014562   0.28687665 -1.0883336 ]]
print(sess.run(b))
# [-0.05884542  0.10988893 -0.05104348]
