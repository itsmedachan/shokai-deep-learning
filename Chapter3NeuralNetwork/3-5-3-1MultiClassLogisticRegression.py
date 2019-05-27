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
# [[6.5313103e-03 9.8681277e-01 6.6559250e-03]
#  [5.4406030e-08 4.4718280e-02 9.5528162e-01]
#  [9.6815646e-01 3.1843551e-02 2.3380778e-08]
#  [2.9949013e-10 2.4116028e-03 9.9758840e-01]
#  [9.9679178e-01 3.2082470e-03 9.8642283e-10]
#  [9.1035688e-01 8.9643046e-02 8.1150816e-08]
#  [9.9912506e-01 8.7493187e-04 5.0535107e-11]
#  [1.3475924e-09 1.7383253e-03 9.9826163e-01]
#  [2.0237109e-02 9.7786146e-01 1.9014322e-03]
#  [9.9905962e-01 9.4034022e-04 6.8459155e-11]]

# また、クラス1とクラス2の分類直線は
# w11*x1 + w12*x2 + b1 = w21*x1 + w22*x2 + b2
# を満たす直線である
print(sess.run(W))
# [[-1.0887656   0.29475296  0.7940129 ]
#  [ 0.79717684  0.2882816  -1.0854582 ]]
print(sess.run(b))
# [-0.04716736  0.09135975 -0.04419243]
