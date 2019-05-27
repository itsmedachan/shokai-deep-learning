# 多層パーセプトロンでXORゲートを実装
import numpy as np
import tensorflow as tf

# XORのデータを用意
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# XORゲートは入力層が2次元、出力層が1次元である。それぞれに対応するplaceholderを定義
x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])

# 各層の出力を表現する
# 入力層 - 隠れ層
W = tf.Variable(tf.truncated_normal([2, 2]))
# truncated_normal()は切断正規分布(truncated normal distribution)に従うデータを生成するmethod
b = tf.Variable(tf.zeros([2]))
h = tf.nn.sigmoid(tf.matmul(x, W) + b)

# 隠れ層 - 出力層
V = tf.Variable(tf.truncated_normal([2, 1]))
c = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(h, V) + c)

# 誤差関数の設定
# 今回は2値分類なので、交差エントロピー誤差関数を以下を用いる
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1-t) * tf.log(1-y))

# 確率的勾配降下法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

# 学習
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(4000):
  sess.run(train_step, feed_dict={
    x: X,
    t: Y
  })
  if epoch % 1000 == 0:
    print('epoch:', epoch)


# 学習結果の確認
classified = correct_prediction.eval(session=sess, feed_dict={
  x: X,
  t: Y
})
prob = y.eval(session=sess, feed_dict={
  x: X
})

print('classified:')
print(classified)
# classified:
#   [[ True]
#  [ True]
#  [ True]
#  [ True]]
print()
print('output probability:')
print(prob)
# [[0.00724628]
#  [0.9902805 ]
#  [0.9932968 ]
#  [0.00617734]]