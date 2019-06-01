# 活性化関数を工夫することで勾配消化問題は解決できたが、過学習(over fitting)問題は解決できていない
# モデルの汎化性能を向上させて過学習を防ぎたい(未知のデータに対しての予測精度をあげることを汎化(generalization)という)
# そこでドロップアウト(Dropout)を用いる
# ドロップアウトとは学習の際にランダムにニューロンをドロップアウト(除外)させること
# ドロップアウトによって擬似的に複数のモデルで学習を行なうことができる
# 複数のモデルを生成して学習を行うことをアンサンブル学習(ensemble learning)と言う
# つまり、ドロップアウトによって擬似的にアンサンブル学習ができる
# 実装はニューロンにランダムで0か1の値をとる「マスク」をかけることで実現できる

# TensorFlowによる実装

import numpy as np
import tensorflow as tf

# モデル定義

x = tf.placeholder(tf.float32, shape=[None, n_in])
t = tf.placeholder(tf.float32, shape=[None, n_out])
keep_prob = tf.placeholder(tf.float32) # ドロップアウトしない確率

# 入力層 - 隠れ層
W0 = tf.Variable(tf.truncated_normal([n_in, n_hidden], stddev=0.01))
b0 = tf.Variable(tf.zeros([n_hidden]))
h0 = tf.nn.relu(tf.matmul(x, W0) + b0)
h0_drop = tf.nn.dropout(h0, keep_prob) # keep_probはドロップアウトしない確率(=1-p)、学習時は0.5、テスト時は1.0

# 隠れ層 - 出力層
W1 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
b1 = tf.Variable(tf.zeros([n_hidden]))
h1 = tf.nn.relu(tf.matmul(h0_drop, W1) + b1)
h1_drop = tf.nn.dropout(h1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
b2 = tf.Variable(tf.zeros([n_hidden]))
h2 = tf.nn.relu(tf.matmul(h1_drop, W2) + b2)
h2_drop = tf.nn.dropout(h2, keep_prob)

# 隠れ層 - 出力層
W3 = tf.Variable(tf.truncated_normal([n_hidden, n_out], stddev=0.01))
b3 = tf.Variable(tf.zeros([n_out]))
y = tf.nn.softmax(tf.matmul(h2_drop, W3) + b3)

# 学習
for epoch in range(epochs):
  X_, Y_ = shuffle(X_train, Y_train )

  for i in range(n_batches):
    start = i * batch_size
    end = start + batch_size

    sess.run(train_step, feed_dict={
      x: X_[start:end],
      t: Y_[start:end],
      keep_prob: 0.5
    })

# 学習後のテスト
accuracy_rate = accuracy.eval(session.sess, feed_dict={
  x: X_test,
  t: Y_test,
  keep_prob: 1.0
})