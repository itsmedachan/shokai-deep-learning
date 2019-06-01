# ReLUではx<=0における微分が0(勾配が0)になってしまうことが問題だった
# そこで登場するのがLReLU(leaky relu) : f(x) = max(αx, x) (αは0.01など小さい定数)
# TensorFlowによる実装
import tensorflow as tf

def lrelu(x, alpha=0):
  return tf.maximum(alpha * x, x)

# 入力層 - 隠れ層
W0 = tf.Variable(tf.truncated_normal([n_in, n_hidden], stddev=0.01))
b0 = tf.Variable(tf.zeros([n_hidden]))
h0 = lrelu(tf.matmul(x, W0) + b0)

W1 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
b1 = tf.Variable(tf.zeros([n_hidden]))
h1 = lrelu(tf.matmul(h0, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
b2 = tf.Variable(tf.zeros([n_hidden]))
h2 = lrelu(tf.matmul(h1, W2) + b2)

W3 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
b3 = tf.Variable(tf.zeros([n_hidden]))
h3 = lrelu(tf.matmul(h2, W3) + b3)

# 隠れ層 - 出力層
W4 = tf.Variable(tf.truncated_normal([n_hidden, n_out], stddev=0.01))
b4 = tf.Variable(tf.zeros([n_out]))
y = tf.nn.softmax(tf.matmul(h3, W4) + b4)



# x>0における微分が1なので、xがどんなに大きい値をとっても勾配が消失することがない(ReLUと同様)
# x<=0における微分はα(!=0)なので、x<=0においても勾配が0にならず、学習が進む
# と期待されたが、実際に効果が出るときと出ないときがあり未解明