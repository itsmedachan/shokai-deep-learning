# LReLUではx<=0における微分がα(勾配がα)で固定だった
# このαも学習によって最適化しようというアプローチがParametric ReLU(PReLU) : f(pj) = max(0, pj) + αj * min(0, pj)
# TensorFlowによる実装
import tensorflow as tf

def prelu(x, alpha):
  return tf.maximum(tf.zeros(tf.shape(x)), x) + alpha * tf.minimum(tf.zeros(tf.shape(x)), x)


# 入力層 - 隠れ層
W0 = tf.Variable(tf.truncated_normal([n_in, n_hidden], stddev=0.01))
b0 = tf.Variable(tf.zeros([n_hidden]))
alpha0 = tf.Variable(tf.zeros([n_hidden]))
h0 = prelu(tf.matmul(x, W0) + b0, alpha0)

W1 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
b1 = tf.Variable(tf.zeros([n_hidden]))
alpha1 = tf.Variable(tf.zeros([n_hidden]))
h1= tf.lrelu(tf.matmul(h0, W1) + b1, alpha1)

W2 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
b2 = tf.Variable(tf.zeros([n_hidden]))
alpha2 = tf.Variable(tf.zeros([n_hidden]))
h2 = tf.lrelu(tf.matmul(h1, W2) + b2, alpha2)

W3 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
b3 = tf.Variable(tf.zeros([n_hidden]))
alpha3 = tf.Variable(tf.zeros([n_hidden]))
h3 = tf.lrelu(tf.matmul(h2, W3) + b3, alpha3)

# 隠れ層 - 出力層
W4 = tf.Variable(tf.truncated_normal([n_hidden, n_out], stddev=0.01))
b4 = tf.Variable(tf.zeros([n_out]))
y = tf.nn.softmax(tf.matmul(h3, W4) + b4)



# 他にもRandomized ReLU(RReLU)やExponetial Linear Units(ELU)など、たくさんのReLUをベースにした活性化関数が提案されている