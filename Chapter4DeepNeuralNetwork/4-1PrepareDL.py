from sklearn import datasets
mnist = datasets.fetch_mldata('MNIST original', data_home='.')

n =len(mnist.data)
N = 10000 # MNISTの一部のデータで実験
indices = np.random.permutation(range(n))[:N] # ランダムにN枚を選択
X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)] # 1-of-K表現に変換

X_train, X_test, Y_train, Y_test = train_test_split(x, Y, train_size=0.8)

# Kerasを用いて実装

# モデルの設定
n_in = len(X[0]) # 784
n_hidden = 200
n_out = len(Y[0]) # 10

model = Sequential()
model.add(Dense(n_hidden, input_dim=n_in))
model.add(Activation('sigmoid'))

model.add(Dense(n_out))
model.add(Activation('softmax'))

model.compile(loss='categorial_crossentropy',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])

# モデル学習
epochs = 1000
batch_size = 100

model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

# 予測精度の評価
loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)

# ニューロン数を闇雲に増やしても予測精度は上がらず、ただ計算時間が増えてしまう
# 隠れ層の数自体を増やすアプローチはどうか
# 各隠れ層の内のニューロン数が全て同じ200とすると実装は
model.add(Dense(n_hidden))
model.add(Activation('sigmoid'))
# を追加していくだけ
# i.e. モデル全体は以下のようになる(隠れ層が3つの場合)
model = Sequential()
model.add(Dense(n_hidden, input_dim=n_in))
model.add(Activation('sigmoid'))

model.add(Dense(n_hidden))
model.add(Activation('sigmoid'))

model.add(Dense(n_hidden))
model.add(Activation('sigmoid'))

model.add(Dense(n_out))
model.add(Activation('softmax'))
# 結果は
# 隠れ層の数 | 正解率(%)
# 1        | 87.30
# 2        | 87.30
# 3        | 82.20
# 4        | 36.20
# 予測精度は上がるどころか下がっている
# この問題の究明と解決は次節