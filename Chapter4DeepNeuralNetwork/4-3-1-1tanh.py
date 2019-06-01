# 出力層における活性化関数は確率を出力する関数でなければならないので、通常シグモイド関数かソフトマックス関数を用いる
# しかし、隠れ層における活性化関数は、受け取る値が小さければ小さい値、大きければ大きい値を出力する関数であれば問題ない
# シグモイド関数を用いると勾配が消失してしまう問題を解決するため別の活性化関数(tanh)を用いる
# Kerasによる実装

model = Sequential()
model.add(Dense(n_hidden, input_dim=n_in))
model.add(Activation('tanh')) # instead of 'sigmoid'、 sigmoidと形が似ており、勾配が消失しにくい性質あり

model.add(Dense(n_hidden))
model.add(Activation('tanh'))

model.add(Dense(n_hidden))
model.add(Activation('tanh'))

model.add(Dense(n_hidden))
model.add(Activation('tanh'))

model.add(Dense(n_out))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])