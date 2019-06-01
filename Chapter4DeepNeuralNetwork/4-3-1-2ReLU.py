# 双曲線正接関数(tanh)を用いることで勾配は確かに消滅しにくくなった
# しかし依然として、高次元のデータを扱う場合など関数の中身の値が大きくなる場合は勾配が消えてしまう
# そこで登場するのがReLU関数(rectified linear unit) : f(x) = max(0, x)
# Kerasによる実装

model = Sequential()
model.add(Dense(n_hidden, input_dim=n_in))
model.add(Activation('relu')) # instead of 'tanh'

model.add(Dense(n_hidden))
model.add(Activation('relu'))

model.add(Dense(n_hidden))
model.add(Activation('relu'))

model.add(Dense(n_hidden))
model.add(Activation('relu'))

model.add(Dense(n_out))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])

# x>0における微分が1なので、xがどんなに大きい値をとっても勾配が消失することがない
# x<=0における微分は0なので、x<=0における勾配も0になり、学習の間ずっと0(不活性)になり得る
# 特に学習率を大きい値に設定すると、最初の誤差逆伝播でニューロンの値が小さくなりすぎてしまい、そのニューロンがほとんど存在しない状態になり得るという欠点もあり
# とはいえ便利ゆえよく使われる活性化関数の１つ