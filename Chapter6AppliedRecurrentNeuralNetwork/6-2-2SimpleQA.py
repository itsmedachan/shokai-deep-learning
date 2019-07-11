# 最大3桁同士の加算をNLPとして解かせるtoy problemの実装

def n(digitd=3):
  number = ''
  for i in range(np.random.randint(1, digits + 1)):
    number += np.random.choice(list('0123456789'))
    return int(number)

def padding(chars, maxlen):
  return chars + ' ' * (maxlen - len(chars))



digits = 3 #最大桁数
input_digits = digits * 2 + 1 # eg. 123+456
output_digits = digits + 1 # 500+500 = 1000 以上で4桁になる

added = set()
questions = []
answers = []

while len(questions) < N:
  a, b = n(), n() # 適当な数を2つ生成

  pair = tuple(sorted((a, b)))
  if pair in added:
    continue

  question = '{}+{}'.format(a, b)
  question = padding(question, input_digits) # 足りない桁を穴埋め
  answer = str(a + b)
  answer = padding(answer, output_digits) # 足りない桁を穴埋め

  added.add(pair)
  questions.append(question)
  answers.append(answer)


chars = '0123456789+ '
char_indices = dict((c, i) for i, c in enumerate(chars)) # 文字からベクトルの次元に
indices_char = dict((i, c) for i, c in enumerate(chars)) # ベクトルの次元から文字に

X = np.zeros((len(questions), input_digits, len(chars)), dtype=np.integer)
Y = np.zeros((len(questions), digits + 1, len(chars)), dtype=np.integer)

for i in range(N):
  for t, char in enumerate(questions[i]):
    X[i, t, char_indices[char]] = 1
  for t, char in enumerate(answers[i]):
    Y[i, t, char_indices[char]] = 1
  
X_train, X_validation, Y_train, Y_validation = \
train_test_split(X, Y, train_size=N_train)

