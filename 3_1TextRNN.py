import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        # 단어로 만들기 위해 문장 분리
        input = [word_dict[n] for n in word[:-1]]
        # 맨끝의 단어를 예측하기 위한 방법임으로 맨끝 단어 전까지 단어에 대해 고유한 인덱스를 저장
        target = word_dict[word[-1]]
        # target단어

        input_batch.append(np.eye(n_class)[input])
        # 각 단어를 원핫방식으로 표현

        target_batch.append(target)

    return input_batch, target_batch

class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()

        self.rnn = nn.RNN(input_size = n_class, hidden_size = n_hidden)
        # RNN모듈을 사용해서 I와 O를 지정
        # n_class로 한 이유가 원핫으로 표현했기 떄문에

        self.W = nn.Linear(n_hidden, n_class, bias = False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, hidden, X):

        X = X.transpose(0, 1)
        # X =  [n_step, batch_size, n_class]
        # RNN의 특성으로 Sequence한 다음 데이터를 예측하기 위해서 앞단어와 다음 단어를 분리한 것


        outputs, hidden = self.rnn(X, hidden)
        # hidden은 왜 있는거지 모르겠음

        outputs = outputs[-1]
        # 다음 단어
        model = self.W(outputs) + self.b
        # model : [batch_size, n_class]

        return model


if __name__ == '__main__':
    n_step = 2
    n_hidden = 5

    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)
    batch_size = len(sentences)

    model = TextRNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    for epoch in range(5000):
        optimizer.zero_grad()

        hidden = torch.zeros(1, batch_size, n_hidden)
        output = model(hidden, input_batch)

        loss = criterion(output, target_batch)

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    input = [sen.split()[:2] for sen in sentences]

    # Predict
    hidden = torch.zeros(1, batch_size, n_hidden)
    predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])

