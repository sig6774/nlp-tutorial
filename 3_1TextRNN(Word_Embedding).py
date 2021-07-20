import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

''' 긍정 부정 학습 
'''
def make_batch():
    x = []
    o = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word]

        x.append(input)

    o = [i for i in target]

    return x, o

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()

        self.C = nn.Embedding(vocab_size, embedding_size)



        self.rnn = nn.RNN(input_size= 3, hidden_size = hidden)
        # 각 단어에 대해 3차원의 벡터를 입력으로 받게 되고 output은 hidden에 맞도록 도출

        self.W  = nn.Linear(hidden*embedding_size , 1)

    def forward(self, x):
        # X = x.transpose(0,1)

        x = self.C(input_batch)

        outputs, hidden = self.rnn(x)

        outputs = outputs.view(4,-1)

        model = self.W(outputs)

        return model

if __name__ == '__main__':
    hidden = 10
    embedding_size = 3
    sentences = ["i like dog", "i love coffee", "i hate milk", "he dislike coffee"]
    target = [0, 0, 1, 1]

    word_list = ' '.join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w : i for i,w in enumerate(word_list)}
    vocab_size = len(word_dict)

    batch_size = len(sentences)

    model = RNN()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)


    x, o = make_batch()
    input_batch = torch.LongTensor(x)
    target_batch = torch.FloatTensor(o)

    for epoch in range(500):
        optimizer.zero_grad()


        output = model(input_batch)

        output = output.squeeze()
        loss = criterion(output, target_batch)

        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

''' 정리 
one-hot방식은 별로라고 생각하여 word_embedding을 사용하여 각 단어에 대해서 벡터를 부여 
각 단어에 대해 rnn 거친 결과를 각 batch마다 하나로 합쳐 linear함수를 통해 최종적인 라벨값이 나오게 설계 
epoch가 350회가 넘어갔을때 loss값은 2%로써 좋은 결과를 보여줌 
model로 return하기전 1을 더해주면 조금 더 좋은 결과가 나올것이라 예쌍 
끄읏
'''