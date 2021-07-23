import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 끝 단어를 맞추기 위해 데이터 구성
def make_batch():
    input_batch, target_batch = [], []

    for seq in seq_data:
        input = [word_dict[n] for n in seq[:-1]]
        target = word_dict[seq[-1]]
        input_batch.append(np.eye(n_class)[input])
        # 원핫으로 구성
        target_batch.append(target)

    return input_batch, target_batch
# input이 3개이므로 [batch_size, 3, n_class]가 됨


class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size = n_class, hidden_size= n_hidden)
        # 클래스 안에서 사용할 모델 지정
        # I : n_class, O : n_hidden
        self.W = nn.Linear(n_hidden, n_class, bias = True)
        # I : n_hidden, O : n_class

        self.b = nn.Parameter(torch.ones([n_class]))
        # 1을 더해주는 것
        # 근데 이걸 왜 하지?

    def forward(self, X):
        input = X.transpose(0,1)
        # 차원 바꿔줌, 학습에 적절하도록 차원 변경
        # input : [number of input, batch_size, n_class]

        hidden_state = torch.zeros(1, len(X), n_hidden)
        cell_state = torch.zeros(1, len(X), n_hidden)
        # input의 차원에 맞도록 생성
        # 이걸 왜 생성하는걸까 어차피 0으로 구성된 mat인데
            # outputs에 한번에 정보가 전부 들어갈까봐 안써도 따로 나눠주는건가
        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))

        outputs = outputs[-1]
        # 끝을 맞추기 위해서 사용하는거니 모델을 통해 나온 끝만 추출
        model = self.W(outputs) + self.b
        # model = self.W(outputs)

        return model
        # model = [batch_size, n_class]



if __name__ == '__main__':
    n_step = 3
    n_hidden = 128

    char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
    word_dict ={n:i for i, n in enumerate(char_arr)}
    number_dict = {i: w for i, w in enumerate(char_arr)}
    n_class = len(word_dict)

    seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

    model = TextLSTM()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    for epoch in range(1000):
        optimizer.zero_grad()

        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    inputs = [sen[:3] for sen in seq_data]

    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(inputs, '->', [number_dict[n.item()] for n in predict.squeeze()])

''' 정리 
목표 : 단어의 끝을 예측하는 문제 
1. 하나의 단어를 예측하기 때문에 알파벳을 vocab으로 지정 
2. 모델에 들어가기전 input은 [batch_size, number of input, n_class]
3. lstm모델과 linear모델을 사용하여 최종적인 output을 도출 
4. lstm모델이 실제 단어의 sequence특징을 잘 파악한다는 것을 알 수 있음 
5. 이러한 방법으로 다른 문제를 풀어볼 예정
'''