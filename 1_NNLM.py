import torch
import torch.nn as nn
import torch.optim as optim

def batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        # 문장을 단어단위로 쪼깸
        input = [word_dict[n] for n in word[:-1]]
        # word의 마지막 단어를 제외한 단어들을 dict로 넣어주고 해당 단어의 인덱스를 input변수에 넣어줌

        target = word_dict[word[-1]]
        # 마지막 단어에 대한 인덱스 값을 target변수에 저장

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch
# input_batch : 각 단어의 인덱스 번호를 넣어준 변수
#


class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()

        self.C = nn.Embedding(n_class, m)
        # 각 단어에 대해서 정해진 차원m만큼 임베딩
        self.H = nn.Linear(n_step*m, n_hidden, bias = False)
        # input : n_step * m, output : n_hidden
        self.d = nn.Parameter(torch.ones(n_hidden))
        # n_hidden만큼의 크기로 1인 tensor 생성

        self.U = nn.Linear(n_hidden, n_class, bias = False)

        self.W = nn.Linear(n_step * m, n_class, bias = False)

        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, X):
        X = self.C(X)
        # X : [batch_size, n_step, m]
        X = X.view(-1, n_step*m)
        # view 차원을 줄여줌 X : [batch_size, n_step*m]
        # self.H가 n_step*m이라서 그것에 맞도록 줄여준건감???

        tanh = torch.tanh(self.d + self.H(X))
        # self.H를 통해서 tanh는 [batch_size, n_hidden]
        # n_step이 왜 필요한거지?, n_step을 뺴고 해볼까

        output = self.b + self.W(X) + self.U(tanh)
        # output : [batch_size, n_class]
        return output
    # 헤당 입력을 받고 forward과정을 거치면서 중복되지 않은 단어 중 어느단어에 해당하는지를 분류


if __name__ == '__main__':
    n_step = 2
    n_hidden = 2
    m = 2

    sentences = ['i like cat', 'i love coffee', 'i hate sushi']
    # 입력문장

    word_list = ' '.join(sentences).split()
    # ,구분자를 추가하여 한칸 띄어서 각 단어를 문장에서 분리

    word_list = list(set(word_list))
    # 중복 단어 제거

    word_dict = {w:i for i, w in enumerate(word_list)}
    # key : value 형태로 위에서 자른 단어를 넣어줌

    number_dict = {i : w for i, w in enumerate(word_list)}

    n_class = len(word_dict)
    # 단어 개수

    model = NNLM()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = 0.001)
    # 손실함수, 최적화 함수 설정

    input_batch, target_batch = batch()

    input_batch = torch.LongTensor(input_batch)
    # 각 단어를 고유한 인덱스로 되어 있는것을 부호가 포함된 tensor로 변환

    target_batch = torch.LongTensor(target_batch)

    for epoch in range(3000):
        optimizer.zero_grad()
        output = model(input_batch)

        # output : [batch_size, n_class], target_Batch : [batch_size]

        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    predict = model(input_batch).data.max(1, keepdim = True)[1]



''' 정리 
1. 각 문장의 타겟 단어를 제외한 나머지를 고유한 인덱스 번호로 부여함
2. 부여된 단어들을 설명할 수 있는 벡터차원을 사용자가 지정 (Ex : 1개의 단어가 있는데 단어를 표현하고 싶은 차원을 3으로 한다면 love : 0.1 0.2 0.5 요렇게)
3. Linear함수를 통해서 input, output의 dimension을 맞춰주면서 연산 진행 
4. 학습을 거치면서 타겟 단어를 맞추도록 함 
5. Batch_size만큼의 결과가 나오게 되고 그 결과와 실제 결과를 비교하며 성능 검증 
6. 끝 

'''

