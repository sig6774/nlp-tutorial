import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def random_batch():
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace = False)
    # np.random.choice(replace = False) : 비복원추출
    # range(len(skip_grams))사이즈에서 batch_size만큼 비복원 추출로 인덱스 부여
    # 비복원 추출로 하는 이유는 인덱스는 고유해야함으로
    # batch_size만큼 skip_gram안에 있는 랜덤한 인덱스 부여
    #


    for i in random_index:
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]]) # target 단어
        # 해당 인덱스에 해당하는 값만 1로 해주고 나머지는 0으로 해줌
        # 아마 이걸로 나중에 w2v에서 각 단어의 가중치값을 뽑아낼듯?
        random_labels.append(skip_grams[i][1]) # word 단어, +1,-1로 지정했기 때문에



    return random_inputs, random_labels

class W2V(nn.Module):

    def __init__(self):
        super(W2V, self).__init__()

        self.W = nn.Linear(voc_size, embedding_size, bias = False)
        # I : voc_size, O : embedding_size
        self.WT = nn.Linear(embedding_size, voc_size, bias = False)
        # I : embedding_size, O : voc_size

        # W2V의 구조로 생각해보면 위의 구조와 동일


    def forward(self,X):

        hidden_layer = self.W(X)
        # hidden_layer = [batch_size, embedding_size]

        output_layer = self.WT(hidden_layer)
        # output_layer = [batch_size, voc_size]

        return output_layer

if __name__ == '__main__':
    batch_size = 2
    embedding_size = 2

    sentences =  ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]

    word_sequence = ' '.join(sentences).split()
    # sentences를 전부 합쳐서 띄어쓰기 기준으로 자름

    word_list = ' '.join(sentences).split()

    word_list = list(set(word_list))
    # 중복 제거

    word_dict = {w : i for i, w in enumerate(word_list)}
    # key : value형태로 저장

    voc_size = len(word_list)


    skip_grams = []
    # skip-gram : 중심단어로 주변단어 예측
    for i in range(1, len(word_sequence) - 1):
        # 왜 여기서 word_sequence개수로 하지 않고 -2를 한것으로 반복했을까?
            # 다시보니 인덱스를 맞추기 위해서 word_sequence의 인덱스가 17까지 있으며 -1을 앞,뒤 단어를 봐야하기 때문에
            # 0부터 시작하지 않고 1부터 시작하여 word_sequence의 인덱스를 맞추기 위함



        target = word_dict[word_sequence[i]]
        # 중심단어 인덱스

        context = [word_dict[word_sequence[i-1]], word_dict[word_sequence[i+1]]]
        # 중심 단어에서 앞 뒤 단어 인덱스

        for w in context:
            skip_grams.append([target, w])

    # skip_grams는 [앞단어, 중심단어], [뒷단어, 중심단어] 이렇게 저장되기 때문에 총 32개

    model = W2V()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(5000):
        input_batch, target_batch = random_batch()
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)
        # random_batch에서 랜덤한 인덱스를 도출하여서 해당 인덱스에 해당하는 word와 target을 batch_size만큼 도출


        optimizer.zero_grad()
        output = model(input_batch)
        # output은 voc_size대로 해당 값이 나오는데 target_batch는 batch_size만큼의 실제값이 나온다

        loss = criterion(output, target_batch)
        # criterion에서 정의한 crossentropy가 위의 문제를 해결해줌

        if (epoch +1 ) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()


''' 정리 
가진 voc에 의해 랜덤 인덱스를 부여 
타겟단어와 주위 단어를 매칭해주는 리스트 구축 
구축한 리스트를 바탕으로 타겟 단어와 주변 단어를 도출 
해당 인덱스에 맞는 단어를 input으로 만들어줌 (input은 0,0,1,0...이런 형식, 왜냐하면 학습할때 원하는 단어에 대한 가중치를 학습해야하니깐)
두개의 Linear함수를 통해서 input을 내가 원하는 output형태로 변환
함수를 통해 나온 output과 target단어가 맞는지 비교해보며 가중치 변환하며 학습 

W2V는 가중치를 학습하는것
'''
# 시각화 코드 안돌아감
    # for i, label in enumerate(word_list):
    #     W, WT = model.parameters()
    #     x, y = W[0][i].item(), W[1][i].item()
    #     plt.scatter(x, y)
    #     plt.annotate(label, xy = (x, y), xytext = (5,2), textcoords = 'offset points', ha = 'right', va = 'bottom')
    # plt.show()

