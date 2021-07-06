import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()

        self.num_filters_total = num_filters * len(filter_sizes)
        # 일단 보류

        self.W = nn.Embedding(vocab_size, embedding_size)
        # vocab_size : 최대 길이
        # 한 vocab을 설명하기 위한 벡터차원

        self.Weight = nn.Linear(self.num_filters_total, num_classes, bias = False)
        # I : num_filters_total, O : num_classes
        self.Bias = nn.Parameter(torch.ones([num_classes]))
        # num_classes만큼의 1로된 벡터

        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])
        # kernel size = (size, embedding_size), size별로 해당 문장의 단어를 읽으며 특징 추출

    def forward(self, X):
        embedded_chars = self.W(X)
        # [batch_size, number of sentences, embedding_size]
        embedded_chars = embedded_chars.unsqueeze(1)
        # 1인 위치에 차원을 하나 추가 아마, Conv2d Layer에서 입력을 1로 지정했기 때문에 했을거라 생각
        # embedded_char = [batch, channel(1), number of sentences, embedding_size]
        pooled_outputs = []

        for i, conv in enumerate(self.filter_list):
            # conv : [input_channel, output_channel, (filter_height, filter_width)]
            # filter_height : 몇개의 단어를 한번에 볼 것인가
            # filter_width : 각 단어의 의미를 표현하는 벡터의 수

            h = F.relu(conv(embedded_chars))
            # w는 각 단어에 대해 사용자가 지정한 emb vec으로 구성
            # filter가 2개의 단어에 대한 특징을 도출
            # 각 단어에 대해 2개의 output이 도출되고 그것을 relu함수를 통해 결과 나옴
            # 즉, h는 각 단어에 대해서 2개의 output을 도출하고 3개의 단어가 하나의 배치이고 총 6개의 문장이 있으므로 6*3*2가 된다
            mp = nn.MaxPool2d((sequence_length - filter_sizes[i] + 1, 1))
            # mp를 통해 각 값의 최대값만 도출

            pooled = mp(h).permute(0,3,2,1)
            # 차원의 순서를 바꿔줌
            # 상당한 이해가 필요
            # pooled = [batch_size, output_height, output_width, output_channel]

            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(filter_sizes))
        # cat : 텐서를 연결하는것
        #

        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total])
        # flat해서 최종적으로 결과를 도출하는건 알겠는데 이게 차원이 어떻게 되는걸까

        model = self.Weight(h_pool_flat) + self.Bias

        return model

if __name__ == '__main__':

    embedding_size = 2
    sequence_length = 3
    num_classes = 2
    filter_sizes = [2, 2, 2]
    num_filters = 3

    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0]

    word_list = ' '.join(sentences).split()
    # 데이터가 ,로 구분되어 있기 때문에 join을 사용해서 스페이스를 구분자로 하여 하나의 리스트로 만들어주고 스페이스를 기준으로 잘라줌
    word_list = list(set(word_list))

    word_dict = {w : i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)

    model = TextCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    inputs = torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
    # 해당 문장에 있는 단어를 고유한 인덱스로 바꿔줌

    targets = torch.LongTensor([out for out in labels])
    # 요건 그냥 리스트를 tensor로 바꿔주는거


    for epoch in range(10):

        optimizer.zero_grad()

        output = model(inputs)
        # output이 아마 [batch_size, num_classes]로 나올듯

        loss = criterion(output, targets)
        # 실제값과 추정값 비교

        if (epoch+1) % 1000 == 0:
            print('epoch : ', '%04d' % (epoch + 1), 'cost :', '{:6f}'.format(loss))

        loss.backward()
        optimizer.step()

