import numpy as np
import torch
import torch.nn as nn

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

def batch():
    input_batch, output_batch, target_batch = [], [], []

    for seq in data:
        for i in range(2):
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))
            # 고정된 텍스트 길이를 맞춰주기 위해 사용 (P를 고정된 길이에 맞도록 추가)
            # Use 'P' for fixed text length

        input = [num_dict[n] for n in seq[0]]
        # input to unique index
        # Encoder Input

        output = [num_dict[n] for n in ('S' + seq[1])]
        # Encoder Ouput

        target = [num_dict[n] for n in (seq[1] + 'E')]
        # Decoder Output


        input_batch.append(np.eye(n_class)[input])
        output_batch.append(np.eye(n_class)[output])
        # One-Hot Encoding

        target_batch.append(target)
        # # One-Hot Encoding

    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)
# 원핫으로 되어 있는 output을 가지고 target을 맞춰야함
# 'S'와 'E'가 추가되어 있기 때문에 나중에 +1 해야함

def testbatch(input_word):
    input_batch, output_batch = [], []

    input_w = input_word + 'P' * (n_step - len(input_word))
    input = [num_dict[n] for n in input_w]
    output = [num_dict[n] for n in 'S' + 'P' * n_step]

    input_batch = np.eye(n_class)[input]
    output_batch = np.eye(n_class)[output]

    return torch.FloatTensor(input_batch).unsqueeze(0), torch.FloatTensor(output_batch).unsqueeze(0)


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()

        self.enc_cell = nn.RNN(input_size = n_class, hidden_size = n_hidden, dropout = 0.5)
        self.dec_cell = nn.RNN(input_size = n_class, hidden_size = n_hidden, dropout = 0.5)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):

        enc_input = enc_input.transpose(0,1)
        dec_input = dec_input.transpose(0,1)

        _, enc_states = self.enc_cell(enc_input, enc_hidden)
        outputs, _ = self.dec_cell(dec_input, enc_states)

        model = self.fc(outputs)
        return model


if __name__ == '__main__':
    n_step = 5
    n_hidden = 128

    char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
    num_dict = {n: i for i, n in enumerate(char_arr)}
    data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

    n_class = len(num_dict)
    batch_size = len(data)

    model = Seq2Seq()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    input_batch, output_batch, target_batch = batch()

    for epoch in range(5000):

        hidden = torch.zeros(1, batch_size, n_hidden)

        optimizer.zero_grad()

        output = model(input_batch, hidden, output_batch)

        output = output.transpose(0, 1)
        loss = 0
        for i in range(0, len(target_batch)):

            loss += criterion(output[i], target_batch[i])
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()



