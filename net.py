# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_num):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=layer_num) 
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embed = self.embedding(x)
        lstm_in = F.relu(embed)
        lstm_out, (h, c) = self.lstm(lstm_in)
        y = self.output(lstm_out[:, -1, :])
        p = self.softmax(y)
        return p

class ScoreEstimater(object):
    def __init__(self, input_size, hidden_size, output_size, lstm_layer=1, learning_rate=0.01):
        self.hidden_size = hidden_size
        self.model = LSTM(input_size, hidden_size, output_size, layer_num=lstm_layer)
        self.loss_function  = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
    
    def train(self, input_corpus, target_corpus, epoch_size, save_model):
        for epoch in range(epoch_size):
            batch_num = len(input_corpus)
            random = np.random.randint(0, batch_num, (batch_num*9//10))
            input_batch = [input_corpus[r] for r in random]
            target_batch = [target_corpus[r] for r in random]

            total_loss = 0
            for i, t in zip(input_batch, target_batch):
                self.optimizer.zero_grad()
                x = torch.tensor(i)
                t = torch.tensor(t)
                p = self.model(x)
                loss = self.loss_function(p, t)
                loss.backward()
                self.optimizer.step()
                total_loss += loss
            if save_model:
                torch.save(self.model.state_dict(), './model/'+save_model+'_%d.model' % (epoch+1))
            print('epoch: %d \t loss: %.8f' % ((epoch+1), total_loss.item()))
    
    def test(self, input_batch, target_batch, model_path):
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        
        with torch.no_grad():
            Y = []
            acc, total = 0, 0
            for i, t in zip(input_batch, target_batch):
                x = torch.tensor(i)
                t = torch.tensor(t)
                p = self.model(x)
                y = p.argmax(dim=1)
                
                Y.append(y.tolist())
                acc += (t==y).sum()
                total += t.size(0)
                
            print('acc: %.5f' % (acc.item()/total))
            return Y