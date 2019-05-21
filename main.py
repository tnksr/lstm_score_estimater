# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from net import LSTM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScoreEstimater(object):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.hidden_size = hidden_size
        self.model = LSTM(input_size, hidden_size, output_size)
        self.loss_function  = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
    
    def run_epoch(self, input_tensor, target_tensor):
        batch_size = input_tensor.size(1)
        loss = 0
        self.optimizer.zero_grad()
        a5
        # initialize
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        for x, t in zip(input_tensor, target_tensor):
            x = torch.tensor(x)
            y, h = self.model(x, h)
                
        t = torch.tensor(t)
        loss += self.loss_function(y, t)
        loss.backward()
        self.optimizer.step()
        
        return y, loss.items()/target_tensor.size(0)
    
    def train(self, input_batch, target_batch, epoch_size, save_model_path):
        total_loss = 0
        for epoch in range(epoch_size):
            input_tensor = torch.tensor(input_batch[epoch])
            target_tensor = torch.tensor(target_batch[epoch])
            _, loss = self.run_epoch(input_tensor, target_tensor)
            total_loss += loss
        if save_model_path:
            torch.save(self.model.state_dict(), save_model_path)
    
    def test(self, input_batch, target_batch, load_model_path):
        if load_model_path:
            self.model.load_state_dict(torch.load(save_model_path))
        with torch.no_grad():
            input_tensor = torch.tensor(input_batch)
            target_tensor = torch.tensor(target_batch)
            y, loss = self.run_epoch(input_tensor, target_tensor)