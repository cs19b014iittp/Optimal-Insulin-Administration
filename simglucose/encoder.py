import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# obs_data = []

# with open('obs_hist1.pkl', 'rb') as inp:
#     obs_data = pickle.load(inp)
# dlen = len(obs_data)

# print(dlen, len(obs_data[0]))

# train_data = torch.FloatTensor(obs_data[0:int(dlen*0.7)])
# test_data = torch.FloatTensor(obs_data[int(dlen*0.7):])
# '''
# define the NN architecture
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(300,200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 6),
        )
    
        self.decoder = torch.nn.Sequential(  
            torch.nn.Linear(6, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 300),
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

if __name__ == "__main__":
    model = AE().to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 100

    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        model.train()

        for data in train_data:
            data = data.to(device)
            
            encoded, decoded = model(data)
            loss = criterion(decoded, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
                
        # print avg training statistics 
        train_loss = train_loss/len(train_data)
        print('Epoch: {} \n\tTraining Loss: {:.4g}'.format(epoch, train_loss))

        model.eval()
        test_loss = 0

        with torch.no_grad():
            for data in test_data:
                data = data.to(device)
                encoded, decoded = model(data)
                test_loss += criterion(decoded, data).item()

        test_loss /= len(test_data)
        print('\tAvg Loss: {:.4g}'.format(test_loss))

    torch.save(model, 'ae_model1.pt')
    # '''
