import torch
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"using PyTOrch version: {torch.__version__}, Device: {DEVICE}")

FEATURE_NUMS = 4 # 입력층으로 들어가는 데이터 개수 feature
SEO_LENGTH = 5 # 정답을 만들기 위해 필요한 시점 개수 time_step
HIDDEN_SIZE=4 # RNN 계열 계층을 구성하는 hidden state 개수
NUM_LAYERS = 1 # RNN 계열 계층이 몇 겹으로 쌓였는지 나타냄
LEARNING_RATE = 1e-3 # 학습율
BATCH_SIZE = 20 # 학습을 위한 배치사이즈 개수

import FinanceDataReader as fdr
df = fdr.DataReader('005930', '2020-01-01', '2024-06-30')
df = df[['Open', 'High', 'Low', 'Volume', 'Close']]

df.head(10)

# 데이터 스케일링/ 시퀀스 데이터 만들기
SPLIT = int(0.7*len(df)) # train: test = 7:3

train_df = df[:SPLIT]
test_df = df[SPLIT:]

from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()

train_df.iloc[:, :-1] = scaler_x.fit_transform(train_df.iloc[:, :-1])
test_df.iloc[:, :-1] = scaler_x.fit_transform(test_df.iloc[:, :-1])

scaler_y = MinMaxScaler()
train_df.iloc[:, -1] = scaler_y.fit_transform(train_df.iloc[:, [-1]])
test_df.iloc[:, -1] = scaler_y.fit_transform(test_df.iloc[:,[-1]])

import numpy as np
def MakeSeqNumpyData(data, seq_length):
    x_seq_list = []
    y_seq_list = []

    for i in range(len(data)- seq_length):
        x_seq_list.append(data[ i: i+seq_length, :-1])
        y_seq_list.append(data[ i+seq_length, [-1]])

    x_seq_numpy = np.array(x_seq_list)
    y_seq_numpy = np.array(y_seq_list)

    return x_seq_numpy, y_seq_numpy


x_train_data, y_train_data = MakeSeqNumpyData(np.array(train_df), SEO_LENGTH)
x_test_data, y_test_data = MakeSeqNumpyData(np.array(test_df), SEO_LENGTH)

print(x_train_data.shape, y_train_data.shape)
print(x_test_data.shape, y_test_data.shape)

X_train_tensor = torch.FloatTensor(x_train_data).to(DEVICE)
y_train_tensor = torch.FloatTensor(y_train_data).to(DEVICE)

X_test_tensor = torch.FloatTensor(x_test_data).to(DEVICE)
y_test_tensor = torch.FloatTensor(y_test_data).to(DEVICE)

from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset= train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset= test_dataset, batch_size=BATCH_SIZE, shuffle=False)

import torch.nn as nn

class MyLSTMMOdel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,1)

    def forward(self, data):
        h0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(DEVICE)

        outputs, _ = self.lstm(data, (h0, c0))
        last_hs = outputs[:, -1, :]  # 마지막 시점의 hidden_state
        prediction = self.fc(last_hs)

        return prediction
    

model = MyLSTMMOdel(FEATURE_NUMS,HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def model_train(dataloader, model, loss_function, optimizer):

    model.train()
    train_loss_sum = 0

    total_train_batch = len(dataloader)

    for inputs, labels in dataloader:
        x_train = inputs.to(DEVICE)
        y_train = labels.to(DEVICE)

        outputs = model(x_train)
        loss = loss_function(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()

    train_avg_loss = train_loss_sum / total_train_batch


    return train_avg_loss

train_loss_list = []

EPOCHS = 200

for epoch in range(EPOCHS):

    avg_loss= model_train(train_loader, model, loss_function, optimizer)

    train_loss_list.append(avg_loss)

train_loss_list

import seaborn as sns

sns.lineplot(train_loss_list)

test_pred_tensor = model(X_test_tensor)
test_pred_numpy = test_pred_tensor.cpu().detach().numpy()
pred_inverse = scaler_y.inverse_transform(test_pred_numpy)
y_test_numpy = y_test_tensor.cpu().detach().numpy()
y_test_inverse = scaler_y.inverse_transform(y_test_numpy)

import matplotlib.pyplot as plt

plt.plot(y_test_inverse, label="actual")
plt.plot(pred_inverse, label="prediction")
plt.grid()
plt.legend()

plt.show()