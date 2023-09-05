import os
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as Data
from torch import nn
from scipy.signal import savgol_filter

# from sdtw import soft_dtw
filepath = os.path.join(os.getcwd(), 'output')


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class Lossfun(nn.Module):
    def __init__(self):
        super(Lossfun, self).__init__()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.pdst = nn.PairwiseDistance(p=2)

    def forward(self, predict, label):
        loss1 = self.mse(predict, label)
        loss2 = torch.sum(self.relu(predict - 1))
        loss4 = torch.mean(self.relu(predict - label))
        loss5 = torch.mean(self.relu(label - predict - 0.1))
        loss6 = torch.mean(torch.abs(predict - label))
        # loss = loss1 + 2 * loss2 + 0.5 * loss5 + 0.1 * loss4 + loss6
        loss = loss1
        return loss


def nn_seq_ms(bs, window_length):
    length = window_length
    print('data processing...')
    # split
    df = pd.read_csv('ETTh1.csv', index_col=0)
    df['target'] = df['OT'].shift(-168)
    # # print(df)
    df = df[['OT', 'target']]
    df = df.dropna()
    for i in range(len(df.columns)):
        df[df.columns[i]] = savgol_filter(df[df.columns[i]], 21, 1)
    # df['target'] = savgol_filter(df['target'], 21, 1)

    # train = df[:int(len(df) * 0.7)]
    train = df.loc['2016-07-01':'2017-07-01']
    print(len(train))
    train.to_csv(filepath +'/train_data.csv')
    # val = df[int(len(df) * 0.7):int(len(df) * 0.8)]
    val = pd.concat(
        [df.loc['2017-07-01':'2017-07-31'], df.loc['2017-09-01':'2017-09-30'], df.loc['2017-12-01':'2017-12-31'], df.loc['2018-03-01':'2018-03-31']])
    print(len(val))
    val.to_csv(filepath +'/val_data.csv')
    # test = df[int(len(df) * 0.8):]
    test = pd.concat(
        [df.loc['2017-08-01':'2017-08-31'], df.loc['2017-10-01':'2017-11-30'], df.loc['2018-01-01':'2018-03-01'], df.loc['2018-04-01':'2018-05-31']])
    print(len(test))
    test.to_csv(filepath +'/test_data.csv')
    train_max = train.max()
    train_min = train.min()
    train_mean = train.mean()
    train_std = train.std()

    train = (train - train_min) / (train_max - train_min)
    val = (val - train_min) / (train_max - train_min)
    test = (test - train_min) / (train_max - train_min)

    # train = (train - train_mean) / train_std
    # val = (val - train_mean) / train_std
    # test = (test - train_mean) / train_std

    train_max = pd.DataFrame(train_max)
    train_max.columns = ['max']
    train_min = pd.DataFrame(train_min)
    train_min.columns = ['min']
    train_std = pd.DataFrame(train_std)
    train_std.columns = ['std']
    train_mean = pd.DataFrame(train_mean)
    train_mean.columns = ['mean']
    max_min = pd.concat([train_max, train_min, train_mean, train_std], axis=1)
    max_min = train_max.join(train_min, how='outer')

    print(max_min)
    max_min.to_csv(filepath + '/normalization.csv')
    data_col = train.shape[1]

    # df_0 = pd.DataFrame(0, index=range(64), columns=range(8))
    # df_0.columns = df.columns
    # df = pd.concat([df, df_0])

    def process(data, batch_size):
        load = data[data.columns[data_col - 1]]
        load = load.tolist()

        data = data.values.tolist()
        seq = []
        for i in range(len(data) - length + 1):
            train_seq = []
            train_label = []
            for j in range(i, i + length):
                x = []
                for c in range(0, data_col - 1):
                    x.append(data[j][c])
                train_seq.append(x)
            # print('train_seq:', train_seq)
            train_label.append(load[i + length - 1])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

        # return seq, [m, n]
        return seq

    Dtr = process(train, bs)
    Dte = process(test, bs)
    Dev = process(val, bs)
    Dall = process(df, bs)
    # lstm_all = df['x1']
    # return Dtr, Dte, lis1, lis2
    return Dtr, Dev, Dte, Dall
    # return Dtr, Dte, Dev, train_mean, train_std


def evaluate(eval_model, criterion, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    cur_loss = 0.
    mse_lst = []
    loss_lst = []
    with torch.no_grad():
        for data, targets in data_source:
            data = data.to(0)
            targets = targets.to(0)
            output = eval_model(data)
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
            # mse = nn.MSELoss(output, targets)
            # mse_lst.append(mse)
    return total_loss / len(data_source)


def train(model, optimizer, scheduler, criterion, train_data, epoch):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    loss_list = []
    for batch, (data, targets) in enumerate(train_data):
        optimizer.zero_grad()
        data = data.to(0)
        targets = targets.to(0)
        output = model(data)
        # print('output:',output.size())
        # print('target:',targets.size())
        loss = criterion(output, targets)
        optimizer.zero_grad()
        # loss = Lossfun(output, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

        cur_loss = total_loss
        loss_list.append(cur_loss)
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d}/{:5d} batches | '
              'lr {:02.6f} | {:5.2f} ms | '
              'loss {:5.5f}'.format(
            epoch, batch + 1, len(train_data), scheduler.get_last_lr()[0],
                   elapsed * 1000,
            cur_loss))
        total_loss = 0
        start_time = time.time()

    return np.array(loss_list)

def train1(model, optimizer, scheduler, criterion, train_data, epoch):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    loss_list = []
    for batch, (data, targets) in enumerate(train_data):
        optimizer.zero_grad()
        data = data.to(0)
        targets = targets.to(0)
        output = model(data)
        loss = criterion(output, targets)
        # loss = Lossfun(output, targets)
        optimizer.zero_grad()
        model.hidden = model.init_hidden()
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        cur_loss = total_loss
        loss_list.append(cur_loss)
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d}/{:5d} batches | '
              'lr {:02.6f} | {:5.2f} ms | '
              'loss {:5.5f}'.format(
            epoch, batch + 1, len(train_data), scheduler.get_last_lr()[0],
                   elapsed * 1000,
            cur_loss))
        total_loss = 0
        start_time = time.time()

    return np.array(loss_list)

def predict_future(eval_model, data_source, st, title="ARK-ODENet"):
    def rmse(pred, truth):
        return np.sqrt(np.sum(np.square(pred - truth)) / len(pred))

    def mse(pred, truth):
        return np.sum(np.square(pred - truth)) / len(pred)

    def mape(pred, truth):
        return np.mean(np.abs((pred - truth) / truth))

    def mae(pred, truth):
        return np.sum(np.abs(pred - truth)) / len(pred)

    eval_model.eval()

    pred = []
    truth = []
    with torch.no_grad():
        for data, label in data_source:
            data = data.to(0)
            label = label.to(0)
            output = eval_model(data)

            truth.extend(label.cpu().numpy()[:, -1].flatten())

            pred.extend(output.cpu().numpy()[:, -1].flatten())
        pred = np.array(pred)

        truth = np.array(truth)

    # correc_ref = truth
    # for i in range(len(pred)):
    #     if correc_ref[i] == 0:
    #         pred[i] = 0
    # for i in range(len(pred)):
    #     if pred[i] < 0:
    #         pred[i] = 0
    # for i in range(len(pred)):
    #     if pred[i] > 1:
    #         pred[i] = np.mean(np.array(pred[i]))

    pred1 = []
    truth1 = []
    for i in range(len(pred)):
        if truth[i] != 0:
            pred1.append(pred[i])
            truth1.append(truth[i])
    pred1 = np.array(pred1)
    truth1 = np.array(truth1)

    plt.figure(figsize=(9, 5))
    plt.plot(truth, label='actual', color='purple', linestyle='-', linewidth=2.5, markersize='6'
             )
    plt.plot(pred, label='prediction', color='blue', linestyle='dashdot', linewidth=2.5,
             markersize='6')
    # plt.fill_between(truth, pred, color='purple', alpha=0.5)
    plt.title('RMSE: {:.5f}| RMSE-peak:{:.5f}|MAE:{:.5f}|peak:{:.5f}'.format(rmse(pred, truth), rmse(pred1, truth1),
                                                                             mae(pred, truth), mae(pred1, truth1)))

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(ncol=2, loc='best', fontsize=14)
    plt.xticks(fontproperties='Times New Roman', fontsize=18)
    plt.xlabel('Time', fontproperties='Times New Roman', fontsize=16)
    plt.ylabel('Power/Normalizated', fontproperties='Times New Roman', fontsize=16)
    plt.yticks(fontproperties='Times New Roman', fontsize=18)
    plt.tight_layout()
    plt.savefig(filepath + '/{}.png'.format(str(st)))

    return pred, truth


def step_predict(model, data, n=1):
    input_data = data
    result = []
    with torch.no_grad():
        for _ in range(n):
            output = model(input_data)
            input_data = input_data.squeeze(2)
            input_data = input_data.squeeze(0)
            result.append(output.numpy().flatten()[0])
            input_data = torch.concat((input_data[1:], output.flatten()))
            input_data = input_data.unsqueeze(1)
            input_data = input_data.unsqueeze(0)
    return np.array(result)
