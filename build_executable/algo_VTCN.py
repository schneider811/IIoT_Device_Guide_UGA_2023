#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

####  below is the original TCN code #####
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
 

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        # self.relu = nn.ReLU()

        ### added beyond TCN
        self.bn3 = nn.BatchNorm1d(n_outputs)
        self.relu3 = nn.ReLU()
        # self.dropout3 = nn.Dropout(dropout)
        ###

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        ### added beyond TCN
        res = out + res        
        res = self.bn3(res)
        res = self.relu3(res)
        # res = self.dropout3(res)
        return res
        ###

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

####  above is the original TCN code #####

class VTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(VTCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = x #self.batchnorm(x)
        y = self.tcn(y)
        y = self.linear(y[:, :, -1])
        return y


import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys, os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import nvidia_smi #pip3 install nvidia-ml-py3
from torch.autograd import Variable
# from utils import *
# from algo_dsp import low_pass_filter, band_pass_filter
from sklearn.preprocessing import StandardScaler

# list1: label; list2: prediction
def plot_2vectors(label, pred, name, mae):
    list1 = label
    list2 = np.array(pred)
    # mae = calc_mae(list1, list2)
    sorted_id = sorted(range(len(list1)), key=lambda k: list1[k])

    plt.clf()
    plt.text(0,np.min(list2),f'MAE={mae}')
    plt.scatter(np.arange(list2.shape[0]),list2[sorted_id],s = 1, alpha=0.5,label=f'{name} prediction', color='blue')
    plt.scatter(np.arange(list1.shape[0]),list1[sorted_id],s = 1, alpha=0.5,label=f'{name} label', color='red')

    plt.legend()
    plt.savefig(f'{name}.png')
    # print(f'Saved plot to {name}.png')
    plt.show()

def get_label_index(label_name, labels_list):
    # print(labels_list.index(label_name))
    index = labels_list.index(label_name)
    if index < 0:
        print(f"the label_name <{label_name}> does not exist in the labels_list <{labels_list}>!")
        exit()
    return index-len(labels_list)  

def preprocess_data(X_data, label_name):
    # if label_name == 'R':
    #     print("apply low_pass_filter!")
    #     for i in range(X_data.shape[0]):
    #         X_data[i, :] = low_pass_filter(X_data[i, :], 100, 0.6, 3)
    # elif label_name == 'S' or label_name == 'D':
    #     print("apply band_pass_filter!")
    #     for i in range(X_data.shape[0]):
    #         X_data[i, :] = band_pass_filter(X_data[i, :], Fs = 100, low = 0.5, high = 25, order = 3)
    # X_data = standardize_data (X_data)
    # scaler =  StandardScaler()
    # X_data = scaler.fit_transform (X_data)
    return X_data

# input_channels = 1
def data_generator(data_file, data_length, label_name, labels_list):
    label_index = get_label_index(label_name, labels_list) # label_index = range(-2, 0) # this works, but need to figure out l1_loss
    print(f"Reading data with label: {args.label_name} index {label_index} from {data_file}")
    data_set = np.load(data_file)
    np.random.shuffle(data_set)
    X_data = data_set[:, :data_length]
    Y_data = data_set[:, label_index]
    return X_data, Y_data

def tensor_converter(X_data, Y_data, input_channels):
    # data_set = np.load(data_file)
    data_length = X_data.shape[1]
    chunk_length = int (data_length/input_channels)

    start_index = 0
    while start_index + chunk_length <= data_length:
        end_index = start_index + chunk_length
        data_X = np.reshape( X_data[:, start_index:end_index], (-1, 1, chunk_length))

        if start_index == 0:
            X = data_X
        else:
            X = np.concatenate((X, data_X), axis=1)
        start_index = end_index

    X = torch.from_numpy(X).to(dtype=torch.float)

    data_y = np.reshape(Y_data, (-1, 1))
    Y = torch.from_numpy(data_y).to(dtype=torch.float)

    print(X.size(), Y.size())
    return Variable(X), Variable(Y)

# >>> A
# array([[81, 79, 48, 63, 93, 65,  4, 54],
#        [30, 17, 57, 77, 94, 27, 65, 87],
#        [31,  5, 77, 39, 59, 90, 38, 39]])
# >>> A1 = A[:, 0:4]
# >>> A2 = A[:, 4:8]
# >>> B1 = np.reshape( A1, (-1, 1, 4))
# >>> B2 = np.reshape( A2, (-1, 1, 4))
# >>> X = np.concatenate((B1, B2), axis=1)
# >>> X.shape
# (3, 2, 4)
# >>> X
# array([[[81, 79, 48, 63],
#         [93, 65,  4, 54]],

#        [[30, 17, 57, 77],
#         [94, 27, 65, 87]],

#        [[31,  5, 77, 39],
#         [59, 90, 38, 39]]])

def train(device, epoch, X_data, Y_data, model, optimizer, log_interval, batch_size, clip, verbose=False):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    cur_loss = 0
    for i in range(0, X_data.size(0), batch_size):
        if i + batch_size > X_data.size(0):
            x, y = X_data[i:], Y_data[i:]
        else:
            x, y = X_data[i:(i+batch_size)], Y_data[i:(i+batch_size)]
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = F.l1_loss(output, y)
        opt_loss = F.mse_loss(output, y)
        
        # add regularization from https://androidkt.com/how-to-add-l1-l2-regularization-in-pytorch-loss-function/
        # #Replaces pow(2.0) with abs() for L1 regularization
        # l2_lambda = 0.001 # L2 normal will generate nan and cannot continue
        # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        # opt_loss = opt_loss + l2_lambda * l2_norm
        l1_lambda = 0.001
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        opt_loss = opt_loss + l1_lambda * l1_norm
        ### 

        opt_loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()
        cur_loss += loss.item()

        if batch_idx % log_interval == 0:
            cur_loss = cur_loss / log_interval
            processed = min(i+batch_size, X_data.size(0))
            if verbose: 
                print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, X_data.size(0), 100.*processed/X_data.size(0), lr, cur_loss))
            cur_loss = 0
    
    mae = total_loss*batch_size/X_data.size(0)
    if verbose:
        print(f'\nTrain_MAE: {mae:.2f}\n')
    return mae


def test(device, X_data, Y_data, data_name, batch_size, verbose=False):
    model.eval()
    Y_pred = np.asarray([])
    with torch.no_grad():
        batch_idx = 1
        total_loss = 0
        for i in range(0, X_data.size(0), batch_size):
            if i + batch_size > X_data.size(0):
                x, y = X_data[i:], Y_data[i:]
            else:
                x, y = X_data[i:(i+batch_size)], Y_data[i:(i+batch_size)]
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = F.l1_loss(output, y)
            batch_idx += 1
            total_loss += loss.item()
            output = output.cpu().detach().numpy()
            if i == 0:
                Y_pred = output
            else:
                Y_pred = np.vstack((Y_pred, output))

    mae = total_loss*batch_size/X_data.size(0)
    
    Y_gt = Y_data.cpu().detach().numpy()
    plot_2vectors(Y_gt, Y_pred, data_name, mae)
    if verbose: 
        print(f'\nTest_MAE: {mae:.2f}\n')
        print(f'Saved plot to {data_name}.png')

    return mae

def find_best_device():
    if not torch.cuda.is_available():
        return "cpu"
    # elif not args.cuda:
    #     print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    #     return "cpu"

    nvidia_smi.nvmlInit()
    best_gpu_id = 0
    best_free = 0 
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
        if info.free > best_free:
            best_free = info.free
            best_gpu_id = i
    nvidia_smi.nvmlShutdown()
    print(f"Best GPU to use is cuda:{best_gpu_id}!")
    return f"cuda:{best_gpu_id}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequence Modeling - Vital Signs')
    parser.add_argument('--input_channels', type=int, default=1,
                        help='the number of input channels')   
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA (default: True)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout applied to layers (default: 0.0)')
    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clip, -1 means no clip (default: -1)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='upper epoch limit (default: 1000)')
    parser.add_argument('--ksize', type=int, default=7,
                        help='kernel size (default: 7)')
    parser.add_argument('--nhid', type=int, default=30,
                        help='number of hidden units per layer (default: 30)')
    parser.add_argument('--levels', type=int, default=8,
                        help='# of levels (default: 8)')
    parser.add_argument('--seq_len', type=int, default=1000,
                        help='sequence length (default: 1000)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval (default: 100')
    parser.add_argument('--lr', type=float, default=4e-3,
                        help='initial learning rate (default: 4e-3)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed (default: 1111)')
    parser.add_argument('--train', type=str, default='../../data/ctru_20_21_22_good_minute_none_train.npy',
                        help='the dataset to run (default: )')
    parser.add_argument('--test', type=str, default='../../data/ctru_20_21_22_good_minute_none_test.npy',
                        help='the dataset to run (default: )')
    parser.add_argument('--mode', type=str, default='fit',
                        help='the running mode (fit or predict or prefit)')
    parser.add_argument('--model_file', type=str, default='../models/bp/S1:8:7_866_3.99_4.98.pt',
                        help='the AI mode file path and name')
    parser.add_argument('--model_path', type=str, default='../models/train_model/minute',
                        help='the model path for saved models')
    parser.add_argument('--labels_list', type=str, default='ITHRSD',
                        help='the list of labels in string format')
    parser.add_argument('--label_name', type=str, default='S',
                        help='the label name to fit or predict')
    parser.add_argument('--verbose', type=int, default=1,
                        help='turn on verbose/debug print or not')
                    

    args = parser.parse_args()
    print(args)
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    input_channels = args.input_channels
    n_classes = 1
    batch_size = args.batch_size
    seq_length = args.seq_len
    epochs = args.epochs
    channel_sizes = [args.nhid]*args.levels
    kernel_size = args.ksize
    dropout = args.dropout
    if args.verbose == 1:
        verbose = True
    else:
        verbose = False

    print("Producing data...")
        
    if args.mode == 'fit' or 'prefit':
        X_train, Y_train = data_generator(args.train, seq_length, args.label_name, args.labels_list)
        X_train = preprocess_data (X_train, args.label_name)
        X_train, Y_train = tensor_converter(X_train, Y_train, input_channels)

    X_test, Y_test = data_generator(args.test, seq_length, args.label_name, args.labels_list)
    X_test = preprocess_data (X_test, args.label_name)
    X_test, Y_test = tensor_converter(X_test, Y_test, input_channels)

    device = torch.device(find_best_device())
    print(f"The program will run on {device}!")

    if args.mode == 'fit' or args.mode == 'prefit':
        print(f"Start training and saving the checkpoints at {args.model_path} ...")
        if args.mode == 'prefit': 
            print(f"Loading pre-trained model file {args.model_file} ...")
            model = torch.load(open(args.model_file, "rb"))
        else:
            model = VTCN(input_size = input_channels, output_size = n_classes, 
                num_channels = channel_sizes, kernel_size = kernel_size, dropout = dropout)
        print(model)

        lr = args.lr
        optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
        log_interval=args.log_interval

        model.to(device)

        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
            print(f"The new model path {args.model_path} is created!")
        fig_name = f"{args.label_name}"

        for ep in range(1, epochs+1):
            aloss = train(device, ep, X_train, Y_train, model, optimizer, log_interval, batch_size, args.clip, verbose)
            tloss = test(device, X_test, Y_test, fig_name, batch_size, verbose)
            print(f'Epoch: {ep},\tTrain_MAE: {aloss:.2f},\tTest_MAE: {tloss:.2f}')
            model_file = f"{args.model_path}/{args.label_name}{input_channels}:{args.levels}:{kernel_size}_{ep}_{aloss:.2f}_{tloss:.2f}.pt"

            with open(model_file, "wb") as f:
                torch.save(model, f)
                # print(f"Saved model to {model_file}!\n")
    else:
        model_file = args.model_file
        [out_name, third] = os.path.splitext(model_file)
        fig_name = f"{out_name}_{args.label_name}"
        print(f"Start testing with the model file {args.model_file} ...")
        model = torch.load(open(model_file, "rb"))
        print(model)
        model.to(device)
        tloss = test(device, X_test, Y_test, fig_name, batch_size, verbose)