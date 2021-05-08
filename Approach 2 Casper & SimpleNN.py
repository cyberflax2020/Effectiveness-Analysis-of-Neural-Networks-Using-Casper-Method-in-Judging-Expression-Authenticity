import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import nn
import matplotlib.pyplot as plt
import sys

'''
Yuliang Ma
u6462980

Dataset: Anger
Approach: Casper

This code uses the second data processing method mentioned in the paper which performs worse then the first approach, 
each video containing 20 frames is regarded as a data point, and the label is determined based on all the frames in 
the video. 
'''

'''
Part - 0: A simple report generator which saves all console info in to a .txt file
'''


class Logger(object):
    def __init__(self, fileN="default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger("running_report_V2.txt")

'''
Part - 1: Data preprocessing
'''

# load in data from .xlsx file
data = pd.read_excel(r'Anger.xlsx', sheet_name=0, header=0)

# delete the redundant columns, leaving only pure data, and set column names
coln = list(data)
coln[0] = "0"
data.columns = coln
data = data.drop(labels="0", axis=1)

# Labelling the labels using 0 or 1
le = preprocessing.LabelEncoder()
le.fit(data["Label"].values.tolist())
integer_mapping = {l: i for i, l in enumerate(le.classes_)}
data["Label"] = le.transform(data["Label"].values.tolist())

# check the labels
print("The labels have been set as: ", integer_mapping)


# this part is a different approach of data processing
df_T1 = data.loc[data["Video"] == 'T1'].sample(frac=1).reset_index(drop=True).values
df_T2 = data.loc[data["Video"] == 'T2'].sample(frac=1).reset_index(drop=True).values
df_T3 = data.loc[data["Video"] == 'T3'].sample(frac=1).reset_index(drop=True).values
df_T4 = data.loc[data["Video"] == 'T4'].sample(frac=1).reset_index(drop=True).values
df_T5 = data.loc[data["Video"] == 'T5'].sample(frac=1).reset_index(drop=True).values
df_T6 = data.loc[data["Video"] == 'T6'].sample(frac=1).reset_index(drop=True).values
df_T7 = data.loc[data["Video"] == 'T7'].sample(frac=1).reset_index(drop=True).values
df_T8 = data.loc[data["Video"] == 'T8'].sample(frac=1).reset_index(drop=True).values
df_T9 = data.loc[data["Video"] == 'T9'].sample(frac=1).reset_index(drop=True).values
df_T10 = data.loc[data["Video"] == 'T10'].sample(frac=1).reset_index(drop=True).values

df_F1 = data.loc[data["Video"] == 'F1'].sample(frac=1).reset_index(drop=True).values
df_F2 = data.loc[data["Video"] == 'F2'].sample(frac=1).reset_index(drop=True).values
df_F3 = data.loc[data["Video"] == 'F3'].sample(frac=1).reset_index(drop=True).values
df_F4 = data.loc[data["Video"] == 'F4'].sample(frac=1).reset_index(drop=True).values
df_F5 = data.loc[data["Video"] == 'F5'].sample(frac=1).reset_index(drop=True).values
df_F6 = data.loc[data["Video"] == 'F6'].sample(frac=1).reset_index(drop=True).values
df_F7 = data.loc[data["Video"] == 'F7'].sample(frac=1).reset_index(drop=True).values
df_F8 = data.loc[data["Video"] == 'F8'].sample(frac=1).reset_index(drop=True).values
df_F9 = data.loc[data["Video"] == 'F9'].sample(frac=1).reset_index(drop=True).values
df_F10 = data.loc[data["Video"] == 'F10'].sample(frac=1).reset_index(drop=True).values

data_T1 = []
for i in range(20):
    for j in range(1, len(df_T1[i]) - 1):
        data_T1.append(df_T1[i][j])
data_T1.append(df_T1[0][-1])

data_T2 = []
for i in range(20):
    for j in range(1, len(df_T2[i]) - 1):
        data_T2.append(df_T2[i][j])
data_T2.append(df_T2[0][-1])

data_T3 = []
for i in range(20):
    for j in range(1, len(df_T3[i]) - 1):
        data_T3.append(df_T3[i][j])
data_T3.append(df_T3[0][-1])

data_T4 = []
for i in range(20):
    for j in range(1, len(df_T4[i]) - 1):
        data_T4.append(df_T4[i][j])
data_T4.append(df_T4[0][-1])

data_T5 = []
for i in range(20):
    for j in range(1, len(df_T5[i]) - 1):
        data_T5.append(df_T5[i][j])
data_T5.append(df_T5[0][-1])

data_T6 = []
for i in range(20):
    for j in range(1, len(df_T6[i]) - 1):
        data_T6.append(df_T1[i][j])
data_T6.append(df_T1[0][-1])

data_T7 = []
for i in range(20):
    for j in range(1, len(df_T7[i]) - 1):
        data_T7.append(df_T7[i][j])
data_T7.append(df_T7[0][-1])

data_T8 = []
for i in range(20):
    for j in range(1, len(df_T8[i]) - 1):
        data_T8.append(df_T8[i][j])
data_T8.append(df_T8[0][-1])

data_T9 = []
for i in range(20):
    for j in range(1, len(df_T9[i]) - 1):
        data_T9.append(df_T9[i][j])
data_T9.append(df_T9[0][-1])

data_T10 = []
for i in range(20):
    for j in range(1, len(df_T10[i]) - 1):
        data_T10.append(df_T10[i][j])
data_T10.append(df_T10[0][-1])

data_F1 = []
for i in range(20):
    for j in range(1, len(df_F1[i]) - 1):
        data_F1.append(df_F1[i][j])
data_F1.append(df_F1[0][-1])

data_F2 = []
for i in range(20):
    for j in range(1, len(df_F2[i]) - 1):
        data_F2.append(df_F2[i][j])
data_F2.append(df_F2[0][-1])

data_F3 = []
for i in range(20):
    for j in range(1, len(df_F3[i]) - 1):
        data_F3.append(df_F3[i][j])
data_F3.append(df_F3[0][-1])

data_F4 = []
for i in range(20):
    for j in range(1, len(df_F4[i]) - 1):
        data_F4.append(df_F4[i][j])
data_F4.append(df_F4[0][-1])

data_F5 = []
for i in range(20):
    for j in range(1, len(df_F5[i]) - 1):
        data_F5.append(df_F5[i][j])
data_F5.append(df_F5[0][-1])

data_F6 = []
for i in range(20):
    for j in range(1, len(df_F6[i]) - 1):
        data_F6.append(df_F1[i][j])
data_F6.append(df_F1[0][-1])

data_F7 = []
for i in range(20):
    for j in range(1, len(df_F7[i]) - 1):
        data_F7.append(df_F7[i][j])
data_F7.append(df_F7[0][-1])

data_F8 = []
for i in range(20):
    for j in range(1, len(df_F8[i]) - 1):
        data_F8.append(df_F8[i][j])
data_F8.append(df_F8[0][-1])

data_F9 = []
for i in range(20):
    for j in range(1, len(df_F9[i]) - 1):
        data_F9.append(df_F9[i][j])
data_F9.append(df_F9[0][-1])

data_F10 = []
for i in range(20):
    for j in range(1, len(df_F10[i]) - 1):
        data_F10.append(df_F10[i][j])
data_F10.append(df_F10[0][-1])

data1 = pd.DataFrame(np.array(data_T1).reshape(1, 121))
data2 = pd.DataFrame(np.array(data_T2).reshape(1, 121))
data3 = pd.DataFrame(np.array(data_T3).reshape(1, 121))
data4 = pd.DataFrame(np.array(data_T4).reshape(1, 121))
data5 = pd.DataFrame(np.array(data_T5).reshape(1, 121))
data6 = pd.DataFrame(np.array(data_T6).reshape(1, 121))
data7 = pd.DataFrame(np.array(data_T7).reshape(1, 121))
data8 = pd.DataFrame(np.array(data_T8).reshape(1, 121))
data9 = pd.DataFrame(np.array(data_T9).reshape(1, 121))
data10 = pd.DataFrame(np.array(data_T10).reshape(1, 121))
data11 = pd.DataFrame(np.array(data_F1).reshape(1, 121))
data12 = pd.DataFrame(np.array(data_F2).reshape(1, 121))
data13 = pd.DataFrame(np.array(data_F3).reshape(1, 121))
data14 = pd.DataFrame(np.array(data_F4).reshape(1, 121))
data15 = pd.DataFrame(np.array(data_F5).reshape(1, 121))
data16 = pd.DataFrame(np.array(data_F6).reshape(1, 121))
data17 = pd.DataFrame(np.array(data_F7).reshape(1, 121))
data18 = pd.DataFrame(np.array(data_F8).reshape(1, 121))
data19 = pd.DataFrame(np.array(data_F9).reshape(1, 121))
data20 = pd.DataFrame(np.array(data_F10).reshape(1, 121))

data = pd.concat(
    [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15,
     data16, data17, data18, data19, data20])

# data normalization
for i in range(119):
    data[i] = (data[i] - data[i].mean()) / data[i].std()

data.columns = list(map(str, list(range(121))))

# train, validation, test set split

# the 1st approach: using train_test_split with a random state

# df_train, df_test_vali = train_test_split(data, train_size=0.8)
# df_vali, df_test = train_test_split(df_test_vali, train_size=0.5)

# the 2nd approach: construction balanced datasets manually:

# extract all data with the same label:
df_G = data.loc[data["120"] == 1].sample(frac=1).reset_index(drop=True)
df_F = data.loc[data["120"] == 0].sample(frac=1).reset_index(drop=True)

# reconstruct the dataset and let the two labels alternate
data = data.drop(index=data.index)
for i in range(len(df_G)):
    data.loc[2 * i] = df_G.loc[i]
    data.loc[2 * i + 1] = df_F.loc[i]

# set params: tr_vte for train size/(vali size + test size), v_te for vali size / test size
tr_vte = 0.8
v_te = 0.9
data_size = len(data)

# preform the split, note that only in train set, keep the labels alternate
df_train = data.loc[0:tr_vte * data_size - 1].reset_index(drop=True)
df_vali = data.loc[tr_vte * data_size:v_te * data_size - 1].sample(frac=1, random_state=None).reset_index(drop=True)
df_test = data.loc[v_te * data_size:data_size].sample(frac=1, random_state=None).reset_index(drop=True)

# train set:
n_features = df_train.shape[1] - 1
train_input = df_train.iloc[:, :n_features]
train_target = df_train.iloc[:, n_features]
X = torch.Tensor(train_input.values).float()
Y = torch.Tensor(train_target.values).long()

# validation set:
vali_input = torch.Tensor(df_vali.iloc[:, :n_features].values).float()
vali_target = torch.Tensor(df_vali.iloc[:, n_features].values).long()

# test set:
test_input = torch.Tensor(df_test.iloc[:, :n_features].values).float()
test_target = torch.Tensor(df_test.iloc[:, n_features].values).long()

final_acc_Casper = 0
final_acc_SimpleNN = 0

'''
Part - 2: Parameter setting
'''

# set regular params:
n_input = n_features;
n_output = 2
n_epochs = 1500
# n_epochs = 5000
batch_size = 2

# p is used to determine the interval between checkpoints while training, larger p results in less checkpoints.
p = 5
# p = 3

# k is the maximum number of hidden neurons, once Casper's hidden units is about to reach to k+1, training stops.
k = 20
# k = 25

# 3 different learning rates for different parts of the network according to the definition of Casper
l1 = 0.02
l2 = 0.005
l3 = 0.0001

'''
Part - 3: Building model - Casper network
'''


class Casper(nn.Module):

    # initialize a fully connected neural network
    def __init__(self):
        super().__init__()
        self.hidden_counts = 0
        self.hiddens = nn.ModuleList()
        self.outputs = nn.ModuleList()
        # set components inside every unit : just a simple linear function
        self.function = nn.Sequential(nn.Linear(n_input, n_output))

    # a function that adds one neuron to the existing network
    def add_neuron(self):
        self.hiddens.append(nn.Sequential(
            nn.Dropout(p=0.3),  # set it to have a 30% probability of being muted to improve the accuracy of the model
            nn.Linear(n_input + self.hidden_counts, 1),  # linked with the previous added neurons and the input neurons
            # nn.Sigmoid(),
            # nn.ReLU(),
            nn.LeakyReLU()  # activation function
        ))
        self.outputs.append(nn.Sequential(
            nn.Linear(1, n_output)  # linked with the out put neurons
        ))
        self.hidden_counts += 1

    # once a new neuron is added, the optimizer with a learning rate setting should be updated
    # this function returns an updated optimizer
    def update_lr_opt(self):
        # if this is the first neuron to be added, return the initialized the optimizer with l1
        if self.hidden_counts - 1 == 0:
            optimizer = torch.optim.RMSprop(self.parameters(), lr=l1, momentum=0.1)
        else:
            # this part is to decide which parts should apply what learning rates according to the Casper definition
            all_paras = list(self.parameters())
            l1_paras = list(self.hiddens[-1].parameters())
            l2_paras = list(self.outputs[-1].parameters())
            l3_paras = []
            idx = []
            for para in l1_paras:
                idx.append(id(para))
            for para in l2_paras:
                idx.append(id(para))
            for para in all_paras:
                if id(para) not in idx:
                    l3_paras.append(para)

            # update the optimizer using the new learning rate setting
            optimizer = torch.optim.RMSprop(
                [{'params': l1_paras, 'lr': l1}, {'params': l2_paras, 'lr': l2}, {'params': l3_paras, 'lr': l3}], lr=l1,
                momentum=0.1)

        return optimizer

    def forward(self, x):
        # first calculate from the input layer to the hidden layer
        hidden_out = []
        for i in range(self.hidden_counts):
            if len(hidden_out) != 0:
                hidden_out.append(self.hiddens[i](torch.cat([x] + hidden_out[:i], dim=1)))
            else:
                hidden_out.append(self.hiddens[i](x))

        # then calculate from the hidden layer to the output layer
        out = [self.function(x)]
        for i in range(self.hidden_counts):
            out.append(self.outputs[i](hidden_out[i]))
        sum_output = sum(out)

        # note that we use CrossEntropyLoss as loss to do classification, so no need to apply softmax.
        return sum_output


'''
Part - 4: Training and validation  - Casper network
'''

# initialize the network and some basic parameters
net = Casper()
optimizer = torch.optim.RMSprop(net.parameters(), lr=l1, momentum=0.1)
checkpoint = 15 + p * net.hidden_counts
batch_idx = list(range(batch_size))
previous_loss = float('inf')  # used to decided when to add a new neuron
stop = False  # a simple indicator used to decided when to stop training
final_epoch = 0

# data records used to plot figures
plt_loss_train = []  # train loss
plt_loss_vali1 = []  # another version of train loss used for comparison with validation loss
plt_loss_vali2 = []  # validation loss

plt_loss_train.append(previous_loss)
plt_loss_vali1.append(previous_loss)
plt_loss_vali2.append(previous_loss)

print(
    'Training for the Casper network starts, params have been set as: maximum hidden neurons: %d, maximum epochs: %d, p: %d' % (
        k, n_epochs, p))

# epoch training
for epoch in range(n_epochs):

    # initialize records
    total = 0
    correct = 0
    total_loss = 0

    for i in range(int(len(X) / batch_size)):

        # select some data form training set according to batch_size
        idx = []
        for j in batch_idx:
            idx.append(i + j)
        x = X[idx]
        y = Y[idx]

        # some regular processes
        optimizer.zero_grad()
        output = net(x)
        loss = nn.CrossEntropyLoss()(output, y)  # use CrossEntropyLoss, no need to apply softmax
        loss.backward()
        optimizer.step()

        # if this epoch is a checkpoint, record the loss and accuracy
        if epoch == checkpoint:
            _, predicted = torch.max(output, 1)
            total = total + predicted.size(0)
            correct = correct + sum(predicted.data.numpy() == y.data.numpy())
            total_loss = total_loss + loss

    # also if this epoch is a checkpoint, check if new neuron should be added
    if epoch == checkpoint:

        # update the checkpoint
        N = net.hidden_counts
        checkpoint += 15 + p * N

        # once Casper's hidden units is about to reach to k+1, training stops.
        if net.hidden_counts == k:
            stop = True
            final_epoch = epoch

        # add a neuron when the loss is not decreasing enough (l3)
        if previous_loss <= total_loss + l3:
            if stop:
                break

            # the training continue, neuron is about to be added, but we should first validating the loss.
            else:
                all_vali_loss = 0
                for i in range(int(len(vali_input) / batch_size)):

                    # get some data from the validation set
                    idx = []
                    for j in batch_idx:
                        idx.append(i + j)
                    x_vali = vali_input[idx]
                    y_vali = vali_target[idx]

                    # preform the validation and record the loss
                    y_pred_vali = net(x_vali)
                    vali_loss = nn.CrossEntropyLoss()(y_pred_vali, y_vali)
                    all_vali_loss = all_vali_loss + vali_loss

                # save the loss for plotting
                plt_loss_vali1.append(all_vali_loss.item())
                plt_loss_vali2.append(total_loss.item())

                # after validation, add a new neuron and get the new optimizer
                net.add_neuron()
                optimizer = net.update_lr_opt()

        # save the loss for plotting
        plt_loss_train.append(total_loss.item())

        # update the previous loss
        previous_loss = total_loss

        print('The number of hidden neurons in the Casper network is now: ', net.hidden_counts)
        print('Casper network: Epoch [%d/%d], Training loss: %.4f, Training accuracy: %.2f %%'
              % (epoch + 1, n_epochs,
                 total_loss, 100 * correct / total))

# training ends
print('The training for the Casper network is finished at epoch %d, testing starts' % final_epoch)

# plot 1st figure: loss changes with the increase of checkpoints
plt.figure()
plt.plot(plt_loss_train)
plt.xlabel('checkpoints')
plt.ylabel('loss')
plt.show()
print('A figure of training loss has been generated')

# plot 2nd figure: the validation loss and training loss change with the increase of hidden neurons added
# this picture is used to determine the setting of the hyper parameter k
axis = []
for i in range(len(plt_loss_vali1)):
    axis.append(list(range(len(plt_loss_vali1)))[i] + 1)

plt.plot(axis, plt_loss_vali1, linestyle="--", label="vali loss")
plt.plot(axis, plt_loss_vali2, label="train loss")
plt.xlabel('added neuron')
plt.ylabel('loss')
plt.legend()
plt.show()

# validation ends
print('A figure of validation loss has been generated')

'''
Part - 5: Testing and evaluation  - Casper network
'''

all_test_pred = []
all_test_loss = 0
correct_test = 0
for i in range(int(len(test_input) / batch_size)):

    # get some data from testing set
    idx = []
    for j in batch_idx:
        idx.append(i + j)
    x_test = test_input[idx]
    y_test = test_target[idx]

    # regular processes
    y_pred_test = net(x_test)
    test_loss = nn.CrossEntropyLoss()(y_pred_test, y_test)
    all_test_loss = all_test_loss + test_loss

    _, predicted_test = torch.max(y_pred_test, 1)
    for j in range(len(predicted_test.data)):
        all_test_pred.append(predicted_test.data[j])

    correct_test = correct_test + sum(predicted_test.data.numpy() == y_test.data.numpy())


# this part is inspired by the COMP8420 Lab 2 code, it is used to generate a confusion matrix
def plot_confusion(input_sample, num_classes, des_output, actual_output):
    confusion = torch.zeros(num_classes, num_classes)
    for i in range(input_sample):
        actual_class = actual_output[i]
        predicted_class = des_output[i]

        confusion[actual_class][predicted_class] += 1

    return confusion


final_acc_Casper = 100 * correct_test / len(test_input)

print("The confusion matrix of testing result of the Casper network is:")
print(plot_confusion(len(test_target.data), n_output, all_test_pred, test_target.data))
print('Casper network: Testing loss: %.4f, Testing accuracy: %.2f %%' % (
    all_test_loss, final_acc_Casper))
print('Testing for the Casper network is finished')

'''
Part - 6: Building model  - A simple fully connected 2-layer NN For comparison
'''

# this simple shares almost all params setting with the Casper model and uses the same datasets for training and testing
lr = l1
n_hidden = k  # same number of hidden neurons with the Casper network
n_epochs = 100  # since the Casper algorithm usually reaches the upper limit of hidden neurons before going through


# all epochs and stops, therefore set a smaller number of epochs here


class FCN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(FCN, self).__init__()
        self.lf1 = nn.Linear(n_input, n_hidden)
        # self.sigmoid = nn.Sigmoid()
        self.LeakyRelu = nn.LeakyReLU()  # same activation function with the Casper network
        self.lf2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        out = nn.Dropout(p=0.3)(x)  # same dropping rate with the Casper network
        out = self.lf1(out)
        # out = self.sigmoid(out)
        out = self.LeakyRelu(out)
        out = self.lf2(out)
        return out


'''
Part - 7: Training - A simple fully connected 2-layer NN For comparison
'''

# just some regular processes
net = FCN(n_input, n_hidden, n_output)
loss_func = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(net.parameters(), lr=lr)
all_losses = []

print('Training for the simple NN starts, params have been set as: hidden neurons: %d, total epochs: %d' % (
    k, n_epochs))

# epoch training, the number of epochs is the same with the Casper network
for epoch in range(n_epochs):
    Y_pred = net(X)
    loss = loss_func(Y_pred, Y)
    all_losses.append(loss.item())

    if epoch % 50 == 0:
        _, predicted = torch.max(Y_pred, 1)

        # calculate and print accuracy
        total = predicted.size(0)
        correct = predicted.data.numpy() == Y.data.numpy()

        print('Simple NN: Epoch [%d/%d] Training loss: %.4f  Training accuracy: %.2f %%'
              % (epoch + 1, n_epochs, loss.item(), 100 * sum(correct) / total))

    net.zero_grad()
    loss.backward()
    optimiser.step()

# plot the training loss
plt.figure()
plt.plot(all_losses)
plt.show()

print('A figure of training loss has been generated')
print('The training for the simple NN is finished, testing starts')

'''
Part - 8: Testing and evaluation - A simple fully connected 2-layer NN For comparison
'''

outputs = net(test_input)
_, predicted = torch.max(outputs, 1)
test_loss = nn.CrossEntropyLoss()(output, test_target)
total = predicted.size(0)
correct = predicted.data.numpy() == test_target.data.numpy()

final_acc_SimpleNN = 100 * sum(correct) / total

print('The confusion matrix of testing result of the simple NN is')
print(plot_confusion(test_input.shape[0], n_output, predicted.long().data, test_target.data))
print('Simple NN: Testing loss: %.4f, Testing accuracy: %.2f %%' % (test_loss, final_acc_SimpleNN))
print('Testing for simple NN is finished')

'''
Part - 9: Comparison
'''

print(
    "Comparison: for the same dataset and setting, the testing accuracies are: Casper network: %.2f %%, Simple NN: "
    "%.2f %%" % (
        final_acc_Casper, final_acc_SimpleNN))

print('All console info has been saved into running_report_V2.txt')
