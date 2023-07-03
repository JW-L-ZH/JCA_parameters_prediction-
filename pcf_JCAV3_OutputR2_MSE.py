# Simple introduction for model construction https://www.pythonf.cn/read/72352

import time
start_time = time.time()
# print('start_time: ', start_time)

#import PySimpleGUI as sg

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
# import matplotlib.backends.tkagg as tkagg
import tkinter as Tk


import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sys
import pickle
import torch
from torch import nn, optim
from torchvision import transforms
from collections import OrderedDict


# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## Transforms features by scaling each feature to a given range.
## This estimator scales and translates each feature individually such that it is in the given range on the training set, i.e. between zero and one.
## This transformation is often used as an alternative to zero mean, unit variance scaling.
## fit(X[, y])	Compute the minimum and maximum to be used for later scaling.
## transform(X)	Scaling features of X according to feature_range.
## fit_transform(X[, y])	Fit to data, then transform it.
## inverse_transform(X)	Undo the scaling of X according to feature_range.
scaler1 = MinMaxScaler()  
scaler2 = MinMaxScaler()  

no_of_output_nodes = 5
# porosity, flow resistance, tortuosity, characteristic viscous length, characteristic thermal length

df_1 = pd.read_excel('JCA_data_array_ForTrainingV3.xlsx', sheet_name='sheet')
datafile_1 = df_1.values                  ## stored data from xlsx file
print(datafile_1.shape)

# print(datafile_1)
# print(len(datafile_1))
# print()


########   just to see output variable values   ##########
out_var_datafile_1 = datafile_1[:,range(2,7)]      ## range(2,7) 2 3 4 5 6           stored output_variable (4th column) from xlsx file
print(out_var_datafile_1.shape)
out_var_datafile_1 = out_var_datafile_1.reshape((-1,no_of_output_nodes))    ## one column with unknown no. of rows
print(out_var_datafile_1.shape)
print(out_var_datafile_1)
print('no. of training points: ', len(out_var_datafile_1))

scaler1.fit(datafile_1)
scaler2.fit(out_var_datafile_1)


scaler_datafile_1 = scaler1.transform(datafile_1)
X = scaler_datafile_1[:,range(0,2)]                 ## input variables columns
y = scaler_datafile_1[:,range(2,7)]                ## output variables columns

# print(X)
# print()
# print(y)

X, y = shuffle(X, y)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1) # test_size: percentage of data sepeartion
X_train = X_train.reshape(-1, 2)  ## 2nd column value is = no. of input variables columns
y_train = y_train.reshape(-1, no_of_output_nodes)  ## 2nd column value is = no. of output variables columns
X_validation = X_validation.reshape(-1, 2)  ## 2nd column value is = no. of input variables columns
y_validation = y_validation.reshape(-1, no_of_output_nodes)  ## 2nd column value is = no. of output variables columns
print('no. of training points: ', len(X_train))
print('no. of validation points: ', len(X_validation))

###########     manual testing    #########
df_2 = pd.read_excel('JCA_data_array_manual.xlsx', sheet_name='JCA_data_array_manual')
datafile_2 = df_2.values  ## stored data from xlsx file
print(datafile_2)
scaler_datafile_2 = scaler1.transform(datafile_2)
X_test = scaler_datafile_2[:, range(0, 2)]  ## input variables columns
y_test = scaler_datafile_2[:, range(2, 7)]  ## output variables columns
print(X_test)
print()
print(y_test)
print('no. of test points: ', len(X_test))
X_test = X_test.reshape(-1, 2)  ## 2nd column value is = no. of input variables columns
y_test = y_test.reshape(-1, no_of_output_nodes)  ## 2nd column value is = no. of output variables columns
###########################################


input_dim = 2  ## = no. of input variables columns
output_dim = no_of_output_nodes  ## = no. of output variables columns
from collections import OrderedDict

# ############     model without dropout     #####################
# nodes_hidden_1 = 20
# nodes_hidden_2 = 20
# ## nn.Linear() is fully connected layer
# model = nn.Sequential(OrderedDict([
#                         ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
#                         ('relu', nn.ReLU()),
#                         ('fc2', nn.Linear(nodes_hidden_1, nodes_hidden_2)),
#                         ('relu', nn.ReLU()),
#                         ('fc3', nn.Linear(nodes_hidden_2, output_dim)),
#                         ]))


############     model with dropout - 3 layers    #####################
####             dropout_prob leads to variations in mse curve      #########
# construct training model by using pytorch
dropout_prob = 0.1
num=150
nodes_hidden_1 = num
nodes_hidden_2 = num
nodes_hidden_3 = num
nodes_hidden_4 = num
nodes_hidden_5 = num
f = open("L3N150_R2_MSE_Output_20210721.dat", 'w')  # 只读方式打开同目录下的text.txt
## nn.Linear() is fully connected layer


# 5layer
# model = nn.Sequential(OrderedDict([
#     ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc2', nn.Linear(nodes_hidden_1, nodes_hidden_2)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc3', nn.Linear(nodes_hidden_2, nodes_hidden_3)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc4', nn.Linear(nodes_hidden_3, nodes_hidden_4)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc5', nn.Linear(nodes_hidden_4, nodes_hidden_5)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc6', nn.Linear(nodes_hidden_5, output_dim)),
# ]))


# 4layer
# model = nn.Sequential(OrderedDict([
#     ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc2', nn.Linear(nodes_hidden_1, nodes_hidden_2)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc3', nn.Linear(nodes_hidden_2, nodes_hidden_3)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc4', nn.Linear(nodes_hidden_3, nodes_hidden_4)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc5', nn.Linear(nodes_hidden_4, output_dim)),
# ]))


# 3layer
model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(dropout_prob)),
    ('fc2', nn.Linear(nodes_hidden_1, nodes_hidden_2)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(dropout_prob)),
    ('fc3', nn.Linear(nodes_hidden_2, nodes_hidden_3)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(dropout_prob)),
    ('fc4', nn.Linear(nodes_hidden_3, output_dim)),
]))

# 2layer
# model = nn.Sequential(OrderedDict([
#     ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc2', nn.Linear(nodes_hidden_1, nodes_hidden_2)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc3', nn.Linear(nodes_hidden_2, output_dim)),
# ]))

# 1layer
# model = nn.Sequential(OrderedDict([
#     ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
#     ('relu', nn.ReLU()),
#     ('dropout', nn.Dropout(dropout_prob)),
#     ('fc2', nn.Linear(nodes_hidden_1, output_dim)),
# ]))


# ############     model with dropout - 2 layers     #####################
# ####             dropout_prob leads to variations in mse curve    ###########
# dropout_prob = 0.1           # 0.5 - used in nvidia model-behavioural cloning
# nodes_hidden_1 = 50
# nodes_hidden_2 = 50
# ## nn.Linear() is fully connected layer
# model = nn.Sequential(OrderedDict([
#                         ('fc1', nn.Linear(input_dim, nodes_hidden_1)),
#                         ('relu', nn.ReLU()),
#                         ('dropout', nn.Dropout(dropout_prob)),
#                         ('fc2', nn.Linear(nodes_hidden_1, nodes_hidden_2)),
#                         ('relu', nn.ReLU()),
#                         ('dropout', nn.Dropout(dropout_prob)),
#                         ('fc3', nn.Linear(nodes_hidden_2, output_dim)),
#                         ]))


print(model)
# model.double()
# print(X_train)
print(X_train.shape, y_train.shape)

criterion = nn.MSELoss()
learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(device)
## move model to gpu if available, else cpu
# model.to(device)


epochs = 5000  #
# Convert numpy array to torch Variable
# inputs = torch.from_numpy(X_train).requires_grad_()
# labels = torch.from_numpy(y_train)
inputs = torch.Tensor((X_train))
labels = torch.Tensor((y_train))
inputs_validation = torch.Tensor((X_validation))
labels_validation = torch.Tensor((y_validation))
running_loss = []
running_loss_validation = []
for epoch in range(epochs):
    epoch += 1

    #################   train the model   ######################
    model.train()  # prep model for training
    # Clear gradients w.r.t. parameters, else gradients will be added up with every previous pass
    optimizer.zero_grad()
    # Forward to get output
    outputs = model(inputs)
    # Calculate Loss
    loss = criterion(outputs, labels)  ## mean squared error
    # Getting gradients w.r.t. parameters
    loss.backward()
    # Updating parameters
    optimizer.step()  ## take a step with optimizer to update the weights
    running_loss.append(loss.item())

    # ###############    validate the model (not showing fluctuations)      ###################
    # # Turn off gradients for validation, saves memory and computations
    # with torch.no_grad():
    #     ## this turns off dropout for evaluation mode of model
    #     model.eval()      # prep model for evaluation
    #     outputs_validation = model(inputs_validation)
    #     loss_validation = criterion(outputs_validation, labels_validation)
    #     running_loss_validation.append(loss_validation.item())

    # ###############    validate the model (showing fluctuations)      ###################
    outputs_validation = model(inputs_validation)
    loss_validation = criterion(outputs_validation, labels_validation)
    running_loss_validation.append(loss_validation.item())

    print('epoch: {}, mse_loss: {:.6f}, mse_loss_validation: {:.6f}'.format(epoch, loss.item(), loss_validation.item()))
    # print(mean_squared_error(outputs_validation,labels_validation))

    # if (epoch == 1000):
    #     torch.save(model.state_dict(), 'checkpoint_1000.pth')
    # elif (epoch == 2500):
    #     torch.save(model.state_dict(), 'checkpoint_2500.pth')
    # elif (epoch == 5000):
    #     torch.save(model.state_dict(), 'checkpoint_5000.pth')
    # elif (epoch == 7500):
    #     torch.save(model.state_dict(), 'checkpoint_7500.pth')
    # elif (epoch == 10000):
    #     torch.save(model.state_dict(), 'checkpoint_10000.pth')
    # elif (epoch == 12500):
    #     torch.save(model.state_dict(), 'checkpoint_12500.pth')
    # elif (epoch == 15000):
    #     torch.save(model.state_dict(), 'checkpoint_15000.pth')

# save the model, as weights & parameters are stored in model.state_dict()
# print(model.state_dict().keys())
# print(model.state_dict())
#### torch.save(model.state_dict(), 'checkpoint-epochs-{}.pth'.format(epochs))
torch.save(model.state_dict(), 'model.pkl')
# # load the saved model at particular epochs to compare
state_dict = torch.load('model.pkl')
# load the saved model
#### state_dict = torch.load('checkpoint-epochs-{}.pth'.format(epochs))
# state_dict = torch.load('checkpoint.pth')
# state_dict = torch.load('checkpoint-simple_waveguide_neff_pytorch_1_epochs-5000.pth')
model.load_state_dict(state_dict)

# Purely inference
# predicted_on_X_train = model(torch.Tensor(X_train).requires_grad_()).data.numpy()
# predicted_on_X_validation = model(torch.Tensor(X_validation).requires_grad_()).data.numpy()
# predicted_on_X_test = model(torch.Tensor(X_test).requires_grad_()).data.numpy()
with torch.no_grad():
    ## this turns off dropout for evaluation mode of model
    model.eval()
    predicted_on_X_train = model(torch.Tensor(X_train)).data.numpy()
    predicted_on_X_validation = model(torch.Tensor(X_validation)).data.numpy()
    predicted_on_X_test = model(torch.Tensor(X_test)).data.numpy()
    # print(predicted)

end_time = time.time()
print('end_time: ', end_time)
print('time taken to train in sec: ', (end_time - start_time))

## make axis bold
plt.rcParams.update({'font.size': 10})
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

mse_training_interval = 10
mse_validation_interval = 10
running_loss = running_loss[::mse_training_interval]
running_loss_index = [i for i in range(1, epochs, mse_training_interval)]
running_loss_validation = running_loss_validation[::mse_validation_interval]
running_loss_validation_index = [i for i in range(1, epochs, mse_validation_interval)]
print('mse lengths: ', len(running_loss), len(running_loss_validation))
# print('running_loss_index: ', running_loss_index)
# print('running_loss_validation_index: ', running_loss_validation_index)


###############################################################
#################   plotting graphs together - porosity  ################
###############################################################

plt.figure()
plt.suptitle('JCA - porosity - (epochs-{}) - pyTorch'.format(epochs), fontsize=25,
             color='r', fontweight='bold')  ## giving title on top of all subplots

plt.subplot(231)
plt.plot(running_loss_index, running_loss, 'r-', linewidth=3, label='mse_loss_train')
plt.plot(running_loss_validation_index, running_loss_validation, 'b-', linewidth=3, label='mse_loss_validation')
plt.legend(loc='best', fontsize=10)
plt.xlabel('epochs#', fontsize=15)

# plt.figure()
plt.subplot(232)
# Plot true data
plt.plot(scaler2.inverse_transform(y_train)[:, 0], 'ro', markersize=12, label='y_train')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_train)[:, 0], 'b*', markersize=12, label='predicted_on_X_train')
# Legend and plot
plt.legend(loc='best', fontsize=10)

# plt.figure()
plt.subplot(233)
# Plot true data
plt.plot(scaler2.inverse_transform(y_validation)[:, 0], 'ro', markersize=12, label='y_validation')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_validation)[:, 0], 'b*', markersize=12,
         label='predicted_on_X_validation')
# Legend and plot
plt.legend(loc='best', fontsize=10)

# plt.figure()
plt.subplot(234)
# Plot true data
plt.plot(scaler2.inverse_transform(y_test)[:, 0], 'ro', markersize=12, label='y_test')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_test)[:, 0], 'b*', markersize=12, label='predicted_on_X_test')
# Legend and plot
plt.legend(loc='best', fontsize=10)

# plt.figure()
plt.subplot(235)
xx = scaler2.inverse_transform(y_train)[:, 0]
yy = scaler2.inverse_transform(predicted_on_X_train)[:, 0]
# print('JCA - porosity - r2_score')
# print(r2_score(xx, yy))  #r2_score(y_true, y_pred)

f.write("JCA - Training porosity - r2_score")
f.write('\n')
f.write(str(r2_score(xx, yy)))
f.write('\n')
f.write("JCA - Training porosity - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx, yy)))
f.write('\n')
f.write("JCA - Training porosity - root mean_squared_error") #mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx, yy))))
f.write('\n')
f.write("JCA - Training porosity - mean_absolute_error") #mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx, yy)))

xx_validation = scaler2.inverse_transform(y_validation)[:, 0]
yy_validation = scaler2.inverse_transform(predicted_on_X_validation)[:, 0]

f.write('\n')
f.write("JCA - Validation porosity - r2_score")
f.write('\n')
f.write(str(r2_score(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation porosity - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation porosity - root mean_squared_error") #mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx_validation, yy_validation))))
f.write('\n')
f.write("JCA - Validation porosity - mean_absolute_error") #mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx_validation, yy_validation)))

xx_test = scaler2.inverse_transform(y_test)[:, 0]
yy_test = scaler2.inverse_transform(predicted_on_X_test)[:, 0]
bubble_plot_line_x1y1 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
bubble_plot_line_x2y2 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx, yy, label='train', marker='o', facecolors='none', edgecolors='red', s=50)
plt.scatter(xx_validation, yy_validation, label='validation', marker='o', facecolors='none', edgecolors='blue', s=50)
plt.scatter(xx_test, yy_test, label='test', marker='o', facecolors='none', edgecolors='black', s=50)
plt.legend(loc='best', fontsize=10)
plt.xlabel('true-values', fontsize=15)
plt.ylabel('predicted', fontsize=15)

f.write('\n')
f.write("JCA - AdditionalTest porosity - r2_score")
f.write('\n')
f.write(str(r2_score(xx_test, yy_test)))
f.write('\n')
f.write("JCA - AdditionalTest porosity - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_test, yy_test)))
f.write('\n')
f.write("JCA - AdditionalTest porosity - root mean_squared_error") #mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx_test, yy_test))))
f.write('\n')
f.write("JCA - AdditionalTest porosity - mean_absolute_error") #mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx_test, yy_test)))

# plt.figure()
# plt.subplot(236)
# true_values = scaler2.inverse_transform(y_test)[:, 0]
# predicted_values = scaler2.inverse_transform(predicted_on_X_test)[:, 0]
# x_index = [i for i in range(len(true_values))]
# error_values = predicted_values - true_values
# plt.errorbar(x=x_index, y=true_values, yerr=error_values, fmt='o', color='black',
#              ecolor='black', elinewidth=2, capsize=10);
# plt.grid(linestyle='--', linewidth=1)

# print()
# print("o/p of test set:           \n", (scaler2.inverse_transform(y_test)[:, 0]))
# print("predicted o/p of test set: \n", (scaler2.inverse_transform(predicted_on_X_test)[:, 0]))
# print("mse_test_set: ", mean_squared_error(y_test, predicted_on_X_test))

###############################################################
#################   plotting graphs together - sigma  ################
###############################################################

plt.figure()
plt.suptitle('JCA - flow resistance - (epochs-{}) - pyTorch'.format(epochs), fontsize=25,
             color='r', fontweight='bold')  ## giving title on top of all subplots

plt.subplot(231)
plt.plot(running_loss_index, running_loss, 'r-', linewidth=3, label='mse_loss_train')
plt.plot(running_loss_validation_index, running_loss_validation, 'b-', linewidth=3, label='mse_loss_validation')
plt.legend(loc='best', fontsize=10)
plt.xlabel('epochs#', fontsize=15)

# plt.figure()
plt.subplot(232)
# Plot true data
plt.plot(scaler2.inverse_transform(y_train)[:, 1], 'ro', markersize=12, label='y_train')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_train)[:, 1], 'b*', markersize=12, label='predicted_on_X_train')
# Legend and plot
plt.legend(loc='best', fontsize=10)

# plt.figure()
plt.subplot(233)
# Plot true data
plt.plot(scaler2.inverse_transform(y_validation)[:, 1], 'ro', markersize=12, label='y_validation')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_validation)[:, 1], 'b*', markersize=12,
         label='predicted_on_X_validation')
# Legend and plot
plt.legend(loc='best', fontsize=10)

# plt.figure()
plt.subplot(234)
# Plot true data
plt.plot(scaler2.inverse_transform(y_test)[:, 1], 'ro', markersize=12, label='y_test')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_test)[:, 1], 'b*', markersize=12, label='predicted_on_X_test')
# Legend and plot
plt.legend(loc='best', fontsize=10)

# plt.figure()
plt.subplot(235)
xx = scaler2.inverse_transform(y_train)[:, 1]
yy = scaler2.inverse_transform(predicted_on_X_train)[:, 1]
xx_validation = scaler2.inverse_transform(y_validation)[:, 1]
yy_validation = scaler2.inverse_transform(predicted_on_X_validation)[:, 1]

f.write('\n')
f.write("JCA - Training flow resistance - r2_score")
f.write('\n')
f.write(str(r2_score(xx, yy)))
f.write('\n')
f.write("JCA - Training flow resistance - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx, yy)))
f.write('\n')
f.write("JCA - Training flow resistance - root mean_squared_error") #mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx, yy))))
f.write('\n')
f.write("JCA - Training flow resistance - mean_absolute_error") #mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx, yy)))
f.write('\n')
f.write("JCA - Validation flow resistance - r2_score")
f.write('\n')
f.write(str(r2_score(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation flow resistance - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation flow resistance - root mean_squared_error") #mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx_validation, yy_validation))))
f.write('\n')
f.write("JCA - Validation flow resistance - mean_absolute_error") #mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx_validation, yy_validation)))

xx_test = scaler2.inverse_transform(y_test)[:, 1]
yy_test = scaler2.inverse_transform(predicted_on_X_test)[:, 1]
bubble_plot_line_x1y1 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
bubble_plot_line_x2y2 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx, yy, label='train', marker='o', facecolors='none', edgecolors='red', s=50)
plt.scatter(xx_validation, yy_validation, label='validation', marker='o', facecolors='none', edgecolors='blue', s=50)
plt.scatter(xx_test, yy_test, label='test', marker='o', facecolors='none', edgecolors='black', s=50)
plt.legend(loc='best', fontsize=10)
plt.xlabel('true-values', fontsize=15)
plt.ylabel('predicted', fontsize=15)

f.write('\n')
f.write("JCA - AdditionalTest flow resistance - r2_score")
f.write('\n')
f.write(str(r2_score(xx_test, yy_test)))
f.write('\n')
f.write("JCA - AdditionalTest flow resistance - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_test, yy_test)))
f.write('\n')
f.write("JCA - AdditionalTest flow resistance - root mean_squared_error") #mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx_test, yy_test))))
f.write('\n')
f.write("JCA - AdditionalTest flow resistance - mean_absolute_error") #mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx_test, yy_test)))

# plt.figure()
# plt.subplot(236)
# true_values = scaler2.inverse_transform(y_test)[:, 1]
# predicted_values = scaler2.inverse_transform(predicted_on_X_test)[:, 1]
# x_index = [i for i in range(len(true_values))]
# error_values = predicted_values - true_values
# plt.errorbar(x=x_index, y=true_values, yerr=error_values, fmt='o', color='black',
#              ecolor='black', elinewidth=2, capsize=10);
# plt.grid(linestyle='--', linewidth=1)


# print()
# print("o/p of test set:           \n", (scaler2.inverse_transform(y_test)[:, 1]))
# print("predicted o/p of test set: \n", (scaler2.inverse_transform(predicted_on_X_test)[:, 1]))
# print("mse_test_set: ", mean_squared_error(y_test, predicted_on_X_test))

# ####################################################################################################
# ###########    saving predicted data to excel file   ##############
# plt.show()
# n1 = xx
# n2 = yy
# n3 = true_values
# n4 = predicted_values
# n5 = error_values
# ## convert your array into a dataframe
# # df = pd.DataFrame(l1, columns=['a'])
# df1 = pd.DataFrame(OrderedDict({'y_train':n1, 'predicted_on_X_train':n2}))
# df2 = pd.DataFrame({'y_test':n3, 'predicted_on_X_test':n4, 'error_values':n5},
#                         columns=['y_test', 'predicted_on_X_test', 'error_values'])
# ## save to xlsx file
# # filepath_1 = 'test_excel_file_1.xlsx'
# df1.to_excel('test_excel_file_1.xlsx', sheet_name='sheet1', index=False)
# df2.to_excel('test_excel_file_2.xlsx', sheet_name='sheet1', index=False)
# # sys.exit()
# ####################################################################################################

###############################################################
#################   plotting graphs together -  tortuosity  \alpha_{\infty}  ################
###############################################################

plt.figure()
plt.suptitle('JCA - tortuosity - (epochs-{}) - pyTorch'.format(epochs), fontsize=25,
             color='r', fontweight='bold')  ## giving title on top of all subplots

plt.subplot(231)
plt.plot(running_loss_index, running_loss, 'r-', linewidth=3, label='mse_loss_train')
plt.plot(running_loss_validation_index, running_loss_validation, 'b-', linewidth=3, label='mse_loss_validation')
plt.legend(loc='best', fontsize=10)
plt.xlabel('epochs#', fontsize=15)

# plt.figure()
plt.subplot(232)
# Plot true data
plt.plot(scaler2.inverse_transform(y_train)[:, 2], 'ro', markersize=12, label='y_train')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_train)[:, 2], 'b*', markersize=12, label='predicted_on_X_train')
# Legend and plot
plt.legend(loc='best', fontsize=10)

# plt.figure()
plt.subplot(233)
# Plot true data
plt.plot(scaler2.inverse_transform(y_validation)[:, 2], 'ro', markersize=12, label='y_validation')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_validation)[:, 2], 'b*', markersize=12,
         label='predicted_on_X_validation')
# Legend and plot
plt.legend(loc='best', fontsize=10)

# plt.figure()
plt.subplot(234)
# Plot true data
plt.plot(scaler2.inverse_transform(y_test)[:, 2], 'ro', markersize=12, label='y_test')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_test)[:, 2], 'b*', markersize=12, label='predicted_on_X_test')
# Legend and plot
plt.legend(loc='best', fontsize=10)

# plt.figure()
plt.subplot(235)
xx = scaler2.inverse_transform(y_train)[:, 2]
yy = scaler2.inverse_transform(predicted_on_X_train)[:, 2]
xx_validation = scaler2.inverse_transform(y_validation)[:, 2]
yy_validation = scaler2.inverse_transform(predicted_on_X_validation)[:, 2]

f.write('\n')
f.write("JCA - Training  tortuosity - r2_score")
f.write('\n')
f.write(str(r2_score(xx, yy)))
f.write('\n')
f.write("JCA - Training  tortuosity - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx, yy)))
f.write('\n')
f.write("JCA - Training  tortuosity - root mean_squared_error") #mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx, yy))))
f.write('\n')
f.write("JCA - Training  tortuosity - mean_absolute_error") #mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx, yy)))
f.write('\n')
f.write("JCA - Validation  tortuosity - r2_score")
f.write('\n')
f.write(str(r2_score(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation  tortuosity - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation  tortuosity - root mean_squared_error") #mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx_validation, yy_validation))))
f.write('\n')
f.write("JCA - Validation  tortuosity - mean_absolute_error") #mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx_validation, yy_validation)))

xx_test = scaler2.inverse_transform(y_test)[:, 2]
yy_test = scaler2.inverse_transform(predicted_on_X_test)[:, 2]
bubble_plot_line_x1y1 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
bubble_plot_line_x2y2 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx, yy, label='train', marker='o', facecolors='none', edgecolors='red', s=50)
plt.scatter(xx_validation, yy_validation, label='validation', marker='o', facecolors='none', edgecolors='blue', s=50)
plt.scatter(xx_test, yy_test, label='test', marker='o', facecolors='none', edgecolors='black', s=50)
plt.legend(loc='best', fontsize=10)
plt.xlabel('true-values', fontsize=15)
plt.ylabel('predicted', fontsize=15)

f.write('\n')
f.write("JCA - AdditionalTest tortuosity - r2_score")
f.write('\n')
f.write(str(r2_score(xx_test, yy_test)))
f.write('\n')
f.write("JCA - AdditionalTest tortuosity - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_test, yy_test)))
f.write('\n')
f.write("JCA - AdditionalTest tortuosity - root mean_squared_error") #mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx_test, yy_test))))
f.write('\n')
f.write("JCA - AdditionalTest tortuosity - mean_absolute_error") #mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx_test, yy_test)))

# plt.figure()
# plt.subplot(236)
# true_values = scaler2.inverse_transform(y_test)[:, 2]
# predicted_values = scaler2.inverse_transform(predicted_on_X_test)[:, 2]
# x_index = [i for i in range(len(true_values))]
# error_values = predicted_values - true_values
# plt.errorbar(x=x_index, y=true_values, yerr=error_values, fmt='o', color='black',
#              ecolor='black', elinewidth=2, capsize=10);
# plt.grid(linestyle='--', linewidth=1)

# print()
# print("o/p of test set:           \n", (scaler2.inverse_transform(y_test)[:, 2]))
# print("predicted o/p of test set: \n", (scaler2.inverse_transform(predicted_on_X_test)[:, 2]))
# print("mse_test_set: ", mean_squared_error(y_test, predicted_on_X_test))

###############################################################
#################   plotting graphs together - viscous length  ################
###############################################################

plt.figure()
plt.suptitle('JCA - viscous length - (epochs-{}) - pyTorch'.format(epochs), fontsize=25,
             color='r', fontweight='bold')  ## giving title on top of all subplots

plt.subplot(231)
plt.plot(running_loss_index, running_loss, 'r-', linewidth=3, label='mse_loss_train')
plt.plot(running_loss_validation_index, running_loss_validation, 'b-', linewidth=3, label='mse_loss_validation')
plt.legend(loc='best', fontsize=10)
plt.xlabel('epochs#', fontsize=15)

# plt.figure()
plt.subplot(232)
# Plot true data
plt.plot(scaler2.inverse_transform(y_train)[:, 3], 'ro', markersize=12, label='y_train')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_train)[:, 3], 'b*', markersize=12, label='predicted_on_X_train')
# Legend and plot
plt.legend(loc='best', fontsize=10)

# plt.figure()
plt.subplot(233)
# Plot true data
plt.plot(scaler2.inverse_transform(y_validation)[:, 3], 'ro', markersize=12, label='y_validation')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_validation)[:, 3], 'b*', markersize=12,
         label='predicted_on_X_validation')
# Legend and plot
plt.legend(loc='best', fontsize=10)

# plt.figure()
plt.subplot(234)
# Plot true data
plt.plot(scaler2.inverse_transform(y_test)[:, 3], 'ro', markersize=12, label='y_test')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_test)[:, 3], 'b*', markersize=12, label='predicted_on_X_test')
# Legend and plot
plt.legend(loc='best', fontsize=10)

# plt.figure()
plt.subplot(235)
xx = scaler2.inverse_transform(y_train)[:, 3]
yy = scaler2.inverse_transform(predicted_on_X_train)[:, 3]
xx_validation = scaler2.inverse_transform(y_validation)[:, 3]
yy_validation = scaler2.inverse_transform(predicted_on_X_validation)[:, 3]

f.write('\n')
f.write("JCA - Training viscous length - r2_score")
f.write('\n')
f.write(str(r2_score(xx, yy)))
f.write('\n')
f.write("JCA - Training viscous length - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx, yy)))
f.write('\n')
f.write("JCA - Training viscous length - root mean_squared_error") #mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx, yy))))
f.write('\n')
f.write("JCA - Training viscous length - mean_absolute_error") #mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx, yy)))
f.write('\n')
f.write("JCA - Validation viscous length - r2_score")
f.write('\n')
f.write(str(r2_score(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation viscous length - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation viscous length - root mean_squared_error") #mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx_validation, yy_validation))))
f.write('\n')
f.write("JCA - Validation viscous length - mean_absolute_error") #mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx_validation, yy_validation)))

xx_test = scaler2.inverse_transform(y_test)[:, 3]
yy_test = scaler2.inverse_transform(predicted_on_X_test)[:, 3]
bubble_plot_line_x1y1 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
bubble_plot_line_x2y2 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx, yy, label='train', marker='o', facecolors='none', edgecolors='red', s=50)
plt.scatter(xx_validation, yy_validation, label='validation', marker='o', facecolors='none', edgecolors='blue', s=50)
plt.scatter(xx_test, yy_test, label='test', marker='o', facecolors='none', edgecolors='black', s=50)
plt.legend(loc='best', fontsize=10)
plt.xlabel('true-values', fontsize=15)
plt.ylabel('predicted', fontsize=15)

f.write('\n')
f.write("JCA - AdditionalTest viscous length - r2_score")
f.write('\n')
f.write(str(r2_score(xx_test, yy_test)))
f.write('\n')
f.write("JCA - AdditionalTest viscous length - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_test, yy_test)))
f.write('\n')
f.write("JCA - AdditionalTest viscous length - root mean_squared_error") #mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx_test, yy_test))))
f.write('\n')
f.write("JCA - AdditionalTest viscous length - mean_absolute_error") #mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx_test, yy_test)))


# plt.figure()
# plt.subplot(236)
# true_values = scaler2.inverse_transform(y_test)[:, 3]
# predicted_values = scaler2.inverse_transform(predicted_on_X_test)[:, 3]
# x_index = [i for i in range(len(true_values))]
# error_values = predicted_values - true_values
# plt.errorbar(x=x_index, y=true_values, yerr=error_values, fmt='o', color='black',
#              ecolor='black', elinewidth=2, capsize=10);
# plt.grid(linestyle='--', linewidth=1)

# print()
# print("o/p of test set:           \n", (scaler2.inverse_transform(y_test)[:, 3]))
# print("predicted o/p of test set: \n", (scaler2.inverse_transform(predicted_on_X_test)[:, 3]))
# print("mse_test_set: ", mean_squared_error(y_test, predicted_on_X_test))

###############################################################
#################   plotting graphs together - thermal length ################
###############################################################

plt.figure()
plt.suptitle('JCA - thermal length - (epochs-{}) - pyTorch'.format(epochs), fontsize=25,
             color='r', fontweight='bold')  ## giving title on top of all subplots

plt.subplot(231)
plt.plot(running_loss_index, running_loss, 'r-', linewidth=3, label='mse_loss_train')
plt.plot(running_loss_validation_index, running_loss_validation, 'b-', linewidth=3, label='mse_loss_validation')
plt.legend(loc='best', fontsize=10)
plt.xlabel('epochs#', fontsize=15)

# plt.figure()
plt.subplot(232)
# Plot true data
plt.plot(scaler2.inverse_transform(y_train)[:, 4], 'ro', markersize=12, label='y_train')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_train)[:, 4], 'b*', markersize=12, label='predicted_on_X_train')
# Legend and plot
plt.legend(loc='best', fontsize=10)

# plt.figure()
plt.subplot(233)
# Plot true data
plt.plot(scaler2.inverse_transform(y_validation)[:, 4], 'ro', markersize=12, label='y_validation')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_validation)[:, 4], 'b*', markersize=12,
         label='predicted_on_X_validation')
# Legend and plot
plt.legend(loc='best', fontsize=10)

# plt.figure()
plt.subplot(234)
# Plot true data
plt.plot(scaler2.inverse_transform(y_test)[:, 4], 'ro', markersize=12, label='y_test')
# Plot predictions
plt.plot(scaler2.inverse_transform(predicted_on_X_test)[:, 4], 'b*', markersize=12, label='predicted_on_X_test')
# Legend and plot
plt.legend(loc='best', fontsize=10)

# plt.figure()
plt.subplot(235)
xx = scaler2.inverse_transform(y_train)[:, 4]
yy = scaler2.inverse_transform(predicted_on_X_train)[:, 4]
xx_validation = scaler2.inverse_transform(y_validation)[:, 4]
yy_validation = scaler2.inverse_transform(predicted_on_X_validation)[:, 4]

f.write('\n')
f.write("JCA - Training thermal length - r2_score")
f.write('\n')
f.write(str(r2_score(xx, yy)))
f.write('\n')
f.write("JCA - Training thermal length - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx, yy)))
f.write('\n')
f.write("JCA - Training thermal length - root mean_squared_error") #mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx, yy))))
f.write('\n')
f.write("JCA - Training thermal length - mean_absolute_error") #mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx, yy)))
f.write('\n')
f.write("JCA - Validation thermal length - r2_score")
f.write('\n')
f.write(str(r2_score(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation thermal length - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_validation, yy_validation)))
f.write('\n')
f.write("JCA - Validation thermal length - root mean_squared_error") #mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx_validation, yy_validation))))
f.write('\n')
f.write("JCA - Validation thermal length - mean_absolute_error") #mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx_validation, yy_validation)))


xx_test = scaler2.inverse_transform(y_test)[:, 4]
yy_test = scaler2.inverse_transform(predicted_on_X_test)[:, 4]
bubble_plot_line_x1y1 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
bubble_plot_line_x2y2 = [min(np.minimum(xx, yy)), max(np.maximum(xx, yy))]
plt.xlim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.ylim(bubble_plot_line_x1y1[0], bubble_plot_line_x1y1[1])
plt.plot(bubble_plot_line_x1y1, bubble_plot_line_x2y2, 'k-', linewidth=2)
plt.grid(linestyle='--', linewidth=1)
plt.scatter(xx, yy, label='train', marker='o', facecolors='none', edgecolors='red', s=50)
plt.scatter(xx_validation, yy_validation, label='validation', marker='o', facecolors='none', edgecolors='blue', s=50)
plt.scatter(xx_test, yy_test, label='test', marker='o', facecolors='none', edgecolors='black', s=50)
plt.legend(loc='best', fontsize=10)
plt.xlabel('true-values', fontsize=15)
plt.ylabel('predicted', fontsize=15)


f.write('\n')
f.write("JCA - AdditionalTest thermal length - r2_score")
f.write('\n')
f.write(str(r2_score(xx_test, yy_test)))
f.write('\n')
f.write("JCA - AdditionalTest thermal length - mean_squared_error")
f.write('\n')
f.write(str(mean_squared_error(xx_test, yy_test)))
f.write('\n')
f.write("JCA - AdditionalTest thermal length - root mean_squared_error") #mean_squared_error(y_true, y_pred)
f.write('\n')
f.write(str(sqrt(mean_squared_error(xx_test, yy_test))))
f.write('\n')
f.write("JCA - AdditionalTest thermal length - mean_absolute_error") #mean_absolute_error(y_true, y_pred)
f.write('\n')
f.write(str(mean_absolute_error(xx_test, yy_test)))

f.write('\n')
f.write('\n')
f.write("time taken to train in sec:")
f.write('\n')
f.write(str((end_time - start_time)))
# print('time taken to train in sec: ', (end_time - start_time))
f.close() # close output statistic file

# plt.figure()
# plt.subplot(236)
# true_values = scaler2.inverse_transform(y_test)[:, 4]
# predicted_values = scaler2.inverse_transform(predicted_on_X_test)[:, 4]
# x_index = [i for i in range(len(true_values))]
# error_values = predicted_values - true_values
# plt.errorbar(x=x_index, y=true_values, yerr=error_values, fmt='o', color='black',
#              ecolor='black', elinewidth=2, capsize=10);
# plt.grid(linestyle='--', linewidth=1)

# print()
# print("o/p of test set:           \n", (scaler2.inverse_transform(y_test)[:, 4]))
# print("predicted o/p of test set: \n", (scaler2.inverse_transform(predicted_on_X_test)[:, 4]))
# print("mse_test_set: ", mean_squared_error(y_test, predicted_on_X_test))
# print()

###############################################################
#################   plotting graphs together - conf-loss-without/with-log10  ################
###############################################################

# plt.figure()
# plt.subplot(121)
# true_values = scaler2.inverse_transform(y_test)[:, 3]
# predicted_values = scaler2.inverse_transform(predicted_on_X_test)[:, 3]
# print(true_values)
# print(predicted_values)
# x_index = [i for i in range(len(true_values))]
# error_values = predicted_values - true_values
# plt.errorbar(x=x_index, y=true_values, yerr=error_values, fmt='o', color='black',
#              ecolor='black', elinewidth=2, capsize=10)
# plt.yscale('log')
# plt.grid(linestyle='--', linewidth=1)
# plt.title('conf-loss-without-log10', fontsize=25)
#
# plt.subplot(122)
# true_values = 10 ** (scaler2.inverse_transform(y_test)[:, 4])
# predicted_values = 10 ** (scaler2.inverse_transform(predicted_on_X_test)[:, 4])
# print(true_values)
# print(predicted_values)
# x_index = [i for i in range(len(true_values))]
# error_values = predicted_values - true_values
# plt.errorbar(x=x_index, y=true_values, yerr=error_values, fmt='o', color='black',
#              ecolor='black', elinewidth=2, capsize=10)
# plt.yscale('log')
# plt.grid(linestyle='--', linewidth=1)
# plt.title('conf-loss-with-log10', fontsize=25)

# ####################################################################################################
# ###########    saving predicted data to excel file   ##############
# plt.show()
# n1 = xx
# n2 = yy
# n3 = true_values
# n4 = predicted_values
# n5 = error_values
# ## convert your array into a dataframe
# # df = pd.DataFrame(l1, columns=['a'])
# df1 = pd.DataFrame(OrderedDict({'y_train':n1, 'predicted_on_X_train':n2}))
# df2 = pd.DataFrame({'y_test':n3, 'predicted_on_X_test':n4, 'error_values':n5},
#                         columns=['y_test', 'predicted_on_X_test', 'error_values'])
# ## save to xlsx file
# # filepath_1 = 'test_excel_file_1.xlsx'
# df1.to_excel('test_excel_file_1.xlsx', sheet_name='sheet1', index=False)
# df2.to_excel('test_excel_file_2.xlsx', sheet_name='sheet1', index=False)
# # sys.exit()
# ####################################################################################################


plt.show()

###########    saving predicted data to excel file   ##############
l1 = 10 ** (scaler2.inverse_transform(y_test)[:, 4])  #### check multiple of 10
l2 = 10 ** (scaler2.inverse_transform(predicted_on_X_test)[:, 4])  #### check multiple of 10
###########    saving mse data to excel file   ##############
l3 = running_loss
l4 = running_loss_validation
l5 = running_loss_index
l6 = running_loss_validation_index

## convert your array into a dataframe
# df = pd.DataFrame(l1, columns=['a'])
df = pd.DataFrame({'a': l3, 'b': l4})

## save to xlsx file
filepath = 'test_excel_file.xlsx'
df.to_excel(filepath, sheet_name='sheet1', index=False)

# sys.exit()


##########################################
#######           GUI           ##########
##########################################

