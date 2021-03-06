

import argparse
from data_utils import *
from Neural_Networks import *
from log_regression import *
from svm_baselines import *
from data_utils2 import *




# In[4]:


parser = argparse.ArgumentParser(description='PyTorch QGen')
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')

# Dataset related
parser.add_argument('--path_to_data', default='data/', type=str,
                    help='path to train data')
parser.add_argument('--load_data', default='', type=str,
                    help='Load pickled data')
parser.add_argument('--save_data', default='', type=str,
                    help='save pickled data')
parser.add_argument('--example_to_train', type=int, default=1000,
                    help='example taken to train')
parser.add_argument('--split_ratio', type=float, default=0.7,
                    help='ratio of training data')

# Hyperparameters
parser.add_argument('--hidden_size', type=int, default=300,
                    help='RNN hidden size')
parser.add_argument('--lr', type=float, default=3e-4,
                    help='Learning rate')

# Model
parser.add_argument('--save', default='', type=str,
                    help='save the model after training')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--no_train', action='store_true', default=False,
                    help="don't start training")
parser.add_argument('--no_eval', action='store_true', default=False,
                    help="don't evaluate")
parser.add_argument('--algorithm', default="NN", type=str,
                    help="don't evaluate")


args = parser.parse_args()
print(args)


print("Split ratio------->", args.split_ratio)
print("Learning rate ------->", args.lr)
print("Epochs ------->", args.num_epochs)
print("Path to data ------->", args.path_to_data)
print("Algo -------->", args.algorithm)


season_list = [2005,2006,2007,2008,2009,2010,2011]
#season_list = [2005,2006]

#X_DATA,Y_DATA = createData(season_list, args.path_to_data)

#Considering previous years
X_DATA,Y_DATA = createData2(season_list, args.path_to_data)
X_UNSEEN, Y_UNSEEN = createData2([2012,2013],args.path_to_data)
#X_DATA,Y_DATA = createData(season_list, args.path_to_data)


print("X-DATA: ", X_DATA.shape)
#######PRINT #######
#print("X-DATA", X_DATA[0])

print("Y-DATA: ", Y_DATA.shape)

#Baseline1:
indicator_col = X_DATA[:,10:11]
print("ind: ", indicator_col.shape)


#X_DATA = np.delete(X_DATA,4,1)
#X_DATA = np.delete(X_DATA,4,1)
#X_DATA = X_DATA[:,:2]
#X_DATA = np.concatenate((X_DATA,indicator_col), axis=1)
print("X-DATA: ", X_DATA.shape)


Y_DATA = Y_DATA[:,0:1]
Y_UNSEEN = Y_UNSEEN[:,0:1]
print("Y_data : ", Y_DATA)

split_index = int(args.split_ratio * X_DATA.shape[0])

X_test = X_DATA[split_index:]
X_train = X_DATA[:split_index]
Y_test = Y_DATA[split_index:]
Y_train = Y_DATA[:split_index]


learningModel = run_neural_network

if(args.algorithm == "LR"):
    learningModel = runLogRegression
elif(args.algorithm == "SVM"):
    learningModel = Svm().run_svm
else:
    learningModel = run_neural_network

print("Model Selected",learningModel)

loss_train, loss_test, loss_unseen, accuracy_train, accuracy_test, accuracy_unseen, correct_predictions_train, correct_predictions_test, correct_predictions_unseen = learningModel(X_train, Y_train, X_test, Y_test, X_UNSEEN, Y_UNSEEN, args.num_epochs, args.lr)

Y_ones = Y_train[Y_train>0.5]
Y_zeros = Y_train[Y_train<0.5]

print("Number of ones",Y_ones.shape)
print("Number of zeros",Y_zeros.shape)

#for i in range(X_train.shape[1]):
#    loss_train, loss_test, accuracy_train, accuracy_test, correct_predictions_train, correct_predictions_test = learningModel(X_train[:,i].reshape((X_train.shape[0],1)), Y_train, X_test[:,i].reshape((X_test.shape[0],1)), Y_test, args.num_epochs, args.lr)
    #loss_train, loss_test, accuracy_train, accuracy_test, correct_predictions_train, correct_predictions_test = learningModel(X_train[:,i:i+1].reshape((X_train.shape[0],1)), Y_train, X_test[:,i].reshape((X_test.shape[0],1)), Y_test, args.num_epochs, args.lr)
#    print("Num of Features", X_train.shape[1])
#    print("Training Loss: ",loss_train)
#    print("Test Loss: ",loss_test) 
#    print("Training Accuracy: ",accuracy_train)
#    print("Test Accuracy: ",accuracy_test)
#    print("Correct predictions train: ",correct_predictions_train)
#    print("Correct predictions test: ",correct_predictions_test)

#outputs = learningModel(X_train, Y_train, X_test, Y_test, args.num_epochs, args.lr)

print("Num of Features", X_train.shape[1])
print("Training Loss: ",loss_train)
print("Test Loss: ",loss_test)
print("Test Unseen: ",loss_unseen)
print("Training Accuracy: ",accuracy_train)
print("Test Accuracy: ",accuracy_test)
print("Unseen Accuracy: ",accuracy_unseen)
print("Correct predictions train: ",correct_predictions_train)
print("Correct predictions test: ",correct_predictions_test)
print("Correct predictions unseen: ",correct_predictions_unseen)