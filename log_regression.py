
# coding: utf-8

# In[1]:


from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torch import autograd,nn, optim
import numpy as np

"""
# In[2]:


np.set_printoptions(threshold=np.nan)


# In[3]:


SeasonList = [2005,2006]
SeasonList = [2005,2006,2007,2008,2009,2010,2011,2012,2013]

X_data, Y_data = createData(SeasonList)

X_data = X_data[:,:]

X_data.shape


# In[4]:


# Splitting the data
splitIndex = int(0.7*X_data.shape[0])
X_test = X_data[splitIndex:,:]
X_train = X_data[:splitIndex,:]
Y_test = Y_data[splitIndex:]
Y_train = Y_data[:splitIndex]

print(Y_train)
print(X_test)

"""
# In[5]:


def runLogRegression(X_train, Y_train, X_test, Y_test, num_epochs=1000, learning_rate=0.001, num_classes=1):
        # Hyper Parameters 
    input_size = X_train.shape[1]
    num_classes = 1
    num_epochs = 1000
    batch_size = X_train.shape[0]
    learning_rate = 0.001
    
    train_input = Variable(torch.Tensor(X_train))
    train_target = Variable(torch.Tensor(Y_train).long())
    test_input = Variable(torch.Tensor(X_test))
    test_target = Variable(torch.Tensor(Y_test).long())
    
        # Model
    class LogisticRegression(nn.Module):
        def __init__(self, input_size, num_classes):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(input_size, num_classes)
    
        def forward(self, x):
            #out = self.linear(x)
            out = F.sigmoid(self.linear(x))
            return out
    

    def Accuracy(Y_label,Yhat):
        CorrectPredictions = 0
        print(len(Y_label))
        print(len(Yhat))
        for i,current in enumerate(Y_label):
            if Y_label[i] == Yhat[i]:
                CorrectPredictions = CorrectPredictions + 1
        return(100 * CorrectPredictions/len(Y_label), CorrectPredictions)


    model = LogisticRegression(input_size, num_classes)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss()
    criterion = F.nll_loss
    criterion = nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

    # Training the Model
    for epoch in range(num_epochs):
    #    for i, (images, labels) in enumerate(train_loader):
    #       images = Variable(images.view(-1, 28*28))
    #        labels = Variable(labels)
        
            # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(train_input)
        loss = criterion(outputs.float(), train_target.float())
        loss.backward()
        optimizer.step()
        
        print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' 
                   % (epoch+1, num_epochs, 1, len(train_input)//batch_size, loss.data[0]))
    

    lossOnTrain = loss.data[0]
    outputs = model(train_input)
    train_prediction = np.round(outputs.data.numpy())

    accuracyTrain, CorrectPredictionsTrain = Accuracy(Y_train, train_prediction)
    print("Accuracy on train data = ",accuracyTrain,"\n Correct predictions on train data = ",CorrectPredictionsTrain)
    

    # Test the Model
    outputs_test = model(test_input)
    test_prediction = np.round(outputs_test.data.numpy())
    loss_test = criterion(outputs_test.float(), test_target.float())
    
    lossOnTest = loss_test.data[0]

    print(outputs_test)

    test_prediction

    accuracyTest, CorrectPredictionsTest = Accuracy(Y_test, test_prediction)

    print("Accuracy on test data = ",accuracyTest,"\n Correct predictions on test data = ",CorrectPredictionsTest)
    return lossOnTrain, lossOnTest, accuracyTrain, accuracyTest, CorrectPredictionsTrain, CorrectPredictionsTest


# In[6]:


#runLogRegression(X_train, Y_train, X_test, Y_test)

