
# coding: utf-8

# In[1]:


from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torch import autograd,nn, optim
import numpy as np
import matplotlib.pyplot as plt
USE_CUDA = False


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


def filterIQR(X):
    q75,q25 = np.percentile(X,[75,25])
    iqr = q75 - q25
    minq = q25 - 1.5 * iqr
    maxq = q75 + 1.5 * iqr
    
    b1 = X < maxq
    b2  =  X > minq
    
    return b1 & b2

def runLogRegression(X_train, Y_train, X_test, Y_test, X_unseen, Y_unseen, num_epochs=1000, learning_rate=0.001, num_classes=1 ):
        # Hyper Parameters 
    X_train2 = np.copy(X_train)
    
    #X_train = X_train[:,[0,1,2,3,6,7,9,10,11,12,13,14,15,16,17,18]]
    #X_test = X_test[:,[0,1,2,3,6,7,9,10,11,12,13,14,15,16,17,18]]
    
    #plt.boxplot(X_train[:,:])
   # plt.show()
    
    features = ["Win ratio of HT","Win ratio of VT","Point difference of HT","Point difference of VT",
               "Time of possession of HT","Time of possession of VT","HT Penalty points","VT Penalty points","Kickoff yard HT","Kickoff yard VT","HT fumbles","VT fumbles","Rush yards per carry of HT","Rush yards per carry of VT","Yards per pass attempt of HT","Yards per pass attempt of VT","Goal Ratio of HT","Goal Ratio of VT","Scores diff, HT - VT"]
       
    #features = ["Win ratio of HT","Win ratio of VT","Point difference of HT","Point difference of VT","HT Penalty points","VT Penalty points","Kickoff yard HT","Kickoff yard VT","HT fumbles","VT fumbles","Rush yards per carry of HT","Rush yards per carry of VT","Yards per pass attempt of HT","Yards per pass attempt of VT","Goal Ratio of HT","Goal Ratio of VT","Scores diff, HT - VT"]    

    
    #Normalisation
    print("SHAPE OF X_TRAIN",X_train.shape[0])
    #mask = np.ones(X_train.shape[1],dtype =bool)
    #mask[[4,5]] = False
    #X_train = X_train[:,mask]
    #X_test = X_test[:,mask]
    #X_unseen = X_unseen[:,mask]
    #features = features[mask]
    #X_train = X_train[: , 0:10]
    #X_test = X_test[:, 0:10]
    #X_unseen = X_unseen[:, 0:10]
    for i in range(X_train.shape[1]):
        minf = np.min(X_train[:,i])
        maxf = np.max(X_train[:,i])
        X_train[:,i] = (X_train[:,i] -minf)/ (maxf-minf)
        X_train2[:,i] = (X_train2[:,i] -minf)/ (maxf-minf)
        #X_train[:,i] = (X_train[:,i] -np.mean(X_train[:,i]))/ np.std(X_train[:,i])
        #X_train2[:,i] = (X_train2[:,i] -np.mean(X_train2[:,i]))/ np.std(X_train2[:,i])
        #temp = X_train[filterIQR(X_train[:,i])]
        toRemove = filterIQR(X_train[:,i])
        X_test[:,i] = (X_test[:,i] -minf)/ (maxf-minf)
        X_unseen[:,i] = (X_unseen[:,i] -minf)/ (maxf-minf)
        #X_test[:,i] = (X_test[:,i] -np.mean(X_train[:,i]))/ np.std(X_train[:,i])
        #X_unseen[:,i] = (X_unseen[:,i] -np.mean(X_train[:,i]))/ np.std(X_train[:,i])
        X_train = X_train[toRemove]
        Y_train = Y_train[toRemove]
        print("TEMP SHAPE for feature",i,":",X_train.shape[0])
        #X_train[:,i] = X_train[:,i] / np.linalg.norm(X_train[:,i])
        #X_test[:,i] = X_test[:,i] / np.linalg.norm(X_train[:,i])
        
    #print(X_train[])
    
    

    print("############ PLOTTING BOX PLOT####")
    
    fig, ax1 = plt.subplots(figsize=(20,12))
    ax1.set_title('Box Plot for Normalised features with outliers', fontsize=20)
    ax1.boxplot(X_train2[:,:])
    ax1.set_xticklabels(features,rotation = 45, fontsize =12)
    plt.show()

    
    fig, ax1 = plt.subplots(figsize=(20,12))
    ax1.set_title('Box Plot for Normalised features after reducing outliers', fontsize=20)
    ax1.boxplot(X_train[:,:])
    ax1.set_xticklabels(features,rotation = 45, fontsize =12)
    plt.show()
    
    #return
    input_size = X_train.shape[1]
    batch_size = X_train.shape[0]
    
    train_input = Variable(torch.Tensor(X_train))
    train_target = Variable(torch.Tensor(Y_train).long())
    test_input = Variable(torch.Tensor(X_test))
    test_target = Variable(torch.Tensor(Y_test).long())
    unseen_input = Variable(torch.Tensor(X_unseen))
    unseen_target = Variable(torch.Tensor(Y_unseen).long())
    
    print("########",unseen_input.shape)
    print(unseen_target.shape)
    
    if torch.cuda.is_available() and USE_CUDA:
        train_input = train_input.cuda()
        train_target = train_target.cuda()
        test_input = test_input.cuda()
        test_target = test_target.cuda()
    
        # Model
    class LogisticRegression(nn.Module):
        def __init__(self, input_size, num_classes):
            super(LogisticRegression, self).__init__()
            self.linear1 = nn.Linear(input_size, 1)
            torch.nn.init.xavier_uniform(self.linear1.weight)
    
        def forward(self, x):
            out = self.linear1(x)
            out = F.sigmoid(out)
            return out
    

    def Accuracy(Y_label,Yhat):
        CorrectPredictions = 0
        #print(len(Y_label))
        #print(len(Yhat))
        for i,current in enumerate(Y_label):
            if Y_label[i] == Yhat[i]:
                CorrectPredictions = CorrectPredictions + 1
        return(100 * CorrectPredictions/len(Y_label), CorrectPredictions)


    model = LogisticRegression(input_size, num_classes)
    if torch.cuda.is_available() and USE_CUDA:
        model = model.cuda();
    

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    #criterion = nn.CrossEntropyLoss()
    #criterion = F.nll_loss
    criterion = nn.BCELoss(size_average=True)
    #criterion = nn.MSELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

    test_acc_grp = []
    train_acc_grp = []
    # Training the Model
    for epoch in range(num_epochs):
        #print("Weights : ", model.linear1.weight)
        #print("Input--->",X_train[0])
        
    #    for i, (images, labels) in enumerate(train_loader):
    #       images = Variable(images.view(-1, 28*28))
    #        labels = Variable(labels)
        
            # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(train_input)
        #print("outputs : " , outputs.data)
        #print(np.unique(outputs.data))
        loss = criterion(outputs.float(), train_target.float())
        loss.backward()
        optimizer.step()
        
        #print("Grad--->",optimizer)
        if epoch%1000 == 0:
            print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' 
                   % (epoch+1, num_epochs, 1, len(train_input)//batch_size, loss.data[0]))
        
        lossOnTrain = loss.data[0]
        outputs = model(train_input)
        train_prediction = np.round(outputs.data.numpy())

        accuracyTrain, CorrectPredictionsTrain = Accuracy(Y_train, train_prediction)
        #print("Accuracy on train data = ",accuracyTrain,"\n Correct predictions on train data = ",CorrectPredictionsTrain)
        train_acc_grp.append(accuracyTrain)

        # Test the Model
        outputs_test = model(test_input)
        test_prediction = np.round(outputs_test.data.numpy())
        loss_test = criterion(outputs_test.float(), test_target.float())
    
        lossOnTest = loss_test.data[0]

        #print(outputs_test)

        #test_prediction

        accuracyTest, CorrectPredictionsTest = Accuracy(Y_test, test_prediction)
        #print("Accuracy on test data = ",accuracyTest,"\n Correct predictions on test data = ",CorrectPredictionsTest)
        test_acc_grp.append(accuracyTest)        
    

    plt.plot(train_acc_grp,label="train")
    plt.plot(test_acc_grp,label="test")
    plt.legend()
    plt.show()
    
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
    
    # Test the Model on Unseen data
    outputs_unseen = model(unseen_input)
    unseen_prediction = np.round(outputs_unseen.data.numpy())
    loss_unseen = criterion(outputs_unseen.float(), unseen_target.float())
    
    lossOnUnseen = loss_unseen.data[0]

    print(outputs_unseen)

    unseen_prediction

    accuracyUnseen, CorrectPredictionsUnseen = Accuracy(Y_unseen, unseen_prediction)

    print("Accuracy on test data = ",accuracyTest,"\n Correct predictions on test data = ",CorrectPredictionsTest)
    return lossOnTrain, lossOnTest, lossOnUnseen, accuracyTrain, accuracyTest, accuracyUnseen, CorrectPredictionsTrain, CorrectPredictionsTest, CorrectPredictionsUnseen


# In[6]:


#runLogRegression(X_train, Y_train, X_test, Y_test)

