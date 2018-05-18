
import numpy as np
import torch
from torch import autograd,nn, optim
import torch.nn.functional as F

def filterIQR(X):
    q75,q25 = np.percentile(X,[75,25])
    iqr = q75 - q25
    minq = q25 - 1.5 * iqr
    maxq = q75 + 1.5 * iqr
    
    b1 = X < maxq
    b2  =  X > minq
    
    return b1 & b2

def run_neural_network(X_train, Y_train, X_test, Y_test, X_unseen, Y_unseen, num_epochs=100, learning_rate=0.001, num_classes=1, hidden_size=50):

    for i in range(X_train.shape[1]):
        minf = np.min(X_train[:,i])
        maxf = np.max(X_train[:,i])
        X_train[:,i] = (X_train[:,i] -minf)/ (maxf-minf)
        #X_train2[:,i] = (X_train2[:,i] -minf)/ (maxf-minf)
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
    
    input_size = X_train.shape[1]
    #input = autograd.Variable(torch.rand(batch_size,input_size)
    model_input = autograd.Variable(torch.Tensor(X_train))

    #input[0][1] = 2
    target = autograd.Variable(torch.Tensor(Y_train).long())


    # In[11]:


    class Net(nn.Module):
        def __init__(self, input_size, hidden_size,num_classes):
            super().__init__()
            #create hidden layers and assign it to self
            self.h1 = nn.Linear(input_size,hidden_size)
            self.h2 = nn.Linear(hidden_size,num_classes)
        def forward(self,x):
            #run forward propogation
            x = self.h1(x)
            x = F.tanh(x)
            x = self.h2(x)
            x = F.sigmoid(x)
            return x


    # In[12]:


    model = Net(input_size = input_size, hidden_size = hidden_size, num_classes = num_classes)
    opt = optim.Adam(params=model.parameters(),lr = learning_rate)


    # In[13]:


    for epoch in range(num_epochs):
        out = model(model_input)
        #print(out)
        #_, pred = out.max(1)
       # print("Target: ",str(target.view(1,-1)).split('\n')[1])
       # print("Pred: ", str(pred.view(1,-1)).split('\n')[1])
        loss = F.binary_cross_entropy(out,target.float())
        #loss = F.nll_loss(out,target)
        print("Loss: ",loss.data[0])
        model.zero_grad()
        loss.backward()
        opt.step()

    loss_train = loss

    # In[14]:


    pred = np.round(out.data.numpy())
    t = target.data.numpy()
    torch.save(model.state_dict(), "mymodel2")


    # In[15]:


    def Accuracy(Y_label,Y_hat):
        CorrectPredictions = 0
        print(len(Y_label))
        print(len(Y_hat))
        for i,current in enumerate(Y_label):
            if(Y_label[i] == Y_hat[i]):
                CorrectPredictions = CorrectPredictions + 1
        return(100 * CorrectPredictions/len(Y_label), CorrectPredictions)


    # In[16]:


    accuracy_train, correct_predictions_train = Accuracy(t,pred)


    # In[17]:


    print(accuracy_train)
    print(correct_predictions_train)


    # In[18]:


    the_model = Net(input_size = input_size, hidden_size = hidden_size, num_classes = num_classes)
    the_model.load_state_dict(torch.load("mymodel2"))


    # In[19]:


    test_input = autograd.Variable(torch.Tensor(X_test))
    test_target = autograd.Variable(torch.Tensor(Y_test).long())
    unseen_input = autograd.Variable(torch.Tensor(X_unseen))
    unseen_target = autograd.Variable(torch.Tensor(Y_unseen).long())

    # In[20]:


    out2 = the_model(test_input)
    pred2 = np.round(out2.data.numpy())
    loss_test = F.binary_cross_entropy(out2,test_target.float())
    
    
    accuracy_test, correct_predictions_test = Accuracy(test_target.data.numpy(), pred2)
    
    out_unseen = the_model(unseen_input)
    pred_unseen = np.round(out_unseen.data.numpy())
    loss_unseen = F.binary_cross_entropy(out_unseen,unseen_target.float())


    accuracy_unseen, correct_predictions_unseen = Accuracy(unseen_target.data.numpy(), pred_unseen)
    # In[21]:




    # In[22]:


    print(accuracy_test)
    print(correct_predictions_test)

    
    return(loss_train, loss_test, loss_unseen, accuracy_train, accuracy_test, accuracy_unseen, correct_predictions_train, correct_predictions_test, correct_predictions_unseen)