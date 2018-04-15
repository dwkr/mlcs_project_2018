
import numpy as np
import torch
from torch import autograd,nn, optim
import torch.nn.functional as F


def run_neural_network(X_train, Y_train, X_test, Y_test, num_epochs=1000, learning_rate=0.001, num_classes=1, hidden_size=50):

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


    for epoch in range(500):
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


    # In[20]:


    out2 = the_model(test_input)
    pred2 = np.round(out2.data.numpy())
    loss_test = F.binary_cross_entropy(out2,test_target.float())


    # In[21]:


    accuracy_test, correct_predictions_test = Accuracy(test_target.data.numpy(), pred2)


    # In[22]:


    print(accuracy_test)
    print(correct_predictions_test)

    
    return(loss_train, loss_test, accuracy_train, accuracy_test, correct_predictions_train, correct_predictions_test)