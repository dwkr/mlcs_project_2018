from sklearn import datasets
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, hinge_loss, log_loss
from sklearn.svm import SVC
from data_utils import *
import torch.nn.functional as F
from torch import autograd,nn, optim
import torch

class Svm(object):
    from sklearn.svm import LinearSVC, SVC
    def baseline_1(self, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, num_epochs =1000, learning_rate = 0.001, num_classes=1):
        X = X_TRAIN[:, :2]
        y = Y_TRAIN
        svc_C = 0.1
        lin_C = 0.01
        rbf_C = 0.1
        poly_C =1
        # SVC with linear kernel
        svc = svm.SVC(kernel='linear', C=svc_C, probability=True).fit(X, y)
        # LinearSVC (linear kernel)
        lin_svc = svm.LinearSVC(C=lin_C).fit(X, y)
        # SVC with RBF kernel
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=rbf_C, probability=True).fit(X, y)
        # SVC with polynomial (degree 3) kernel
        poly_svc = svm.SVC(kernel='poly', degree=2, C=poly_C, probability=True).fit(X, y)

        h = .02  # step size parameter in the mesh
        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
        # title for the plots
        titles = ['SVC with linear kernel','LinearSVC (linear kernel)','SVC with RBF kernel','SVC with polynomial (degree 2) kernel']

        for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            plt.subplot(2, 2, i + 1)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            pred = clf.predict(X_TEST[:,:2])
            print (titles[i])
            accuracyTest = accuracy_score(Y_TEST, pred)
            print ( "Accuracy on Test Data:", accuracyTest)
            correctPredictionsTest = accuracy_score(Y_TEST, pred, normalize = False)
            print ("Correct Predictions on Test Data:", correctPredictionsTest)
            lossOnTest = hinge_loss(Y_TEST, pred)
            print ("Hinge Loss on Test Data", lossOnTest)
            if(i!=1):
                print(clf.predict_proba(X_TEST[:,:2])[:,1])
                print(Y_TEST)
                pred = autograd.Variable(torch.Tensor(clf.predict_proba(X_TEST[:,:2])[:,1]))
                truelabel = autograd.Variable(torch.Tensor(Y_TEST))
                print("Binary Cross Entropy on Test Data",F.binary_cross_entropy(pred, truelabel))
            
            predt = clf.predict(X_TRAIN[:,:2])
            accuracyTrain = accuracy_score(Y_TRAIN, predt)
            print ("Accuracy on Train Data:", accuracyTrain)
            correctPredictionsTrain = accuracy_score(Y_TRAIN, predt, normalize = False)
            print ("Correct Predictions on Train Data:", correctPredictionsTrain)
            lossOnTrain = hinge_loss(Y_TRAIN, predt)
            print ("Hinge Loss on Train Data", lossOnTrain)
            if(i!=1):
                pred = autograd.Variable(torch.Tensor(clf.predict_proba(X_TRAIN[:,:2])[:,1]))
                print(len(pred))
                truelabel = autograd.Variable(torch.Tensor(Y_TRAIN))
                print(len(truelabel))
                print("Binary Cross Entropy on Train Data",F.binary_cross_entropy(pred, truelabel))

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

            # Plot also the training points
            plt.scatter(X_TRAIN[:, 0], X_TRAIN[:, 1], c=y, cmap=plt.cm.coolwarm)
            plt.xlabel('HTeam Win Ratio')
            plt.ylabel('VTeam Win Ratio')
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.xticks(())
            plt.yticks(())
            plt.title(titles[i])
            plt.show()
        return  lossOnTrain, lossOnTest, accuracyTrain, accuracyTest, correctPredictionsTrain, correctPredictionsTest

    def filterIQR(self, X):
    	q75,q25 = np.percentile(X,[75,25])
    	iqr = q75 - q25
    	minq = q25 - 1.5 * iqr
    	maxq = q75 + 1.5 * iqr
    	b1 = X < maxq
    	b2  =  X > minq
    	return b1 & b2

    def run_svm(self, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,  X_UNSEEN, Y_UNSEEN, num_epochs =1000, learning_rate = 0.001, num_classes=1):        
        X = X_TRAIN
        y = Y_TRAIN
        
        for i in range(X_TRAIN.shape[1]):
        	X_TRAIN[i] = X_TRAIN[i]/np.linalg.norm(X_TRAIN[:,i])
        	X_TEST[i] = X_TEST[i]/np.linalg.norm(X_TRAIN[:,i])
        
        for i in range(X_UNSEEN.shape[1]):
        	X_TRAIN[i] = X_UNSEEN[i]/np.linalg.norm(X_UNSEEN[:,i])
        
        #for i in range(X_TRAIN.shape[1]):
        #	X_TRAIN[i] = (X_TRAIN[i] - np.min(X_TRAIN[:,1]))/(np.max(X_TRAIN[:,1]) - np.min(X_TRAIN[:,1]))
        #	X_TEST[i] = (X_TEST[i]- np.min(X_TRAIN[:,1]))/(np.max(X_TRAIN[:,1]) - np.min(X_TRAIN[:,1]))
        	
        #print("SHAPE OF X_TRAIN",X_TRAIN.shape[0])
        #for i in range(X_TRAIN.shape[1]):
        #	minf = np.min(X_TRAIN[:,i])
        #	maxf = np.max(X_TRAIN[:,i])
        #	X_TRAIN[:,i] = (X_TRAIN[:,i] - minf)/ (maxf-minf)
        #	#temp = X_train[filterIQR(X_train[:,i])]
        #	toRemove = self.filterIQR(X_TRAIN[:,i])
        #	X_TRAIN = X_TRAIN[toRemove]
        #	Y_TRAIN = Y_TRAIN[toRemove]
        #	print("TEMP SHAPE for feature",i,":",X_TRAIN.shape[0])
        #	X_TEST[:,i] = (X_TEST[:,i] - minf)/ (maxf-minf)
                
        titles = ['SVC with linear kernel','LinearSVC (linear kernel)','SVC with RBF kernel','SVC with polynomial (degree 3) kernel']
    
        SVC_C = 0.1
        clf = svm.SVC(kernel='linear', C=SVC_C,  max_iter=num_epochs, probability = True)
        clf.fit(X, y) 
        pred = clf.predict(X_TEST)
        
        print ("\nSVC")
        accuracyTest = accuracy_score(Y_TEST, pred)
        print ( "Accuracy on Test Data:", accuracyTest)#, normalize=False))
        correctPredictionsTest = accuracy_score(Y_TEST, pred, normalize = False)
        print ("Correct Predictions on Test Data:", correctPredictionsTest)
        #lossOnTest = hinge_loss(Y_TEST[:,0], pred)
        #print ("Hinge Loss on Test Data", lossOnTest)
        print(len(clf.predict_proba(X_TEST)[:,1]))
        print(len(Y_TEST))
        
        pred = autograd.Variable(torch.Tensor(clf.predict_proba(X_TEST)[:,1]))
        truelabel = autograd.Variable(torch.Tensor(Y_TEST))
        lossOnTest = F.binary_cross_entropy(pred, truelabel)
        #print("Binary Cross Entropy on Test Data",F.binary_cross_entropy(pred, truelabel))
        
        predu = clf.predict(X_UNSEEN)
        print ("\nSVC")
        accuracyTest = accuracy_score(Y_UNSEEN, predu)
        print ( "Accuracy on UNSEEN DATA:", accuracyTest)#, normalize=False))
        correctPredictionsTest = accuracy_score(Y_UNSEEN, predu, normalize = False)
        print ("Correct Predictions on UNSEEN DATA:", correctPredictionsTest)
        #lossOnTest = hinge_loss(Y_TEST[:,0], pred)
        #print ("Hinge Loss on Test Data", lossOnTest)
        print(len(clf.predict_proba(X_UNSEEN)[:,1]))
        print(len(Y_UNSEEN))
        
        pred = autograd.Variable(torch.Tensor(clf.predict_proba(X_UNSEEN)[:,1]))
        truelabel = autograd.Variable(torch.Tensor(Y_UNSEEN))
        loss_unseen = F.binary_cross_entropy(pred, truelabel)
        print("Unseen loss", loss_unseen)
        #print("Binary Cross Entropy on Test Data",F.binary_cross_entropy(pred, truelabel))


        predt = clf.predict(X_TRAIN)
        accuracyTrain = accuracy_score(Y_TRAIN, predt)
        print ( "Accuracy on Train Data:", accuracyTrain)
        correctPredictionsTrain = accuracy_score(Y_TRAIN, predt, normalize = False)
        print ("Correct Predictions on Train Data:", correctPredictionsTrain)
        #lossOnTrain = hinge_loss(Y_TRAIN[:,0], predt)
        #print ("Hinge Loss on Train Data", lossOnTrain)
        pred = autograd.Variable(torch.Tensor(clf.predict_proba(X_TRAIN)[:,1]))
        truelabel = autograd.Variable(torch.Tensor(Y_TRAIN))
        lossOnTrain = F.binary_cross_entropy(pred, truelabel)
        print("Binary Cross Entropy on Train Data",F.binary_cross_entropy(pred, truelabel))
    
    
        LIN_C = 1.0
        clf = svm.LinearSVC(C=LIN_C).fit(X, y)
        clf.fit(X, y) 
        pred = clf.predict(X_TEST)
        
        print ("\nLinear SVC")
        accuracyTest = accuracy_score(Y_TEST, pred)
        print ( "Accuracy on Test Data:", accuracyTest)
        correctPredictionsTest = accuracy_score(Y_TEST, pred, normalize = False)
        print ("Correct Predictions on Test Data:", correctPredictionsTest)
        lossOnTest = hinge_loss(Y_TEST[:,0], pred)
        print ("Hinge Loss on Test Data", lossOnTest)
        #pred = autograd.Variable(torch.Tensor(clf.predict_proba(X_TEST)[:,1]))
        #truelabel = autograd.Variable(torch.Tensor(Y_TEST[:,0]))
        #print("Binary Cross Entropy on Test Data",F.binary_cross_entropy(pred, truelabel))
        
        predt = clf.predict(X_TRAIN)
        accuracyTrain = accuracy_score(Y_TRAIN, predt)
        print ( "Accuracy on Train Data:", accuracyTrain)
        correctPredictionsTrain = accuracy_score(Y_TRAIN, predt, normalize = False)
        print ("Correct Predictions on Train Data:", correctPredictionsTrain)
        lossOnTrain = hinge_loss(Y_TRAIN, predt)
        print ("Hinge Loss on Train Data", lossOnTrain)
        #pred = autograd.Variable(torch.Tensor(clf.predict_proba(X_TRAIN)[:,1]))
        #truelabel = autograd.Variable(torch.Tensor(Y_TRAIN[:,0]))
        #print("Binary Cross Entropy on Train Data",F.binary_cross_entropy(pred, truelabel))
    
     
        RBF_C = 0.1
        clf = svm.SVC(kernel='rbf', gamma=0.7, C=RBF_C,  probability=True).fit(X, y)
        clf.fit(X, y) 
        pred = clf.predict(X_TEST)
        
        print ("\nRBF Kernel")
        accuracyTest = accuracy_score(Y_TEST, pred)
        print ( "Accuracy on Test Data:", accuracyTest)
        correctPredictionsTest = accuracy_score(Y_TEST, pred, normalize = False)
        print ("Correct Predictions on Test Data:", correctPredictionsTest)
        #lossOnTest = hinge_loss(Y_TEST[:,0], pred)
        #print ("Hinge Loss on Test Data", lossOnTest)
        pred = autograd.Variable(torch.Tensor(clf.predict_proba(X_TEST)[:,1]))
        truelabel = autograd.Variable(torch.Tensor(Y_TEST))
        lossOnTest = F.binary_cross_entropy(pred, truelabel)
        print("Binary Cross Entropy on Test Data",F.binary_cross_entropy(pred, truelabel))
        
        predt = clf.predict(X_TRAIN)
        accuracyTrain = accuracy_score(Y_TRAIN, predt)
        print ( "Accuracy on Train Data:", accuracyTrain)
        correctPredictionsTrain = accuracy_score(Y_TRAIN, predt, normalize = False)
        print ("Correct Predictions on Train Data:", correctPredictionsTrain)
        #lossOnTrain = hinge_loss(Y_TRAIN[:,0], predt)
        #print ("Hinge Loss on Train Data", lossOnTrain)
        pred = autograd.Variable(torch.Tensor(clf.predict_proba(X_TRAIN)[:,1]))
        truelabel = autograd.Variable(torch.Tensor(Y_TRAIN))
        lossOnTrain = F.binary_cross_entropy(pred, truelabel)
        print("Binary Cross Entropy on Train Data",F.binary_cross_entropy(pred, truelabel))
              
              
        POLY_C = 1
        clf = svm.SVC(kernel='poly', degree=1, C=POLY_C, probability = True).fit(X, y)
        clf.fit(X, y) 
        pred = clf.predict(X_TEST)
        
        print("\nPolynomial")
        

        
        accuracyTest = accuracy_score(Y_TEST, pred)
        print ( "Accuracy on Test Data:", accuracyTest)#, normalize=False))
        correctPredictionsTest = accuracy_score(Y_TEST, pred, normalize = False)
        print ("Correct Predictions on Test Data:", correctPredictionsTest)
        #lossOnTest = hinge_loss(Y_TEST[:,0], pred)
        #print ("Hinge Loss on Test Data", lossOnTest)
        pred = autograd.Variable(torch.Tensor(clf.predict_proba(X_TEST)[:,1]))
        truelabel = autograd.Variable(torch.Tensor(Y_TEST))
        lossOnTest = F.binary_cross_entropy(pred, truelabel)
        print("Binary Cross Entropy on Test Data",F.binary_cross_entropy(pred, truelabel))
        
        predt = clf.predict(X_TRAIN)
        accuracyTrain = accuracy_score(Y_TRAIN, predt)
        print ( "Accuracy on Train Data:", accuracyTrain)#, normalize=False))
        correctPredictionsTrain = accuracy_score(Y_TRAIN, predt, normalize = False)
        print ("Correct Predictions on Train Data:", correctPredictionsTrain)
        #lossOnTrain = hinge_loss(Y_TRAIN[:,0], predt)
        #print ("Hinge Loss on Train Data", lossOnTrain)
        pred = autograd.Variable(torch.Tensor(clf.predict_proba(X_TRAIN)[:,1]))
        truelabel = autograd.Variable(torch.Tensor(Y_TRAIN))
        lossOnTrain = F.binary_cross_entropy(pred, truelabel)
        print("Binary Cross Entropy on Train Data",F.binary_cross_entropy(pred, truelabel))
        
        
        predu = clf.predict(X_UNSEEN)
        accuracyunseen = accuracy_score(Y_UNSEEN, predu)
        print ( "Accuracy on Unseen Data:", accuracyunseen)#, normalize=False))
        correctPredictionsUnseen = accuracy_score(Y_UNSEEN, predu, normalize = False)
        print ("Correct Predictions on Unseen Data:", correctPredictionsUnseen)
        #lossOnTrain = hinge_loss(Y_TRAIN[:,0], predt)
        #print ("Hinge Loss on Train Data", lossOnTrain)
        predu = autograd.Variable(torch.Tensor(clf.predict_proba(X_UNSEEN)[:,1]))
        truelabel = autograd.Variable(torch.Tensor(Y_UNSEEN))
        lossOnUnseen = F.binary_cross_entropy(predu, truelabel)
        print("Binary Cross Entropy on Unseen Data",F.binary_cross_entropy(predu, truelabel))
        
        return  lossOnTrain, lossOnTest, lossOnUnseen, accuracyTrain, accuracyTest, accuracyunseen, correctPredictionsTrain, correctPredictionsTest, correctPredictionsUnseen