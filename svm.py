from sklearn import datasets
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, hinge_loss
from sklearn.svm import SVC
from data_utils import *


class Svm(object):
    from sklearn.svm import LinearSVC, SVC
    def baseline_1(self, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, num_epochs =1000, learning_rate = 0.001, num_classes=1):
        X = X_TRAIN[:, :2]
        y = Y_TRAIN[:,0]
        svc_C = 0.1
        lin_C = 0.01
        rbf_C = 0.1
        poly_C =1
        # SVC with linear kernel
        svc = svm.SVC(kernel='linear', C=svc_C).fit(X, y)
        # LinearSVC (linear kernel)
        lin_svc = svm.LinearSVC(C=lin_C).fit(X, y)
        # SVC with RBF kernel
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=rbf_C).fit(X, y)
        # SVC with polynomial (degree 3) kernel
        poly_svc = svm.SVC(kernel='poly', degree=2, C=poly_C).fit(X, y)

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
            accuracyTest = accuracy_score(Y_TEST[:,0], pred)
            print ( "Accuracy on Test Data:", accuracyTest)
            correctPredictionsTest = accuracy_score(Y_TEST[:,0], pred, normalize = False)
            print ("Correct Predictions on Test Data:", correctPredictionsTest)
            lossOnTest = hinge_loss(Y_TEST[:,0], pred)
            print ("Hinge Loss on Test Data", lossOnTest)
            
            predt = clf.predict(X_TRAIN[:,:2])
            accuracyTrain = accuracy_score(Y_TRAIN[:,0], predt)
            print ("Accuracy on Train Data:", accuracyTrain)
            correctPredictionsTrain = accuracy_score(Y_TRAIN[:,0], predt, normalize = False)
            print ("Correct Predictions on Train Data:", correctPredictionsTrain)
            lossOnTrain = hinge_loss(Y_TRAIN[:,0], predt)
            print ("Hinge Loss on Train Data", lossOnTrain)

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
    
    def baseline_2(self, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, num_epochs =1000, learning_rate = 0.001, num_classes=1):        
        X = X_TRAIN
        y=Y_TRAIN[:,0]

        titles = ['SVC with linear kernel','LinearSVC (linear kernel)','SVC with RBF kernel','SVC with polynomial (degree 3) kernel']
    
        SVC_C = 0.1
        clf = svm.SVC(kernel='linear', C=SVC_C,  max_iter=num_epochs)
        clf.fit(X, y) 
        pred = clf.predict(X_TEST)
        
        print ("\nSVC")
        accuracyTest = accuracy_score(Y_TEST[:,0], pred)
        print ( "Accuracy on Test Data:", accuracyTest)#, normalize=False))
        correctPredictionsTest = accuracy_score(Y_TEST[:,0], pred, normalize = False)
        print ("Correct Predictions on Test Data:", correctPredictionsTest)
        lossOnTest = hinge_loss(Y_TEST[:,0], pred)
        print ("Hinge Loss on Test Data", lossOnTest)
        
        predt = clf.predict(X_TRAIN)
        accuracyTrain = accuracy_score(Y_TRAIN[:,0], predt)
        print ( "Accuracy on Train Data:", accuracyTrain)
        correctPredictionsTrain = accuracy_score(Y_TRAIN[:,0], predt, normalize = False)
        print ("Correct Predictions on Train Data:", correctPredictionsTrain)
        lossOnTrain = hinge_loss(Y_TRAIN[:,0], predt)
        print ("Hinge Loss on Train Data", lossOnTrain)
    
    
        LIN_C = 1.0
        clf = svm.LinearSVC(C=LIN_C).fit(X, y)
        clf.fit(X, y) 
        pred = clf.predict(X_TEST)
        
        print ("\nLinear SVC")
        accuracyTest = accuracy_score(Y_TEST[:,0], pred)
        print ( "Accuracy on Test Data:", accuracyTest)
        correctPredictionsTest = accuracy_score(Y_TEST[:,0], pred, normalize = False)
        print ("Correct Predictions on Test Data:", correctPredictionsTest)
        lossOnTest = hinge_loss(Y_TEST[:,0], pred)
        print ("Hinge Loss on Test Data", lossOnTest)
        
        predt = clf.predict(X_TRAIN)
        accuracyTrain = accuracy_score(Y_TRAIN[:,0], predt)
        print ( "Accuracy on Train Data:", accuracyTrain)
        correctPredictionsTrain = accuracy_score(Y_TRAIN[:,0], predt, normalize = False)
        print ("Correct Predictions on Train Data:", correctPredictionsTrain)
        lossOnTrain = hinge_loss(Y_TRAIN[:,0], predt)
        print ("Hinge Loss on Train Data", lossOnTrain)
    
     
        RBF_C = 0.1
        clf = svm.SVC(kernel='rbf', gamma=0.7, C=RBF_C).fit(X, y)
        clf.fit(X, y) 
        pred = clf.predict(X_TEST)
        
        print ("\nRBF Kernel")
        accuracyTest = accuracy_score(Y_TEST[:,0], pred)
        print ( "Accuracy on Test Data:", accuracyTest)
        correctPredictionsTest = accuracy_score(Y_TEST[:,0], pred, normalize = False)
        print ("Correct Predictions on Test Data:", correctPredictionsTest)
        lossOnTest = hinge_loss(Y_TEST[:,0], pred)
        print ("Hinge Loss on Test Data", lossOnTest)
        
        predt = clf.predict(X_TRAIN)
        accuracyTrain = accuracy_score(Y_TRAIN[:,0], predt)
        print ( "Accuracy on Train Data:", accuracyTrain)
        correctPredictionsTrain = accuracy_score(Y_TRAIN[:,0], predt, normalize = False)
        print ("Correct Predictions on Train Data:", correctPredictionsTrain)
        lossOnTrain = hinge_loss(Y_TRAIN[:,0], predt)
        print ("Hinge Loss on Train Data", lossOnTrain)
    
    
        POLY_C = 1
        clf = svm.SVC(kernel='poly', degree=1, C=POLY_C).fit(X, y)
        clf.fit(X, y) 
        pred = clf.predict(X_TEST)
        
        print("\nPolynomial")
        accuracyTest = accuracy_score(Y_TEST[:,0], pred)
        print ( "Accuracy on Test Data:", accuracyTest)#, normalize=False))
        correctPredictionsTest = accuracy_score(Y_TEST[:,0], pred, normalize = False)
        print ("Correct Predictions on Test Data:", correctPredictionsTest)
        lossOnTest = hinge_loss(Y_TEST[:,0], pred)
        print ("Hinge Loss on Test Data", lossOnTest)
        
        predt = clf.predict(X_TRAIN)
        accuracyTrain = accuracy_score(Y_TRAIN[:,0], predt)
        print ( "Accuracy on Train Data:", accuracyTrain)#, normalize=False))
        correctPredictionsTrain = accuracy_score(Y_TRAIN[:,0], predt, normalize = False)
        print ("Correct Predictions on Train Data:", correctPredictionsTrain)
        lossOnTrain = hinge_loss(Y_TRAIN[:,0], predt)
        print ("Hinge Loss on Train Data", lossOnTrain)
    
        return  lossOnTrain, lossOnTest, accuracyTrain, accuracyTest, correctPredictionsTrain, correctPredictionsTest

############CALL###############
s = Svm()
print("BASELINE ALGORITHM -1")
s.baseline_1(X_TRAIN = X_TRAIN, Y_TRAIN = Y_TRAIN, X_TEST = X_TEST, Y_TEST=Y_TEST)
print("######################################")
print("BASELINE ALGORITHM -2")
s.baseline_2(X_TRAIN = X_TRAIN, Y_TRAIN = Y_TRAIN, X_TEST = X_TEST, Y_TEST=Y_TEST)



    


