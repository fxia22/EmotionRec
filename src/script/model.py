import os
import fnmatch
import subprocess
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn import cross_validation
import pickle as pkl

def hello_world():
    print 'Hello World from Python'

class Model:
    def __init__(self):
        print 'instance created'
        pass
    
    def load_data(self):
        with open('data.pkl') as f:
            self.data = pkl.load(f)
        f.close()
        print 'data loaded'
    def train_model(self):
        X = self.data['X']
        y = self.data['y']
        self.model = OneVsRestClassifier(LinearSVC()).fit(X, y)
        print 'model trained'
    def test_performance(self):
        X = self.data['X']
        y = self.data['y']
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2, random_state=8)
        l = [len(item) for item in X]
        y_train_pred = OneVsRestClassifier(LinearSVC()).fit(X_train, y_train).predict(X_train)
        print y_train_pred,y_train
        y_test_pred = OneVsRestClassifier(LinearSVC()).fit(X_train, y_train).predict(X_test)
        print y_test_pred,y_test

        corr = 0;
        for i in range(len(y_train)):
            if y_train_pred[i] == y_train[i]:
                corr += 1
        print 'training set accuracy:',corr/float(len(y_train))

        corr = 0;
        for i in range(len(y_test)):
            if y_test_pred[i] == y_test[i]:
                corr += 1
        print 'testing set accuracy:',corr/float(len(y_test))
    def make_prediction(self,x_test):
        #print x_test
        #print len(x_test)
        pred = self.model.predict(x_test)
        print pred[0]
        return pred[0]
    def fuck(self,a):
        print a[0]
        print a[1]
        