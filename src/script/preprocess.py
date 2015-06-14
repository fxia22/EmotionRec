import os
import fnmatch
import subprocess
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from sklearn import cross_validation
import pickle as pkl


r = '/Users/xiafei/Desktop/FaceTracker/build/bin/dataset/cohn-kanade-images/video'
for root, dir, files in os.walk(r):
        #print root
        mfiles = fnmatch.filter(files, "*.mp4")
        filenum = len(mfiles)
        if filenum > 0:
            for filename in mfiles:
                infile = root+'/'+filename
                outfile = infile.split('.')[0]+'.txt'
                command = './tracker -i '+infile+' -o' + outfile
                #print command
                #os.system(command)
                
X = []
y = []
       
                
for root, dir, files in os.walk(r):
        print root
        mfiles = fnmatch.filter(files, "*.txt")
        filenum = len(mfiles)
        if filenum > 0:
            for filename in mfiles:
                infile = root+'/'+filename
                with open(infile,'r') as f:
                    lines = f.readlines()
                    feature = [int(line.strip().split(',')[-1]) for line in lines]
                    label = int(filename.split('emotion')[-1][0])
                f.close()
                print feature,label
                if len(feature) == 132:
                    X.append(feature)
                    y.append(label)
                
                
print X
print y
X = np.array(X)
y = np.array(y)

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

pkl.dump({'X':X,'y':y},open('data.pkl','w'))