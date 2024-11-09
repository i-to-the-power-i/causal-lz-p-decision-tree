from DistanceDecisionTree import DistanceDecisionTree
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
PATH = os.getcwd()+'/trees/'
def getTestResults(X_train, Y_train, min_samples, max_depth, X_test, Y_test,filename,list = None):
    '''Input: X_train, Y_train, min samples, max depth, X test, Y test
    
    Output: accuracy, F1 score, precision score and recall score'''

    clf = DistanceDecisionTree(min_samples_split=min_samples, max_depth=max_depth)
    clf.fit(X_train,Y_train)
    prediction = clf.predict(X_test)
    dot = clf.export_graphviz(list=list)
    dot.render(PATH+filename,format="png")


    def accuracy(y_test,y_pred):
        return (np.sum(y_pred==y_test)/len(y_test))

    acc=accuracy(Y_test, prediction)
    f1 = f1_score(Y_test, prediction,average = 'macro')
    prec = precision_score(Y_test,prediction,average='macro')
    recall = recall_score(Y_test,prediction,average='macro')

    return acc, f1, prec, recall


  
    