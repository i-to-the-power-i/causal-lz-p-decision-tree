import numpy as np
from sklearn.metrics import f1_score
import numpy as np
from DistanceDecisionTree import DistanceDecisionTree
from sklearn.model_selection import KFold, TimeSeriesSplit, StratifiedKFold

K_FOLD = 5

def getF1(X,y,min_samples,max_depth):   
    '''Input: Train data, Test data, minimum number of samples at each node, maximum depth of tree
    
    Output: Mean and variance of F1 score after training'''

    tscv = StratifiedKFold(n_splits=K_FOLD)

    model = DistanceDecisionTree(min_samples_split = min_samples, max_depth = max_depth)
    macro_f1_scores = []

    # Perform 5-fold cross-validation

    for train_index, test_index in tscv.split(X,y):

        # Split the data into training and testing sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print(X_train.size)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred, average='macro')  # Calculate macro F1
        macro_f1_scores.append(f1)  # Append to the list


    macro_f1_scores = np.array(macro_f1_scores)

    average_macro_f1 = np.mean(macro_f1_scores)
    variance_f1 = np.var(macro_f1_scores)

    # print("Macro F1 scores for each fold:", macro_f1_scores)
    # print("Average Macro F1 score across 5 folds:", average_macro_f1)
    print(f"Macro F1 scores for min samples: {min_samples} and max depth: {max_depth} :", macro_f1_scores)
    return average_macro_f1, variance_f1


def getParams(X_train,Y_train):
    '''Input: Training Data
    
    Output: Tuned hyperparameters: min samples, at a node, depth of tree as well as mean and variance of f1 score'''

    maximum_f1_mean = 0
    f1_var=0
    min_samples = 0
    max_depth = 0
    # print("hi")
    for no_samples in range(2,10):
        for depth in range(4,21):
            curr_f1,var = getF1(X_train,Y_train,no_samples,depth)
            # print("meow")
            if (curr_f1 > maximum_f1_mean):
                maximum_f1_mean = curr_f1
                f1_var = var
                min_samples = no_samples
                max_depth = depth

            if (curr_f1 >= maximum_f1_mean):
                min_samples = no_samples          
 
    return min_samples, max_depth,maximum_f1_mean, f1_var