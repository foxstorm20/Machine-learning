import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from operator import itemgetter

diamonds = pd.read_csv("diamonds.csv")


def task1(data):
    groups = data.groupby(['cut', 'color', 'clarity']).groups
    bos = []
    list_of_features = []
    list_of_target = []
    for i in groups:
        if len(data[(data['cut'] == list(i)[0]) & (data['color'] == list(i)[1]) & (data['clarity'] == list(i)[2])]) > 800:
            kps = data[(data['cut'] == list(i)[0]) & (data['color'] == list(i)[1]) & (data['clarity'] == list(i)[2])]
            feature = kps[['carat', 'depth', 'table']]
            target = kps['price']
            print(str(i) + ": " + str(len(kps)))
            bos.append(list(i))
            list_of_features.append(feature)
            list_of_target.append(target)
    return bos, list_of_features, list_of_target


labels_set, features_set, targets_set = task1(diamonds)


def calculate_model_function(deg, data, p):
    result = np.zeros(data.shape[0])
    k = 0
    for n in range(deg+1):
        for i in range(n+1):
            for j in range(n+1):
                for v in range(n+1):
                    if i+j+v == n:
                        result += p[k] * (data.iloc[:, 0]**i)*(data.iloc[:, 1]**j)*(data.iloc[:, 2]**v)
                        k += 1
    return result


def num_coefficients_3(d):
    t = 0
    for n in range(d+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k == n:
                        t = t+1
    return t


def linearize(deg, data, p0):
    f0 = calculate_model_function(deg, data, p0)
    l = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = calculate_model_function(deg, data, p0)
        p0[i] -= epsilon
        di = (fi - f0)/epsilon
        l[:,i] = di
    return f0, l


def calculate_update(y, f0, J):
    l=1e-2
    N = np.matmul(J.T, J) + l*np.eye(J.shape[1])
    r = y-f0
    n = np.matmul(J.T,r)
    dp = np.linalg.solve(N, n)
    return dp


def regression(deg, target, feature):
    max_iter = 10
    p0 = np.zeros(num_coefficients_3(deg))
    for i in range(max_iter):
        f0, J = linearize(deg, feature, p0)
        dp = calculate_update(target, f0, J)
        p0 += dp
    return p0


def model_selection(combination, feature_datasets, target_datasets):
    zip_it_up = []
    split=3
    for comb, feature, target in zip(combination, feature_datasets, target_datasets):
        difference_tested = []
        kf2 = KFold(n_splits=split, shuffle=False, random_state=None)
        print('===============\ntesting for dataset'+str(comb)+"\n=========")
        for degree in range(3 + 1):
            ave = 0
            print('testing for degree:' + str(degree))
            for training_index, validation_index in kf2.split(feature, target):
                j = regression(degree, target.iloc[training_index], feature.iloc[training_index])
                test_ld = calculate_model_function(degree, feature.iloc[validation_index], j)
                difference = np.mean(np.abs(test_ld - target.iloc[validation_index]))
                ave += difference
                print(difference)
            ave = ave/split
            difference_tested.append([degree, ave])
        difference_tested = np.array(difference_tested)
        print("Ideal degree for " + str(comb) + " :"+str(np.argmin(difference_tested[:,1])))
        zip_it_up.append(np.argmin(difference_tested[:,1]))
    return zip_it_up, feature_datasets, target_datasets


optimal_degrees, tested_features, tested_target = model_selection(labels_set, features_set, targets_set)


def visualization(degrees, label_datasets, features_datasets, target_datasets):
    for degree, label, feature, target in zip(degrees, label_datasets, features_datasets, target_datasets):
        X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.33, random_state=42)
        j = regression(degree, y_train, X_train)
        test_ld = calculate_model_function(degree, X_test, j)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(test_ld, y_test, c='r')
        plt.title(str(label))
        plt.xlabel('Estimated target vectors')
        plt.ylabel('Actual target vectors')
        plt.show()


visualization(optimal_degrees, labels_set, tested_features, tested_target)
