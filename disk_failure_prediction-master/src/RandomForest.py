import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import warnings
import argparse
from utils.loadData import read_data1 as read_data
from utils.evaluation import calMetrix

# from sklearn import tree
# from sklearn.externals.six import StringIO
# import pydot
# from IPython.display import Image

warnings.filterwarnings("ignore", category=DeprecationWarning)


def classifyRes(arr):
    arr = arr.reshape(arr.size,)
    for i in range(arr.size):
        arr[i] = 1 if arr[i]>0.5 else 0
    return arr


####RandomForest regression####
def random_forest_regressor(train_x, train_y):
    from sklearn import ensemble
    model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=2000)
    # 2000 decision trees are used here
    model_RandomForestRegressor.fit(train_x, train_y)
    return model_RandomForestRegressor

####RandomForest classification####
def random_forest_classifier(train_x, train_y):
    from sklearn import ensemble
    model_RandomForestClassifier = ensemble.RandomForestClassifier(n_estimators=2000)
    model_RandomForestClassifier.fit(train_x, train_y)
    return model_RandomForestClassifier


def model_fit(train_X, train_y, test_X, test_y):
    model_save_file = ''
    model_save = {}

    test_regressor = ['RF']
    regressors = {
                   'RF': random_forest_classifier,
                   }
    print('reading training and testing data...')
    # train_x, train_y, test_x, test_y = read_data()

    for regressor in test_regressor:
        start_time = time.time()
        model = regressors[regressor](train_X, train_y)
        print('training took %fs!' % (time.time() - start_time))
        t0 = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        pickle.dump(model, open('../model-2/RandomForest_model_%s.h5' % t0, 'wb'))
        predict = model.predict(test_X)
        # print('test_y: {}\npredict: {}'.format(test_y, predict))
        # score = model.score(test_x, test_y)
        calMetrix(__file__, predict, test_y)
        plt.figure()
        plt.plot(np.arange(len(predict)), test_y, 'go-', label='true value')
        plt.plot(np.arange(len(predict)), predict, 'ro-', label='predict value')
        plt.title('%s' % regressor)
        plt.legend()
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="train data path")
    parser.add_argument("--xpath", help="Xdata Path", default='', type=str,  required=True)
    parser.add_argument("--ypath", help="ydata Path", default='', type=str, required=True)
    parser.add_argument("--group", help="group", default='', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    for train_X, train_y, test_X, test_y in read_data(args.xpath, args.ypath, args.group):
        model_fit(train_X, train_y, test_X, test_y)
