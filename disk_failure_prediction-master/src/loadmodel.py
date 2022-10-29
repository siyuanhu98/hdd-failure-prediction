from keras.models import load_model
import pickle
import argparse
import numpy as np
from sklearn import metrics

def model_pre(model_path,x_path,y_path):
    #model = load_model(model_path)
    model = pickle.load(open(model_path,'rb'))
    x_sample = np.load(x_path)
    y_sample = np.load(y_path)
    L = x_sample.shape[0]
    X = x_sample.reshape(L, -1)
    Y = y_sample.reshape(L,)
    predict_y = model.predict(X)
    print(predict_y)
    print(Y)
    print(metrics.confusion_matrix(Y, predict_y))
    print('precision_score: ')
    print(metrics.precision_score(Y, predict_y))
    print(metrics.recall_score(Y, predict_y, average='binary'))


def parse_args():
    parser = argparse.ArgumentParser(description="train data path")
    parser.add_argument("--modelpath", help="model Path", type=str, required=True)
    parser.add_argument("--xpath", help="Xdata Path", default='X_sample.npy', type=str,  required=True)
    parser.add_argument("--ypath", help="Ydata Path", default='Y_sample.npy', type=str, required=True)
    return parser.parse_args()
#model = load_model('./CNN_LSTM_model_20211110111259.h5')
#model = pickle.load('./CNN_LSTM_model_20211110111259.h5')
if __name__ == '__main__':
    args = parse_args()
    model_pre(args.modelpath, args.xpath, args.ypath)
