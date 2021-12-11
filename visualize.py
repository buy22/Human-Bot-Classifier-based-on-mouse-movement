import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def visual1(path='data/dsjtzs_txfz_training.txt'):
    train = pd.read_csv(path, sep=' ', header=None, encoding='utf-8', names=['id', 'data', 'target', 'label'])
    data = train['data'].apply(lambda x: [list(map(float, point.split(',')))[:-1] for point in x.split(';')[:-1]])
    human = list(data[:2600])
    bot = list(data[2600:])
    for line in bot:
        x = [xy[0] for xy in line]
        y = [xy[1] for xy in line]
        plt.figure()
        plt.plot(x, y)
        plt.show()


def visual2(path='Mousecollector/records.txt'):
    train = pd.read_csv(path, sep=' ', header=None, encoding='utf-8', names=['data'])
    data = train['data'].apply(lambda x: [list(map(float, point.split(','))) for point in x.split(';')[:-1]])
    for line in data:
        x = [xy[0] for xy in line]
        y = [xy[1] for xy in line]
        plt.figure()
        plt.plot(x, y)
        plt.show()


def visual3(path='data/gc.csv'):
    train = pd.read_csv(path, sep=' ', header=None, encoding='utf-8', names=['id','data','_','1','end'])
    data = train['data'].apply(lambda x: [list(map(float, point.split(';'))) for idx, point in enumerate(x.split(',')) if idx % 2 == 0][2:-2])
    for line in data:
        x = [xy[0] for xy in line]
        y = [xy[1] for xy in line]
        plt.figure()
        plt.plot(x, y)
        plt.show()


if __name__ == '__main__':
    visual1()
