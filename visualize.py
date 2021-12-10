import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def visual(path='Mousecollector/records.txt'):
    train = pd.read_csv(path, sep=' ', header=None, encoding='utf-8', names=['data'])
    data = train['data'].apply(lambda x: [list(map(float, point.split(','))) for point in x.split(';')[:-1]])
    for line in data:
        x = [xy[0] for xy in line]
        y = [xy[1] for xy in line]
        plt.figure()
        plt.plot(x, y)
        plt.show()


if __name__ == '__main__':
    visual()
