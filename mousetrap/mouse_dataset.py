"""
dataloader.py - PyTorch dataloader for our dataset.
"""

import numpy as np
import pandas as pd
import torch
import torch.utils.data


# length of padded data if that option is chosen
_UNIFORM_DATA_LENGTH = 300

# number of classes
_NUM_CLASSES = 2


def _load_data_yu(filepath: str) -> pd.DataFrame:
    """Loads the data from the csv in `filepath`."""
    df = pd.read_csv(filepath, sep=' ', header=None, encoding='utf-8', names=['id', 'data', 'target', 'label'])
    df['data'] = df['data'].apply(lambda x: [list(map(float, point.split(','))) for point in x.split(';')[:-1]])

    return df


def _preprocess(df: pd.DataFrame, column_name: str = 'data') -> pd.DataFrame:
    # TODO
    return df


def _pad(l):
    if (len(l) < _UNIFORM_DATA_LENGTH):
        endp = l[-1]
        for i in range(len(l), 300):
            l.append(endp)
    return l


def _label_to_probs(d): 
    outs = np.zeros(_NUM_CLASSES, dtype=np.float32)
    outs[d] = 1.0
    outs = torch.from_numpy(outs)
    return outs

def _process_labels(l):
    if l == 1:
        return 0
    else:
        return 1
    

class MouseMovementDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for mouse cursor movements.
    """

    def __init__(self, filepath: str, pad: bool = True, transform=None):
        self.transform = transform
        self.df = _load_data_yu(filepath)
        
        if (pad):
            self.df['data'] = self.df['data'].apply(_pad)
        
        self.df['label'] = self.df['label'].apply(_process_labels)
        
        # make the labels a prob distribution between classes
        # self.df['label'] = self.df['label'].apply(_label_to_probs)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get time series data from dataframe
        if isinstance(idx, list):
            data = [np.array(x, dtype=np.float32).T for x in self.df.iloc[idx]['data'].to_list()]
        else: 
            data = [np.array(self.df.iloc[idx]['data'], dtype=np.float32).T]

        # get labels from dataframe
        label = self.df.iloc[idx]['label'].to_list() if isinstance(idx, list) else self.df.iloc[idx]['label']

        # apply transforms
        if (self.transform):
            for i, t in enumerate(data):
                data[i] = self.transform(t)
                data[i] = torch.squeeze(data[i])
        
        sample = {'data': data, 'label': label}
        return sample

