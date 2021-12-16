# Human-Bot-Classifier-based-on-mouse-movement

Proposed name: Mousetrap

A simple classifier that uses machine learning methods to classify a certain string of mouse movements to be either performed by a human or a bot. 

For basic CNN in PyTorch, see `sandbox.ipynb`.
For GBDT, Ramdom Forest and feature extraction, see 'GBDT.py'.
For VAR, see 'VAR.ipynb'.
For LSTM, see 'LSTM.ipynb'

## Dataset Description
### Dataset 1
Located in data/dsjtzs_txfz_training.txt and dsjtzs_txfz_test1.txt.

Our dataset includes 3000 entries of labelled training data and 102 entries of *unlabeled* testing data.

Each entry in the training set contains: id, mouse trajectory(x,y,t separated by ",", and each data point is separated by ";"), a final (x,y) destination coordinate, and label (1 for human, 0 for bot).

Each entry in the testing set contains: id, mouse trajectory(x,y,t separated by ",", and each data point is separated by ";"), and a final (x,y) destination coordinate.

### Dataset 2
For human data, see data/records1.txt. For bot data, see data/gc.csv(200 entries) and data/gc2.cvs(1000 entries).

This dataset contains roughly 300 entries of labeled human mouse trajectories and 1200 entries of labeled synthesized human-like mouse trajectories.

Each entry in the human dataset contains: mouse trajectory(x,y separated by ",", and each data point is separated by ";")

Each entry in the bot dataset contains: id, mouse trajectory(x,y,t separated by ",", and each data point is separated by ";"), and a final (x,y) destination coordinate.
