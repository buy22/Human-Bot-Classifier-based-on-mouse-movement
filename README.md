# Human-Bot-Classifier-based-on-mouse-movement

Proposed name: Mousetrap

A simple classifier that uses machine learning methods to classify a certain string of mouse movements to be either performed by a human or a bot. 

## Dataset Description

Our dataset includes 3000 entries of labelled training data and 102 entries of *unlabeled* testing data.

Each entry in the training set contains: id, mouse trajectory(x,y,t separated by ",", and each data point is separated by ";"), a final (x,y) destination coordinate, and label (1 for human, 0 for bot).

Each entry in the testing set contains: id, mouse trajectory(x,y,t separated by ",", and each data point is separated by ";"), and a final (x,y) destination coordinate.
