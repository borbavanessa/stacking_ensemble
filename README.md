# stacking_ensemble
## **A Deep Learning Ensemble to Classify Anxiety, Depression, and their Comorbidity from Texts of Social Networks** ##

This repository documents experiments and results of the proposing to an ensemble stacking classifier to automatically identify depression, anxiety, and comorbidity, using data extracted from Reddit. The stacking is composed of specialized single-label binary classifiers that distinguish between specific disorders and control users. For the development of the weak classifiers, we experiment with alternative architectures (LSTM, CNN, and their combination), and word embeddings. A meta-learner explores these weak classifiers as a context for reaching a multi-label, multi-class decision. 
![stacking_ensemble_topology.png](https://github.com/borbavanessa/stacking_ensemble/blob/master/images/stacking_ensemble_topology.png)

To run the project it is necessary to install an environment containing the packages:

* Python 3.6 or higher and its set of libraries for machine learning (Sckit-learning, NumPy, Pandas)
* Keras 2.2.5
* Tensorflow 1.14.0
* Jupyter Notebook (if you want to run the experiments of the Jupyter files)

The dataset used for developing this model was made available for this work under a data usage contract and, for this reason, is not available with the project.
