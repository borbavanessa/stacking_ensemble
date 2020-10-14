
# Definition using for network models

from enum import Enum

## Use_embedding_pre_train values:
## None - without embedding layer
## Rand - embedding learning join at network;
## Static - to used embedding pre-train;
## NonStatic - to used embedding pre-train and continue learning with network
class UseEmbedding(Enum):
		NONE = 1
		RAND = 2
		STATIC = 3
		NON_STATIC = 4

class EmbeddingType(Enum):
		NONE = 1
		GLOVE_6B = 2
		GLOVE_TWITTER = 3
		WORD2VEC = 4
		DOMAIN_DICTIONARY = 5
		WORD2VEC_CUSTOM = 6
		GLOVE_CUSTOM = 7
		
## Input Data Type
class InputData(Enum):
		POSTS_ONLY_TEXT = 1 #Tensor 2D
		POSTS_LIST = 2      #Tensor 3D

class LoadDataset(Enum):
		TRAIN_DATA_MODEL = 1
		TEST_DATA_MODEL = 2
		ALL_DATA_MODEL = 3
		
class TypePredictionLabel(Enum):
		BINARY = 1
		BINARY_CATEGORICAL = 2
		MULTI_LABEL_CATEGORICAL = 3
		SINGLE_LABEL_CATEGORICAL = 4
		
class OptimizerFunction(Enum):
		ADAM = 0
		ADAMAX = 1
		NADAM = 2
		ADADELTA = 3
		ADAGRAD = 4

OPTIMIZER_FUNCTIONS = ['adam','adamax','nadam','adadelta','adagrad']

# Enum for performance analysis ml models
class ExplainerTypeShap(Enum):
		DEEP = 1
		KERNEL = 2
		TREE = 3

class TypeModel(Enum):
		STACKED_ENSEMBLE = 1
		CUSTOM_ENSEMBLE = 2

USE_TENSOR_BOARD = False

# General Data
# SMHD dep, if batch-size=40 and total=2120, if batch-size=20 and total=2140
TOTAL_REGISTERS = 5

K_FOLD = 10
REPEAT_VALIDATION = 1 #minimun 1

SEED = 1234  # random seed

TOTAL_GPUS = 2

LEARNING_RATE = 0.01

# For multilabel classifier, the categoricals are compose according pathologies order, exemple
# ['control', 'anxiety', 'depression'] => this case define the subdirectory in PreprocessData.
# ['control', 'anxiety,depression'] ==> this case the multilabel is more one class (dataset original),
#                                       it's not need subdirectory
LABEL_SET = ['control', 'anxiety', 'depression']

DATASET_NAME = 'SMHD' #RSDD #SMHD

MIN_TOTAL_POSTS_USER = 50

PATH_PROJECT = "/home/deep-learning-for-mental-health"
