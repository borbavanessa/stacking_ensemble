# Load libs standard python and custom
import numpy as np
import datetime
import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from network_model.model_class import ModelClass
from utils.experiment_processes import ExperimentProcesses

import utils.definition_network as dn

# MULTILABEL Not VALID train
# Model lstm_exp9_var_L3_N16_B40_E32_D0.2 static glove com kernel_initializer=glorot_uniform=xavier_uniform
# For SMHD dep, anx
def generate_model_nv_1(exp, name_model):
    exp.pp_data.vocabulary_size = 5000

    exp.pp_data.embedding_size = 300
    exp.pp_data.max_posts = 1750
    exp.pp_data.max_terms_by_post = 300
    exp.pp_data.binary_classifier = True
    exp.pp_data.format_input_data = dn.InputData.POSTS_ONLY_TEXT
    exp.pp_data.remove_stopwords = False
    exp.pp_data.delete_low_tfid = False
    exp.pp_data.min_df = 0
    exp.pp_data.min_tf = 0
    exp.pp_data.random_posts = False
    exp.pp_data.random_users = False
    exp.pp_data.tokenizing_type = 'WE'
    exp.pp_data.word_embedding_custom_file = ''
    exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_6B
    exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
    exp.pp_data.word_embedding_custom_file = ''
    exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
    exp.pp_data.type_prediction_label= dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL

    exp.use_custom_metrics = False
    exp.use_valid_set_for_train = False
    exp.valid_split_from_train_set = 0.2
    exp.imbalanced_classes = True

    we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                                 '_EF_' + 'glove6B300d_glorot'

    lstm = ModelClass(1)
    lstm.loss_function = 'binary_crossentropy'
    lstm.optmizer_function = 'adam'
    lstm.epochs = 15
    lstm.batch_size = 32
    lstm.patience_train = 10
    lstm.use_embedding_pre_train = exp.pp_data.use_embedding
    lstm.embed_trainable = (lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

    neuronios_by_layer = [16]
    epochs = [64]
    batch_sizes = [40]
    dropouts = [0.2]

    np.random.seed(dn.SEED)

    time_ini_rep = datetime.datetime.now()
    x_train, y_train, x_valid, y_valid, num_words, embedding_matrix = exp.pp_data.load_data()
    exp.set_period_time_end(time_ini_rep, 'Load data')

    for neuronios in neuronios_by_layer:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for dropout in dropouts:
                    lstm.epochs = epoch
                    lstm.batch_size = batch_size
                    lstm.patience_train = epoch / 2
                    exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
                            batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name

                    lstm.model = Sequential()
                    lstm.model.add(Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size,
                                             trainable=lstm.embed_trainable, name='emb_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        return_sequences=True, name='dense_1_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        return_sequences=True, name='dense_2_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        name='dense_3_' + name_model))
                    lstm.model.add(Dense(3,
                                         activation='sigmoid',
                                         name='dense_4_' + name_model))

                    time_ini_exp = datetime.datetime.now()
                    exp.generate_model_hypeparams(lstm, x_train, y_train, x_valid, y_valid, embedding_matrix)
                    exp.set_period_time_end(time_ini_exp, 'Total experiment')

    del x_train, y_train, x_valid, y_valid, num_words, embedding_matrix

    # Test
    exp.pp_data.load_dataset_type = dn.LoadDataset.TEST_DATA_MODEL
    np.random.seed(dn.SEED)
    time_ini_rep = datetime.datetime.now()
    x_test, y_test = exp.pp_data.load_data()
    exp.set_period_time_end(time_ini_rep, 'Load data')

    for neuronios in neuronios_by_layer:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for dropout in dropouts:
                    lstm.epochs = epoch
                    lstm.batch_size = batch_size
                    lstm.patience_train = epoch / 2
                    exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
                            batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name

                    lstm.model = exp.load_model(dn.PATH_PROJECT + exp.experiment_name + '.h5')
                    exp.save_geral_configs()
                    exp.save_summary_model(lstm.model)
                    exp.predict_samples(lstm, x_test, y_test)

    del x_test, y_test, lstm, exp

# Model lstm_exp9_var_L3_N16_B40_E32_D0.2 static glove com kernel_initializer=lecun_uniform
# SMHD dep, anx
def generate_model_nv_2(exp, name_model):
    exp.pp_data.vocabulary_size = 5000

    exp.pp_data.embedding_size = 300
    exp.pp_data.max_posts = 1750
    exp.pp_data.max_terms_by_post = 300
    exp.pp_data.binary_classifier = True
    exp.pp_data.format_input_data = dn.InputData.POSTS_ONLY_TEXT
    exp.pp_data.remove_stopwords = False
    exp.pp_data.delete_low_tfid = False
    exp.pp_data.min_df = 0
    exp.pp_data.min_tf = 0
    exp.pp_data.random_posts = False
    exp.pp_data.random_users = False
    exp.pp_data.tokenizing_type = 'WE'
    exp.pp_data.word_embedding_custom_file = ''
    exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_6B
    exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
    exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
    exp.pp_data.type_prediction_label= dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL

    exp.use_custom_metrics = False
    exp.use_valid_set_for_train = False
    exp.valid_split_from_train_set = 0.2
    exp.imbalanced_classes = True

    we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                                 '_EF_' + 'glove6B300d_lecun'

    lstm = ModelClass(1)
    lstm.loss_function = 'binary_crossentropy'
    lstm.optmizer_function = 'adam'
    lstm.epochs = 15
    lstm.batch_size = 32
    lstm.patience_train = 10
    lstm.use_embedding_pre_train = exp.pp_data.use_embedding
    lstm.embed_trainable = (lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

    neuronios_by_layer = [16]
    epochs = [64]
    batch_sizes = [40]
    dropouts = [0.2]

    np.random.seed(dn.SEED)

    time_ini_rep = datetime.datetime.now()
    x_train, y_train, x_valid, y_valid, num_words, embedding_matrix = exp.pp_data.load_data()
    exp.set_period_time_end(time_ini_rep, 'Load data')

    for neuronios in neuronios_by_layer:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for dropout in dropouts:
                    lstm.epochs = epoch
                    lstm.batch_size = batch_size
                    lstm.patience_train = epoch / 2
                    exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
                            batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name

                    lstm.model = Sequential()
                    lstm.model.add(Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size,
                                             trainable=lstm.embed_trainable, name='emb_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        kernel_initializer='lecun_uniform', recurrent_initializer='orthogonal',
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        return_sequences=True, name='dense_1_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        kernel_initializer='lecun_uniform', recurrent_initializer='orthogonal',
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        return_sequences=True, name='dense_2_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        kernel_initializer='lecun_uniform', recurrent_initializer='orthogonal',
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        name='dense_3_' + name_model))
                    lstm.model.add(Dense(3,
                                         kernel_initializer='lecun_uniform',
                                         activation='sigmoid',
                                         name='dense_4_' + name_model))

                    time_ini_exp = datetime.datetime.now()
                    exp.generate_model_hypeparams(lstm, x_train, y_train, x_valid, y_valid, embedding_matrix)
                    exp.set_period_time_end(time_ini_exp, 'Total experiment')

    del x_train, y_train, x_valid, y_valid, num_words, embedding_matrix

    # Test
    exp.pp_data.load_dataset_type = dn.LoadDataset.TEST_DATA_MODEL
    np.random.seed(dn.SEED)
    time_ini_rep = datetime.datetime.now()
    x_test, y_test = exp.pp_data.load_data()
    exp.set_period_time_end(time_ini_rep, 'Load data')

    for neuronios in neuronios_by_layer:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for dropout in dropouts:
                    lstm.epochs = epoch
                    lstm.batch_size = batch_size
                    lstm.patience_train = epoch / 2
                    exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
                            batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name

                    lstm.model = exp.load_model(dn.PATH_PROJECT + exp.experiment_name + '.h5')
                    exp.save_geral_configs()
                    exp.save_summary_model(lstm.model)
                    exp.predict_samples(lstm, x_test, y_test)

    del x_test, y_test, lstm, exp

# MULTILABEL With VALID train
# Model lstm_exp9_var_L3_N16_B40_E32_D0.2 static glove com kernel_initializer=glorot_uniform=xavier_uniform
# SMHD anx, dep
def generate_model_wv_1(exp, name_model):
    exp.pp_data.vocabulary_size = 5000

    exp.pp_data.embedding_size = 300
    exp.pp_data.max_posts = 1750
    exp.pp_data.max_terms_by_post = 300
    exp.pp_data.binary_classifier = True
    exp.pp_data.format_input_data = dn.InputData.POSTS_ONLY_TEXT
    exp.pp_data.remove_stopwords = False
    exp.pp_data.delete_low_tfid = False
    exp.pp_data.min_df = 0
    exp.pp_data.min_tf = 0
    exp.pp_data.random_posts = False
    exp.pp_data.random_users = False
    exp.pp_data.tokenizing_type = 'WE'
    exp.pp_data.word_embedding_custom_file = ''
    exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_6B
    exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
    exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
    exp.pp_data.type_prediction_label= dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL

    exp.use_custom_metrics = False
    exp.use_valid_set_for_train = True
    exp.valid_split_from_train_set = 0.0
    exp.imbalanced_classes = True

    we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                                 '_EF_' + 'glove6B300d_glorot'

    lstm = ModelClass(1)
    lstm.loss_function = 'binary_crossentropy'
    lstm.optmizer_function = 'adam'
    lstm.epochs = 15
    lstm.batch_size = 32
    lstm.patience_train = 10
    lstm.use_embedding_pre_train = exp.pp_data.use_embedding
    lstm.embed_trainable = (lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

    neuronios_by_layer = [16]
    epochs = [32]
    batch_sizes = [40]
    dropouts = [0.2]

    np.random.seed(dn.SEED)

    time_ini_rep = datetime.datetime.now()
    x_train, y_train, x_valid, y_valid, num_words, embedding_matrix = exp.pp_data.load_data()
    exp.set_period_time_end(time_ini_rep, 'Load data')

    for neuronios in neuronios_by_layer:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for dropout in dropouts:
                    lstm.epochs = epoch
                    lstm.batch_size = batch_size
                    lstm.patience_train = epoch / 2
                    exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
                            batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name

                    lstm.model = Sequential()
                    lstm.model.add(Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size,
                                             trainable=lstm.embed_trainable, name='emb_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        return_sequences=True, name='dense_1_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        return_sequences=True, name='dense_2_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        name='dense_3_' + name_model))
                    lstm.model.add(Dense(3,
                                         activation='sigmoid',
                                         name='dense_4_' + name_model))

                    time_ini_exp = datetime.datetime.now()
                    exp.generate_model_hypeparams(lstm, x_train, y_train, x_valid, y_valid, embedding_matrix)
                    exp.set_period_time_end(time_ini_exp, 'Total experiment')

    del x_train, y_train, x_valid, y_valid, num_words, embedding_matrix

    # Test
    exp.pp_data.load_dataset_type = dn.LoadDataset.TEST_DATA_MODEL
    np.random.seed(dn.SEED)
    time_ini_rep = datetime.datetime.now()
    x_test, y_test = exp.pp_data.load_data()
    exp.set_period_time_end(time_ini_rep, 'Load data')

    for neuronios in neuronios_by_layer:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for dropout in dropouts:
                    lstm.epochs = epoch
                    lstm.batch_size = batch_size
                    lstm.patience_train = epoch / 2
                    exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
                            batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name

                    lstm.model = exp.load_model(dn.PATH_PROJECT + exp.experiment_name + '.h5')
                    exp.save_geral_configs()
                    exp.save_summary_model(lstm.model)
                    exp.predict_samples(lstm, x_test, y_test)

    del x_test, y_test, lstm, exp

# Model lstm_exp9_var_L3_N16_B40_E32_D0.2 static glove com kernel_initializer=lecun_uniform
# SMHD anx, dep
def generate_model_wv_2(exp, name_model):
    exp.pp_data.vocabulary_size = 5000

    exp.pp_data.embedding_size = 300
    exp.pp_data.max_posts = 1750
    exp.pp_data.max_terms_by_post = 300
    exp.pp_data.binary_classifier = True
    exp.pp_data.format_input_data = dn.InputData.POSTS_ONLY_TEXT
    exp.pp_data.remove_stopwords = False
    exp.pp_data.delete_low_tfid = False
    exp.pp_data.min_df = 0
    exp.pp_data.min_tf = 0
    exp.pp_data.random_posts = False
    exp.pp_data.random_users = False
    exp.pp_data.tokenizing_type = 'WE'
    exp.pp_data.word_embedding_custom_file = ''
    exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_6B
    exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
    exp.pp_data.word_embedding_custom_file = ''
    exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
    exp.pp_data.type_prediction_label= dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL

    exp.use_custom_metrics = False
    exp.use_valid_set_for_train = True
    exp.valid_split_from_train_set = 0.0
    exp.imbalanced_classes = True

    we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                                 '_EF_' + 'glove6B300d_lecun'

    lstm = ModelClass(1)
    lstm.loss_function = 'binary_crossentropy'
    lstm.optmizer_function = 'adam'
    lstm.epochs = 15
    lstm.batch_size = 32
    lstm.patience_train = 10
    lstm.use_embedding_pre_train = exp.pp_data.use_embedding
    lstm.embed_trainable = (lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

    neuronios_by_layer = [16]
    epochs = [32]
    batch_sizes = [40]
    dropouts = [0.2]

    np.random.seed(dn.SEED)

    time_ini_rep = datetime.datetime.now()
    x_train, y_train, x_valid, y_valid, num_words, embedding_matrix = exp.pp_data.load_data()
    exp.set_period_time_end(time_ini_rep, 'Load data')

    for neuronios in neuronios_by_layer:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for dropout in dropouts:
                    lstm.epochs = epoch
                    lstm.batch_size = batch_size
                    lstm.patience_train = epoch / 2
                    exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
                            batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name

                    lstm.model = Sequential()
                    lstm.model.add(Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size,
                                             trainable=lstm.embed_trainable, name='emb_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        kernel_initializer='lecun_uniform', recurrent_initializer='orthogonal',
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        return_sequences=True, name='dense_1_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        kernel_initializer='lecun_uniform', recurrent_initializer='orthogonal',
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        return_sequences=True, name='dense_2_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        kernel_initializer='lecun_uniform', recurrent_initializer='orthogonal',
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        name='dense_3_' + name_model))
                    lstm.model.add(Dense(3,
                                         kernel_initializer='lecun_uniform',
                                         activation='sigmoid',
                                         name='dense_4_' + name_model))

                    time_ini_exp = datetime.datetime.now()
                    exp.generate_model_hypeparams(lstm, x_train, y_train, x_valid, y_valid, embedding_matrix)
                    exp.set_period_time_end(time_ini_exp, 'Total experiment')

    del x_train, y_train, x_valid, y_valid, num_words, embedding_matrix

    # Test
    exp.pp_data.load_dataset_type = dn.LoadDataset.TEST_DATA_MODEL
    np.random.seed(dn.SEED)
    time_ini_rep = datetime.datetime.now()
    x_test, y_test = exp.pp_data.load_data()
    exp.set_period_time_end(time_ini_rep, 'Load data')

    for neuronios in neuronios_by_layer:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for dropout in dropouts:
                    lstm.epochs = epoch
                    lstm.batch_size = batch_size
                    lstm.patience_train = epoch / 2
                    exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
                            batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name

                    lstm.model = exp.load_model(dn.PATH_PROJECT + exp.experiment_name + '.h5')
                    exp.save_geral_configs()
                    exp.save_summary_model(lstm.model)
                    exp.predict_samples(lstm, x_test, y_test)

    del x_test, y_test, lstm, exp

# SINGLE-LABEL With VALID train
# Model lstm_exp9_var_L3_N16_B40_E32_D0.2 static glove com kernel_initializer=glorot_uniform=xavier_uniform
# SMHD anx, dep
def generate_model_sl_wv_1(exp, name_model):
    exp.pp_data.vocabulary_size = 5000

    exp.pp_data.embedding_size = 300
    exp.pp_data.max_posts = 1750
    exp.pp_data.max_terms_by_post = 300
    exp.pp_data.binary_classifier = False
    exp.pp_data.format_input_data = dn.InputData.POSTS_ONLY_TEXT
    exp.pp_data.remove_stopwords = False
    exp.pp_data.delete_low_tfid = False
    exp.pp_data.min_df = 0
    exp.pp_data.min_tf = 0
    exp.pp_data.random_posts = False
    exp.pp_data.random_users = False
    exp.pp_data.tokenizing_type = 'WE'
    exp.pp_data.word_embedding_custom_file = ''
    exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_6B
    exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
    exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
    exp.pp_data.type_prediction_label= dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL

    exp.use_custom_metrics = False
    exp.use_valid_set_for_train = True
    exp.valid_split_from_train_set = 0.0
    exp.imbalanced_classes = True

    we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                                 '_EF_' + 'glove6B300d_glorot'

    lstm = ModelClass(1)
    lstm.loss_function = 'categorical_crossentropy'
    lstm.main_metric = 'accuracy'
    lstm.optmizer_function = 'adam'
    lstm.epochs = 15
    lstm.batch_size = 32
    lstm.patience_train = 10
    lstm.use_embedding_pre_train = exp.pp_data.use_embedding
    lstm.embed_trainable = (lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

    neuronios_by_layer = [16]
    epochs = [32]
    batch_sizes = [40]
    dropouts = [0.2]

    np.random.seed(dn.SEED)

    time_ini_rep = datetime.datetime.now()
    x_train, y_train, x_valid, y_valid, num_words, embedding_matrix = exp.pp_data.load_data()
    exp.set_period_time_end(time_ini_rep, 'Load data')

    for neuronios in neuronios_by_layer:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for dropout in dropouts:
                    lstm.epochs = epoch
                    lstm.batch_size = batch_size
                    lstm.patience_train = epoch / 2
                    exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
                            batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name

                    lstm.model = Sequential()
                    lstm.model.add(Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size,
                                             trainable=lstm.embed_trainable, name='emb_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        return_sequences=True, name='dense_1_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        return_sequences=True, name='dense_2_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        name='dense_3_' + name_model))
                    lstm.model.add(Dense(3,
                                         activation='sigmoid',
                                         name='dense_4_' + name_model))

                    time_ini_exp = datetime.datetime.now()
                    exp.generate_model_hypeparams(lstm, x_train, y_train, x_valid, y_valid, embedding_matrix)
                    exp.set_period_time_end(time_ini_exp, 'Total experiment')

    del x_train, y_train, x_valid, y_valid, num_words, embedding_matrix

    # Test
    exp.pp_data.load_dataset_type = dn.LoadDataset.TEST_DATA_MODEL
    np.random.seed(dn.SEED)
    time_ini_rep = datetime.datetime.now()
    x_test, y_test = exp.pp_data.load_data()
    exp.set_period_time_end(time_ini_rep, 'Load data')

    for neuronios in neuronios_by_layer:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for dropout in dropouts:
                    lstm.epochs = epoch
                    lstm.batch_size = batch_size
                    lstm.patience_train = epoch / 2
                    exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
                            batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name

                    lstm.model = exp.load_model(dn.PATH_PROJECT + exp.experiment_name + '.h5')
                    exp.save_geral_configs()
                    exp.save_summary_model(lstm.model)
                    exp.predict_samples(lstm, x_test, y_test)

    del x_test, y_test, lstm, exp

# Model lstm_exp9_var_L3_N16_B40_E32_D0.2 static glove com kernel_initializer=lecun_uniform
# SMHD anx, dep
def generate_model_sl_wv_2(exp, name_model):
    exp.pp_data.vocabulary_size = 5000

    exp.pp_data.embedding_size = 300
    exp.pp_data.max_posts = 1750
    exp.pp_data.max_terms_by_post = 300
    exp.pp_data.binary_classifier = False
    exp.pp_data.format_input_data = dn.InputData.POSTS_ONLY_TEXT
    exp.pp_data.remove_stopwords = False
    exp.pp_data.delete_low_tfid = False
    exp.pp_data.min_df = 0
    exp.pp_data.min_tf = 0
    exp.pp_data.random_posts = False
    exp.pp_data.random_users = False
    exp.pp_data.tokenizing_type = 'WE'
    exp.pp_data.word_embedding_custom_file = ''
    exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_6B
    exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
    exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
    exp.pp_data.type_prediction_label= dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL

    exp.use_custom_metrics = False
    exp.use_valid_set_for_train = True
    exp.valid_split_from_train_set = 0.0
    exp.imbalanced_classes = True

    we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                                 '_EF_' + 'glove6B300d_lecun'

    lstm = ModelClass(1)
    lstm.loss_function = 'categorical_crossentropy'
    lstm.main_metric = 'accuracy'
    lstm.optmizer_function = 'adam'
    lstm.epochs = 15
    lstm.batch_size = 32
    lstm.patience_train = 10
    lstm.use_embedding_pre_train = exp.pp_data.use_embedding
    lstm.embed_trainable = (lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

    neuronios_by_layer = [16]
    epochs = [32]
    batch_sizes = [40]
    dropouts = [0.2]

    np.random.seed(dn.SEED)

    time_ini_rep = datetime.datetime.now()
    x_train, y_train, x_valid, y_valid, num_words, embedding_matrix = exp.pp_data.load_data()
    exp.set_period_time_end(time_ini_rep, 'Load data')

    for neuronios in neuronios_by_layer:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for dropout in dropouts:
                    lstm.epochs = epoch
                    lstm.batch_size = batch_size
                    lstm.patience_train = epoch / 2
                    exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
                            batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name

                    lstm.model = Sequential()
                    lstm.model.add(Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size,
                                             trainable=lstm.embed_trainable, name='emb_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        kernel_initializer='lecun_uniform', recurrent_initializer='orthogonal',
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        return_sequences=True, name='dense_1_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        kernel_initializer='lecun_uniform', recurrent_initializer='orthogonal',
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        return_sequences=True, name='dense_2_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        kernel_initializer='lecun_uniform', recurrent_initializer='orthogonal',
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        name='dense_3_' + name_model))
                    lstm.model.add(Dense(3,
                                         kernel_initializer='lecun_uniform',
                                         activation='sigmoid',
                                         name='dense_4_' + name_model))

                    time_ini_exp = datetime.datetime.now()
                    exp.generate_model_hypeparams(lstm, x_train, y_train, x_valid, y_valid, embedding_matrix)
                    exp.set_period_time_end(time_ini_exp, 'Total experiment')

    del x_train, y_train, x_valid, y_valid, num_words, embedding_matrix

    # Test
    exp.pp_data.load_dataset_type = dn.LoadDataset.TEST_DATA_MODEL
    np.random.seed(dn.SEED)
    time_ini_rep = datetime.datetime.now()
    x_test, y_test = exp.pp_data.load_data()
    exp.set_period_time_end(time_ini_rep, 'Load data')

    for neuronios in neuronios_by_layer:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for dropout in dropouts:
                    lstm.epochs = epoch
                    lstm.batch_size = batch_size
                    lstm.patience_train = epoch / 2
                    exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
                            batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name

                    lstm.model = exp.load_model(dn.PATH_PROJECT + exp.experiment_name + '.h5')
                    exp.save_geral_configs()
                    exp.save_summary_model(lstm.model)
                    exp.predict_samples(lstm, x_test, y_test)

    del x_test, y_test, lstm, exp

# Model lstm_exp9_var_L3_N16_B40_E32_D0.2 static custom w2v Skipgram
# SMHD anx, dep
def SMHD_anx_dep_singlelabel_wv_4(exp, name_model):
    exp.pp_data.vocabulary_size = 5000

    exp.pp_data.embedding_size = 300
    exp.pp_data.max_posts = 1750
    exp.pp_data.max_terms_by_post = 300
    exp.pp_data.binary_classifier = False
    exp.pp_data.format_input_data = dn.InputData.POSTS_ONLY_TEXT
    exp.pp_data.remove_stopwords = False
    exp.pp_data.delete_low_tfid = False
    exp.pp_data.min_df = 0
    exp.pp_data.min_tf = 0
    exp.pp_data.random_posts = False
    exp.pp_data.random_users = False
    exp.pp_data.tokenizing_type = 'WE'
    exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
    exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
    exp.pp_data.word_embedding_custom_file = 'SMHD-Skipgram-AllUsers-300.bin'
    exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
    exp.pp_data.type_prediction_label= dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL

    exp.use_custom_metrics = False
    exp.use_valid_set_for_train = True
    exp.valid_split_from_train_set = 0.0
    exp.imbalanced_classes = True

    we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                                 '_EF_' + exp.pp_data.word_embedding_custom_file

    ## Gera dados conforme configuração
    lstm = ModelClass(1)
    lstm.loss_function = 'categorical_crossentropy'
    lstm.main_metric = 'accuracy'
    lstm.optmizer_function = 'adam'
    lstm.epochs = 15
    lstm.batch_size = 32
    lstm.patience_train = 10
    lstm.use_embedding_pre_train = exp.pp_data.use_embedding
    lstm.embed_trainable = (lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

    neuronios_by_layer = [16]
    epochs = [32]
    batch_sizes = [40]
    dropouts = [0.2]

    np.random.seed(dn.SEED)

    time_ini_rep = datetime.datetime.now()
    x_train, y_train, x_valid, y_valid, num_words, embedding_matrix = exp.pp_data.load_data()
    exp.set_period_time_end(time_ini_rep, 'Load data')

    for neuronios in neuronios_by_layer:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for dropout in dropouts:
                    lstm.epochs = epoch
                    lstm.batch_size = batch_size
                    lstm.patience_train = epoch / 2
                    exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
                            batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name

                    lstm.model = Sequential()
                    lstm.model.add(Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size,
                                             trainable=lstm.embed_trainable, name='emb_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        return_sequences=True, name='dense_1_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        return_sequences=True, name='dense_2_' + name_model))
                    lstm.model.add(LSTM(neuronios,
                                        activation='tanh', dropout=dropout, recurrent_dropout=dropout,
                                        name='dense_3_' + name_model))
                    lstm.model.add(Dense(3,
                                         activation='sigmoid',
                                         name='dense_4_' + name_model))

                    time_ini_exp = datetime.datetime.now()
                    exp.generate_model_hypeparams(lstm, x_train, y_train, x_valid, y_valid, embedding_matrix)
                    exp.set_period_time_end(time_ini_exp, 'Total experiment')

    del x_train, y_train, x_valid, y_valid, num_words, embedding_matrix

    # Test
    exp.pp_data.load_dataset_type = dn.LoadDataset.TEST_DATA_MODEL
    np.random.seed(dn.SEED)
    time_ini_rep = datetime.datetime.now()
    x_test, y_test = exp.pp_data.load_data()
    exp.set_period_time_end(time_ini_rep, 'Load data')

    for neuronios in neuronios_by_layer:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for dropout in dropouts:
                    lstm.epochs = epoch
                    lstm.batch_size = batch_size
                    lstm.patience_train = epoch / 2
                    exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + str(
                            batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + we_file_name

                    lstm.model = exp.load_model(dn.PATH_PROJECT + exp.experiment_name + '.h5')
                    exp.save_geral_configs()
                    exp.save_summary_model(lstm.model)
                    exp.predict_samples(lstm, x_test, y_test)

    del x_test, y_test, lstm, exp
    
def main():
    for arg in sys.argv[1]:
        if arg == '1':
            print('Initializer experiment 1 - model SMHD_anx_dep_multilabel_nv_1')
            exp = ExperimentProcesses('lstm_exp9_var_L3')
            exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
                           total_registers=3160, subdirectory="anx_dep_multilabel")
            generate_model_nv_1(exp, 'SMHD_anx_dep_multilabel_nv_1')

        elif arg == '2':
            print('Initializer experiment 2 - model SMHD_anx_dep_multilabel_nv_2')
            exp = ExperimentProcesses('lstm_exp9_var_L3')
            exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
                           total_registers=3160, subdirectory="anx_dep_multilabel")
            generate_model_nv_2(exp, 'SMHD_anx_dep_multilabel_nv_2')

        elif arg == '3':
            print('Initializer experiment 1 - model SMHD_anx_dep_multilabel_wv_1')
            exp = ExperimentProcesses('lstm_exp9_var_L3')
            exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
                           total_registers=3160, subdirectory="anx_dep_multilabel")
            generate_model_wv_1(exp, 'SMHD_anx_dep_multilabel_wv_1')

        elif arg == '4':
            print('Initializer experiment 2 - model SMHD_anx_dep_multilabel_wv_2')
            exp = ExperimentProcesses('lstm_exp9_var_L3')
            exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
                           total_registers=3160, subdirectory="anx_dep_multilabel")
            generate_model_wv_2(exp, 'SMHD_anx_dep_multilabel_wv_2')

        elif arg == '5':
            print('Initializer experiment 1 - model SMHD_anx_dep_singlelabel_wv_1')
            exp = ExperimentProcesses('lstm_exp9_var_L3')
            exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
                           total_registers=3160, subdirectory="anx_dep_multilabel")
            generate_model_sl_wv_1(exp, 'SMHD_anx_dep_singlelabel_wv_1')

        elif arg == '6':
            print('Initializer experiment 1 - model SMHD_anx_dep_singlelabel_wv_2')
            exp = ExperimentProcesses('lstm_exp9_var_L3')
            exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
                           total_registers=3160, subdirectory="anx_dep_multilabel")
            generate_model_sl_wv_2(exp, 'SMHD_anx_dep_singlelabel_wv_2')

        elif arg == '7':
            print('Initializer experiment 1 - model SMHD_anx_dep_singlelabel_wv_4')
            exp = ExperimentProcesses('lstm_exp9_var_L3')
            exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
                                           total_registers=3160, subdirectory="anx_dep_multilabel")
            SMHD_anx_dep_singlelabel_wv_4(exp, 'SMHD_anx_dep_singlelabel_wv_4')
if __name__ == '__main__':
    main()
