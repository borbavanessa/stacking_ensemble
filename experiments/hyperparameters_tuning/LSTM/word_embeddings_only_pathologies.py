import numpy as np
import sys

import utils.definition_network as dn

import datetime

# LAYERS
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from network_model.model_class import ModelClass
from utils.experiment_processes import ExperimentProcesses

def generate_model(exp, name_model, we_file_name):
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
    exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL

    exp.use_custom_metrics = True
    exp.use_valid_set_for_train = True
    exp.valid_split_from_train_set = 0.0
    exp.imbalanced_classes = False

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
                    exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' +\
                                          str(batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' +\
                                          we_file_name

                    lstm.model = Sequential()
                    lstm.model.add(Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size,
                                             trainable=lstm.embed_trainable, name='emb_' + name_model))
                    lstm.model.add(LSTM(neuronios, kernel_initializer='lecun_uniform', recurrent_initializer='orthogonal',
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
                    lstm.model.add(Dense(1,
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

    del x_test, y_test, lstm


def generate_anxiety_model(arg):
    if arg == '1':
        print('Initializer experiment WE anxiety 1 (model SMHD_ml_gl_1040)\n'+\
              'Set: dataset=SMHD_1040 single-label SMHD-Skipgram-A-D-ADUsers-300.bin Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety'], total_registers=1040,
                                       subdirectory="anxiety")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-Skipgram-A-D-ADUsers-300.bin'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_1040', we_file_name)
    
    elif arg == '2':
        print('Initializer experiment WE anxiety 2 (model SMHD_ml_gl_1040)\n' + \
              'Set: dataset=SMHD_1040 single-label SMHD-Skipgram-A-D-ADUsers-300.bin Non-Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety'], total_registers=1040,
                                       subdirectory="anxiety")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.NON_STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-Skipgram-A-D-ADUsers-300.bin'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_1040', we_file_name)
    
    elif arg == '3':
        print('Initializer experiment WE anxiety 3 (model SMHD_ml_gl_1040)\n' + \
              'Set: dataset=SMHD_1040 single-label SMHD-CBOW-A-D-ADUsers-300.bin Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety'], total_registers=1040,
                                       subdirectory="anxiety")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-CBOW-A-D-ADUsers-300.bin'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_1040', we_file_name)
    
    elif arg == '4':
        print('Initializer experiment WE anxiety 4 (model SMHD_ml_gl_1040)\n' + \
              'Set: dataset=SMHD_1040 single-label SMHD-CBOW-A-D-ADUsers-300.bin Non-Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety'], total_registers=1040,
                                       subdirectory="anxiety")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.NON_STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-CBOW-A-D-ADUsers-300.bin'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_1040', we_file_name)
    
    elif arg == '5':
        print('Initializer experiment WE anxiety 5 (model SMHD_ml_gl_1040)\n' + \
              'Set: dataset=SMHD_1040 single-label SMHD-glove-A-D-ADUsers-300.pkl Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety'], total_registers=1040,
                                       subdirectory="anxiety")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-glove-A-D-ADUsers-300.pkl'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'SMHD_anx_3_' + we_file_name[0:13] + we_file_name[18:30] + '_1040', we_file_name)
    
    elif arg == '6':
        print('Initializer experiment WE anxiety 6 (model SMHD_ml_gl_1040)\n' + \
              'Set: dataset=SMHD_1040 single-label SMHD-glove-A-D-ADUsers-300.pkl Non-Static')

        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety'], total_registers=1040,
                                       subdirectory="anxiety")

        exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.NON_STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-glove-A-D-ADUsers-300.pkl'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL

        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'

        generate_model(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_1040', we_file_name)

    else:
        print('Initializer experiment WE anxiety - (model SMHD_ml_gl_1040)\n' + \
              'Set: dataset=SMHD_1040 single-label Glove6B Static')

        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety'], total_registers=1040,
                                       subdirectory="anxiety")

        exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_6B
        exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
        exp.pp_data.word_embedding_custom_file = ''
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL

        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_lecun'

        generate_model(exp, 'exp9_' + 'glove6B' + '_1040', we_file_name)


def generate_depression_model(arg):
    if arg == '1':
        print('Initializer experiment WE depression 1 (model SMHD_ml_gl_2160)\n' + \
              'Set: dataset=SMHD_2160 single-label SMHD-Skipgram-A-D-ADUsers-300.bin Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'depression'], total_registers=2160,
                                       subdirectory="depression")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-Skipgram-A-D-ADUsers-300.bin'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_2160', we_file_name)
    
    elif arg == '2':
        print('Initializer experiment WE depression 2 (model SMHD_ml_gl_2160)\n' + \
              'Set: dataset=SMHD_2160 single-label SMHD-Skipgram-A-D-ADUsers-300.bin Non-Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'depression'], total_registers=2160,
                                       subdirectory="depression")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.NON_STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-Skipgram-A-D-ADUsers-300.bin'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_2160', we_file_name)
    
    elif arg == '3':
        print('Initializer experiment WE depression 3 (model SMHD_ml_gl_2160)\n' + \
              'Set: dataset=SMHD_2160 single-label SMHD-CBOW-A-D-ADUsers-300.bin Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'depression'], total_registers=2160,
                                       subdirectory="depression")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-CBOW-A-D-ADUsers-300.bin'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_2160', we_file_name)
    
    elif arg == '4':
        print('Initializer experiment WE depression 4 (model SMHD_ml_gl_2160)\n' + \
              'Set: dataset=SMHD_2160 single-label SMHD-CBOW-A-D-ADUsers-300.bin Non-Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'depression'], total_registers=2160,
                                       subdirectory="depression")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.NON_STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-CBOW-A-D-ADUsers-300.bin'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'SMHD_dep_3_' + we_file_name[0:13] + we_file_name[18:30] + '_2160', we_file_name)
    
    elif arg == '5':
        print('Initializer experiment WE depression 5 (model SMHD_ml_gl_2160)\n' + \
              'Set: dataset=SMHD_2160 single-label SMHD-glove-A-D-ADUsers-300.pkl Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'depression'], total_registers=2160,
                                       subdirectory="depression")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-glove-A-D-ADUsers-300.pkl'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_2160', we_file_name)
    
    elif arg == '6':
        print('Initializer experiment WE depression 6 (model SMHD_ml_gl_2160)\n' + \
              'Set: dataset=SMHD_2160 single-label SMHD-glove-A-D-ADUsers-300.pkl Non-Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'depression'], total_registers=2160,
                                       subdirectory="depression")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.NON_STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-glove-A-D-ADUsers-300.pkl'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_2640', we_file_name)

    else:
        print('Initializer experiment WE depression - (model SMHD_ml_gl_2160)\n' + \
              'Set: dataset=SMHD_2160 single-label Glove6B Static')

        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'depression'], total_registers=2160,
                                       subdirectory="depression")

        exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_6B
        exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
        exp.pp_data.word_embedding_custom_file = ''
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL

        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_lecun'

        generate_model(exp, 'exp9_' + 'glove6B' + '_2160', we_file_name)


def generate_anx_dep_model(arg):
    if arg == '1':
        print('Initializer experiment WE anx_dep 1 (model SMHD_ml_gl_880)\n' + \
              'Set: dataset=SMHD_880 single-label SMHD-Skipgram-A-D-ADUsers-300.bin Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety,depression'],
                                       total_registers=880, subdirectory="anxiety,depression")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-Skipgram-A-D-ADUsers-300.bin'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_880', we_file_name)
    
    elif arg == '2':
        print('Initializer experiment WE anx_dep 2 (model SMHD_ml_gl_880)\n' + \
              'Set: dataset=SMHD_880 single-label SMHD-Skipgram-A-D-ADUsers-300.bin Non-Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety,depression'],
                                       total_registers=880, subdirectory="anxiety,depression")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.NON_STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-Skipgram-A-D-ADUsers-300.bin'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_880', we_file_name)
    
    elif arg == '3':
        print('Initializer experiment WE anx_dep 3 (model SMHD_ml_gl_880)\n' + \
              'Set: dataset=SMHD_880 single-label SMHD-CBOW-A-D-ADUsers-300.bin Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety,depression'],
                                       total_registers=880, subdirectory="anxiety,depression")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-CBOW-A-D-ADUsers-300.bin'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_880', we_file_name)
    
    elif arg == '4':
        print('Initializer experiment WE anx_dep 4 (model SMHD_ml_gl_880)\n' + \
              'Set: dataset=SMHD_880 single-label SMHD-CBOW-A-D-ADUsers-300.bin Non-Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety,depression'],
                                       total_registers=880, subdirectory="anxiety,depression")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.NON_STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-CBOW-A-D-ADUsers-300.bin'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_880', we_file_name)
    
    elif arg == '5':
        print('Initializer experiment WE anx_dep 5 (model SMHD_ml_gl_880)\n' + \
              'Set: dataset=SMHD_880 single-label SMHD-glove-A-D-ADUsers-300.pkl Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety,depression'],
                                       total_registers=880, subdirectory="anxiety,depression")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-glove-A-D-ADUsers-300.pkl'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_880', we_file_name)
    
    elif arg == '6':
        print('Initializer experiment WE anx_dep 6 (model SMHD_ml_gl_880)\n' + \
              'Set: dataset=SMHD_880 single-label SMHD-glove-A-D-ADUsers-300.pkl Non-Static')
        
        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety,depression'],
                                       total_registers=880, subdirectory="anxiety,depression")
        
        exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_CUSTOM
        exp.pp_data.use_embedding = dn.UseEmbedding.NON_STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-glove-A-D-ADUsers-300.pkl'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
        
        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'
        
        generate_model(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_880', we_file_name)

    elif arg == '7':
        print('Initializer experiment WE anx_dep 7 (model SMHD_ml_gl_880)\n' + \
              'Set: dataset=SMHD_880 single-label SMHD-CBOW-AllUsers-300.pkl Non-Static')

        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety,depression'],
                                       total_registers=880, subdirectory="anxiety,depression")

        exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC
        exp.pp_data.use_embedding = dn.UseEmbedding.NON_STATIC
        exp.pp_data.word_embedding_custom_file = 'SMHD-CBOW-AllUsers-300.pkl'
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL

        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_lecun'

        generate_model(exp, 'SMHD_anx_dep_3_' + we_file_name[0:13] + we_file_name[18:30] + '_880', we_file_name)

    else:
        print('Initializer experiment WE anx_dep - (model SMHD_ml_gl_880)\n' + \
              'Set: dataset=SMHD_880 single-label Glove6B Static')

        exp = ExperimentProcesses('lstm_exp9_var_L3')
        exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety,depression'],
                                       total_registers=880, subdirectory="anxiety,depression")

        exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_6B
        exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
        exp.pp_data.word_embedding_custom_file = ''
        exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL

        we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
                       '_lecun'

        generate_model(exp, 'exp9_' + 'glove6B' + '_880', we_file_name)


if __name__ == '__main__':

    fn = sys.argv[1]
    arg = sys.argv[2]

    if fn == 'anx':
        generate_anxiety_model(arg)
    elif fn == 'dep':
        generate_depression_model(arg)
    else:
        generate_anx_dep_model(arg)
