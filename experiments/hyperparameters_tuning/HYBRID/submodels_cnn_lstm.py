# Load libs standard python and custom
import numpy as np
import datetime
import sys

from keras.models import Sequential
from keras import regularizers
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM, Conv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers import Embedding, TimeDistributed, Input
from network_model.model_class import ModelClass
from utils.experiment_processes import ExperimentProcesses

import utils.definition_network as dn

def generate_model(exp, name_model, kernel_function, set_params):
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
		exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_6B
		exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
		exp.pp_data.word_embedding_custom_file = ''
		exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
		
		exp.pp_data.type_prediction_label = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL
		
		exp.use_custom_metrics = False
		exp.use_valid_set_for_train = True
		exp.valid_split_from_train_set = 0.0
		exp.imbalanced_classes = False
		
		we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
									 '_EF_' + 'glove6B300d' + kernel_function
		
		cnn_lstm = ModelClass(1)
		cnn_lstm.loss_function = 'binary_crossentropy'
		cnn_lstm.optmizer_function = 'adadelta'
		cnn_lstm.epochs = 15
		cnn_lstm.batch_size = 32
		cnn_lstm.patience_train = 10
		cnn_lstm.use_embedding_pre_train = exp.pp_data.use_embedding
		cnn_lstm.embed_trainable = (cnn_lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))
		
		# set_params is empty
		if not bool(set_params):
				filters_by_layer = [32, 64, 128]
				neuronios_by_lstm_layer = [64, 128, 256]
				dropouts = [0.2, 0.5]
				dropouts_lstm = [0.2, 0.5]
		else:
				filters_by_layer = set_params['filters_by_layer']
				neuronios_by_lstm_layer = set_params['neuronios_by_lstm_layer']
				dropouts = set_params['dropouts']
				dropouts_lstm = set_params['dropouts_lstm']
		
		kernels_size = [5]
		epochs = [10]
		batch_sizes = [20]

		np.random.seed(dn.SEED)
		time_ini_rep = datetime.datetime.now()
		x_train, y_train, x_valid, y_valid, num_words, embedding_matrix = exp.pp_data.load_data()
		exp.set_period_time_end(time_ini_rep, 'Load data')
		
		for filter in filters_by_layer:
				for kernel_size in kernels_size:
						for batch_size in batch_sizes:
								for epoch in epochs:
										for dropout in dropouts:
												for dropout_lstm in dropouts_lstm:
														for neuronios in neuronios_by_lstm_layer:
																cnn_lstm.epochs = epoch
																cnn_lstm.batch_size = batch_size
																cnn_lstm.patience_train = epoch
																exp.experiment_name = name_model + '_cnn_lstm' + '_F' + str(filter) + '_K' + \
																											str(kernel_size) + '_P' + 'None' + '_B' + str(batch_size) + \
																											'_E' + str(epoch) + '_D' + str(dropout) + '_HLN' + \
																											str(filter) + '_LSTM_N' + str(neuronios) + \
																											'_D'+ str(dropout_lstm) +	'_' + we_file_name
																
																cnn_lstm.model = Sequential()
																cnn_lstm.model.add(Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size,
																												trainable=cnn_lstm.embed_trainable, name='emb_' + name_model))
																cnn_lstm.model.add(Dropout(dropout, name='dropout_1_' + name_model))
																cnn_lstm.model.add(Conv1D(filters=filter, kernel_size=kernel_size,
																										 kernel_initializer='glorot_uniform',
																										 # kernel_regularizer=regularizers.l2(0.03),
																										 padding='valid', activation='relu',
																										 name='conv_1_' + name_model))
																cnn_lstm.model.add(MaxPooling1D(name='max_pool_1_' + name_model))
																cnn_lstm.model.add(LSTM(neuronios,
																												activation='tanh', dropout=dropout_lstm,
																												recurrent_dropout=dropout_lstm,
																												return_sequences=True, name='lstm_1_' + name_model))
																cnn_lstm.model.add(LSTM(neuronios,
																												activation='tanh', dropout=dropout_lstm,
																												recurrent_dropout=dropout_lstm,
																												return_sequences=True, name='lstm_2_' + name_model))
																cnn_lstm.model.add(LSTM(neuronios,
																												activation='tanh', dropout=dropout_lstm,
																												recurrent_dropout=dropout_lstm,
																												name='lstm_3_' + name_model))
																cnn_lstm.model.add(Dense(3, activation='sigmoid', name='dense_1_' + name_model))
										
																time_ini_exp = datetime.datetime.now()
																exp.generate_model_hypeparams(cnn_lstm, x_train, y_train, x_valid, y_valid, embedding_matrix)
																exp.set_period_time_end(time_ini_exp, 'Total experiment')
		
		del x_train, y_train, x_valid, y_valid, num_words, embedding_matrix
		
		# Test
		exp.pp_data.load_dataset_type = dn.LoadDataset.TEST_DATA_MODEL
		np.random.seed(dn.SEED)
		time_ini_rep = datetime.datetime.now()
		x_test, y_test = exp.pp_data.load_data()
		exp.set_period_time_end(time_ini_rep, 'Load data')
		
		for filter in filters_by_layer:
				for kernel_size in kernels_size:
						for batch_size in batch_sizes:
								for epoch in epochs:
										for dropout in dropouts:
												for dropout_lstm in dropouts_lstm:
														for neuronios in neuronios_by_lstm_layer:
																cnn_lstm.epochs = epoch
																cnn_lstm.batch_size = batch_size
																cnn_lstm.patience_train = epoch
																exp.experiment_name = name_model + '_cnn_lstm' + '_F' + str(filter) + '_K' + \
																											str(kernel_size) + '_P' + 'None' + '_B' + str(batch_size) + \
																											'_E' + str(epoch) + '_D' + str(dropout) + '_HLN' + \
																											str(filter) + '_LSTM_N' + str(neuronios) + \
																											'_D'+ str(dropout_lstm) +	'_' + we_file_name
										
																cnn_lstm.model = exp.load_model(dn.PATH_PROJECT + exp.experiment_name + '.h5')
																exp.save_geral_configs('Experiment Specific Configuration: ' + exp.experiment_name)
																exp.save_summary_model(cnn_lstm.model)
																exp.predict_samples(cnn_lstm, x_test, y_test)
		
		del x_test, y_test, cnn_lstm, exp

def test(option):
		set_params = dict()

		if option == '1':
				print('Initializer experiment 1 (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 'SMHD_cnn_lstm_gl_880', '_glorot', set_params)

		elif option == '2':
				print('Initializer experiment 2 (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 'SMHD_cnn_lstm_gl_1040', '_glorot', set_params)

		elif option == '3':
				print('Initializer experiment 3 (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 'SMHD_cnn_lstm_gl_2160', '_glorot', set_params)


def test_outperformance(option):
		set_params = dict()
		
		# Anxiety variations
		# 1,2 best performances; 3,4 second best; 5,6
		if option == '1':
				set_params.update({'filters_by_layer': [32],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [10, 15, 20, 25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
		
		elif option == '2':
				set_params.update({'filters_by_layer': [32],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [15, 20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
		
		elif option == '3':
				set_params.update({'filters_by_layer': [32],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
		
		elif option == '4':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [10, 15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
		
		elif option == '5':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
		
		elif option == '6':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [25, 50],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
		
		elif option == '7':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
		
		elif option == '8':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
		
		elif option == '9':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [25, 50],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
		
		elif option == '10':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [10],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
		elif option == '11':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
		elif option == '12':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
		elif option == '13':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
		
		
		elif option == '14':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
		elif option == '15':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
		elif option == '16':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params)
		
		# Depression
		elif option == '17':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [128],
													 'dropouts_lstm': [0.2],
													 'epochs': [10],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		elif option == '18':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [128],
													 'dropouts_lstm': [0.2],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		elif option == '19':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [128],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		elif option == '20':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [128],
													 'dropouts_lstm': [0.2],
													 'epochs': [25, 50],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		
		elif option == '21':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [128],
													 'dropouts_lstm': [0.2],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		elif option == '22':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [128],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		elif option == '23':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [128],
													 'dropouts_lstm': [0.2],
													 'epochs': [25, 50],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		
		elif option == '24':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [10],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		elif option == '25':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		elif option == '26':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		elif option == '27':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		
		elif option == '28':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		elif option == '29':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		elif option == '30':
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		
		elif option == '31':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [10],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		elif option == '32':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		elif option == '33':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		elif option == '34':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		
		elif option == '35':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		elif option == '36':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		elif option == '37':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params)
		
		# Comorbidity
		elif option == '38':
				set_params.update({'filters_by_layer': [32],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [10],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '39':
				set_params.update({'filters_by_layer': [32],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '40':
				set_params.update({'filters_by_layer': [32],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '41':
				set_params.update({'filters_by_layer': [32],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		
		elif option == '42':
				set_params.update({'filters_by_layer': [32],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '43':
				set_params.update({'filters_by_layer': [32],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '44':
				set_params.update({'filters_by_layer': [32],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		
		elif option == '45':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [10],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '46':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '47':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '48':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		
		elif option == '49':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [15, 20, 25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '50':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [15, 20, 25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '51':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [15, 20, 25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		
		elif option == '52':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [10],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '53':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '54':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '55':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		
		elif option == '56':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '57':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '58':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		
		elif option == '59':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [10],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '60':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '61':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '62':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		
		elif option == '63':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [5],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '64':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [5],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		elif option == '65':
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [5],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.2],
													 'epochs': [25, 50],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params)
		
		# Multi-label
		elif option == '66':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [10],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		elif option == '67':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		elif option == '68':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		elif option == '69':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [25, 50],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		
		elif option == '70':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		elif option == '71':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		elif option == '72':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [256],
													 'dropouts_lstm': [0.5],
													 'epochs': [25, 50],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		
		elif option == '73':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [128],
													 'dropouts_lstm': [0.2],
													 'epochs': [10],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		elif option == '74':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [128],
													 'dropouts_lstm': [0.2],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		elif option == '75':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [128],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		elif option == '76':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [4],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [128],
													 'dropouts_lstm': [0.2],
													 'epochs': [25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		
		elif option == '77':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [128],
													 'dropouts_lstm': [0.2],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		elif option == '78':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [128],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		elif option == '79':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [128],
													 'kernels_size': [5],
													 'dropouts': [0.2],
													 'neuronios_by_lstm_layer': [128],
													 'dropouts_lstm': [0.2],
													 'epochs': [25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		
		elif option == '80':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [10],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		elif option == '81':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		elif option == '82':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		elif option == '83':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [4],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [25, 50],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		
		elif option == '84':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [5],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [15],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		elif option == '85':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [5],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [20],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)
		elif option == '86':  # division test 4 to run in PCAD, with total time experiment not excceding 24h
				set_params.update({'filters_by_layer': [64],
													 'kernels_size': [5],
													 'dropouts': [0.5],
													 'neuronios_by_lstm_layer': [64],
													 'dropouts_lstm': [0.2],
													 'epochs': [25],
													 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params)


if __name__ == '__main__':
		option = sys.argv[1]
		# test(option)
		test_outperformance(option)
