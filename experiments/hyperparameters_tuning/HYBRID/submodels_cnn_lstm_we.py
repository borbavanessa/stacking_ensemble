# Load libs standard python and custom
import numpy as np
import datetime
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Conv1D, MaxPooling1D
from keras.layers import Embedding
from network_model.model_class import ModelClass
from utils.experiment_processes import ExperimentProcesses

import utils.definition_network as dn


# Generate hybrid model
def generate_model(exp, name_model, kernel_name, set_params, function):
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
		exp.pp_data.type_prediction_label = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL
		
		exp.use_custom_metrics = False
		exp.use_valid_set_for_train = True
		exp.valid_split_from_train_set = 0.0
		exp.imbalanced_classes = False
		
		cnn_lstm = ModelClass(1)
		cnn_lstm.loss_function = 'binary_crossentropy'
		cnn_lstm.optmizer_function = 'adadelta'
		cnn_lstm.epochs = 15
		cnn_lstm.batch_size = 32
		cnn_lstm.patience_train = 10
		
		filters_by_layer = set_params['filters_by_layer']
		neuronios_by_lstm_layer = set_params['neuronios_by_lstm_layer']
		dropouts = set_params['dropouts']
		dropouts_lstm = set_params['dropouts_lstm']
		kernels_size = set_params['kernels_size']
		epochs = set_params['epochs']
		batch_sizes = set_params['batch_sizes']
		
		np.random.seed(dn.SEED)
		time_ini_rep = datetime.datetime.now()

		for embedding_type in set_params['embedding_types']:
				for embedding_custom_file in set_params['embedding_custom_files']:
						for use_embedding in set_params['use_embeddings']:
								exp.pp_data.embedding_type = embedding_type
								exp.pp_data.word_embedding_custom_file = embedding_custom_file
								exp.pp_data.use_embedding = use_embedding
								exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL

								exp.set_period_time_end(time_ini_rep, 'Load data')
								x_train, y_train, x_valid, y_valid, num_words, embedding_matrix = exp.pp_data.load_data()

								cnn_lstm.use_embedding_pre_train = exp.pp_data.use_embedding
								cnn_lstm.embed_trainable = (
														cnn_lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

								emb_name = function

								if embedding_custom_file != '':
										emb_name = exp.pp_data.word_embedding_custom_file.split('.')[0]

								we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + \
															 str(exp.pp_data.use_embedding.value) + '_EF_' + emb_name + kernel_name

								for filter in filters_by_layer:
										for kernel_size in kernels_size:
												for batch_size in batch_sizes:
														for epoch in epochs:
																for dropout in dropouts:
																		for dropout_lstm in dropouts_lstm:
																				for neuronios in neuronios_by_lstm_layer:
																						cnn_lstm.epochs = epoch
																						cnn_lstm.batch_size = batch_size
																						cnn_lstm.patience_train = epoch/2
																						exp.experiment_name = name_model + '_cnn_lstm' + '_F' + str(filter) + '_K' +\
																																	str(kernel_size) + '_P' + 'None' + '_B' + str(batch_size) +\
																																	'_E' + str(epoch) + '_D' + str(dropout) + '_HLN' + str(filter) +\
																																	'_LSTM_N' + str(neuronios) + '_D' + str(dropout_lstm) + '_' + we_file_name

																						cnn_lstm.model = Sequential()
																						cnn_lstm.model.add(
																								Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size,
																													trainable=cnn_lstm.embed_trainable,
																													name='emb_' + name_model))
																						cnn_lstm.model.add(Dropout(dropout, name='dropout_1_' + name_model))
																						cnn_lstm.model.add(Conv1D(filters=filter, kernel_size=kernel_size,
																																			kernel_initializer='glorot_uniform',
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
																						cnn_lstm.model.add(
																								Dense(3, activation='sigmoid', name='dense_1_' + name_model))

																						time_ini_exp = datetime.datetime.now()
																						exp.generate_model_hypeparams(cnn_lstm, x_train, y_train, x_valid, y_valid,
																																					embedding_matrix)
																						exp.set_period_time_end(time_ini_exp, 'Total experiment')

								del x_train, y_train, x_valid, y_valid, num_words, embedding_matrix
		
		# Test
		np.random.seed(dn.SEED)
		time_ini_rep = datetime.datetime.now()

		for embedding_type in set_params['embedding_types']:
				for embedding_custom_file in set_params['embedding_custom_files']:
						for use_embedding in set_params['use_embeddings']:
								exp.pp_data.embedding_type = embedding_type
								exp.pp_data.word_embedding_custom_file = embedding_custom_file
								exp.pp_data.use_embedding = use_embedding
								exp.pp_data.load_dataset_type = dn.LoadDataset.TEST_DATA_MODEL
								
								exp.set_period_time_end(time_ini_rep, 'Load data')
								x_test, y_test = exp.pp_data.load_data()

								cnn_lstm.use_embedding_pre_train = exp.pp_data.use_embedding
								cnn_lstm.embed_trainable = (
														cnn_lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))
								
								emb_name = function
								
								if embedding_custom_file != '':
										emb_name = exp.pp_data.word_embedding_custom_file.split('.')[0]
								
								we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + \
															 str(exp.pp_data.use_embedding.value) + '_EF_' + emb_name + kernel_name
								
								for filter in filters_by_layer:
										for kernel_size in kernels_size:
												for batch_size in batch_sizes:
														for epoch in epochs:
																for dropout in dropouts:
																		for dropout_lstm in dropouts_lstm:
																				for neuronios in neuronios_by_lstm_layer:
																						cnn_lstm.epochs = epoch
																						cnn_lstm.batch_size = batch_size
																						cnn_lstm.patience_train = epoch/2
																						exp.experiment_name = name_model + '_cnn_lstm' + '_F' + str(filter) + '_K' +\
																																	str(kernel_size) + '_P' + 'None' + '_B' + str(batch_size) +\
																																	'_E' + str(epoch) + '_D' + str(dropout) + '_HLN' + str(filter) +\
																																	'_LSTM_N' + str(neuronios) + '_D' + str(dropout_lstm) + '_' + we_file_name
																						
																						cnn_lstm.model = exp.load_model(dn.PATH_PROJECT + exp.experiment_name + '.h5')
																						exp.save_geral_configs('Experiment Specific Configuration: ' + exp.experiment_name)
																						exp.save_summary_model(cnn_lstm.model)
																						exp.predict_samples(cnn_lstm, x_test, y_test)

								del x_test, y_test
								
		del cnn_lstm, exp

def test_glove6b(option, function, dataset):
		name_model = 't' + option + '_' + function
		set_params = dict({'embedding_types': [dn.EmbeddingType.GLOVE_6B],
											 'embedding_custom_files': [''],
											 'use_embeddings': [dn.UseEmbedding.STATIC, dn.UseEmbedding.NON_STATIC]})
		
		if dataset == 'anx':
				if option == '1':
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [4],
															 'dropouts': [0.2],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.2],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
						
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params, function)

		elif dataset == 'dep':
				if option == '1':
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses(name_model +'_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp,  name_model +'_SMHD_cnn_lstm_gl_2160', '_glorot', set_params, function)

		elif dataset == 'anx_dep':
				if option == '1':
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																	 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, name_model +  '_SMHD_cnn_lstm_gl_880', '_glorot', set_params, function)

		else: #multi
				if option == '1':
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.5],
															 'epochs': [50],
															 'batch_sizes': [20]})
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params, function)


def test_glove_twitter_emb(option, function, dataset):
		name_model = 't' + option + '_' + function
		set_params = dict({'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
											 'embedding_custom_files': [''],
											 'use_embeddings': [dn.UseEmbedding.STATIC, dn.UseEmbedding.NON_STATIC]})
		
		if dataset == 'anx':
				if option == '1':
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [4],
															 'dropouts': [0.2],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.2],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params, function)
		
		elif dataset == 'dep':
				if option == '1':
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params, function)
		
		elif dataset == 'anx_dep':
				if option == '1':
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params, function)
		
		else:  # multi
				if option == '1':
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.5],
															 'epochs': [50],
															 'batch_sizes': [20]})
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params, function)


def test_google_news_emb(option, function, dataset):
		name_model = 't' + option + '_' + function
		set_params = dict({'embedding_types': [dn.EmbeddingType.WORD2VEC],
											 'embedding_custom_files': [''],
											 'use_embeddings': [dn.UseEmbedding.STATIC, dn.UseEmbedding.NON_STATIC]})
		
		if dataset == 'anx':
				if option == '1':
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [4],
															 'dropouts': [0.2],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.2],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params, function)
		
		elif dataset == 'dep':
				if option == '1':
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params, function)
		
		elif dataset == 'anx_dep':
				if option == '1':
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params, function)
		
		else:  # multi
				if option == '1':
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.5],
															 'epochs': [50],
															 'batch_sizes': [20]})
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params, function)


def test_w2v_custom_emb(option, function, dataset):
		name_model = 't' + option + '_' + function
		set_params = dict({'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
											 'embedding_custom_files': ['SMHD-Skipgram-AllUsers-300.bin', 'SMHD-CBOW-AllUsers-300.bin',
																									'SMHD-Skipgram-A-D-ADUsers-300.bin', 'SMHD-CBOW-A-D-ADUsers-300.bin'],
											 'use_embeddings': [dn.UseEmbedding.STATIC, dn.UseEmbedding.NON_STATIC]})

		if dataset == 'anx':
				if option == '1':
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [4],
															 'dropouts': [0.2],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.2],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params, function)
		
		elif dataset == 'dep':
				if option == '1':
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params, function)
		
		elif dataset == 'anx_dep':
				if option == '1':
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params, function)
		
		else:  # multi
				if option == '1':
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.5],
															 'epochs': [50],
															 'batch_sizes': [20]})
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params, function)


def test_glove_custom_emb(option, function, dataset):
		name_model = 't' + option + '_' + function
		set_params = dict({'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
											 'embedding_custom_files': ['SMHD-glove-AllUsers-300.pkl', 'SMHD-glove-A-D-ADUsers-300.pkl'],
											 'use_embeddings': [dn.UseEmbedding.STATIC, dn.UseEmbedding.NON_STATIC]})
		
		if dataset == 'anx':
				if option == '1':
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [4],
															 'dropouts': [0.2],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.2],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params, function)
		
		elif dataset == 'dep':
				if option == '1':
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params, function)
		
		elif dataset == 'anx_dep':
				if option == '1':
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params, function)
		
		else:  # multi
				if option == '1':
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.5],
															 'epochs': [50],
															 'batch_sizes': [20]})
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params, function)


def test_none_emb(option, function, dataset):
		name_model = 't' + option + '_' + function
		set_params = dict({'embedding_types': [dn.EmbeddingType.NONE],
											 'embedding_custom_files': [''],
											 'use_embeddings': [dn.UseEmbedding.RAND]})
		
		if dataset == 'anx':
				if option == '1':
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [4],
															 'dropouts': [0.2],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.2],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_1040', '_glorot', set_params, function)
		
		elif dataset == 'dep':
				if option == '1':
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [128],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_2160', '_glorot', set_params, function)
		
		elif dataset == 'anx_dep':
				if option == '1':
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_880', '_glorot', set_params, function)
		
		else:  # multi
				if option == '1':
						set_params.update({'filters_by_layer': [64],
															 'kernels_size': [4],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [64],
															 'dropouts_lstm': [0.2],
															 'epochs': [50],
															 'batch_sizes': [20]})
				else:
						set_params.update({'filters_by_layer': [128],
															 'kernels_size': [5],
															 'dropouts': [0.5],
															 'neuronios_by_lstm_layer': [256],
															 'dropouts_lstm': [0.5],
															 'epochs': [50],
															 'batch_sizes': [20]})
				print('Initializer experiment ' + option + ' (model SMHD_cnn_lstm_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses(name_model + '_cnn_L1_lstm_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, name_model + '_SMHD_cnn_lstm_gl_2640', '_glorot', set_params, function)


if __name__ == '__main__':
		#Divide for unit test to run on pcad
		function = sys.argv[1]
		option = sys.argv[2]
		dataset = sys.argv[3]
		
		if function == 'glove6B300d':
				test_glove6b(option, function, dataset)
		elif function == 'gloveTwitter':
				test_glove_twitter_emb(option, function, dataset)
		elif function == 'googleNews':
				test_google_news_emb(option, function, dataset)
		elif function == 'w2vCustom':
				test_w2v_custom_emb(option, function, dataset)
		elif function == 'gloveCustom':
				test_glove_custom_emb(option, function, dataset)
		else: #None
				test_none_emb(option, function, dataset)