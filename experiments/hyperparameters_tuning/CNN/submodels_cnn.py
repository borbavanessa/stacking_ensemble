# Load libs standard python and custom
import numpy as np
import datetime
import sys

from keras.models import Sequential
from keras import regularizers
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers import Embedding, TimeDistributed, Input
from network_model.model_class import ModelClass
from utils.experiment_processes import ExperimentProcesses

import utils.definition_network as dn


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
		
		cnn = ModelClass(1)
		cnn.loss_function = 'binary_crossentropy'
		cnn.optmizer_function = 'adadelta'
		
		filters_by_layer = set_params['filters_by_layer']
		kernels_size = set_params['kernels_size']
		epochs = set_params['epochs']
		batch_sizes = set_params['batch_sizes']
		dropouts = set_params['dropouts']
		
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

								cnn.use_embedding_pre_train = exp.pp_data.use_embedding
								cnn.embed_trainable = (
														cnn.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

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
																		cnn.epochs = epoch
																		cnn.batch_size = batch_size
																		cnn.patience_train = epoch
																		exp.experiment_name = name_model + '_cnn' + '_F' + str(filter) + '_K' + str(kernel_size) +\
																													'_P' + 'None' + '_B' + str(batch_size) + '_E' +\
																													str(epoch) + '_D' + str(dropout) + '_HLN' + str(filter)  + '_' + \
																													we_file_name

																		cnn.model = Sequential()
																		cnn.model.add(Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size,
																														 trainable=cnn.embed_trainable, name='emb_' + name_model))
																		cnn.model.add(Dropout(dropout, name='dropout_1_' + name_model))
																		cnn.model.add(Conv1D(filters=filter, kernel_size=kernel_size,
																												 kernel_initializer='glorot_uniform',
																												 padding='valid', activation='relu',
																												 name='conv_1_' + name_model))
																		cnn.model.add(GlobalAveragePooling1D(name='gloval_avg_pool_1_' + name_model))
																		cnn.model.add(Dense(filter, activation='relu', kernel_initializer='glorot_uniform',
																												name='dense_1_' + name_model))
																		cnn.model.add(Dropout(dropout, name='dropout_2_' + name_model))
																		cnn.model.add(Dense(3, activation='sigmoid', name='dense_2_' + name_model))

																		time_ini_exp = datetime.datetime.now()
																		exp.generate_model_hypeparams(cnn, x_train, y_train, x_valid, y_valid, embedding_matrix)
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

								cnn.use_embedding_pre_train = exp.pp_data.use_embedding
								cnn.embed_trainable = (
														cnn.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

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
																		cnn.epochs = epoch
																		cnn.batch_size = batch_size
																		cnn.patience_train = epoch
																		exp.experiment_name = name_model + '_cnn' + '_F' + str(filter) + '_K' + str(kernel_size) +\
																													'_P' + 'None' + '_B' + str(batch_size) + '_E' +\
																													str(epoch) + '_D' + str(dropout) + '_HLN' + str(filter)  + '_' + \
																													we_file_name
																		
																		cnn.model = exp.load_model(dn.PATH_PROJECT + exp.experiment_name + '.h5')
																		exp.save_geral_configs('Experiment Specific Configuration: ' + exp.experiment_name)
																		exp.save_summary_model(cnn.model)
																		exp.predict_samples(cnn, x_test, y_test)
		
								del x_test, y_test
								
		del cnn, exp

def test_glove6b(option, function):
		set_params = dict({'filters_by_layer': [100, 250],
											 'kernels_size': [3, 4, 5],
											 'epochs': [10],
											 'batch_sizes': [20, 40],
											 'dropouts': [0.2, 0.5],
											 'embedding_types': [dn.EmbeddingType.GLOVE_6B],
											 'embedding_custom_files': [''],
											 'use_embeddings': [dn.UseEmbedding.STATIC, dn.UseEmbedding.NON_STATIC]})
		
		if option == '1':
				print('Initializer experiment '+option+' (model SMHD_cnn_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't'+option+'_SMHD_cnn_gl_880', '_glorot', set_params, function)

		elif option == '2':
				print('Initializer experiment '+option+' (model SMHD_cnn_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't'+option+'_SMHD_cnn_gl_1040', '_glorot', set_params, function)

		elif option == '3':
				print('Initializer experiment '+option+' (model SMHD_cnn_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't'+option+'_SMHD_cnn_gl_2160', '_glorot', set_params, function)

		elif option == '4':
				print('Initializer experiment '+option+' (model SMHD_cnn_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t'+option+'_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't'+option+'_SMHD_cnn_gl_2460', '_glorot', set_params, function)


def test_glove_twitter_emb(option, function):
		set_params = dict({'filters_by_layer': [100, 250],
											 'kernels_size': [3, 4, 5],
											 'epochs': [10],
											 'batch_sizes': [20],
											 'dropouts': [0.2, 0.5],
											 'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
											 'embedding_custom_files': [''],
											 'use_embeddings': [dn.UseEmbedding.STATIC, dn.UseEmbedding.NON_STATIC]})
		
		if option == '1':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_gt_SMHD_cnn_gl_880', '_glorot', set_params, function)
		
		elif option == '2':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_gt_SMHD_cnn_gl_1040', '_glorot', set_params, function)
		
		elif option == '3':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_gt_SMHD_cnn_gl_2160', '_glorot', set_params, function)
		
		elif option == '4':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_gt_SMHD_cnn_gl_2460', '_glorot', set_params, function)


def test_google_news_emb(option, function):
		set_params = dict({'filters_by_layer': [100, 250],
											 'kernels_size': [3, 4, 5],
											 'epochs': [10],
											 'batch_sizes': [20],
											 'dropouts': [0.2, 0.5],
											 'embedding_types': [dn.EmbeddingType.WORD2VEC],
											 'embedding_custom_files': [''],
											 'use_embeddings': [dn.UseEmbedding.STATIC, dn.UseEmbedding.NON_STATIC]})
		
		if option == '1':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_gn_SMHD_cnn_gl_880', '_glorot', set_params, function)
		
		elif option == '2':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_gn_SMHD_cnn_gl_1040', '_glorot', set_params, function)
		
		elif option == '3':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_gn_SMHD_cnn_gl_2160', '_glorot', set_params, function)
		
		elif option == '4':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_gn_SMHD_cnn_gl_2460', '_glorot', set_params, function)


def test_w2v_custom_emb(option, function):
		set_params = dict({'filters_by_layer': [100, 250],
											 'kernels_size': [3, 4, 5],
											 'epochs': [10],
											 'batch_sizes': [20],
											 'dropouts': [0.2, 0.5],
											 'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
											 'embedding_custom_files': ['SMHD-Skipgram-AllUsers-300.bin', 'SMHD-CBOW-AllUsers-300.bin',
																									'SMHD-Skipgram-A-D-ADUsers-300.bin', 'SMHD-CBOW-A-D-ADUsers-300.bin'],
											 'use_embeddings': [dn.UseEmbedding.STATIC, dn.UseEmbedding.NON_STATIC]})
		
		if option == '1':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_wc_SMHD_cnn_gl_880', '_glorot', set_params, function)
		
		elif option == '2':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_wc_SMHD_cnn_gl_1040', '_glorot', set_params, function)
		
		elif option == '3':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_wc_SMHD_cnn_gl_2160', '_glorot', set_params, function)
		
		elif option == '4':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_wc_SMHD_cnn_gl_2460', '_glorot', set_params, function)


def test_glove_custom_emb(option, function):
		set_params = dict({'filters_by_layer': [100, 250],
											 'kernels_size': [3, 4, 5],
											 'epochs': [10],
											 'batch_sizes': [20],
											 'dropouts': [0.2, 0.5],
											 'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
											 'embedding_custom_files': ['SMHD-glove-AllUsers-300.pkl', 'SMHD-glove-A-D-ADUsers-300.pkl'],
											 'use_embeddings': [dn.UseEmbedding.STATIC, dn.UseEmbedding.NON_STATIC]})
		
		if option == '1':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_gc_SMHD_cnn_gl_880', '_glorot', set_params, function)
		
		elif option == '2':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_gc_SMHD_cnn_gl_1040', '_glorot', set_params, function)
		
		elif option == '3':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_gc_SMHD_cnn_gl_2160', '_glorot', set_params, function)
		
		elif option == '4':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_gc_SMHD_cnn_gl_2460', '_glorot', set_params, function)


def test_none_emb(option, function):
		set_params = dict({'filters_by_layer': [100, 250],
											 'kernels_size': [3, 4, 5],
											 'epochs': [10],
											 'batch_sizes': [20],
											 'dropouts': [0.2, 0.5],
											 'embedding_types': [dn.EmbeddingType.NONE],
											 'embedding_custom_files': [''],
											 'use_embeddings': [dn.UseEmbedding.RAND]})
		
		if option == '1':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model(exp, 't' + option + '_nw_SMHD_cnn_gl_880', '_glorot', set_params, function)
		
		elif option == '2':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model(exp, 't' + option + '_nw_SMHD_cnn_gl_1040', '_glorot', set_params, function)
		
		elif option == '3':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model(exp, 't' + option + '_nw_SMHD_cnn_gl_2160', '_glorot', set_params, function)
		
		elif option == '4':
				print('Initializer experiment ' + option + ' (model SMHD_cnn_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('t' + option + '_cnn_L1')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model(exp, 't' + option + '_nw_SMHD_cnn_gl_2460', '_glorot', set_params, function)


if __name__ == '__main__':
		function = sys.argv[1]
		option = sys.argv[2]
		if function == 'glove6B300d':
				test_glove6b(option, function)
		elif function == 'gloveTwitter':
				test_glove_twitter_emb(option, function)
		elif function == 'googleNews':
				test_google_news_emb(option, function)
		elif function == 'w2vCustom':
				test_w2v_custom_emb(option, function)
		elif function == 'gloveCustom':
				test_glove_custom_emb(option, function)
		else: #None
				test_none_emb(option, function)