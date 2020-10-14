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

import pandas as pd


# Generate model lstm_exp9_var_L3_N16_B40_E32_D0.2 non-static custom w2v CBOW AllUser
# For SMHD anx_dep
def load_submodel_anx_dep(exp, name_model, kernel_name, set_params):
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
		exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
		exp.pp_data.use_embedding = dn.UseEmbedding.NON_STATIC
		exp.pp_data.word_embedding_custom_file = 'SMHD-CBOW-AllUsers-300.bin'
		exp.pp_data.type_prediction_label = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL
		exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
		
		exp.use_custom_metrics = False
		exp.use_valid_set_for_train = True
		exp.valid_split_from_train_set = 0.0
		exp.imbalanced_classes = False

		we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
									 '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + kernel_name
		
		lstm = ModelClass(1)
		lstm.loss_function = 'binary_crossentropy'
		lstm.optmizer_function = 'adam'
		lstm.epochs = 15
		lstm.batch_size = 32
		lstm.patience_train = 10
		lstm.use_embedding_pre_train = exp.pp_data.use_embedding
		lstm.embed_trainable = (lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))
		
		np.random.seed(dn.SEED)
		
		time_ini_rep = datetime.datetime.now()
		x_train, y_train, x_valid, y_valid, num_words, embedding_matrix = exp.pp_data.load_data()
		exp.set_period_time_end(time_ini_rep, 'Load data')
		
		for neuronios in set_params['neuronios_by_layer']:
				for batch_size in set_params['batch_sizes']:
						for epoch in set_params['epochs']:
								for dropout in set_params['dropouts']:
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
		
		for neuronios in set_params['neuronios_by_layer']:
				for batch_size in set_params['batch_sizes']:
						for epoch in set_params['epochs']:
								for dropout in set_params['dropouts']:
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


def main(arg):
		if arg == '1':
				print('Initializer experiment 1 (model SMHD_ml_gl_880_cbow_alluser)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")

				set_params = dict({'neuronios_by_layer': [8],
													 'epochs': [32],
													 'batch_sizes': [20],
													 'dropouts': [0.1]})
				load_submodel_anx_dep(exp, 'SMHD_ml_gl_880_cbow_alluser', '_glorot', set_params)
		
		elif arg == '2':
				print('Initializer experiment 2 (model SMHD_ml_gl_880_cbow_alluser)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")

				set_params = dict({'neuronios_by_layer': [8],
													 'epochs': [32],
													 'batch_sizes': [20],
													 'dropouts': [0.2]})
				load_submodel_anx_dep(exp, 'SMHD_ml_gl_880_cbow_alluser', '_glorot', set_params)
		
		elif arg == '3':
				print('Initializer experiment 3 (model SMHD_ml_gl_880_cbow_alluser)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				
				set_params = dict({'neuronios_by_layer': [8],
													 'epochs': [64],
													 'batch_sizes': [20],
													 'dropouts': [0.1]})
				load_submodel_anx_dep(exp, 'SMHD_ml_gl_880_cbow_alluser', '_glorot', set_params)
		
		elif arg == '4':
				print('Initializer experiment 4 (model SMHD_ml_gl_880_cbow_alluser)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				
				set_params = dict({'neuronios_by_layer': [8],
													 'epochs': [64],
													 'batch_sizes': [20],
													 'dropouts': [0.2]})
				load_submodel_anx_dep(exp, 'SMHD_ml_gl_880_cbow_alluser', '_glorot', set_params)

		elif arg == '5':
				print('Initializer experiment 5 (model SMHD_ml_gl_880_cbow_alluser)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				
				set_params = dict({'neuronios_by_layer': [8],
													 'epochs': [96],
													 'batch_sizes': [20],
													 'dropouts': [0.1]})
				load_submodel_anx_dep(exp, 'SMHD_ml_gl_880_cbow_alluser', '_glorot', set_params)
		
		elif arg == '6':
				print('Initializer experiment 6 (model SMHD_ml_gl_880_cbow_alluser)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				
				set_params = dict({'neuronios_by_layer': [8],
													 'epochs': [96],
													 'batch_sizes': [20],
													 'dropouts': [0.2]})
				load_submodel_anx_dep(exp, 'SMHD_ml_gl_880_cbow_alluser', '_glorot', set_params)
		
		elif arg == '7':
				print('Initializer experiment 7 (model SMHD_ml_gl_880_cbow_alluser)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				
				set_params = dict({'neuronios_by_layer': [16],
													 'epochs': [32],
													 'batch_sizes': [20],
													 'dropouts': [0.1]})
				load_submodel_anx_dep(exp, 'SMHD_ml_gl_880_cbow_alluser', '_glorot', set_params)
		
		elif arg == '8':
				print('Initializer experiment 8 (model SMHD_ml_gl_880_cbow_alluser)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				
				set_params = dict({'neuronios_by_layer': [16],
													 'epochs': [32],
													 'batch_sizes': [20],
													 'dropouts': [0.2]})
				load_submodel_anx_dep(exp, 'SMHD_ml_gl_880_cbow_alluser', '_glorot', set_params)
		
		elif arg == '9':
				print('Initializer experiment 9 (model SMHD_ml_gl_880_cbow_alluser)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				
				set_params = dict({'neuronios_by_layer': [16],
													 'epochs': [64],
													 'batch_sizes': [20],
													 'dropouts': [0.1]})
				load_submodel_anx_dep(exp, 'SMHD_ml_gl_880_cbow_alluser', '_glorot', set_params)
		
		elif arg == '10':
				print('Initializer experiment 10 (model SMHD_ml_gl_880_cbow_alluser)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				
				set_params = dict({'neuronios_by_layer': [16],
													 'epochs': [64],
													 'batch_sizes': [20],
													 'dropouts': [0.2]})
				load_submodel_anx_dep(exp, 'SMHD_ml_gl_880_cbow_alluser', '_glorot', set_params)
		
		elif arg == '11':
				print('Initializer experiment 11 (model SMHD_ml_gl_880_cbow_alluser)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				
				set_params = dict({'neuronios_by_layer': [16],
													 'epochs': [96],
													 'batch_sizes': [20],
													 'dropouts': [0.1]})
				load_submodel_anx_dep(exp, 'SMHD_ml_gl_880_cbow_alluser', '_glorot', set_params)
		
		elif arg == '12':
				print('Initializer experiment 12 (model SMHD_ml_gl_880_cbow_alluser)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				
				set_params = dict({'neuronios_by_layer': [16],
													 'epochs': [96],
													 'batch_sizes': [20],
													 'dropouts': [0.2]})
				load_submodel_anx_dep(exp, 'SMHD_ml_gl_880_cbow_alluser', '_glorot', set_params)

		elif arg == '13':
				print('Initializer experiment 13 (model SMHD_ml_gl_880_cbow_alluser)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				
				set_params = dict({'neuronios_by_layer': [32],
													 'epochs': [64, 96],
													 'batch_sizes': [20],
													 'dropouts': [0.1, 0.2]})
				load_submodel_anx_dep(exp, 'SMHD_ml_gl_880_cbow_alluser', '_glorot', set_params)
		
		elif arg == '14':
				print('Initializer experiment 13 (model SMHD_ml_gl_880_cbow_alluser)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				
				set_params = dict({'neuronios_by_layer': [64],
													 'epochs': [96, 128],
													 'batch_sizes': [20],
													 'dropouts': [0.1, 0.2]})
				load_submodel_anx_dep(exp, 'SMHD_ml_gl_880_cbow_alluser', '_glorot', set_params)

if __name__ == '__main__':
		option = sys.argv[1]
		main(option)