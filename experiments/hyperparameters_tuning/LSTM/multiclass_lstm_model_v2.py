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


# Generate model lstm_exp9_var_L3_N16_B40_E32_D0.2 static glove com kernel_initializer=glorot_uniform=xavier_uniform
def generate_model(exp, name_model, kernel_function, set_params, function):
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
		
		lstm = ModelClass(1)
		lstm.loss_function = 'binary_crossentropy'

		optimizer_functions = set_params['optimizer_function']
		neuronios_by_layer = set_params['neuronios_by_layer']
		epochs = set_params['epochs']
		batch_sizes = set_params['batch_sizes']
		hidden_layers = set_params['hidden_layers']
		dropouts = [0.2]
		
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

								lstm.use_embedding_pre_train = exp.pp_data.use_embedding
								lstm.embed_trainable = (
														lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))

								emb_name = function
								if embedding_custom_file != '':
										emb_name = exp.pp_data.word_embedding_custom_file.split('.')[0]

								we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + \
															 str(exp.pp_data.use_embedding.value) + '_EF_' + emb_name + kernel_function

								for neuronios in neuronios_by_layer:
										for batch_size in batch_sizes:
												for epoch in epochs:
														for dropout in dropouts:
																for optmizer_function in optimizer_functions:
																		for hidden_layer in hidden_layers:
																				lstm.optmizer_function = dn.OPTIMIZER_FUNCTIONS[optmizer_function]
																				lstm.epochs = epoch
																				lstm.batch_size = batch_size
																				lstm.patience_train = epoch / 2
																				exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + \
																															str(batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_OF'+ \
																															lstm.optmizer_function + '_HL' + str(hidden_layer) + '_' + we_file_name
																				
																				lstm.model = Sequential()
																				lstm.model.add(Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size,
																																 trainable=lstm.embed_trainable, name='emb_' + name_model))
																				
																				for id_hl in range(hidden_layer):
																						lstm.model.add(LSTM(neuronios,
																																activation='tanh', dropout=dropout, recurrent_dropout=dropout,
																																return_sequences=True, name='dense_' + str(id_hl) + '_' + \
																																name_model))
								
																				lstm.model.add(LSTM(neuronios,
																														activation='tanh', dropout=dropout, recurrent_dropout=dropout,
																														name='dense_'+str(id_hl+1)+'_' + name_model))
																				lstm.model.add(Dense(3,
																														 activation='sigmoid',
																														 name='dense_'+str(id_hl+2)+'_' + name_model))
																				
																				time_ini_exp = datetime.datetime.now()
																				exp.generate_model_hypeparams(lstm, x_train, y_train, x_valid, y_valid, embedding_matrix)
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
								
								lstm.use_embedding_pre_train = exp.pp_data.use_embedding
								lstm.embed_trainable = (
												lstm.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))
								
								emb_name = function
								if embedding_custom_file != '':
										emb_name = exp.pp_data.word_embedding_custom_file.split('.')[0]
								
								we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + \
															 str(exp.pp_data.use_embedding.value) + '_EF_' + emb_name + kernel_function
		
								for neuronios in neuronios_by_layer:
										for batch_size in batch_sizes:
												for epoch in epochs:
														for dropout in dropouts:
																for optmizer_function in optimizer_functions:
																		for hidden_layer in hidden_layers:
																				lstm.optmizer_function = dn.OPTIMIZER_FUNCTIONS[optmizer_function]
																				lstm.epochs = epoch
																				lstm.batch_size = batch_size
																				lstm.patience_train = epoch / 2
																				exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + \
																															str(batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_OF' + \
																															lstm.optmizer_function + '_HL' + str(hidden_layer) + '_' + we_file_name
																				
																				lstm.model = exp.load_model(dn.PATH_PROJECT + exp.experiment_name + '.h5')
																				exp.save_geral_configs('Experiment Specific Configuration: ' + exp.experiment_name)
																				exp.save_summary_model(lstm.model)
																				exp.predict_samples(lstm, x_test, y_test)
								
								del x_test, y_test

		del lstm, exp


def test_diff_we_function(function, option):
		if function == 'gloveTwitter':
				set_params = dict({'optimizer_function': [dn.OptimizerFunction.ADAM.value],
														'neuronios_by_layer': [32],
														'hidden_layers': [4],
														'batch_sizes': [40],
														'epochs': [32],
														'embedding_types': [dn.EmbeddingType.GLOVE_TWITTER],
														'embedding_custom_files': ['']})
				if option == '1':
						set_params.update({'use_embeddings': [dn.UseEmbedding.STATIC]})
				else:
						set_params.update({'use_embeddings': [dn.UseEmbedding.NON_STATIC]})

				model_name = 't'+option+'_gt_SMHD_ml_gl_2640_var_hl'
		
		elif function == 'googleNews':
				set_params = dict({'optimizer_function': [dn.OptimizerFunction.ADAM.value],
														'neuronios_by_layer': [32],
														'hidden_layers': [4],
														'batch_sizes': [40],
														'epochs': [32],
														'embedding_types': [dn.EmbeddingType.WORD2VEC],
														'embedding_custom_files': ['']})
				if option == '1':
						set_params.update({'use_embeddings': [dn.UseEmbedding.STATIC]})
				else:
						set_params.update({'use_embeddings': [dn.UseEmbedding.NON_STATIC]})

				model_name = 't'+option+'_gn_SMHD_ml_gl_2640_var_hl'
		
		elif function == 'gloveCustom':
				set_params = dict({'optimizer_function': [dn.OptimizerFunction.ADAM.value],
														'neuronios_by_layer': [32],
														'hidden_layers': [4],
														'batch_sizes': [40],
														'epochs': [32],
														'embedding_types': [dn.EmbeddingType.GLOVE_CUSTOM],
														'embedding_custom_files': ['SMHD-glove-AllUsers-300.pkl', 'SMHD-glove-A-D-ADUsers-300.pkl']})
				if option == '1':
						set_params.update({'use_embeddings': [dn.UseEmbedding.STATIC]})
				else:
						set_params.update({'use_embeddings': [dn.UseEmbedding.NON_STATIC]})

				model_name = 't'+option+'_gc_SMHD_ml_gl_2640_var_hl'
		
		else: # 'w2vCustom'
				set_params = dict({'optimizer_function': [dn.OptimizerFunction.ADAM.value],
														'neuronios_by_layer': [32],
														'hidden_layers': [4],
														'batch_sizes': [40],
														'epochs': [32],
														'embedding_types': [dn.EmbeddingType.WORD2VEC_CUSTOM],
														'embedding_custom_files': ['SMHD-Skipgram-AllUsers-300.bin',
																											 'SMHD-CBOW-AllUsers-300.bin',
																											 'SMHD-Skipgram-A-D-ADUsers-300.bin',
																											 'SMHD-CBOW-A-D-ADUsers-300.bin']})
				if option == '1':
						set_params.update({'use_embeddings': [dn.UseEmbedding.STATIC]})
				else:
						set_params.update({'use_embeddings': [dn.UseEmbedding.NON_STATIC]})

				model_name = 't'+option+'_wc_SMHD_ml_gl_2640_var_hl'

		
		print('Initializer experiment multclass_lstm_model '+function+', option '+option+'\n' + \
					'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label\n' + \
					'Neurons by layer variation')
		exp = ExperimentProcesses('lstm_exp9_var_L3')
		exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																	 total_registers=2640, subdirectory="anx_dep_multilabel")
		generate_model(exp, model_name, '_glorot', set_params, function)
		
if __name__ == '__main__':
		function = sys.argv[1]
		option = sys.argv[2]
		test_diff_we_function(function, option)
