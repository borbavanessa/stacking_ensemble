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

def generate_model_ml_gl(exp, name_model):
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
		exp.pp_data.type_prediction_label = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL
		
		exp.use_custom_metrics = False
		exp.use_valid_set_for_train = True
		exp.valid_split_from_train_set = 0.0
		exp.imbalanced_classes = False
		
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
		batch_sizes = [20]
		dropouts = [0.1, 0.2]
		
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
def generate_model_ml_le(exp, name_model):
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
		batch_sizes = [20]
		dropouts = [0.1, 0.2]
		
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


# Generate model lstm_exp9_var_L3_N16_B40_E32_D0.2 static custom glove A-D-AD
# For SMHD anx
def load_submodel_anx(exp, name_model, kernel_name):
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
		exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_CUSTOM
		exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
		exp.pp_data.word_embedding_custom_file = 'SMHD-glove-A-D-ADUsers-300.pkl'
		exp.pp_data.type_prediction_label = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL
		exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
		
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


# Generate model lstm_exp9_var_L3_N16_B40_E32_D0.2 non-static custom w2v CBOW A-D-AD
# For SMHD dep
def load_submodel_dep(exp, name_model, kernel_name):
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
		exp.pp_data.word_embedding_custom_file = 'SMHD-CBOW-A-D-ADUsers-300.bin'
		exp.pp_data.type_prediction_label = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL
		exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
		
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


# Generate model lstm_exp9_var_L3_N16_B40_E32_D0.2 non-static custom w2v CBOW AllUser
# For SMHD anx_dep
def load_submodel_anx_dep(exp, name_model, kernel_name):
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


def load_best_lstm_mult_label_multi_class(exp, name_model, we_file_name):
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
		
		exp.pp_data.load_dataset_type = dn.LoadDataset.TEST_DATA_MODEL
		np.random.seed(dn.SEED)
		time_ini_rep = datetime.datetime.now()
		x_test, y_test = exp.pp_data.load_dataset_generic(
				"/home/vanessa/PycharmProjects/RecurrentNetworks/dataset/anx_dep_multilabel/SMHD_multi_label_test_test_528.df",
				['control', 'anxiety', 'depression'])
		exp.set_period_time_end(time_ini_rep, 'Load data')
		
		for neuronios in neuronios_by_layer:
				for batch_size in batch_sizes:
						for epoch in epochs:
								for dropout in dropouts:
										lstm.epochs = epoch
										lstm.batch_size = batch_size
										lstm.patience_train = epoch / 2
										exp.experiment_name = name_model + '_lstm_exp9_var_L3' + '_N' + str(neuronios) + '_B' + \
																					str(batch_size) + '_E' + str(epoch) + '_D' + str(dropout) + '_' + \
																					we_file_name
										
										lstm.model = exp.load_model(dn.PATH_PROJECT + exp.experiment_name + '.h5')
										exp.save_geral_configs()
										exp.save_summary_model(lstm.model)
										exp.predict_samples(lstm, x_test, y_test)


def main(arg):
		if arg == '1':
				print('Initializer experiment 1 (model SMHD_ml_gl_1560)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1560 single-label')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1560, subdirectory="anx_dep_multilabel")
				generate_model_ml_gl(exp, 'SMHD_ml_gl_1560')
		
		elif arg == '2':
				print('Initializer experiment 2 (model SMHD_ml_le_1560)\n' + \
							'Set: kernel_initializer=lecun_uniform, dataset=SMHD_1560 single-label')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1560, subdirectory="anx_dep_multilabel")
				generate_model_ml_le(exp, 'SMHD_ml_le_1560')
		
		elif arg == '3':
				print('Initializer experiment 3 (model SMHD_ml_gl_2640)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model_ml_gl(exp, 'SMHD_ml_gl_2640')
		
		elif arg == '4':
				print('Initializer experiment 4 (model SMHD_ml_le_2640)\n' + \
							'Set: kernel_initializer=lecun_uniform, dataset=SMHD_2640 multi-label')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				generate_model_ml_le(exp, 'SMHD_ml_le_2640')
		
		elif arg == '5':
				print('Initializer experiment 5 (model SMHD_ml_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model_ml_gl(exp, 'SMHD_ml_gl_1040')
		
		elif arg == '6':
				print('Initializer experiment 6 (model SMHD_ml_le_1040)\n' + \
							'Set: kernel_initializer=lecun_uniform, dataset=SMHD_1040 single-label')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				generate_model_ml_le(exp, 'SMHD_ml_le_1040')
		
		elif arg == '7':
				print('Initializer experiment 7 (model SMHD_ml_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model_ml_gl(exp, 'SMHD_ml_gl_2160')
		
		elif arg == '8':
				print('Initializer experiment 8 (model SMHD_ml_le_2160)\n' + \
							'Set: kernel_initializer=lecun_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				generate_model_ml_le(exp, 'SMHD_ml_le_2160')
		
		elif arg == '9':
				print('Initializer experiment 9 (model SMHD_ml_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model_ml_gl(exp, 'SMHD_ml_gl_880')
		
		elif arg == '10':
				print('Initializer experiment 10 (model SMHD_ml_le_880)\n' + \
							'Set: kernel_initializer=lecun_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model_ml_le(exp, 'SMHD_ml_le_880')
		
		elif arg == '11':
				print('Initializer experiment 11 (model SMHD_ml_gl_1040)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_1040 multi-label')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=1040, subdirectory="anxiety")
				load_submodel_anx(exp, 'SMHD_ml_gl_1040_glove_a-d-aduser', '_glorot')
		
		elif arg == '12':
				print('Initializer experiment 12 (model SMHD_ml_gl_2160)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_2160 multi-label')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2160, subdirectory="depression")
				load_submodel_dep(exp, 'SMHD_ml_gl_2160_cbow_a-d-aduser', '_glorot')
		
		elif arg == '13':
				print('Initializer experiment 13 (model SMHD_ml_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				load_submodel_anx_dep(exp, 'SMHD_ml_gl_880_cbow_alluser', '_glorot')
		
		elif arg == '14':
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=2640, subdirectory="anx_dep_multilabel")
				
				exp.pp_data.embedding_type = dn.EmbeddingType.WORD2VEC_CUSTOM
				exp.pp_data.use_embedding = dn.UseEmbedding.STATIC
				exp.pp_data.word_embedding_custom_file = 'SMHD-CBOW-A-D-ADUsers-300.bin'
				exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL
				
				we_file_name = 'ET_' + str(exp.pp_data.embedding_type.value) + '_UE_' + str(exp.pp_data.use_embedding.value) + \
											 '_EF_' + exp.pp_data.word_embedding_custom_file.split('.')[0] + '_glove6B300d_glorot'
				
				load_best_lstm_mult_label_multi_class(exp, 'exp9_' + we_file_name[0:13] + we_file_name[18:30] + '_2640',
																							we_file_name)

		elif arg == '15':
				mode_matrix_encoding = ["binary", "count", "tfidf", "freq"]
				for mode in mode_matrix_encoding:
						print('Initializer experiment multclass_lstm_model, text matrix enconding mode = ' + mode + '\n' + \
									'Set: kernel_initializer=lecun, dataset=SMHD_1040 multi-label\n')

						exp = ExperimentProcesses('lstm_exp9_var_L3')
						exp.pp_data.text_matrix_enconding = True
						exp.mode_text_matrix_enconding = mode

						exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																					 total_registers=1040, subdirectory="anxiety")
						generate_model_ml_le(exp, 'SMHD_ml_le_1040_tme_'+mode)

		elif arg == '16':
				print('Initializer experiment 16 (model SMHD_ml_le_880)\n' + \
							'Set: kernel_initializer=lecun_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model_ml_le(exp, 'SMHD_ml_le_880')

		elif arg == '17':
				print('Initializer experiment 9 (model SMHD_ml_gl_880)\n' + \
							'Set: kernel_initializer=glorot_uniform=xavier_uniform, dataset=SMHD_880 multi-label')
				exp = ExperimentProcesses('lstm_exp9_var_L3')
				exp.pp_data.set_dataset_source(dataset_name='SMHD', label_set=['control', 'anxiety', 'depression'],
																			 total_registers=880, subdirectory="anxiety,depression")
				generate_model_ml_gl(exp, 'SMHD_ml_gl_880')

if __name__ == '__main__':
		arg = sys.argv[1]
		main(arg)