"""
    Test Model using RSDD Dataset
"""
import utils.definition_network as dn
import numpy as np

from utils.experiment_processes import ExperimentProcesses
import datetime
# LAYERS
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from network_model.model_class import ModelClass

if __name__ == '__main__':
		exp = ExperimentProcesses('lstm_exp14_L4')
		
		exp.pp_data.vocabulary_size = 5000
		
		exp.pp_data.embedding_size = 300
		exp.pp_data.max_posts = 1750
		exp.pp_data.max_terms_by_post = 300
		exp.pp_data.binary_classifier = True
		exp.pp_data.format_input_data = dn.InputData.POSTS_LIST
		exp.pp_data.remove_stopwords = False
		exp.pp_data.delete_low_tfid = False
		exp.pp_data.min_df = 0
		exp.pp_data.min_tf = 0
		exp.pp_data.random_posts = False  # False = cronological order
		exp.pp_data.random_users = False  # random sample in validation k-fold
		exp.pp_data.tokenizing_type = 'WE'
		exp.pp_data.use_embedding = dn.UseEmbedding.NONE
		exp.pp_data.embedding_type = dn.EmbeddingType.GLOVE_TWITTER
		exp.pp_data.load_dataset_type = dn.LoadDataset.TRAIN_DATA_MODEL

		lstm = ModelClass(1)
		lstm.loss_function = 'binary_crossentropy'
		lstm.optmizer_function = 'adam'
		lstm.use_embedding_pre_train = exp.pp_data.use_embedding
		lstm.embed_trainable = False
		
		# Train
		neuronios_by_layer = [16]
		batch_sizes = [20]
		epochs = [64]

		np.random.seed(dn.SEED)
		time_ini_rep = datetime.datetime.now()
		
		x_train, y_train, x_valid, y_valid, num_words, embedding_matrix = exp.pp_data.load_data()
		exp.set_period_time_end(time_ini_rep, 'Load data')

		for neuronios in neuronios_by_layer:
				for batch_size in batch_sizes:
						for epoch in epochs:
								exp.experiment_name = 'lstm_exp14_L4' + '_N' + str(neuronios) + '_B' + str(batch_size) + '_E' + str(epoch)
								lstm.epochs = epoch
								lstm.batch_size = batch_size
								lstm.patience_train = epoch/2

								data_dim = exp.pp_data.max_terms_by_post
								timesteps = exp.pp_data.max_posts

								lstm.model = Sequential()
								lstm.model.add(LSTM(neuronios, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True, stateful=True,
																		batch_input_shape=(batch_size, timesteps, data_dim)))
								lstm.model.add(LSTM(neuronios, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True, stateful=True))
								lstm.model.add(LSTM(neuronios, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True, stateful=True))
								lstm.model.add(LSTM(neuronios, activation='tanh', dropout=0.2, recurrent_dropout=0.2, stateful=True))
								lstm.model.add(Dense(1, activation='sigmoid'))

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
								exp.experiment_name = 'lstm_exp14_L4' + '_N' + str(neuronios) + '_B' + str(batch_size) + '_E' + str(epoch)
								lstm.epochs = epoch
								lstm.batch_size = batch_size
								lstm.patience_train = epoch/2

								lstm.model = exp.load_model(dn.PATH_PROJECT+exp.experiment_name+'.h5')
								exp.save_geral_configs()
								exp.save_summary_model(lstm.model)
								exp.predict_samples(lstm, x_test, y_test)
