"""
    Test Model using RSDD Dataset
"""
import utils.definition_network as dn

from utils.experiment_processes import ExperimentProcesses
import datetime
# LAYERS
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, TimeDistributed, Input
from keras.layers import LSTM
from network_model.model_class import ModelClass

if __name__ == '__main__':
		exp = ExperimentProcesses('lstm_exp6_1')
		
		exp.pp_data.vocabulary_size = 5000
		
		exp.pp_data.embedding_size = 300
		exp.pp_data.max_posts = 1750
		exp.pp_data.max_terms_by_post = 100
		exp.pp_data.binary_classifier = True
		exp.pp_data.format_input_data = dn.InputData.POSTS_ONLY_TEXT
		exp.pp_data.remove_stopwords = False
		exp.pp_data.delete_low_tfid = False
		exp.pp_data.min_df = 0
		exp.pp_data.min_tf = 0
		exp.pp_data.random_posts = False
		exp.pp_data.random_users = False
		exp.pp_data.tokenizing_type = 'WE'
		exp.pp_data.use_embedding = dn.UseEmbedding.RAND
		exp.pp_data.embedding_type = dn.EmbeddingType.NONE
		
		lstm = ModelClass(1)
		lstm.loss_function = 'binary_crossentropy'
		lstm.optmizer_function = 'adam'
		lstm.epochs = 10
		lstm.batch_size = 32
		lstm.patience_train = 4
		lstm.use_embedding_pre_train = exp.pp_data.use_embedding
		lstm.embed_trainable = True
		
		lstm.model = Sequential()
		lstm.model.add(Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size, trainable=lstm.embed_trainable))
		lstm.model.add(LSTM(64, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
		lstm.model.add(LSTM(64, activation='tanh', dropout=0.2, recurrent_dropout=0.2))
		lstm.model.add(Dense(1, activation='sigmoid'))
		
		time_ini_exp = datetime.datetime.now()
		# exp.k_fold_cross_validation(lstm)
		exp.test_hypeparams(lstm)
		exp.set_period_time_end(time_ini_exp, 'Total experiment')
