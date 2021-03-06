"""
    Rede Teste artigo RSDD
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
		exp = ExperimentProcesses('lstm_exp12')
		
		# Configura pre-processamento dos dados para importação
		exp.pp_data.vocabulary_size = 5000
		
		exp.pp_data.embedding_size = 300  # 300 obrigatório se for usar word_embedding word2vec google neg300
		exp.pp_data.max_posts = 2000
		exp.pp_data.max_terms_by_post = 400
		exp.pp_data.binary_classifier = True
		exp.pp_data.format_input_data = dn.InputData.POSTS_LIST
		exp.pp_data.remove_stopwords = False
		exp.pp_data.delete_low_tfid = False
		exp.pp_data.min_df = 0
		exp.pp_data.min_tf = 0
		exp.pp_data.random_posts = False #False = ordem cronológica
		exp.pp_data.random_users = False #Não usada, as amostras são sempre random no validation k-fold
		exp.pp_data.tokenizing_type = 'WE'
		exp.pp_data.use_embedding = dn.UseEmbedding.RAND
		exp.pp_data.embedding_type = dn.EmbeddingType.NONE
		
		## Gera dados conforme configuração
		lstm = ModelClass(1)
		lstm.loss_function = 'binary_crossentropy'
		lstm.optmizer_function = 'adam'
		lstm.epochs = 10
		lstm.batch_size = 32
		lstm.patience_train = 4
		lstm.use_embedding_pre_train = exp.pp_data.use_embedding
		lstm.embed_trainable = True
		
		input_layer = Input(shape=(exp.pp_data.max_posts, exp.pp_data.max_terms_by_post))

		emb = Embedding(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size, trainable=lstm.embed_trainable)
		embedded = TimeDistributed(emb)(input_layer)
		
		lstm_layer = Sequential()
		lstm_layer.add(LSTM(64, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=False,
												input_shape=(exp.pp_data.vocabulary_size, exp.pp_data.embedding_size)))
		lstm_layer = TimeDistributed(lstm_layer)(embedded)
		
		combined = Flatten()(lstm_layer)
		output_layer = Dense(1, activation='sigmoid')(combined)
		
		lstm.model = Model(inputs=input_layer, outputs=output_layer)

		time_ini_exp = datetime.datetime.now()
		exp.test_hypeparams(lstm)
		exp.set_period_time_end(time_ini_exp, 'Total experiment')
