'''
		Concentrates functions for log file manipulation
'''

# Handling of data structures
import numpy as np
import pandas as pd
import re

import gensim
from glove import Corpus, Glove

from utils.rsdd_file import RsddFile
from utils.smhd_file import SmhdFile

# Tokenizing
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn import preprocessing

# Exclusivo para rotina de leitura dos dados conforme baseline cnn
import gzip
import json
import random
import pickle

import os
import utils.definition_network as dn

class PreprocessData:
		
		def __init__(self):
				# Parameters to generate the dataset
				self.path_project = dn.PATH_PROJECT
				self.dataset_name = dn.DATASET_NAME
				self.label_set = dn.LABEL_SET
				self.subdirectory = self.label_set[1]
				self.total_registers = dn.TOTAL_REGISTERS
				self.src_directory = self.path_project+str("DatasetCLPsych2015/")+self.dataset_name+str("/")
				self.dst_directory = self.path_project+str("dataset/")+str(self.subdirectory)+str("/")
				
				# Parameters to pre-processing data
				self.vocabulary_size = None
				self.embedding_size = 100
				self.max_posts = 1750
				self.max_terms_by_post = 200
				self.format_input_data = dn.InputData.POSTS_LIST
				self.binary_classifier = True
				self.type_prediction_label= dn.TypePredictionLabel.BINARY
				self.remove_stopwords = False
				self.delete_low_tfid=False
				self.min_df=2
				self.min_tf=3
				self.use_embedding = dn.UseEmbedding.RAND
				self.embedding_type = dn.EmbeddingType.NONE
				self.word_embedding_custom_file = ""
				self.load_dataset_type = dn.LoadDataset.ALL_DATA_MODEL
				self.class_weights = {}
				
				## True = random order, False = chronological order
				self.random_posts = False

				## True = random order, False = class name order
				self.random_users = False
				
				
				## Tokenizing_type values
				## OH - One Hot Enconding
				## WE - Word Embedding
				self.tokenizing_type = 'OH'
				self.text_matrix_enconding = False
				self.mode_text_matrix_enconding = 'binary'

		def __set_dataset_file(self, file_path="", verbose=True):
				if self.dataset_name == 'RSDD':
						file = RsddFile(_file_path=file_path, verbose=verbose, label_set=self.label_set)
				else:
						file = SmhdFile(_file_path=file_path, verbose=verbose, label_set=self.label_set)
				
				return file
				
		def __random_dataset(self, dataset):
				return dataset.sample(dataset.shape[0], random_state=dn.SEED)
		
		def __random_list_texts(self, list_texts):
				random.seed(dn.SEED)
				return random.sample(list_texts, list_texts.__len__())
		
		def __concat_dataset(self, dataset_a, dataset_b):
				return pd.concat([dataset_a, dataset_b])
		
		def __remove_low_tfid(self, tokenizer):
				removed = 0
				for term in list(tokenizer.word_index.keys()):
						if tokenizer.word_docs[term] < self.min_df or tokenizer.word_counts[term] < self.min_tf:
								removed += 1
								del tokenizer.word_docs[term]
								del tokenizer.word_counts[term]
								del tokenizer.word_index[term]
				tokenizer.index_docs = None
				idxs = {}
				nexti = 1
				for term, oldi in sorted(tokenizer.word_index.items()):
						idxs[term] = nexti
						nexti += 1
				assert len(tokenizer.word_index) == len(idxs)
				tokenizer.word_index = idxs
				
				print("terms removed: %s; remaining: %s" % (removed, len(tokenizer.word_index)))
				return tokenizer
		
		def __remove_stopwords(self, text_list):
				stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'also', 'an', 'and', 'any', 'are', 'as',
										 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can',
										 'cannot', 'com', 'could', 'did', 'do', 'does', 'doing', 'during', 'each', 'else', 'ever', 'few',
										 'for', 'from', 'further', 'get', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers',
										 'herself', 'him', 'himself', 'his', 'how', 'however', 'http', 'in', 'into', 'is', 'it', 'its',
										 'itself', 'just', 'k', 'like', 'more', 'most', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
										 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'r', 'same', 'shall',
										 'she', 'should', 'since', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them',
										 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
										 'until', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
										 'why', 'with', 'would', 'www', 'you', 'your', 'yours', 'yourself', 'yourselves', "aren't", "can't",
										 "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't", "he'd", "he'll", "he's",
										 "here's", "how's", "isn't", "it's", "let's", "mustn't", "shan't", "she'd", "she'll", "she's",
										 "shouldn't", "that's", "there's", "they'd", "they'll", "they're", "they've", "wasn't", "we'd",
										 "we'll", "we're", "we've", "weren't", "what's", "when's", "where's", "who's",
										 "why's", "won't", "wouldn't", "you'd", "you'll", "you're", "you've"]
				
				# Keep to stop words that reference first person, denial and doubt
				# "i'd","i'll","i'm","i've",'i', 'my','myself', 'me', 'am', 'no','nor','not', 'down', 'if', 'up'
				
				for index in range(len(text_list)):
						text_split = text_list[index].split(' ')
						for stp_word in stopwords:
								for idx_post_word in range(len(text_split)):
										if stp_word == text_split[idx_post_word].lower():
											text_split[idx_post_word] = ""
						
						if "" in text_split:
								text_split.remove("")
						if " " in text_split:
								text_split.remove(" ")
								
						text_list[index] = " ".join(text_split).strip()
						
				return text_list
		
		def __get_dataset_file_name(self, type):
				if self.dataset_name == 'RSDD':
						if type == 'train':
								file_name = 'training.gz'
						elif type == 'test':
								file_name = 'testing.gz'
						else:
								file_name = 'validation.gz'
				else: #SMHD
						if type == 'train':
								file_name = 'SMHD_train.jl.gz'
						elif type == 'test':
								file_name = 'SMHD_test.jl.gz'
						else:
								file_name = 'SMHD_dev.jl.gz'
				
				return file_name
		
		def __generate_dataset(self):
				file_data = self.__set_dataset_file()
				
				file_data.file_path = self.src_directory + self.__get_dataset_file_name('train')
				file_data.set_new_file_log(str(self.dataset_name) + '_train_' + str(self.total_registers))
				file_data.generate_data_to_csv_pandas(self.total_registers,
																							self.dst_directory + str(self.dataset_name) + '_train_' + str(self.total_registers))
				
				file_data.file_path = self.src_directory + self.__get_dataset_file_name('test')
				file_data.set_new_file_log(str(self.dataset_name) + '_test_' + str(self.total_registers))
				file_data.generate_data_to_csv_pandas(self.total_registers,
																							self.dst_directory + str(self.dataset_name) + '_test_' + str(self.total_registers))
				
				file_data.file_path = self.src_directory + self.__get_dataset_file_name('validation')
				file_data.set_new_file_log(str(self.dataset_name) + '_validation_' + str(self.total_registers))
				file_data.generate_data_to_csv_pandas(self.total_registers,
																							self.dst_directory + str(self.dataset_name) + '_validation_' + str(self.total_registers))
		
		def __save_tokenizer_file(self, tokenizer, file_name):
				with open(file_name+'.df', 'wb') as handle:
					pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

		def __load_tokenizer_file(self, file_name):
				tokenizer = None
				with open(file_name+'.df', 'rb') as handle:
						tokenizer = pickle.load(handle)

				return tokenizer
			
		def __build_tokenizer_file_name(self):
				return self.path_project + 'tokenizers/'+str(self.subdirectory)+str("/")+str(self.dataset_name)+'_TR_' + str(self.total_registers) +\
							 '_VS_' + str(self.vocabulary_size) + '_TF_' + str(self.min_tf) + '_DF_' + str(self.min_df) + \
							 '_RSW_' + str(self.remove_stopwords)[0] + '_IT_' + str(self.format_input_data.value) + \
							 '_RP_' + str(self.random_posts)[0]
		
		def __build_embedding_file_name(self):
				name = self.path_project + 'pre_train_embeddings/'+str(self.subdirectory)+str("/")+str(self.dataset_name)+'_TR_' + str(self.total_registers) + \
							 '_VS_' + str(self.vocabulary_size) + '_TF_' + str(self.min_tf) + '_DF_' + str(self.min_df) + \
							 '_RSW_' + str(self.remove_stopwords)[0] + '_IT_' + str(self.format_input_data.value) + \
							 '_RP_' + str(self.random_posts)[0] + '_ET_' + str(self.embedding_type.value)
				
				if self.embedding_type.value > dn.EmbeddingType.WORD2VEC.value:
						name = name + '_' + self.word_embedding_custom_file.split('.')[0]
				
				return name
		
		def __build_dataset_name(self, type):
				return self.dst_directory + str(self.dataset_name) + '_' + str(type) + '_' + str(self.total_registers) + '.df'
		
		def __format_in_text_label(self, data):
				texts = []
				labels = []
				
				for i, row in data.iterrows():
						posts_txt = eval(row.texts)
						if self.random_posts:
								posts_txt = self.__random_list_texts(posts_txt)

						if self.format_input_data == dn.InputData.POSTS_LIST:
								texts.append(posts_txt)
						else:
								texts.append(' '.join(posts_txt))

						if self.type_prediction_label == dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL:
								index_labels = []
								subset_labels = row.label.split(',')
								for label in subset_labels:
										index_labels.append(self.label_set.index(label))

								labels.append(index_labels)
						else:
								labels.append(self.label_set.index(row.label))
						
				
				return texts, labels
		
		def __one_hot_encoding(self, texts_list, labels_list):
				tokenizer = Tokenizer(num_words=self.vocabulary_size)
				tokenizer.fit_on_texts(texts_list)
				
				x_data = tokenizer.texts_to_matrix(texts_list, mode='binary')
				if self.binary_function:
						y_data = labels_list
				else:
						y_data = to_categorical(labels_list)
				
				num_words = len(tokenizer.word_index) + 1
				
				return num_words, x_data, y_data
		
		def __one_hot_encoding_multi_label(self, labels_list):
				dimension = len(self.label_set)
				if dimension == 1:
						results = np.asarray(labels_list).astype('float32')
				else:
						results = np.zeros((len(labels_list), dimension))

						for i, label in enumerate(labels_list):
								results[i, label] = 1

				return results

		def __generate_tokenizing(self, texts_list):
				print('Generate tokenizing')
				# Generate dictionary with total words available in dataset, remove the tokens less using and save.
				tokenizer = Tokenizer(num_words=self.vocabulary_size)
				
				if self.format_input_data == dn.InputData.POSTS_LIST:
						if self.remove_stopwords:
								for index in range(len(texts_list)):
										texts_list[index] = self.__remove_stopwords(texts_list[index])

						tokenizer.fit_on_texts(post for user_posts_lists in texts_list for post in user_posts_lists)
				else:
						if self.remove_stopwords:
								texts_list = self.__remove_stopwords(texts_list)

						tokenizer.fit_on_texts(texts_list)

				if self.delete_low_tfid:
						tokenizer = self.__remove_low_tfid(tokenizer)
				
				return tokenizer
		
		def __format_input_data_tensor_3d(self, tokenizer, texts_list, labels_list):
				# Fit the dictionary generate from dataset's vocabulary at format
				# (samples, n_posts, n_terms)
				x_data = []
				for user_posts in texts_list:
						if self.remove_stopwords:
								user_posts = self.__remove_stopwords(user_posts)
					
						if user_posts.__len__() > self.max_posts:
								user_posts = user_posts[:self.max_posts]
						
						sequence_posts = pad_sequences(tokenizer.texts_to_sequences(user_posts), maxlen=self.max_terms_by_post)
						len_sequence = len(sequence_posts)
						
						if len_sequence < self.max_posts:
								sequence_posts = np.pad(sequence_posts, ((0, self.max_posts - len_sequence), (0, 0)), mode='constant')
						
						x_data.append(sequence_posts)
				
				x_data = np.array(x_data)
				
				if self.type_prediction_label== dn.TypePredictionLabel.BINARY:
						y_data = np.asarray(labels_list).astype('float32')
				elif self.type_prediction_label == dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL:
						y_data = self.__one_hot_encoding_multi_label(labels_list)
				else:
						y_data = to_categorical(labels_list)
				
				# print('Shape of x_data tensor:', x_data.shape)
				# print('Shape of y_data tensor:', y_data.shape)
				
				return x_data, y_data
		
		def __format_input_data_tensor_2d(self, tokenizer, texts_list, labels_list):
				# (samples, text_posts)
				if self.remove_stopwords:
						texts_list = self.__remove_stopwords(texts_list)
				
				if self.text_matrix_enconding:
						x_data = tokenizer.texts_to_matrix(texts_list, mode=self.mode_text_matrix_enconding)
				else:
						x_data = pad_sequences(tokenizer.texts_to_sequences(texts_list), maxlen=self.vocabulary_size)

				if self.type_prediction_label == dn.TypePredictionLabel.BINARY:
						y_data = np.asarray(labels_list).astype('float32')
				elif self.type_prediction_label == dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL:
						y_data = self.__one_hot_encoding_multi_label(labels_list)
				else:
						y_data = to_categorical(labels_list)
				
				# print('Shape of x_data tensor:', x_data.shape)
				# print('Shape of y_data tensor:', y_data.shape)

				return x_data, y_data
		
		def __format_input_data(self, tokenizer, texts_list, labels_list):
				if self.format_input_data == dn.InputData.POSTS_LIST:
						x_data, y_data = self.__format_input_data_tensor_3d(tokenizer, texts_list, labels_list)
				else:
						x_data, y_data = self.__format_input_data_tensor_2d(tokenizer, texts_list, labels_list)
				
				return x_data, y_data
		
		def __generate_data_file(self, prefix_filename):

				train_name = self.__build_dataset_name('train')
				if not os.path.exists(train_name):
						self.__generate_dataset()
				
				test_name = self.__build_dataset_name('test')
				valid_name = self.__build_dataset_name('validation')
				
				train_df = self.__set_dataset_file(train_name).read_data_from_pandas()
				test_df = self.__set_dataset_file(test_name).read_data_from_pandas()
				valid_df = self.__set_dataset_file(valid_name).read_data_from_pandas()
				
				self.__calculate_weights_by_class(train_df)
				
				print('Preprocess data...')
				if self.random_users:
						train_df = self.__random_dataset(train_df)
						test_df = self.__random_dataset(test_df)
						valid_df = self.__random_dataset(valid_df)

				train_texts, train_labels = self.__format_in_text_label(train_df)
				test_texts, test_labels = self.__format_in_text_label(test_df)
				valid_texts, valid_labels = self.__format_in_text_label(valid_df)
				
				tokenizer = self.__generate_tokenizing(train_texts)
				self.__save_tokenizer_file(tokenizer, prefix_filename)
				num_words = len(tokenizer.word_index) + 1
				
				x_train, y_train = self.__format_input_data(tokenizer, train_texts, train_labels)
				x_test, y_test = self.__format_input_data(tokenizer, test_texts, test_labels)
				x_valid, y_valid = self.__format_input_data(tokenizer, valid_texts, valid_labels)
				
				embedding_file = self.__build_embedding_file_name()
				embedding_matrix = self.load_pre_train_embedding(tokenizer)
				self.__save_tokenizer_file(embedding_matrix, embedding_file)
				
				if self.load_dataset_type == dn.LoadDataset.TRAIN_DATA_MODEL:
						return x_train, y_train, x_valid, y_valid, num_words, embedding_matrix
				if self.load_dataset_type == dn.LoadDataset.TEST_DATA_MODEL:
						return x_test, y_test
				else:
						return x_train, y_train, x_test, y_test, x_valid, y_valid, num_words, embedding_matrix
		
		def __load_only_train_data_from_file(self, prefix_filename):
				train_name = self.__build_dataset_name('train')
				valid_name = self.__build_dataset_name('validation')

				train_df = self.__set_dataset_file(train_name).read_data_from_pandas()
				valid_df = self.__set_dataset_file(valid_name).read_data_from_pandas()
				
				self.__calculate_weights_by_class(train_df)

				print('Preprocess data...', train_name)
				if self.random_users:
						train_df = self.__random_dataset(train_df)
						valid_df = self.__random_dataset(valid_df)
				
				train_texts, train_labels = self.__format_in_text_label(train_df)
				valid_texts, valid_labels = self.__format_in_text_label(valid_df)
				
				tokenizer = self.__load_tokenizer_file(prefix_filename)
				num_words = len(tokenizer.word_index) + 1
				
				x_train, y_train = self.__format_input_data(tokenizer, train_texts, train_labels)
				x_valid, y_valid = self.__format_input_data(tokenizer, valid_texts, valid_labels)
				
				embedding_file = self.__build_embedding_file_name()
				if os.path.exists(embedding_file + '.df'):
						embedding_matrix = self.__load_tokenizer_file(embedding_file)
				else:
						embedding_matrix = self.load_pre_train_embedding(tokenizer)
						self.__save_tokenizer_file(embedding_matrix, embedding_file)
				
				return x_train, y_train, x_valid, y_valid, num_words, embedding_matrix
		
		def __load_only_test_data_from_file(self, prefix_filename):
				test_name = self.__build_dataset_name('test')
				test_df = self.__set_dataset_file(test_name).read_data_from_pandas()
				
				print('Preprocess data...', test_name)
				if self.random_users:
						test_df = self.__random_dataset(test_df)
				
				test_texts, test_labels = self.__format_in_text_label(test_df)
				
				tokenizer = self.__load_tokenizer_file(prefix_filename)
				
				x_test, y_test = self.__format_input_data(tokenizer, test_texts, test_labels)
				
				return x_test, y_test
		
		def __load_data_from_file(self, prefix_filename):
				train_name = self.__build_dataset_name('train')
				test_name = self.__build_dataset_name('test')
				valid_name = self.__build_dataset_name('validation')
				
				train_df = self.__set_dataset_file(train_name).read_data_from_pandas()
				test_df = self.__set_dataset_file(test_name).read_data_from_pandas()
				valid_df = self.__set_dataset_file(valid_name).read_data_from_pandas()
				
				self.__calculate_weights_by_class(train_df)
				
				print('Preprocess data...', train_name)
				if self.random_users:
						train_df = self.__random_dataset(train_df)
						test_df = self.__random_dataset(test_df)
						valid_df = self.__random_dataset(valid_df)

				train_texts, train_labels = self.__format_in_text_label(train_df)
				test_texts, test_labels = self.__format_in_text_label(test_df)
				valid_texts, valid_labels = self.__format_in_text_label(valid_df)
				
				tokenizer = self.__load_tokenizer_file(prefix_filename)
				num_words = len(tokenizer.word_index) + 1

				x_train, y_train = self.__format_input_data(tokenizer, train_texts, train_labels)
				x_test, y_test = self.__format_input_data(tokenizer, test_texts, test_labels)
				x_valid, y_valid = self.__format_input_data(tokenizer, valid_texts, valid_labels)
				
				embedding_file = self.__build_embedding_file_name()
				if os.path.exists(embedding_file+'.df'):
					embedding_matrix = self.__load_tokenizer_file(embedding_file)
				else:
					embedding_matrix = self.load_pre_train_embedding(tokenizer)
					self.__save_tokenizer_file(embedding_matrix, embedding_file)

				return x_train, y_train, x_test, y_test, x_valid, y_valid, num_words, embedding_matrix

		def __calculate_weights_by_class(self, x_train_df):
				grouped_labels = x_train_df.groupby("label", sort='count').size().reset_index(name='count')
				total_by_label = grouped_labels.nlargest(self.label_set.__len__(), columns="count")
				total_by_label['class_weight'] = len(x_train_df) / total_by_label['count']
				
				self.class_weights = {}
				for index, label in enumerate(self.label_set):
						#TODO: fixe weigths of the first class when is multilabel 'anxiety,depression'
						if self.type_prediction_label in [dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL, dn.TypePredictionLabel.SINGLE_LABEL_CATEGORICAL]:
								self.class_weights[index] = total_by_label[0:1]['class_weight'].values[0]
						else:
								self.class_weights[index] = total_by_label[total_by_label['label'] == label]['class_weight'].values[0]
				
		def set_dataset_source(self, dataset_name, label_set, total_registers, subdirectory=""):
				self.dataset_name = dataset_name
				self.label_set = label_set

				if subdirectory != "":
						self.subdirectory = subdirectory
				else:
						self.subdirectory = label_set[1]
						
				self.total_registers = total_registers
				self.src_directory = self.path_project + str("DatasetCLPsych2015/") + self.dataset_name + str("/")
				self.dst_directory = self.path_project + str("dataset/") + str(self.subdirectory) + str("/")
		
		def load_tokenizer(self):
				print('Loading tokenizer...')
				prefix_name = self.__build_tokenizer_file_name()
				if os.path.exists(prefix_name + '.df'):
						return self.__load_tokenizer_file(prefix_name)
				else:
						print('Sorry! Tokenizer File {} not found!'.format(prefix_name))
						return None
		
		def generate_word_lookup_tokenizer(self, tokenizer=None):
				if tokenizer == None:
						tokenizer = self.load_tokenizer()

				words = tokenizer.word_index
				word_lookup_lst = list()
				# word_lookup_lst.append(None)
				# word_lookup_lst.append(np.nan)
				for word in words.keys():
						word_lookup_lst.append(word)

				word_lookup_lst = ['<Pad>'] + word_lookup_lst

				return word_lookup_lst, tokenizer

		def load_data(self):
				prefix_name = self.__build_tokenizer_file_name()
				print('Loading data...', prefix_name)

				if os.path.exists(prefix_name+'.df'):
						if self.load_dataset_type == dn.LoadDataset.TEST_DATA_MODEL:
								return self.__load_only_test_data_from_file(prefix_name)
						elif self.load_dataset_type == dn.LoadDataset.TRAIN_DATA_MODEL:
								return self.__load_only_train_data_from_file(prefix_name)
						else:
								return self.__load_data_from_file(prefix_name)
				else:
						return self.__generate_data_file(prefix_name)

		# Function allows to read data for test, it's formatting y
		# in accordance final format (ex ensemble), but it's configuring
		# x in accordance each submodel loaded
		def load_dataset_generic(self, file_path, label_set):
				prefix_name = self.__build_tokenizer_file_name()

				try:
						aux_label_set = self.label_set
						self.label_set = label_set
						data_df = self.__set_dataset_file(file_path).read_data_from_pandas()

						print('Preprocess data...', file_path)
						if self.random_users:
								data_df = self.__random_dataset(data_df)

						texts, labels = self.__format_in_text_label(data_df)
						tokenizer = self.__load_tokenizer_file(prefix_name)
						x_data, y_data = self.__format_input_data(tokenizer, texts, labels)

						self.label_set = aux_label_set

						return x_data, y_data

				except(FileNotFoundError):
						print("Failed to trying load dataset %s.\nCheck that the dataset name is "\
									"correct and that a tokenizer exists %s.df" % (file_path, prefix_name))

		# Function receive dataset and to apply formatting
		# in accordance with submodel configuration loaded
		def load_subdataset_generic(self, data_df, label_set):
				prefix_name = self.__build_tokenizer_file_name()

				try:
						aux_label_set = self.label_set
						self.label_set = label_set

						if self.random_users:
								data_df = self.__random_dataset(data_df)

						texts, labels = self.__format_in_text_label(data_df)
						tokenizer = self.__load_tokenizer_file(prefix_name)
						x_data, y_data = self.__format_input_data(tokenizer, texts, labels)

						self.label_set = aux_label_set

						return x_data, y_data

				except(FileNotFoundError):
						print("Failed to attempt to format data with tokenizer.\nCheck if tokenizer exists %s.df" % (prefix_name))

		def generate_generic_tokenizer(self, data_df):
				texts, labels = self.__format_in_text_label(data_df)

				if self.vocabulary_size is None:
						self.vocabulary_size = 0
						for text in texts:
								total_terms_text = len(text.split())
								if total_terms_text > self.vocabulary_size:
										self.vocabulary_size = total_terms_text

				tokenizer = Tokenizer(num_words=self.vocabulary_size)
				tokenizer.fit_on_texts(texts)

				return tokenizer

		def convert_dataset_to_enconding(self, data_df, generic_tokenizer):
				texts, labels = self.__format_in_text_label(data_df)

				x_data, y_data = self.__format_input_data(generic_tokenizer, texts, labels)

				return x_data, y_data

		def convert_encoding_tokenizer_to_text(self, generic_tokenizer, x_data):
				x_data_text = generic_tokenizer.sequences_to_texts(x_data)

				return x_data_text

		def convert_text_to_encoding_tokenizer(self, generic_tokenizer, x_text):
				x_data = pad_sequences(generic_tokenizer.texts_to_sequences(x_text), maxlen=self.vocabulary_size)

				return x_data

		def load_original_data(self, dataset_name):
				dataset_file = self.__build_dataset_name(dataset_name)
				return self.__set_dataset_file(dataset_file).read_data_from_pandas()
		
		def __read_word_embedding_glove_pre_train(self):
				embeddings_index = {}

				if self.embedding_type == dn.EmbeddingType.GLOVE_CUSTOM:
						glove = Glove().load(self.path_project+'pre_train_embeddings/glove/'+self.word_embedding_custom_file)

						# Normalize for min-max
						# word_vectors = glove.word_vectors
						# word_vectors_max, word_vectors_min = word_vectors.max(), word_vectors.min()
						# word_vectors = (word_vectors - word_vectors_min) / (word_vectors_max - word_vectors_min)

						# Normalize word_vector follow Euclidian Distance
						word_vectors = preprocessing.normalize(glove.word_vectors, norm="l2")

						for idx, word in enumerate(glove.dictionary):
								coefs = np.asarray(word_vectors[idx], dtype='float32')
								embeddings_index[word] = coefs
				else:
						if self.embedding_type == dn.EmbeddingType.GLOVE_TWITTER:
								f = open(self.path_project + 'pre_train_embeddings/glove.twitter/glove.840B.300d.txt', encoding="utf8")
						elif self.embedding_type == dn.EmbeddingType.GLOVE_6B:
								f = open(self.path_project + 'pre_train_embeddings/glove.6B/glove.6B.300d.txt', encoding="utf8")

						for line in f:
								values = line.split()
								word = ''.join(values[:-300])
								coefs = np.asarray(values[-300:], dtype='float32')
								embeddings_index[word] = coefs
						f.close()

				return embeddings_index

		def __load_glove_embedding_matrix(self, tokenizer):
				if tokenizer.num_words == None:
						vocabulary_size = len(tokenizer.word_index) + 1
				else:
						vocabulary_size = tokenizer.num_words

				embeddings_index = self.__read_word_embedding_glove_pre_train()
				
				print('Found %s word vectors.' % len(embeddings_index))
				
				# Preparing the GloVe word-embedding matrix
				embedding_matrix = np.zeros((vocabulary_size, self.embedding_size))
				for word, index in tokenizer.word_index.items():
						if index < vocabulary_size:
								embedding_vector = embeddings_index.get(word)
								if embedding_vector is not None:
										embedding_matrix[index] = embedding_vector
								else:
										embedding_matrix[index] = np.random.uniform(-0.1, 0.1, size=self.embedding_size)
										
				return embedding_matrix
		
		def __load_word2vec_embedding_matrix(self, tokenizer):
				if tokenizer.num_words == None:
						vocabulary_size = len(tokenizer.word_index) + 1
				else:
						vocabulary_size = tokenizer.num_words

				if self.embedding_type == dn.EmbeddingType.WORD2VEC:
						file_name = self.path_project+'pre_train_embeddings/word2vec/GoogleNews-vectors-negative300.bin'
				else:
						file_name = self.path_project+'pre_train_embeddings/word2vec/'+self.word_embedding_custom_file

				word_vectors = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=True)
				
				vocabulary_size = min(len(tokenizer.word_index) + 1, vocabulary_size)
				embedding_matrix = np.zeros((vocabulary_size, self.embedding_size))
				for word, i in tokenizer.word_index.items():
						if i >= vocabulary_size:
								continue
						try:
								embedding_vector = word_vectors[word]
								embedding_matrix[i] = embedding_vector
						except KeyError:
								embedding_matrix[i] = np.random.uniform(-0.1, 0.1, size=self.embedding_size)
				
				del word_vectors
				
				return embedding_matrix
		
		def load_pre_train_embedding(self, tokenizer):
				if self.use_embedding == dn.UseEmbedding.NONE or self.embedding_type == dn.EmbeddingType.NONE:
						return None
				
				print('Load Pre-train embeddings')
				
				# GloVe 6B (dictionary of terms based on wikipedia made by Google)
				if self.embedding_type == dn.EmbeddingType.GLOVE_6B or \
								self.embedding_type == dn.EmbeddingType.GLOVE_TWITTER or \
								self.embedding_type == dn.EmbeddingType.GLOVE_CUSTOM:
						embedding_matrix = self.__load_glove_embedding_matrix(tokenizer)
				elif self.embedding_type == dn.EmbeddingType.WORD2VEC or self.embedding_type == dn.EmbeddingType.WORD2VEC_CUSTOM:
						embedding_matrix = self.__load_word2vec_embedding_matrix(tokenizer)

				return embedding_matrix
