import numpy as np
import pandas as pd
import datetime
import random
import tensorflow as tf # this sets KMP_BLOCKTIME and OMP_PROC_BIND
import keras
import io

import utils.definition_network as dn

from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
from keras.models import load_model
import pandas.io.json as pd_json

from utils.metrics import Metrics
from utils.preprocess_data import PreprocessData
from utils.log import Log
from numpy import savez_compressed
from numpy import load

class ExperimentProcesses:
		def __init__(self, filename):
				self.experiment_name = filename
				self.generic_model_class = False
				self.metrics_test = Metrics()
				self.pp_data = PreprocessData()
				self.log = Log(filename, "txt")
				self.local_device_protos = None
				self.metrics_based_sample = False
				self.total_gpus = dn.TOTAL_GPUS
				self.use_custom_metrics = True
				self.use_valid_set_for_train = True
				self.valid_split_from_train_set = 0
				self.imbalanced_classes = False
				self.iteraction = 1

				self.get_available_gpus()

				# delete the existing values
				# del os.environ['OMP_PROC_BIND']
				# del os.environ['KMP_BLOCKTIME']
		
		def rename_model_layers(self, file_model, sufix_name):
				if self.use_custom_metrics:
						model = load_model(file_model, custom_objects={'f1_m': self.metrics_test.f1_m,
																													 'precision_m': self.metrics_test.precision_m,
																													 'recall_m': self.metrics_test.recall_m},
															 compile=True)
				else:
						model = load_model(file_model, custom_objects=None, compile=True)
				
				for i, layer in enumerate(model.layers):
						if i == 0:
								layer.name = 'emb_' + str(sufix_name)
						elif i == 1:
								layer.name = 'dense_1_' + str(sufix_name)
						elif i == 2:
								layer.name = 'dense_2_' + str(sufix_name)
						elif i == 3:
								layer.name = 'dense_3_' + str(sufix_name)
						elif i == 4:
								layer.name = 'dense_4_' + str(sufix_name)
				
				model.save(file_model)
		
		def save_data_format_train(self, data, data_name):
				savez_compressed(dn.PATH_PROJECT + self.experiment_name + '_' + data_name + '.npz')

		def load_data_format_train(self, file_path):
				dict_data = load(file_path + '.npz')
				return dict_data['arr_0']

		def set_period_time_end(self, time_ini, task_desc):
				time_end = datetime.datetime.now()
				period = time_end - time_ini
				print('%s - Ini: %s\tEnd: %s\tTotal: %s' % (task_desc, time_ini.strftime("%Y-%m-%d %H:%M:%S"),
																										time_end.strftime("%Y-%m-%d %H:%M:%S"),	period))

		def save_geral_configs(self, addition_infors=''):
				config = "\n=== DATASET INFORS ===\n" + \
								 "Dataset type: " + str(self.pp_data.dataset_name) + "\n" + \
								 "Total Registers by group (train, test, valid): " + str(self.pp_data.total_registers) + "\n" + \
								 "Embedding Type: " + str(self.pp_data.embedding_type.value) + "\n" + \
								 "Use Embedding: "+ str(self.pp_data.use_embedding.value) + "\n" + \
								 "File Custom Word Embedding: "+ str(self.pp_data.word_embedding_custom_file) + "\n" + \
								 "Remove stopwords: " + str(self.pp_data.remove_stopwords) + "\n" + \
								 "Use binary function: " + str(self.pp_data.binary_classifier) + "\n" + \
								 "Posts order: " + str(self.pp_data.random_posts) + "\n" + \
								 "User order: " + str(self.pp_data.random_users) + "\n" + \
								 "Tokenizer Type (WE - Word Embedding, OH - OneHot): " + str(self.pp_data.tokenizing_type) + "\n" + \
								 "Total GPUs: "+str(self.total_gpus) + "\n" + \
								 "Devices available: \n"+ str(self.local_device_protos)

				if addition_infors.__len__() > 0:
						config = config + "\n=== ADDITIONAL INFORS ===\n" + addition_infors

				self.log.save(config)
		
		def save_summary_model(self, model):
				with open(self.log.file_path, 'a+') as file_handle:
					model.summary(print_fn=lambda x: file_handle.write(x + '\n'))

		def save_nwk_configs(self, config_nwk):
				self.log.save(config_nwk)
				
		def save_summary(self, model):
				with open(self.log.file_path, 'a+') as file_handle:
						model.summary(print_fn=lambda x: file_handle.write(x + '\n'))
		
		def save_data_model(self, model_class):
				self.save_nwk_configs(model_class.get_config())
				self.save_summary(model_class.model)
				self.log.save("\nLayer's Full Configuration: " + str(model_class.model.get_config()) + "\n")

		def save_history(self, history, file_name, mode_file='w'):
				hist_df = pd.DataFrame(history.history)

				with open(dn.PATH_PROJECT+file_name+'.csv', mode=mode_file) as file:
						hist_df.to_csv(file)

		def save_metrics(self, metrics, file_name):
				metrics.to_csv(dn.PATH_PROJECT + file_name + '.csv', sep=';', encoding='utf-8', index=False, header=True, mode='a')
				
		def save_embedding_weights(self, model_class):
				if dn.USE_TENSOR_BOARD:
						embedded = model_class.model.layers[0]
						embedded_weights = embedded.get_weights()[0]
						print("Embedded weights: " + str(embedded_weights.shape))  # shape: (vocab_size, embedding_dim)
						tokenizer = self.pp_data.load_tokenizer()
						encoder = ['None']
						for key in tokenizer.word_index:
								encoder.append(key)
						
						out_v = io.open(self.experiment_name + '/vecs.tsv', 'w', encoding='utf-8')
						out_m = io.open(self.experiment_name + '/meta.tsv', 'w', encoding='utf-8')
						
						for num, word in enumerate(encoder):
								if num + 1 < len(embedded_weights):
										vec = embedded_weights[num + 1]  # skip 0, it's padding.
										out_m.write(word + "\n")
										out_v.write('\t'.join([str(x) for x in vec]) + "\n")
						out_v.close()
						out_m.close()
				
		def build_callbacks_model(self, model_class, x_train, early_stopping_train, model_checkpoint):
				if dn.USE_TENSOR_BOARD:
						callbacks = [early_stopping_train,
												 model_checkpoint,
												 keras.callbacks.TensorBoard(log_dir=dn.PATH_PROJECT + self.experiment_name,
																										 histogram_freq=1,
																										 batch_size=model_class.batch_size,
																										 write_graph=True,
																										 write_grads=True,
																										 write_images=True,
																										 embeddings_freq=1,
																										 embeddings_data=x_train,
																										 update_freq=model_class.epochs, )]
				else:
						callbacks = [early_stopping_train, model_checkpoint]
				
				return callbacks
		
		def build_metrics(self, model_class):
				if self.use_custom_metrics:
						def_metrics = [model_class.main_metric, self.metrics_test.f1_m, self.metrics_test.precision_m, self.metrics_test.recall_m]
				else:
						def_metrics = [model_class.main_metric]
				
				return def_metrics
				
		def build_model_fit(self, model_class, callbacks, x_train, y_train, x_valid, y_valid):
				if self.use_valid_set_for_train:
						if self.imbalanced_classes:
								history = model_class.model.fit(x_train, y_train,
																								validation_data=(x_valid, y_valid),
																								class_weight=self.pp_data.class_weights,
																								batch_size=model_class.batch_size,
																								epochs=model_class.epochs,
																								verbose=model_class.verbose,
																								callbacks=callbacks)
						else:
								history = model_class.model.fit(x_train, y_train,
																								validation_data=(x_valid, y_valid),
																								batch_size=model_class.batch_size,
																								epochs=model_class.epochs,
																								verbose=model_class.verbose,
																								callbacks=callbacks)
				else:
						if self.imbalanced_classes:
								history = model_class.model.fit(x_train, y_train,
																								validation_split=self.valid_split_from_train_set,
																								class_weight=self.pp_data.class_weights,
																								batch_size=model_class.batch_size,
																								epochs=model_class.epochs,
																								verbose=model_class.verbose,
																								callbacks=callbacks)
						else:
								history = model_class.model.fit(x_train, y_train,
																								validation_split=self.valid_split_from_train_set,
																								batch_size=model_class.batch_size,
																								epochs=model_class.epochs,
																								verbose=model_class.verbose,
																								callbacks=callbacks)
				
				return history

		def format_input_dataset_ensemble(self, model_class, x_data, y_data):
				# Replicates input data each submodel
				X_data = [x_data for _ in range(len(model_class.model.input))]
				Y_data = y_data
				return X_data, Y_data
		
		def get_metrics_log_by_fold(self, text, acc, f1, precision, recall):
				text = '\n=== Model Performance - ' + text + ' ===\n' + \
							 'Measure; Average; Standard Deviation\n' + \
							 str('accura') + str("; %.2f" % acc.mean()) + str("; %.2f\n" % acc.std()) + \
							 str('precis') + str("; %.2f" % precision.mean()) + str("; %.2f\n" % precision.std()) + \
							 str('recall') + str("; %.2f" % recall.mean()) + str("; %.2f\n" % recall.std()) + \
							 str('f1_scr') + str("; %.2f" % f1.mean()) + str("; %.2f" % f1.std())
				print(text)

				self.log.save(text)

		def get_available_gpus(self):
				self.local_device_protos = device_lib.list_local_devices()
				self.total_gpu = len([x.name for x in self.local_device_protos if x.device_type == 'GPU'])

		def load_partial_test_set(self, x_test, y_test):
				folds_test = list(StratifiedKFold(n_splits=2, shuffle=False, random_state=dn.SEED).split(x_test, y_test))
				indexs = folds_test[0][0]
				random.shuffle(indexs)
				return x_test[indexs], y_test[indexs]

		def train_test(self, model_class, x_train, y_train, x_valid, y_valid, x_test, y_test):
				metrics_pds = None
				original_model = model_class.model
				
				self.save_geral_configs()
				self.save_summary_model(model_class.model)
				
				sample_test = x_test.shape[0] // 2
				X_test_train = x_test[:sample_test, :]
				Y_test_train = y_test[:sample_test]
				
				for r in range(dn.REPEAT_VALIDATION - 1):
						print('Repeat ' + str(r+1))
						time_ini_rep = datetime.datetime.now()
						
						# Build Model
						model_class.model = original_model
						model_class.model.compile(loss=model_class.loss_function,	 optimizer=model_class.optmizer_function,
																			metrics=[model_class.main_metric, self.metrics_test.f1_m,
																							 self.metrics_test.precision_m,
																							 self.metrics_test.recall_m])
						
						model_class.model.fit(x_train, y_train,
																	validation_data=(x_valid, y_valid),
																	batch_size=model_class.batch_size,
																	epochs=model_class.epochs,
																	verbose=model_class.verbose)
						
						_, acc, f1, precision, recall = model_class.model.evaluate(X_test_train, Y_test_train,
																																								 model_class.verbose)
						
						if metrics_pds is None:
								metrics_pds = pd.Series([acc, f1, precision, recall],
																				index=[model_class.main_metric, 'f1', 'precision', 'recall'])
						else:
								metrics_pds.append(pd.Series([acc, f1, precision, recall],
																						 index=[model_class.main_metric, 'f1', 'precision', 'recall']),
																	 ignore_index=True, verify_integrity=False)
						self.get_metrics_log_by_fold('Repeat ' + str(r+1), acc, f1, precision, recall)
						
						y_pred = model_class.model.predict(X_test_train, batch_size=model_class.batch_size, verbose=0)
						res = str(self.metrics_test.confusion_matrix(Y_test_train, y_pred, self.pp_data.label_set,
																												 self.pp_data.binary_class))
						self.log.save(res)
						self.set_period_time_end(time_ini_rep, 'Repeat ' + str(r+1))
				# Average weigth of the n repetitions
				if metrics_pds is not None:
						self.get_metrics_log_by_fold('Repeat ' + str(r+1) + ' - Mean Final',
																				 metrics_pds.accuracy.mean(),
																				 metrics_pds.f1.mean(),
																				 metrics_pds.precision.mean(),
																				 metrics_pds.recall.mean())
				
		def final_model(self, model_class, x_train, y_train, x_valid, y_valid, x_test, y_test):
				# Final model training and testing
				print('Train final model')
				time_ini_rep = datetime.datetime.now()
				
				sample_test = x_test.shape[0] // 2
				X_test_final = x_test[sample_test:, :]
				Y_test_final = y_test[sample_test:]
				
				self.save_summary_model(model_class.model)
				model_class.model.compile(loss=model_class.loss_function, optimizer=model_class.optmizer_function,
																	metrics=[model_class.main_metric, self.metrics_test.f1_m, self.metrics_test.precision_m,
																					 self.metrics_test.recall_m])
				model_class.model.fit(x_train, y_train,
															validation_data=(x_valid, y_valid),
															batch_size=model_class.batch_size,
															epochs=model_class.epochs,
															verbose=model_class.verbose)
				
				_, acc, f1, precision, recall = model_class.model.evaluate(X_test_final, Y_test_final, verbose=0)
				self.get_metrics_log_by_fold('Final Model', acc, f1, precision, recall)
				
				y_pred = model_class.model.predict(X_test_final, batch_size=model_class.batch_size, verbose=0)
				self.log.save(str(self.metrics_test.confusion_matrix(Y_test_final, y_pred,
																														 self.pp_data.label_set, self.pp_data.binary_class)))
				self.set_period_time_end(time_ini_rep, 'Generate Final Model')
				
		def test_hypeparams(self, model_class):
				np.random.seed(dn.SEED)
				time_ini_rep = datetime.datetime.now()
				x_train, y_train, x_test, y_test, x_valid, y_valid, num_words, embedding_matrix = self.pp_data.load_data()
				self.set_period_time_end(time_ini_rep, 'Load data')
				
				original_model = model_class.model
				if (model_class.use_embedding_pre_train == (dn.UseEmbedding.STATIC or dn.UseEmbedding.NON_STATIC)):
						original_model.layers[0].set_weights([embedding_matrix])

				self.save_geral_configs()
				self.save_summary_model(model_class.model)

				time_ini_repeat = datetime.datetime.now()

				# generate model
				with tf.device('cpu'):
						model_class.model = original_model

				try:
						model_class.model = multi_gpu_model(model_class.model, gpus=self.total_gpus, cpu_merge=False)
						print("Training using " + str(self.total_gpus) + " GPUs..")
				except:
						print("Training using single GPU or CPU..")

				model_class.model.compile(loss=model_class.loss_function, optimizer=model_class.optmizer_function,
																	metrics=[model_class.main_metric,
																					 self.metrics_test.f1_m,
																					 self.metrics_test.precision_m,
																					 self.metrics_test.recall_m])

				# train model
				early_stopping_train = EarlyStopping(monitor='val_loss', mode='min', verbose=model_class.verbose,
																						 patience=model_class.patience_train)
				model_checkpoint = ModelCheckpoint(dn.PATH_PROJECT + self.experiment_name + '.h5',	monitor='val_acc',
																					 mode='max', save_best_only=True, save_weights_only=False)

				history = model_class.model.fit(x_train, y_train,
																				validation_data=(x_valid, y_valid),
																				batch_size=model_class.batch_size,
																				epochs=model_class.epochs,
																				verbose=model_class.verbose,
																				callbacks=[early_stopping_train, model_checkpoint])

				self.save_history(history, self.experiment_name)

				y_pred = model_class.model.predict(x_test, batch_size=model_class.batch_size, verbose=model_class.verbose)
				res = str(self.metrics_test.confusion_matrix(y_test, y_pred, self.pp_data.label_set,
																										 self.pp_data.binary_class))
				self.log.save(res)

				acc = accuracy_score(y_test, np.round(y_pred))
				f1 = f1_score(y_test, np.round(y_pred))
				precision = precision_score(y_test, np.round(y_pred))
				recall = recall_score(y_test, np.round(y_pred))

				self.get_metrics_log_by_fold('Predition Metrics', acc, f1, precision, recall)

				self.set_period_time_end(time_ini_repeat, 'Generate Model')
				
		def generate_model_hypeparams(self, model_class, x_train, y_train, x_valid, y_valid, embedding_matrix, ensemble=False):
				original_model = model_class.model
				if (model_class.use_embedding_pre_train == (dn.UseEmbedding.STATIC or dn.UseEmbedding.NON_STATIC)):
						original_model.layers[0].set_weights([embedding_matrix])
				
				# self.save_geral_configs('Experiment Specific Configuration: ' + self.experiment_name)
				# self.save_summary_model(model_class.model)
				
				time_ini_repeat = datetime.datetime.now()
				
				with tf.device('cpu'):
						model_class.model = original_model
				
				try:
						model_class.model = multi_gpu_model(model_class.model, gpus=self.total_gpus, cpu_merge=False)
						print("Training using " + str(self.total_gpus) + " GPUs..")
				except:
						print("Training using single GPU or CPU..")
				
				model_class.model.compile(loss=model_class.loss_function, optimizer=model_class.optmizer_function,
																	metrics=self.build_metrics(model_class))
				
				early_stopping_train = EarlyStopping(monitor='val_loss', mode='min', verbose=model_class.verbose,
																						 patience=model_class.patience_train)
				model_checkpoint = ModelCheckpoint(dn.PATH_PROJECT + self.experiment_name + '.h5', monitor='val_acc',
																					 mode='max', save_best_only=True, save_weights_only=False)
				
				if ensemble:
						x_train, y_train = self.format_input_dataset_ensemble(model_class, x_train, y_train)
						if self.use_valid_set_for_train:
								x_valid, y_valid = self.format_input_dataset_ensemble(model_class, x_valid, y_valid)

				callbacks = self.build_callbacks_model(model_class, x_train, early_stopping_train, model_checkpoint)
				history = self.build_model_fit(model_class, callbacks, x_train, y_train, x_valid, y_valid)
				
				self.save_embedding_weights(model_class)
				self.save_history(history, self.experiment_name, mode_file='w+')
				self.set_period_time_end(time_ini_repeat, 'Generate Model')
				
		def load_model(self, file_name):
				if self.use_custom_metrics:
						model = load_model(file_name, custom_objects={'f1_m': self.metrics_test.f1_m,
																																			'precision_m': self.metrics_test.precision_m,
																																			'recall_m': self.metrics_test.recall_m},
															 compile=True)
				else:
						model = load_model(file_name, custom_objects=None, compile=True)
						
				return model
		
		def evaluate_model(self, model_class, x_test, y_test, ensemble=False):
				if ensemble:
						x_test, y_test = self.format_input_dataset_ensemble(model_class, x_test, y_test)

				if self.use_custom_metrics:
						loss, acc, f1, precision, recall = model_class.model.evaluate(x_test, y_test, verbose=0)
						self.get_metrics_log_by_fold('Evaluate Model', loss, acc, f1, precision, recall)
				else:
						loss, acc = model_class.model.evaluate(x_test, y_test, verbose=0)
						self.get_metrics_log_by_fold('Evaluate Model', loss, acc)
			
		def print_confustion_matrix_binary_classifier(self, model_class, x_test, y_test):
				y_pred = model_class.model.predict(x_test, batch_size=model_class.batch_size, verbose=model_class.verbose)
				self.metrics_test.save_predict_results(self.experiment_name + '_' + self.pp_data.dataset_name + '_' + \
																							 self.pp_data.label_set[1] + '_predict_results',
																							 self.pp_data.type_prediction_label, y_pred, y_test)

				res = str(self.metrics_test.confusion_matrix_binary_classifier(y_test, y_pred, self.pp_data.label_set,
																																			 self.pp_data.type_prediction_label))
				self.log.save(res)
				
				if self.pp_data.type_prediction_label == dn.TypePredictionLabel.BINARY_CATEGORICAL:
						average = None
				else:
						average = 'binary'

				acc = accuracy_score(y_test, np.round(y_pred))
				f1 = f1_score(y_test, np.round(y_pred), average=average)
				precision = precision_score(y_test, np.round(y_pred), average=average)
				recall = recall_score(y_test, np.round(y_pred), average=average)

				self.get_metrics_log_by_fold('Predition Metrics', acc, f1, precision, recall)
				
				return y_pred, acc, f1, precision, recall
		
		def print_confusion_matrix_multiclass_classifier(self, model_class, x_test, y_test):
				# average = [None, 'binary'(default), 'micro', 'macro', 'samples', 'weighted']
				y_pred = model_class.model.predict(x_test, batch_size=model_class.batch_size, verbose=model_class.verbose)
				
				final_metrics = self.metrics_test.calc_metrics_multilabel(y_test, y_pred, self.pp_data.label_set,
																																	self.pp_data.type_prediction_label,
																																	self.metrics_based_sample)

				self.log.save('Correct Prediction per Label: ' + str(final_metrics['Correct Prediction per Label']))
				self.log.save('Exact Match Ratio: ' + str(final_metrics['Exact Match Ratio']))
				self.log.save('Hamming Loss: ' + str(final_metrics['Hamming Loss']))
				self.log.save('Confusion Matrix: \n' + str(final_metrics['Multi-label Confusion Matrix']))
				self.log.save('=== Model Performance - Multi-label Metrics ===\n' + str(final_metrics['Multi-label Report']))
				self.log.save('\n\n=== Model Performance - Single-label Metrics ===\n' + str(final_metrics['Single-label Report']))

				self.metrics_test.save_predict_results(self.experiment_name + '_' + self.pp_data.dataset_name + '_' + \
																							 self.pp_data.label_set[1] + '_predict_results',
																							 self.pp_data.type_prediction_label, y_pred, y_test)

				data_pd = pd_json.json_normalize(dict({'test': self.experiment_name,
																							'iteraction': str(self.iteraction),
																							'model': model_class.model_name,
																							'CPLC': final_metrics['Correct Prediction per Label'][0],
																							'CPLA': final_metrics['Correct Prediction per Label'][1],
																							'CPLD': final_metrics['Correct Prediction per Label'][2],
																							'EMR': final_metrics['Exact Match Ratio'],
																						  'HL': final_metrics['Hamming Loss'],
																							'metrics_multilabel': final_metrics['Multi-label Report Dict'],
																						  'metrics_singlelabel': final_metrics['Single-label Report Dict']}))
				data_pd.to_pickle(dn.PATH_PROJECT + self.experiment_name + '_it_' + str(self.iteraction) + str('_metrics.df'))

				return y_pred
		
		def predict_samples(self, model_class, x_test, y_test, ensemble=False):
				acc, f1, precision, recall = None, None, None, None
				
				if ensemble:
						x_test, y_test = self.format_input_dataset_ensemble(model_class, x_test, y_test)
				
				if self.pp_data.type_prediction_label in set([dn.TypePredictionLabel.BINARY,
																											dn.TypePredictionLabel.BINARY_CATEGORICAL]):
						y_pred, acc, f1, precision, recall = self.print_confustion_matrix_binary_classifier(model_class,
																																																x_test, y_test)
				else:
						y_pred = self.print_confusion_matrix_multiclass_classifier(model_class, x_test, y_test)
						
				return y_pred, acc, f1, precision, recall
		
		def k_fold_cross_validation(self, model_class):
				np.random.seed(dn.SEED)
				time_ini_rep = datetime.datetime.now()
				x_train, y_train, x_test, y_test, x_valid, y_valid, num_words, embedding_matrix = self.pp_data.load_data()
				self.set_period_time_end(time_ini_rep, 'Load data')
				
				x_fold_test, y_fold_test = self.load_partial_test_set(x_test, y_test)

				original_model = model_class.model
				if (model_class.use_embedding_pre_train == (dn.UseEmbedding.STATIC or dn.UseEmbedding.NON_STATIC)):
						original_model.layers[0].set_weights([embedding_matrix])

				self.save_geral_configs()
				self.save_summary_model(model_class.model)
				
				for r in range(dn.REPEAT_VALIDATION):
						print('Repeat ' + str(r+1))
						time_ini_rep = datetime.datetime.now()
						
						metrics_by_folds = []
						
						X_train = np.concatenate((x_train, x_valid))
						Y_train = np.concatenate((y_train, y_valid))
						
						folds = list(StratifiedKFold(n_splits=dn.K_FOLD, shuffle=True, random_state=dn.SEED).split(X_train, Y_train))
						
						for fold_idx, (train_idx, val_idx) in enumerate(folds):
								print("Fold " + str(fold_idx+1))
								time_ini_fold = datetime.datetime.now()
								
								X_train_fold = X_train[train_idx]
								Y_train_fold = Y_train[train_idx]
								X_valid_fold = X_train[val_idx]
								Y_valid_fold = Y_train[val_idx]
								
								with tf.device('cpu'):
										model_class.model = original_model

								try:
										model_class.model = multi_gpu_model(model_class.model, gpus=self.total_gpus, cpu_merge=False)
										print("Training using " + str(self.total_gpus) + " GPUs..")
								except:
										print("Training using single GPU or CPU..")

								model_class.model.compile(loss=model_class.loss_function, optimizer=model_class.optmizer_function,
																					metrics=[model_class.main_metric,
																									 self.metrics_test.f1_m,
																									 self.metrics_test.precision_m,
																									 self.metrics_test.recall_m])
								
								early_stopping_train = EarlyStopping(monitor='val_loss', mode='min', verbose=0,
																										 patience=model_class.patience_train)
								model_checkpoint = ModelCheckpoint(dn.PATH_PROJECT + 'fold_' + str(fold_idx+1) + '_' + self.experiment_name + '.h5',
																									 monitor='val_acc', mode='max', save_best_only=True, save_weights_only=False)
								
								history = model_class.model.fit(X_train_fold, Y_train_fold,
																								validation_data=(X_valid_fold, Y_valid_fold),
																								batch_size=model_class.batch_size,
																								epochs=model_class.epochs,
																								verbose=model_class.verbose,
																								callbacks=[early_stopping_train, model_checkpoint])

								self.save_history(history, 'fold_' + str(fold_idx+1) + '_' + self.experiment_name)

								# Analyze overfitting or underfitting about model
								_, acc, f1, precision, recall = model_class.model.evaluate(X_train_fold, Y_train_fold, model_class.verbose)
								self.get_metrics_log_by_fold('Evaluate Fold ' + str(fold_idx+1) + ' Train', acc, f1, precision, recall)

								_, acc, f1, precision, recall = model_class.model.evaluate(X_valid_fold, Y_valid_fold, model_class.verbose)
								self.get_metrics_log_by_fold('Evaluate Fold ' + str(fold_idx+1) + ' Test', acc, f1, precision, recall)

								# This step really tests the model's performance.
								# In some approaches, this step is only performed in the final step.
								y_pred = model_class.model.predict(x_fold_test, batch_size=model_class.batch_size, verbose=0)
								res = str(self.metrics_test.confusion_matrix(y_fold_test, y_pred, self.pp_data.label_set,
																														 self.pp_data.binary_class))
								acc = accuracy_score(y_fold_test, np.round(y_pred))
								f1 = f1_score(y_fold_test, np.round(y_pred))
								precision = precision_score(y_fold_test, np.round(y_pred))
								recall = recall_score(y_fold_test, np.round(y_pred))
								
								metrics_by_folds.append([acc, f1, precision, recall])
								self.get_metrics_log_by_fold('Fold ' + str(fold_idx+1), acc, f1, precision, recall)
								
								self.log.save(res)
								self.set_period_time_end(time_ini_fold, 'K-Fold '+str(fold_idx+1))
						
						# Evaluation: Weighted average of repetition after k-folds
						metrics_pds = pd.DataFrame(data=metrics_by_folds, columns=[model_class.main_metric, 'f1', 'precision', 'recall'])
						self.save_metrics(metrics_pds, 'train_media_metrics_' + self.experiment_name)
						self.get_metrics_log_by_fold('Repeat ' + str(r+1) + ' - Mean CrossValidation Test',
																				 metrics_pds.accuracy,
																				 metrics_pds.f1,
																				 metrics_pds.precision,
																				 metrics_pds.recall)
						
						self.set_period_time_end(time_ini_rep, 'Repeat ' + str(r+1))
				
				# Generates, trains, evaluates and tests final model
				time_ini = datetime.datetime.now()
				
				with tf.device('cpu'):
						model_class.model = original_model
				
				try:
						model_class.model = multi_gpu_model(model_class.model, gpus=self.total_gpus, cpu_merge=False)
						print("Training using " + str(self.total_gpus) + " GPUs..")
				except:
						print("Training using single GPU or CPU..")
				
				model_class.model.compile(loss=model_class.loss_function, optimizer=model_class.optmizer_function,
																	metrics=[model_class.main_metric, self.metrics_test.f1_m,
																					 self.metrics_test.precision_m,
																					 self.metrics_test.recall_m])
				
				early_stopping_train = EarlyStopping(monitor='val_loss', mode='min', patience=model_class.patience_train)
				model_checkpoint = ModelCheckpoint(dn.PATH_PROJECT + 'final_' + self.experiment_name + '.h5',
																					 monitor='val_acc', mode='max', save_best_only=True, save_weights_only=False)
				history = model_class.model.fit(x_train, y_train,
																				validation_data=(x_valid, y_valid),
																				batch_size=model_class.batch_size,
																				epochs=model_class.epochs,
																				verbose=model_class.verbose,
																				callbacks=[early_stopping_train, model_checkpoint])

				self.save_history(history, 'final_' + self.experiment_name)

				_, acc, f1, precision, recall = model_class.model.evaluate(x_test, y_test, verbose=model_class.verbose)
				self.get_metrics_log_by_fold('Modelo final - Evaluate Test', acc, f1, precision, recall)
				
				y_pred = model_class.model.predict(x_test, batch_size=model_class.batch_size, verbose=model_class.verbose)

				acc = accuracy_score(y_test, np.round(y_pred))
				f1 = f1_score(y_test, np.round(y_pred))
				precision = precision_score(y_test, np.round(y_pred))
				recall = recall_score(y_test, np.round(y_pred))
				self.get_metrics_log_by_fold('Final Model - Test', acc, f1, precision, recall)

				self.log.save(str(self.metrics_test.confusion_matrix(y_test, y_pred,
																														 self.pp_data.label_set, self.pp_data.binary_class)))
				
				self.set_period_time_end(time_ini, 'Generate final model')