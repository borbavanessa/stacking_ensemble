import numpy as np
import pandas as pd
import utils.definition_network as dn
import sklearn
import pandas.io.json as pd_json

# import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss

class Metrics:
		def __init__(self, _normalize=False):
				self.normalize = _normalize
		
		#
		# Metric calculation functions compatible with the keras format. Used for the training step of the algorithm
		#
		def recall_m(self, y, y_hat):
				# possible_positives = all that are true
				true_positives = K.sum(K.round(K.clip(y * y_hat, 0, 1)))
				possible_positives = K.sum(K.round(K.clip(y, 0, 1)))
				recall = true_positives / (possible_positives + K.epsilon())
				return recall

		def precision_m(self, y, y_hat):
				true_positives = K.sum(K.round(K.clip(y * y_hat, 0, 1)))
				predicted_positives = K.sum(K.round(K.clip(y_hat, 0, 1)))
				precision = true_positives / (predicted_positives + K.epsilon())
				return precision
		
		def f1_m(self, y, y_hat):
				precision = self.precision_m(y, y_hat)
				recall = self.recall_m(y, y_hat)
				return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
		
		def confusion_matrix_binary_classifier(self, y, y_hat, labels, binary_function=dn.TypePredictionLabel.BINARY):
				# format to use with binary_crossentropy for single label single class
				if binary_function == dn.TypePredictionLabel.BINARY:
						mat_confusion = confusion_matrix(y, np.round(y_hat))
				else:
						mat_confusion = confusion_matrix(y.argmax(axis=1), np.round(y_hat).argmax(axis=1))
				
				return "\nMatriz Confusão:\n\t\t\t\t"+ str(labels[0]) + "(pred)\t"+ str(labels[1]) + "(pred)\n"+\
								str(labels[0]) + "(true)\t\t\t"+ str(mat_confusion[0][0]) + "\t\t" + str(mat_confusion[0][1])+ "\n"+\
								str(labels[1]) + "(true)\t\t"+ str(mat_confusion[1][0]) + "\t\t" + str(mat_confusion[1][1])
		
		#Function metris for multi-class classifier
		def custom_print_metrics(self, labels, p, r, f, s, report_dict):
				# print format dict metrics
				cm_txt = ''
				for label in report_dict:
						cm_txt += str(label) + '\n' + str(report_dict[label]['confusion_matrix']) + '\n'
				
				rows = zip(labels, p, r, f, s)
				headers = ["precision", "recall", "f1-score", "support"]
				head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
				report = head_fmt.format('', *headers, width=20)
				report += '\n\n'
				row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
				for row in rows:
						report += row_fmt.format(*row, width=20, digits=2)
				report += '\n'
				
				report = cm_txt + '\n' + report
				
				return report
		
		def calc_confusion_matrix_by_label(self, y, y_hat):
				report_dict = dict()
				n_samples = y.__len__()
				
				labels = ['control', 'anxiety', 'depression', 'anxiety,depression']
				p = []
				r = []
				f = []
				s = []
		
				for label in labels:
						if label == 'control':
								y_new = [( (y[i][0] == 1) and (y[i][1] == 0) and (y[i][2] == 0) ).astype('int') for i in range(n_samples)]
								y_hat_new = [( (y_hat[i][0] == 1) and (y_hat[i][1] == 0) and (y_hat[i][2] == 0) ).astype('int') for i in range(n_samples)]
						elif label == 'anxiety':
								y_new = [( (y[i][0] == 0) and (y[i][1] == 1) and (y[i][2] == 0) ).astype('int') for i in range(n_samples)]
								y_hat_new = [( (y_hat[i][0] == 0) and (y_hat[i][1] == 1) and (y_hat[i][2] == 0) ).astype('int') for i in range(n_samples)]
						elif label == 'depression':
								y_new = [( (y[i][0] == 0) and (y[i][1] == 0) and (y[i][2] == 1) ).astype('int') for i in range(n_samples)]
								y_hat_new = [( (y_hat[i][0] == 0) and (y_hat[i][1] == 0) and (y_hat[i][2] == 1) ).astype('int') for i in range(n_samples)]
						else: #'comorbidity'
								y_new = [( (y[i][0] == 0) and (y[i][1] == 1) and (y[i][2] == 1) ).astype('int') for i in range(n_samples)]
								y_hat_new = [( (y_hat[i][0] == 0) and (y_hat[i][1] == 1) and (y_hat[i][2] == 1) ).astype('int') for i in range(n_samples)]
				
						p.append(sklearn.metrics.precision_score(y_new, y_hat_new))
						r.append(sklearn.metrics.recall_score(y_new, y_hat_new))
						f.append(sklearn.metrics.f1_score(y_new, y_hat_new))
						s.append(np.sum(y_new))
		
						report_dict.update({label: {'confusion_matrix': confusion_matrix(y_new, y_hat_new),
																				'precision': p[-1],
																				'recall': r[-1],
																				'f1-score': f[-1],
																				'support': s[-1]}})
				
				report = self.custom_print_metrics(labels, p, r, f, s, report_dict)
				return report, report_dict
		
		def exact_match_ratio(self, y, y_hat):
				n_samples = y.__len__()
				mr = (1 / n_samples) * np.sum([(((y[i][0] == y_hat[i][0]) and
																				 (y[i][1] == y_hat[i][1]) and
																				 (y[i][2] == y_hat[i][2]))).astype('float32') for i in range(n_samples)],
																			axis=0)
				return mr
		
		def correct_prediction_by_label(self, y, y_hat):
				n_samples = y.__len__()
				cpl = (1 / n_samples) * np.sum([(y[i] == y_hat[i]).astype('float32') for i in range(n_samples)], axis=0)
				return cpl

		def calc_metrics_multilabel(self, y, y_hat, labels, type_prediction_label, sample_wise=False):
				if type_prediction_label in [dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL, dn.TypePredictionLabel.SINGLE_LABEL_CATEGORICAL]:
						y_hat = (y_hat > 0.5).astype('float32')
						y = (y > 0.5).astype('float32')
				else: # self.pp_data.type_prediction_label== 'dn.TypePredictionLabel.SINGLE_LABEL_CATEGORICAL'
						y_hat = np.argmax(y_hat, axis=1)
						y = np.argmax(y, axis=1)

				label_index = [index for index in range(len(labels))]
				rep_ml_dict = classification_report(y, y_hat, labels=label_index, target_names=labels, zero_division=0, output_dict=True)
				rep_sl, rep_sl_dict = self.calc_confusion_matrix_by_label(y, y_hat)

				final_metrics = dict()
				final_metrics.update({'Exact Match Ratio': self.exact_match_ratio(y, y_hat),
															'Correct Prediction per Label': self.correct_prediction_by_label(y, y_hat),
															'Hamming Loss': hamming_loss(y, y_hat),
															'Multi-label Confusion Matrix': multilabel_confusion_matrix(y, y_hat, samplewise=sample_wise),
															'Multi-label Report': classification_report(y, y_hat, labels=label_index, target_names=labels, zero_division=0),
															'Single-label Report': rep_sl,
															'Multi-label Report Dict': rep_ml_dict,
															'Single-label Report Dict': rep_sl_dict
															})

				return final_metrics
		
		def save_predict_results(self, file_name, type_prediction_label, y_pred, y_test):
				if type_prediction_label in [dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL, dn.TypePredictionLabel.SINGLE_LABEL_CATEGORICAL]:
						predict_values = []
						for index in range(len(y_test)):
								predict_values.append([y_test[index][0], y_test[index][1], y_test[index][2],
																			 y_pred[index][0], y_pred[index][1], y_pred[index][2]])
						
						pd_predict = pd.DataFrame(predict_values, columns=['y_control', 'y_anxiety', 'y_depression',
																															 'yhat_control', 'yhat_anxiety', 'yhat_depression'])

				elif type_prediction_label == dn.TypePredictionLabel.BINARY_CATEGORICAL:
						predict_values = []
						for index in range(len(y_test)):
								predict_values.append([y_test[index][0], y_test[index][1],
																			 y_pred[index][0], y_pred[index][1]])

						pd_predict = pd.DataFrame(predict_values,
																			columns=['y_control', 'y_disorder', 'yhat_control', 'yhat_disorder'])

				else: #type_prediction_label == dn.TypePredictionLabel.BINARY
						labels = np.array([y_test, [res[0] for res in y_pred]])
						pd_predict = pd.DataFrame(labels.transpose(),	columns=['y_test', 'y_pred'])

				pd_predict.to_csv(dn.PATH_PROJECT + file_name + '.csv', index=False, sep=';')

		def computes_metrics_by_iteration(self, path_test, name_test, test_ids):
				for test_id in test_ids:
						print('TEST %s metrics by iteraction' % str(test_id))
						metrics_df = pd.read_pickle(dn.PATH_PROJECT + path_test + 'test_' + str(test_id) + '/' +
																				name_test + str(test_id) + '_metrics.df')

						metrics = metrics_df.columns[3:len(metrics_df.columns)]
						iteractions = metrics_df.iteraction.unique()
						models = metrics_df.model.unique()

						list_report_metrics = []
						for iteraction in iteractions:
								for model in models:
										results = dict()
										for metric in metrics:
												rdf = metrics_df[(metrics_df.iteraction == iteraction) & (metrics_df.model == model)] \
														.agg({metric: ['min', 'max', 'mean']}).T
												results.update({metric: {'min': str(rdf['min'][0]),
																								 'max': str(rdf['max'][0]),
																								 'mean': str(rdf['mean'][0])}})

										list_report_metrics.append(dict({'test': 'test_' + str(test_id),
																										 'iteraction': str(iteraction),
																										 'model': model,
																										 'metric': results}))
						data_pd = pd_json.json_normalize(list_report_metrics)
						data_pd.to_csv(dn.PATH_PROJECT + path_test + 'test_' + str(test_id) + '/' + \
													 name_test + str(test_id) + '_metrics_by_iteration.csv')

		def computes_metrics_by_stage(self, path_test, name_test, stage, test_ids):
				for test_id in test_ids:
						print('TEST %s - stage %s' % (str(test_id), stage))
						metrics_df = pd.read_pickle(dn.PATH_PROJECT + path_test + 'test_' + str(test_id) + '/' +
																				name_test + str(test_id) + '_metrics.df')

						metrics = metrics_df.columns[3:len(metrics_df.columns)]
						models = metrics_df.model.unique()

						list_report_metrics = []
						for model in models:
								results = dict()
								for metric in metrics:
										rdf = metrics_df[(metrics_df['iteraction'].str.contains(stage)) & \
																		 (metrics_df.model == model)].agg({metric: ['min', 'max', 'mean']}).T
										results.update({metric: {'min': str(rdf['min'][0]),
																						 'max': str(rdf['max'][0]),
																						 'mean': str(rdf['mean'][0])}})

										list_report_metrics.append(dict({'test': 'test_' + str(test_id),
																										 'iteraction': stage,
																										 'model': model,
																										 'metric': results}))

						data_pd = pd_json.json_normalize(list_report_metrics)
						data_pd.to_csv(dn.PATH_PROJECT + path_test + 'test_' + str(test_id) + '/' + \
													 name_test + str(test_id) + '_' + stage + '.csv')

