import keras.backend.tensorflow_backend as K
K.set_session

import shap
import numpy as np
import colored
import re

class ShapAnalysePlots:
		
		def __init__(self):
				self.tokenizers_dict = dict()
				self.labels_classifier = []
				self.explainers_dict = dict()
				self.shap_values_dict = dict()
		
		# Auxiliar Functions
		def __clear_dataset(self, data_df):
				data_df['texts'] = data_df.apply(lambda row: row.texts.lower().replace('\xad', ''), axis=1)
				data_df['texts'] = data_df.apply(lambda row: row.texts.lower().replace('\u00ad', ''), axis=1)
				data_df['texts'] = data_df.apply(lambda row: row.texts.lower().replace('\N{SOFT HYPHEN}', ''), axis=1)
				data_df['texts'] = data_df.apply(lambda row: re.sub('[^A-Za-z0-9 \/]+', '', row.texts.lower()), axis=1)
				
				return data_df
		
		def explainer_values_print(self, key_dict, word_lookup, total_words, index_sample=0):
				print('\nPredict submodel ', key_dict)
				print('Explainer expected_value [control, anxiety, depression]: %s' %
							(self.explainers_dict[key_dict].expected_value))
				
				for index_class, class_name in enumerate(self.labels_classifier):
						sig_pos, sig_neg, pos_words, neg_words = self.get_signal_pos_neg_words(key_dict, word_lookup, total_words,
																																									 index_class, index_sample)
						print('\nShap Values to class %s: ' % (class_name))
						print('Positive Words:\n', pos_words)
						print('Negative Words:\n', neg_words)
			
		# Reference https://sararobinson.dev/2019/04/23/interpret-bag-of-words-models-shap.html
		def get_signal_pos_neg_words(self, key_dict, word_lookup, total_words, index_class, index_sample):
				attributions = self.shap_values_dict[key_dict][index_class][index_sample]
				top_signal_pos_words = np.argpartition(attributions, -total_words)[-total_words:]
				pos_words = []
				for word_idx in top_signal_pos_words:
						signal_wd = word_lookup[word_idx]
						pos_words.append(signal_wd)
				
				top_signal_neg_words = np.argpartition(attributions, total_words)[:total_words]
				neg_words = []
				for word_idx in top_signal_neg_words:
						signal_wd = word_lookup[word_idx]
						neg_words.append(signal_wd)
				
				return top_signal_pos_words, top_signal_neg_words, pos_words, neg_words
		
		def color_print_text(self, data_df):
				sig_pos, sig_neg, pos, neg = self.get_signal_pos_neg_words()
				for idx, row in data_df.iterrows():
						posts_txt = eval(row.texts)
						text = ' '.join(posts_txt)
						
						q_arr = []
						q_filtered = filter(None, re.split("[, ().?!]+", text.lower()))
						
						for i in q_filtered:
								q_arr.append(i)
						
						color_str = []
						
						for idx, word in enumerate(q_arr):
								if word in pos:
										color_str.append(colored.bg('blue') + colored.attr(1) + word)
								elif word in neg:
										color_str.append(colored.bg('red') + colored.attr(1) + word)
								else:
										color_str.append(colored.fg('black') + colored.bg('white') + colored.attr(0) + word)
						
						print('Class: %s' % (row.label))
						print(' '.join(color_str))
		
		def prepare_word_lookup(self, key_dict, encoded_x_sample):
				word_lookup = [self.tokenizers_dict[key_dict].index_word[i] if i > 0 else '[PAD]' for i in encoded_x_sample]

				return word_lookup
		
		def generate_word_lookup(self, key_dict, x_test):
				word_lookup = []
				for sample in x_test:
						word_lookup.append(self.prepare_word_lookup(key_dict, sample))
				
				return word_lookup
		
		# Get index features with greater interaction with main term (main_feature). Return inds to samples in
		# features_enconding by class defined n index_class. Important: Total samples would equal between
		# features_enconding with samples informed to generate shap_values.
		def select_features_greater_interactions(self, key_dict, word_lookup, features_enconding, main_feature, index_class):
				# This function execute an algorithm to ordering of the features, second criterion "more interaction with
				# feature (word) of interest. This feature would be a term (pos/neg word) existent in sample features
				# (features_enconding)
				# approximate interactions can be change other function to calculate true approximate interactions
				index_interactions = []
				try:
						index_interactions = shap.approximate_interactions(index=main_feature,
																															 shap_values=self.shap_values_dict[key_dict][index_class],
																															 X=features_enconding,
																															 feature_names=word_lookup)
				except ValueError:
						print("The graph could not be generated. Mistake: ", ValueError)

				return index_interactions

		# Available summary to a set samples
		def feature_importance_plot(self, key_dict, word_lookup, features_encoding, max_features=20, class_inds=None):
				# plot_type : "dot" (default for single output), "bar" (default for multi-output), "violin" or "compact_dot".
				# What type of summary plot to produce. Note that "compact_dot" is only used for SHAP interaction values.
				# Observation: this plot auto identify shap values multi and single output
				# Options: class_inds=[None,"original"], this method determine the color of according to interaction's intensity
				return shap.summary_plot(shap_values=self.shap_values_dict[key_dict],
																 features=features_encoding,
																 max_display=max_features,
																 feature_names=word_lookup,
																 # class_inds=class_inds,
																 plot_type="bar",
																 class_names=self.labels_classifier)

		# Available the impact in one sample, it is considering one feature (term) and relationship with others features
		def dependence_plot(self, key_dict, word_lookup, features_encoding, index_class, index_dependence="rank(0)"):
				try:
						return shap.dependence_plot(ind=index_dependence,
																				shap_values=self.shap_values_dict[key_dict][index_class],
																				features=features_encoding,
																				feature_names=word_lookup,
																				title=self.labels_classifier[index_class])
				except ValueError:
						print("The graph could not be generated. Mistake: ", ValueError)

		# Available the impact in one sample, it is considering one feature (term) and relationship with others features
		def summary_features_plot(self, key_dict, word_lookup, features_encoding, index_class):
				# If matplotlib=True, values figsize and text_rotation are considers

				shap.initjs()
				return shap.force_plot(base_value=self.explainers_dict[key_dict].expected_value[index_class],
															 shap_values=self.shap_values_dict[key_dict][index_class],
															 features=features_encoding,
															 feature_names=word_lookup,
															 out_names=self.labels_classifier,
															 matplotlib=False, figsize=(23,13), text_rotation=90)
