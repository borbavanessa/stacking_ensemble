# Load libs standard python and custom

import utils.definition_network as dn
import sys
from network_model.stacked_ensemble import StackedEnsemble

def set_params_stacked_ensemble(dataset_train_path, dataset_test_path, use_submodel, b_size, h_layer, epochs, neurons):
		epoch = epochs
		batch_size = b_size
		neurons_by_submodel = neurons
		hidden_layer = h_layer

		metric = 'accuracy'
		loss_fn = 'binary_crossentropy'
		activation_output_fn = 'sigmoid'
		optimizer_fn = 'adam'
		activation_hidden_fn = 'tanh'
		kernel_initializer = 'glorot_uniform'
		use_bias = True
		bias_initializer = 'zeros'
		kernel_regularizer = None
		bias_regularizer = None
		activity_regularizer = None
		kernel_constraint = None
		bias_constraint = None
		path_submodels = dn.PATH_PROJECT + "weak_classifiers/"
		type_submodels = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL

		hidden_layers_set = []
		for idx in range(hidden_layer):
						hidden_layers_set.append(
										dict({'units': neurons_by_submodel,
																'activation': activation_hidden_fn,
																'use_bias': use_bias,
																'kernel_initializer': kernel_initializer,
																'bias_initializer': bias_initializer,
																'kernel_regularizer': kernel_regularizer,
																'bias_regularizer': bias_regularizer,
																'activity_regularizer': activity_regularizer,
																'kernel_constraint': kernel_constraint,
																'bias_constraint': bias_constraint}))

		set_network = dict({'epochs': epoch,
																						'batch_size': batch_size,
																						'patient_train': int(
																										epoch / 2),
																						'activation_output_fn': activation_output_fn,
																						'loss_function': loss_fn,
																						'optmizer_function': optimizer_fn,
																						'main_metric': metric,
																						'dataset_train_path': dataset_train_path,
																						'dataset_test_path': dataset_test_path,
																						'path_submodels': path_submodels,
																						'type_submodels': type_submodels,
																						'submodels': use_submodel,
																						'hidden_layers': hidden_layers_set
																						})

		name_test = 'E_' + str(epoch) + '_BS_' + str(batch_size) + \
														'_US_' + str(len(use_submodel)) + '_N_' + str(neurons_by_submodel) + \
														'_HL_' + str(hidden_layer) + '_M_' + str(metric)[0:2] + \
														'_AO_' + str(bias_constraint)[0:2] + \
														'_LF_' + str(loss_fn)[0:2] + '_OP_' + str(optimizer_fn) + \
														'_AH_' + str(activation_hidden_fn)[0:2] + '_KI_' + str(kernel_initializer)[0:2] + \
														'_UB_' + str(use_bias)[0] + '_BI_' + str(bias_initializer)[0:2] + \
														'_KR_' + str(kernel_regularizer) + '_BR_' + str(bias_regularizer) + \
														'_AR_' + str(activity_regularizer) + '_KC_' + str(kernel_constraint)[0:2] + \
														'_BC_' + str(bias_constraint)[0:2]

		return name_test, set_network

def load_stacked_ensemble(name_test, set_network):
		print("Experiment: " + name_test)
		ensemble_stk = StackedEnsemble(name_test, 1, '')

		ensemble_stk.list_report_metrics = []
		ensemble_stk.ensemble_stacked_conf = set_network
		ensemble_stk.k_fold = 5
		ensemble_stk.labels_set = ['control', 'anxiety', 'depression']
		ensemble_stk.labels_ensemble = ['control', 'anxiety', 'depression']

		ensemble_stk.type_predict_label = dn.TypePredictionLabel.MULTI_LABEL_CATEGORICAL
		ensemble_stk.metrics_based_sample = False

		ensemble_stk.set_network_params_ensemble_stack()
		ensemble_stk.model_training()

def generate_test(option):
		dataset_train_path = 'dataset/anx_dep_multilabel/SMHD_multi_label_test_train_2112.df'
		dataset_test_path = 'dataset/samples_for_interpretation/SMHD_multi_label_test_test_352_user_id.df'

		neurons_by_layer = [8, 12, 16]
		h_layers = [3, 4]
		b_sizes = [4, 8, 16]
		epochs = [16, 32, 64]

		for h_layer in h_layers:
				for neurons in neurons_by_layer:
						for b_size in b_sizes:
								for epoch in epochs:
										# 1 - 4: LSTM best results submodels
										if option == '1':
												use_submodel = dict({'CA': [1, 2, 3], 'CD': [1, 2, 3], 'CAD': [1, 2, 3]})
												name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path,
																																						 use_submodel, b_size, h_layer, epoch,
																																						 neurons)
												load_stacked_ensemble('t1_'+name_test, set_network)
										elif option == '2':
												use_submodel = dict({'CA': [1, 2, 3], 'CD': [1, 2, 3]})
												name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path,
																																						 use_submodel, b_size, h_layer, epoch,
																																						 neurons)
												load_stacked_ensemble('t2_' + name_test, set_network)
										elif option == '3':
												use_submodel = dict({'CA': [1, 2], 'CD': [1, 2], 'CAD': [1, 2]})
												name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path,
																																						 use_submodel, b_size, h_layer, epoch,
																																						 neurons)
												load_stacked_ensemble('t3_' + name_test, set_network)
										elif option == '4':
												use_submodel = dict({'CA': [1, 2], 'CD': [1, 2]})
												name_test, set_network = set_params_stacked_ensemble(dataset_train_path, dataset_test_path,
																																						 use_submodel, b_size, h_layer, epoch,
																																						 neurons)
												load_stacked_ensemble('t4_' + name_test, set_network)

if __name__ == '__main__':
		
		arg = sys.argv[1]
		generate_test(arg)
