"""
    Generic Class for models ML
"""
import utils.definition_network as dn

from keras.models import model_from_json
from keras.callbacks import TensorBoard

class ModelClass:
		def __init__(self, _verbose=0):
				self.verbose = _verbose
				self.model = None
				self.model_name = 'model'
				
				# Parameters
				self.loss_function = 'binary_crossentropy'
				self.activation_output_fn = 'sigmoid'
				self.optmizer_function = 'adam'
				self.main_metric = 'accuracy'
				self.epochs = 3  # 6
				self.batch_size = 3  # 64
				self.patience_train = 10
				self.use_embedding_pre_train = dn.UseEmbedding.RAND
				self.embed_trainable = (self.use_embedding_pre_train == (dn.UseEmbedding.RAND or dn.UseEmbedding.NON_STATIC))
				self.callback_tensor_board = [TensorBoard(log_dir='experiments/lstm', histogram_freq=1, embeddings_freq=1, )]

		def get_config(self):
				return "\n=== TRAIN DATA ===\n" + \
							 "batch_size: " + str(self.batch_size) + "\n" + \
							 "epochs: " + str(self.epochs) + "\n" + \
							 "\n=== MODEL DATA ===\n" + \
							 "loss_function: " + self.loss_function + "\n" + \
							 "optmizer_function: " + self.optmizer_function
		
		def save_model(self):
				# serialize model to JSON
				model_json = self.model.to_json()
				with open(self.model_name + ".json", "w") as json_file:
						json_file.write(model_json)
				
				# serialize weights to HDF5
				self.model.save_weights(self.model_name + ".h5")
		
		def load_model(self):
				# load json and create model
				json_file = open(self.model_name + ".json", 'r')
				loaded_model_json = json_file.read()
				json_file.close()
				loaded_model = model_from_json(loaded_model_json)
				
				# load weights into new model
				loaded_model.load_weights(self.model_name + ".h5")
				self.model = loaded_model
