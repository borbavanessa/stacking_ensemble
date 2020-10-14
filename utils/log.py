''' Class for recording process log data'''
import datetime as dt
import utils.definition_network as dn

class Log:
		def __init__(self, file_path="log_process", file_extension="csv"):
				self.file_path = self.__build_file_path(file_path, file_extension)

		def __build_file_path(self, file_path, file_extension):
				return dn.PATH_PROJECT + file_path + dt.datetime.now().strftime("_%y-%m-%d_%H-%M") + "." + file_extension
		
		def set_new_file(self, file_path, file_extension):
				self.file_path = self.__build_file_path(file_path, file_extension)
				
		def save(self, text):
				f = open(self.file_path, "a+")
				f.write(text+'\n')
				f.close()
