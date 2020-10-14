'''
		Class implements import of RSDD dataset saved in Julia format for pandas dataframe
'''

import json
import gzip
import pandas as pd
import numpy as np
import pandas.io.json as pd_json
import datetime
from datetime import datetime as dt

import utils.definition_network as dn
from utils.log import Log
from utils.con_postgres import ConPostgres

# choose a random element from a list
from random import seed
from random import shuffle

class RsddFile:
		def __init__(self, _file_path="",_black_list_users=True, verbose=True, label_set=dn.LABEL_SET):
				self.file_path = _file_path
				self.generate_black_list_users = _black_list_users
				self.verbose = verbose
				self.label_set = label_set
				if self.verbose:
						self.log = Log()

				# Fix initialize
				self.total_class = 0
				self.sample_by_class = 0

		def set_new_file_log(self, _file_path, _file_extension="csv"):
				self.log.set_new_file(_file_path, _file_extension)
				
		def __get_users_other_dataset(self):
				users_list = []
				if self.generate_black_list_users:
						con = ConPostgres('smhd')
						if con.connection():
								pd_users = con.select("SELECT DISTINCT user_id FROM users")
								users_list = pd_users.user_id.unique()
				
				return users_list

		def __generate_log(self, text):
				if self.verbose:
						self.log.save(text)

		def __extract_posts_list(self, posts):
				posts_list = []
				
				# includes only posts with content
				for timestamp, post in posts[0]:
						if post != '':
								posts_list.append(post)
		
				total_posts = len(posts_list)
				total_terms_by_posts = [len(post.split(' ')) for post in posts_list]

				return total_posts, total_terms_by_posts, posts_list

		def __include_sample(self, df, df_aux):
				if df.empty:
						return df_aux
				else:
						return df.append(df_aux, ignore_index=True)

		def __valid_sample_inclusion(self, df, df_aux, posts_list):
				include_sample = False
		
				# insert new column texts with only the list of posts and remove the
				# column id and posts = [[timestamp, post]]
				df_aux.insert(df_aux.columns.__len__(), 'texts', str(posts_list))
				df_aux = df_aux.drop('id', 1)
				df_aux = df_aux.drop('posts', 1)
		
				target_class = df_aux.label.values[0]
		
				if self.sample_by_class.index.isin([target_class]).any():
						if self.sample_by_class[target_class] < self.total_class:
								df = self.__include_sample(df, df_aux)
								self.sample_by_class[target_class] = self.sample_by_class[target_class] + 1
								self.total_register = self.total_register - 1
								include_sample = True
		
				return df, include_sample

		def __load_json_data(self, total_register):
				users_black_list = self.__get_users_other_dataset()
				self.total_register = total_register
				self.total_class = total_register // 2
				self.sample_by_class = pd.Series(np.zeros(len(self.label_set)), index=self.label_set)
				df = pd.DataFrame()

				file = gzip.open(self.file_path, 'rt')
				for i, json_line in enumerate(file):
						if self.total_register > 0:
								df_aux = pd_json.json_normalize(json.loads(json_line)[0])
								user_id = df_aux.id.values[0]
								
								if int(user_id) not in users_black_list:
										total_posts_user, total_terms_by_posts, posts_list = self.__extract_posts_list(df_aux.posts.values)
										df, include_sample = self.__valid_sample_inclusion(df, df_aux, posts_list)
		
										if include_sample:
												self.__generate_log(str(user_id) + ";" +
																						str(df_aux.label.values[0]) + ";" +
																						str(total_posts_user) + ";" +
																						str(total_terms_by_posts))
						else:
								break
				return df

		def __save_dataset_pandas_format(self, dataset, file_name):
				dataset.to_pickle(file_name + str(".df"))

		def __save_dataset_csv_format(self, dataset, file_name):
				dataset.to_csv(file_name+str(".csv"), sep='|', encoding='utf-8', index=False, header=True, mode='a')
				
		def __get_total_lines_file(self):
				file = gzip.open(self.file_path, 'rt')
				total_lines = sum(1 for i, json_line in enumerate(file))
				file.close()
				
				return total_lines

		def __file_to_pandas(self):
				total_lines = 0 # total_lines = self.__get_total_lines_file()
				self.__generate_log('Arquivo %s, total linhas: %s' % (str(self.file_path), str(total_lines)))
		
				df = pd.DataFrame()
				file = gzip.open(self.file_path, 'rt')
		
				for i, json_line in enumerate(file):
						df_aux = pd_json.json_normalize(json.loads(json_line)[0])
				
						id = int(df_aux['id'].values[0])
						label = df_aux['label'].values[0]
						posts = df_aux.posts.values
				
						user_posts = [[id, label, timestamp, post] for timestamp, post in posts[0]]
						if df.empty:
								df = pd.DataFrame(user_posts, columns=['id_user', 'label', 'date', 'post'])
						else:
								df = df.append(pd.DataFrame(user_posts, columns=['id_user', 'label', 'date', 'post']),
															 ignore_index=True)
				
						if (i % 1000) == 0:
								self.__generate_log('Processou %s de %s' % (str(i), str(total_lines)))
		
				return df

		def __insert_one_by_one(self, con, datatype, file):
				for i, json_line in enumerate(file):
						df_aux = pd_json.json_normalize(json.loads(json_line)[0])
				
						label = df_aux['label'].values[0]
						
						if label != None:
								user_id = int(df_aux['id'].values[0])
								posts = df_aux.posts.values
						
								sql_command_txt = "INSERT INTO users (user_id, label, dataset_type) " \
																	"VALUES ({},'{}','{}') RETURNING id;".format(id, str(label), str(datatype))
								
								sql_command = "INSERT INTO users (user_id, label, dataset_type) " \
															"VALUES (%s, %s, %s) RETURNING id;"
								con.insert(sql_command_txt, sql_command, (user_id, label, datatype))
						
								for timestamp, post in posts[0]:
										date_time = dt.fromtimestamp(timestamp)
										
										sql_command_txt = "INSERT INTO posts (user_id, post, date_post) " \
																			"VALUES ({},'{}','{}');".format(user_id, str(post), str(date_time))
										
										sql_command = "INSERT INTO posts (user_id, post, date_post) VALUES (%s, %s, %s);"
										con.insert(sql_command_txt, sql_command, (user_id, post, date_time))
				
						if (i % 1000) == 0:
								self.__generate_log('Number of users processed: ' + str(i))

		def __process_users_and_posts(self, con, file, datatype):
				users = []
				posts_by_user = []
				total_label_none = 0
		
				for i, json_line in enumerate(file):
						df_aux = pd_json.json_normalize(json.loads(json_line)[0])
				
						label = df_aux['label'].values[0]
						if label != None:
								id = int(df_aux['id'].values[0])
								posts = df_aux.posts.values
						
								users.append({'user_id': id, 'label': label, 'dataset_type': datatype})
								for timestamp, post in posts[0]:
										date_time = dt.fromtimestamp(timestamp)
										posts_by_user.append({'user_id': id, 'post': post, 'date_post': date_time})
						else:
								total_label_none = total_label_none + 1
				
						if ((i + 1) % 1000) == 0:
								self.__generate_log('Number of users processed: %s' % (str(i + 1)))
				
						# Realiza aqui a inserção em batch de 10k em 10k por restrições de memória
						if ((i + 1) % 10000) == 0:
								self.__generate_log('Insert users in database: %s, total label none: %s' % (str(i + 1), str(total_label_none)))
								
								con.insert_batch('''INSERT INTO users (user_id, label, dataset_type) VALUES (%(user_id)s, %(label)s, %(dataset_type)s);''', users, 1000)
								con.insert_batch('''INSERT INTO posts (user_id, post, date_post) VALUES (%(user_id)s, %(post)s, %(date_post)s);''', posts_by_user, 1000)
								
								users = []
								posts_by_user = []
								total_label_none = 0
		
				self.__generate_log('Insert last user in database: %s, total label none: %s' % (str(len(users)), str(total_label_none)))
				con.insert_batch('''INSERT INTO users (user_id, label, dataset_type) VALUES (%(user_id)s, %(label)s, %(dataset_type)s);''', users, 1000)
				con.insert_batch('''INSERT INTO posts (user_id, post, date_post) VALUES (%(user_id)s, %(post)s, %(date_post)s);''', posts_by_user, 1000)

		def __insert_by_batch(self, con, datatype, file):
				self.__process_users_and_posts(con, file, datatype)

		def __file_to_postgres(self, con, datatype, insert_batch):
				total_lines = self.__get_total_lines_file()
				self.__generate_log('File %s, total lines: %s' % (str(self.file_path), str(total_lines)))
		
				file = gzip.open(self.file_path, 'rt')
		
				time_ini = datetime.datetime.now()
				if insert_batch:
						self.__insert_by_batch(con, datatype, file)
				else:
						self.__insert_one_by_one(con, datatype, file)
		
				time_end = datetime.datetime.now()
				self.__generate_log('Insert batch: %s - Ini: %s\tEnd: %s\tTotal: %s' % (insert_batch,
																																								time_ini.strftime("%Y-%m-%d %H:%M:%S"),
																																								time_end.strftime("%Y-%m-%d %H:%M:%S"),
																																								(time_end - time_ini)))

		def read_data_from_pandas(self):
				return pd.read_pickle(self.file_path)

		def read_data_from_csv(self, _separator='|'):
				return pd.read_csv(self.file_path, sep=_separator, encoding='utf-8')

		def generate_data_to_csv(self, total_register, file_name):
				self.__generate_log("user_id;label;total_posts;total_terms_by_post")
				
				df = self.__load_json_data(total_register)
				self.__save_dataset_csv_format(self, df, file_name)

		def generate_data_to_pandas(self, total_register, file_name):
				self.__generate_log("user_id;label;total_posts;total_terms_by_post")

				df = self.__load_json_data(total_register)
				self.__save_dataset_pandas_format(df, file_name)

		def generate_data_to_csv_pandas(self, total_register, file_name):
				self.__generate_log("user_id;label;total_posts;total_terms_by_post")

				df = self.__load_json_data(total_register)
				self.__save_dataset_csv_format(df, file_name)
				self.__save_dataset_pandas_format(df, file_name)

		def convert_file_to_pandas(self, dataset_name):
				df = self.__file_to_pandas()
				df.to_pickle(dataset_name)
		
		def convert_file_to_postgres(self, con, datatype, insert_batch):
				if con.connection() == True:
						self.__file_to_postgres(con, datatype, insert_batch)
