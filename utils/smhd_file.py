'''
		Class implements import of dataset data
		SMHD saved in Julia format for dataframe pandas
'''

import json
import gzip
import pandas as pd
import numpy as np
import pandas.io.json as pd_json
import datetime
import csv
from datetime import datetime as dt

import utils.definition_network as dn
from utils.log import Log
from utils.con_postgres import ConPostgres

class SmhdFile:
		def __init__(self, _file_path="", _black_list_users=True, verbose=True, label_set=dn.LABEL_SET):
				self.file_path = _file_path
				self.generate_black_list_users = _black_list_users
				self.label_set = label_set
				self.verbose = verbose
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
						con = ConPostgres('rsdd')
						if con.connection():
								pd_users = con.select(
										"SELECT DISTINCT user_id FROM users")
								users_list = pd_users.user_id.unique()

				return users_list

		def __generate_log(self, text):
				if self.verbose:
						self.log.save(text)

		def __extract_posts_list(self, posts):
				posts_list = []
		
				# include only posts with content
				for idx in range(len(posts[0])):
						if posts[0][idx]['text'] != '':
								post_text = posts[0][idx]['text'].replace('\x00', '')
								posts_list.append(post_text)
		
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
				
				# insert new column texts with only the list of posts and
				# remove the id and posts column = [[timestamp, post]]
				df_aux.insert(df_aux.columns.__len__(), 'texts', str(posts_list))
				df_aux = df_aux.drop('id', 1)
				df_aux = df_aux.drop('posts', 1)
				
				# smhd can contain users with more than one label. Example: label: ['bipolar', 'depression']
				target_class = ','.join(df_aux.label.values[0]) if df_aux.label.values[0].__len__() > 1 else df_aux.label.values[0][0]
		
				if self.sample_by_class.index.isin([target_class]).any():
						if self.sample_by_class[target_class] < self.total_class:
								df_aux.at[0, 'label'] = target_class
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
								df_aux = pd_json.json_normalize(json.loads(json_line))
								
								if df_aux.label.values.__len__() > 0:
										user_id = df_aux.id.values[0]

										if int(user_id) not in users_black_list:
												labels = ','.join(df_aux.label.values[0]) if df_aux.label.values[0].__len__() > 1 else df_aux.label.values[0][0]
												
												total_posts_user, total_terms_by_posts, posts_list = self.__extract_posts_list(df_aux.posts.values)
												
												if total_posts_user > dn.MIN_TOTAL_POSTS_USER:
														df, include_sample = self.__valid_sample_inclusion(df, df_aux, posts_list)
						
														if include_sample:
																self.__generate_log(str(user_id) + ";" +
																										str(labels) + ";" +
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
				total_lines = self.__get_total_lines_file()
				self.__generate_log('File %s, total lines: %s' % (str(self.file_path), str(total_lines)))
		
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
								self.__generate_log('Processed %s to %s' % (str(i), str(total_lines)))
		
				return df

		def __save_list_to_verify(self, list_values):
				with open('posts.csv', 'w', newline='') as file:
						wr = csv.writer(file, quoting=csv.QUOTE_ALL)
						wr.writerow(list_values)

		def __process_users_and_posts(self, con, file, datatype):
				users = []
				posts_by_user = []
				total_label_none = 0
		
				for i, json_line in enumerate(file):
						df_aux = pd_json.json_normalize(json.loads(json_line))
				
						labels = df_aux['label'].values[0]
						if labels != []:
								id = int(df_aux['id'].values[0])
								posts = df_aux.posts.values
								labels_str = ','.join(labels)
								users.append({'user_id': id, 'label': labels_str, 'dataset_type': datatype})

								total_posts = len(posts[0])
								for idx in range(total_posts):
										date_time = str(dt.fromtimestamp(posts[0][idx]['created_utc']))
										post = ''
										selftext = ''
										title = ''
										body = ''

										if 'text' in posts[0][idx].keys():
												post = posts[0][idx]['text'].replace('\x00', '')
												
										if 'selftext' in posts[0][idx].keys():
												selftext = posts[0][idx]['selftext'].replace('\x00', '')

										if 'title' in posts[0][idx].keys():
												title = posts[0][idx]['title'].replace('\x00', '')

										if 'body' in posts[0][idx].keys():
												body = posts[0][idx]['body'].replace('\x00', '')
									 
										posts_by_user.append({'user_id': id, 'post': post, 'date_post': date_time,
																					'selftext': selftext, 'title': title, 'body': body})
						else:
								total_label_none = total_label_none + 1
				
						if ((i + 1) % 1000) == 0:
								self.__generate_log('Number of users processed: %s' % (str(i + 1)))
				
						# Performs insertion in batch of 10k in 10k due to memory restrictions
						if ((i + 1) % 10000) == 0:
								self.__generate_log('Insert users in the base: %s, total label none: %s' % (str(i + 1), str(total_label_none)))
								
								con.insert_batch('''INSERT INTO users (user_id, label, dataset_type) VALUES (%(user_id)s, %(label)s, %(dataset_type)s);''', users, 1000)
								con.insert_batch('''INSERT INTO posts (user_id, post, date_post, selftext, title, body) VALUES (%(user_id)s, %(post)s, %(date_post)s, %(selftext)s, %(title)s, %(body)s);''', posts_by_user, 1000)
								
								users = []
								posts_by_user = []
								total_label_none = 0
		
				self.__generate_log('Insert last users in the base: %s, total label none: %s' % (str(len(users)), str(total_label_none)))
				con.insert_batch(
						'''INSERT INTO users (user_id, label, dataset_type) VALUES (%(user_id)s, %(label)s, %(dataset_type)s);''',
						users, 1000)
				con.insert_batch(
						'''INSERT INTO posts (user_id, post, date_post, selftext, title, body) VALUES (%(user_id)s, %(post)s, %(date_post)s), %(selftext)s, %(title)s, %(body)s;''',
						posts_by_user, 1000)

		def __insert_by_batch(self, con, datatype, file):
				self.__process_users_and_posts(con, file, datatype)

		def __file_to_postgres(self, con, datatype):
				total_lines = self.__get_total_lines_file()
				self.__generate_log('Arquivo %s, total linhas: %s' % (str(self.file_path), str(total_lines)))
		
				file = gzip.open(self.file_path, 'rt')
		
				time_ini = datetime.datetime.now()
				self.__insert_by_batch(con, datatype, file)
		
				time_end = datetime.datetime.now()
				self.__generate_log('Insert batch: Ini: %s\tEnd: %s\tTotal: %s' % (time_ini.strftime("%Y-%m-%d %H:%M:%S"),
																																					 time_end.strftime("%Y-%m-%d %H:%M:%S"),
																																					 (time_end - time_ini)))

		def read_data_from_pandas(self):
				return pd.read_pickle(self.file_path)

		def read_data_from_csv(self):
				return pd.read_csv(self.file_path, sep='|', encoding='utf-8')

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
		
		def convert_file_to_postgres(self, con, datatype):
				if con.connection():
						self.__file_to_postgres(con, datatype)
