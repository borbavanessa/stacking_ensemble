
import psycopg2
import psycopg2.extras
import pandas as pd

class ConPostgres:
		def __init__(self, _dataset='postgres', _host='127.0.0.1', _user='postgres', _passwd='postgres'):
				self.pg_host = _host
				self.pg_database = _dataset
				self.pg_user = _user
				self.pg_password = _passwd
				self.conn = None
				self.conn_cursor = None

		def connection(self):
				try:
						self.conn = psycopg2.connect(user=self.pg_user,
																				 password=self.pg_password,
																				 host=self.pg_host,
																				 port='5432',
																				 database=self.pg_database)
						self.conn_cursor = self.conn.cursor()
						print("Connected!")
						return True
				except(Exception, psycopg2.Error) as error:
						if (self.conn):
								print("Failed to try to connect dataset "+self.pg_database)
				
				return False
				
		def insert(self, sql_command_txt, sql_command, values):
				
				if (self.conn == None):
						print("Start a connection before adding records!")
						return
				
				try:
						self.conn_cursor.execute(sql_command, values)
						self.conn.commit()
						# count = self.conn_cursor.rowcount
						# print(count, "Record inserted successfully!")
				except(Exception, psycopg2.Error) as error:
						print("Failed to attempt to insert record %s\nError msg: %s" % (sql_command_txt, error))
		
		def insert_batch(self, sql_insert, list_values, page_size):
				try:
						psycopg2.extras.execute_batch(self.conn_cursor, sql_insert, list_values, page_size=page_size)
						self.conn.commit()
				except(Exception, psycopg2.Error) as error:
						print("Failed to try to insert batch %s\nError msg: %s" % (sql_insert, error))

		def select(self, sql_command):
				if (self.conn == None):
						print("Start a connection before adding records!")
						return
				
				try:
						# self.conn_cursor.execute(sql_command, values)
						# return self.conn_cursor.fetchall()
						pd_sql_result = pd.read_sql_query(sql_command, self.conn)
						return pd_sql_result

				except(Exception, psycopg2.Error) as error:
						print("Failed to attempt to insert record %s\nError msg: %s" % (sql_command, error))
