'''
   Class implements import of dataset data
   RSDD-Time saved in Julia format for dataframe pandas
'''

import json
import pandas as pd
import pandas.io.json as pd_json
import datetime
from datetime import datetime as dt

import utils.definition_network as dn
from utils.log import Log

# choose a random element from a list
from random import seed
from random import shuffle

class RsddTimeFile:
    def __init__(self, _file_path="", verbose=True):
        self.file_path = _file_path
        self.verbose = verbose
        if self.verbose:
            self.log = Log()

				# Fix initialize
        self.total_class = 0
        self.sample_by_class = 0

    def __generate_log(self, text):
        if self.verbose:
            self.log.save(text)

    def __get_total_lines_file(self):
        with open(self.file_path) as json_file:
            total_lines = sum(1 for i, json_line in enumerate(json_file))
        
        return total_lines

    def __process_users(self, con):
        
        file = open(self.file_path)

        for i, json_line in enumerate(file):
            users = []
            json_line = json_line.replace('"id"', '"user_id"')
            json_line = json_line.replace('True', '"True"')
            json_line = json_line.replace('False', '"False"')
            json_line = json_line.replace(': None', ': "None"')
            print(i, json_line)
            user = json.loads(json_line)
            user['created_utc'] = dt.fromtimestamp(user['created_utc'])

            user['consensus'] = str(user['consensus']).replace('\'', '"')
            user['consensus'] = user['consensus'].replace('I"v', 'Iv')
            user['consensus'] = user['consensus'].replace('i"v', 'iv')
            user['consensus'] = user['consensus'].replace('t"s', 'ts')
            user['consensus'] = user['consensus'].replace('I"m', 'Im')
            user['consensus'] = user['consensus'].replace('e"s', 'es')
            user['consensus'] = user['consensus'].replace('0 " s', '0s')
            user['consensus'] = user['consensus'].replace('True', '"True"')
            user['consensus'] = user['consensus'].replace('False', '"False"')
            user['consensus'] = user['consensus'].replace(': None', ': "None"')

            user['diagnosis'] = str(user['diagnosis']).replace('\'', '"')
            user['diagnosis'] = user['diagnosis'].replace('I"v', 'Iv')
            user['diagnosis'] = user['diagnosis'].replace('i"v', 'iv')
            user['diagnosis'] = user['diagnosis'].replace('t"s', 'ts')
            user['diagnosis'] = user['diagnosis'].replace('e"s', 'es')
            user['diagnosis'] = user['diagnosis'].replace('I"m', 'Im')
            user['diagnosis'] = user['diagnosis'].replace('0 " s', '0s')
            user['diagnosis'] = user['diagnosis'].replace('True', '"True"')
            user['diagnosis'] = user['diagnosis'].replace('False', '"False"')
            user['diagnosis'] = user['diagnosis'].replace(': None', ': "None"')

            user['time'] = str(user['time']).replace('\'', '"')
            user['time'] = user['time'].replace('I"v', 'Iv')
            user['time'] = user['time'].replace('i"v', 'iv')
            user['time'] = user['time'].replace('t"s', 'ts')
            user['time'] = user['time'].replace('e"s', 'es')
            user['time'] = user['time'].replace('I"m', 'Im')
            user['time'] = user['time'].replace('0 " s', '0s')
            user['time'] = user['time'].replace('True', '"True"')
            user['time'] = user['time'].replace('False', '"False"')
            user['time'] = user['time'].replace(': None', ': "None"')

            users.append(user)
            con.insert_batch('''INSERT INTO users (user_id, text, sample_text, created_utc, diagnosis, time, consensus)
                                    VALUES (%(user_id)s, %(text)s, %(sample_text)s, %(created_utc)s, %(diagnosis)s,
                                            %(time)s, %(consensus)s);''', users, 1000)

    def __file_to_postgres(self, con):
        total_lines = self.__get_total_lines_file()
        self.__generate_log('File %s, total lines: %s' % (str(self.file_path), str(total_lines)))
    
        time_ini = datetime.datetime.now()
        self.__process_users(con)
    
        time_end = datetime.datetime.now()
        self.__generate_log('Ini: %s\tEnd: %s\tTotal: %s' % (time_ini.strftime("%Y-%m-%d %H:%M:%S"),
                                                             time_end.strftime("%Y-%m-%d %H:%M:%S"),
                                                             (time_end - time_ini)))
   
    def convert_file_to_postgres(self, con):
        if con.connection() == True:
            self.__file_to_postgres(con)
