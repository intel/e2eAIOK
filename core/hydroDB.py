import sqlite3
import os
from core.hydromodel import *

class HydroDB:
    def __init__(self):
        if not os.path.exists('hydroai.db'):
            self.init_db()
        else:
            self.conn = sqlite3.connect('hydroai.db')
            self.cur = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def init_db(self):
        self.conn = sqlite3.connect('hydroai.db')
        self.cur = self.conn.cursor()
        self.cur.execute('''CREATE TABLE models (learner_id text, hydro_model json)''')
        self.conn.commit()

    def get_model_by_id(self, learner_id):
        hydro_model = None
        for row in self.cur.execute("SELECT hydro_model FROM models WHERE learner_id = '%s'" % learner_id):
            hydro_model = HydroModel(None, row)
        return hydro_model

    def insert_model_by_id(self, learner_id, hydro_model):
        # if learner_id exsits, update original one
        # or insert new one
        do_update = False
        for row in self.cur.execute("SELECT * FROM models WHERE learner_id = '%s'" % learner_id):
            do_update = True
        if do_update:
            self.cur.execute("UPDATE models SET hydro_model = '%s' WHERE learner_id = '%s'" % (hydro_model.to_json(), learner_id))
        else:
            self.cur.execute("INSERT INTO models VALUES ('%s','%s')" % (learner_id, hydro_model.to_json()))
        self.conn.commit()

    def clear():
        os.remove('hydroai.db')