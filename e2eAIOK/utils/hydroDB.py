import sqlite3
import os
from e2eAIOK.utils.hydromodel import *


class HydroDB:
    """
    This class is used to create one instance for hydro database
    """
    def __init__(self, hydrodb_path):
        self.hydrodb_path = hydrodb_path
        if not os.path.exists(self.hydrodb_path):
            self.__init_db()
        else:
            self.conn = sqlite3.connect(self.hydrodb_path)
            self.cur = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def __init_db(self):
        self.conn = sqlite3.connect(self.hydrodb_path)
        self.cur = self.conn.cursor()
        self.cur.execute(
            '''CREATE TABLE models (learner_id text, hydro_model json)''')
        self.conn.commit()

    def print_all(self):
        for row in self.cur.execute("SELECT * FROM models"):
            print(row)

    def get_model_by_id(self, learner_id):
        """
        Get in database model info by learner_id

        Parameters
        ----------
        learner_id : str
            Learning_id is identical to task

        Returns
        -------
        hydro_model: HydroModel
        """
        hydro_model = None
        for row in self.cur.execute(
                "SELECT hydro_model FROM models WHERE learner_id = '%s'" %
                learner_id):
            hydro_model = HydroModel(None, row)
        return hydro_model

    def insert_model_by_id(self, learner_id, hydro_model):
        """
        Insert new or update history model by learner_id

        Parameters
        ----------
        learner_id : str
            Learning_id is identical to task
        hydro_model: HydroModel
        """
        # if learner_id exsits, update original one
        # or insert new one
        do_update = False
        for row in self.cur.execute(
                "SELECT * FROM models WHERE learner_id = '%s'" % learner_id):
            do_update = True
        if do_update:
            self.cur.execute(
                "UPDATE models SET hydro_model = '%s' WHERE learner_id = '%s'"
                % (hydro_model.to_json(), learner_id))
        else:
            self.cur.execute("INSERT INTO models VALUES ('%s','%s')" %
                             (learner_id, hydro_model.to_json()))
        self.conn.commit()

    def clear(self):
        """
        Remove e2eaiok.db file
        """
        os.remove(self.hydrodb_path)