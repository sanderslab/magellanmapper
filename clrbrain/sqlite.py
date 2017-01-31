# SQLite database connection
# Author: David Young, 2017
"""Connects with a SQLite database for segment storage
"""

import os
import datetime
import sqlite3

db_path = "clrbrain.db"

def create_db():
    if os.path.exists(db_path):
        raise FileExistsError("{} already exists; please rename"
                              " or remove it first".format(db_path))
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("CREATE TABLE experiments (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                          "name TEXT, date DATE)")
    cur.execute("CREATE TABLE blobs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                    "experiment_id TEXT, series INTEGER, "
                                    "x INTEGER, y INTEGER, z INTEGER, radius REAL)")
    
    conn.commit()
    return conn, cur

def start_db():
    if not os.path.exists(db_path):
        conn, cur = create_db()
    else:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
    return conn, cur

def insert_experiment(conn, cur, name, date):
    cur.execute("INSERT INTO experiments (name, date) VALUES (?, ?)", (name, date))
    conn.commit()

def select_experiment(cur, name):
    cur.execute("SELECT id, name, date FROM experiments WHERE name = ?", (name, ))
    rows = cur.fetchall()
    return rows
    
def insert_blobs(conn, cur, experiment_id, series, blobs):
    blobs_list = []
    for blob in blobs:
        blobs_list.append((experiment_id, series, blob[0], blob[1], blob[2], blob[3]))
    cur.executemany("INSERT INTO blobs (experiment_id, series, x, y, z, radius) "
                    "VALUES (?, ?, ?, ?, ?, ?)", blobs_list)
    conn.commit()
    
if __name__ == "__main__":
    print("Starting sqlite.py...")
    conn, cur = start_db()
    exp_name = "TextExp"
    exps = select_experiment(cur, exp_name)
    if len(exps) >= 1:
        exp_id = exps[0][0]
    else:
        insert_experiment(conn, cur, exp_name, datetime.date.today())
        exp_id = cur.lastrowid
        print(exp_id)
        #raise LookupError("could not find experiment {}".format(exp_name))
    insert_blobs(conn, cur, exp_id, 12, [[3, 2, 5, 23.4], [2, 3, 7, 13.2]])
    conn.commit()
    conn.close()
    