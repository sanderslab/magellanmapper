# SQLite database connection
# Author: David Young, 2017
"""Connects with a SQLite database for experiment and image
analysis storage.

Attributes:
    db_path: Path to the database.
"""

import os
import datetime
import sqlite3

db_path = "clrbrain.db"

def _create_db():
    """Creates the database including initial schema insertion.
    """
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
    """Starts the database.
    
    Returns:
        conn: The connection.
        cur: Connection's cursor.
    """
    if not os.path.exists(db_path):
        conn, cur = _create_db()
    else:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
    return conn, cur

def insert_experiment(conn, cur, name, date):
    """Inserts an experiment into the database.
    
    Args:
        conn: The connection.
        cur: Connection's cursor.
        name: Name of the experiment.
        date: The date as a SQLite date object.
    """
    cur.execute("INSERT INTO experiments (name, date) VALUES (?, ?)", (name, date))
    print("{} experiment inserted".format(name))
    conn.commit()

def select_experiment(cur, name):
    """Selects an experiment from the given name.
    
    Args:
        cur: Connection's cursor.
        name: Name of the experiment.
    """
    cur.execute("SELECT id, name, date FROM experiments WHERE name = ?", (name, ))
    rows = cur.fetchall()
    return rows

def select_or_insert_experiment(conn, cur, exp_name, date):
    """Selects an experiment from the given name, or inserts the 
    experiment if not found.
    
    Args:
        conn: The connection.
        cur: Connection's cursor.
        exp_name: Name of the experiment.
        date: The date as a SQLite date object.
    """
    exps = select_experiment(cur, exp_name)
    if len(exps) >= 1:
        exp_id = exps[0][0]
    else:
        insert_experiment(conn, cur, exp_name, date)
        exp_id = cur.lastrowid
        #raise LookupError("could not find experiment {}".format(exp_name))
    return exp_id

def insert_blobs(conn, cur, experiment_id, series, blobs):
    """Inserts blobs into the database.
    
    Args:
        conn: The connection.
        cur: Connection's cursor.
        experiment_id: ID of the experiment.
        blobs: Array of blobs arrays, assumes to be in (x, y, z, radius) format.
    """
    blobs_list = []
    for blob in blobs:
        blobs_list.append((experiment_id, series, blob[0], blob[1], blob[2], blob[3]))
    cur.executemany("INSERT INTO blobs (experiment_id, series, x, y, z, radius) "
                    "VALUES (?, ?, ?, ?, ?, ?)", blobs_list)
    print("{} blobs inserted".format(cur.rowcount))
    conn.commit()
    
if __name__ == "__main__":
    print("Starting sqlite.py...")
    # simple database test
    conn, cur = start_db()
    exp_name = "TextExp"
    exp_id = select_or_insert_experiment(conn, cur, exp_name, datetime.datetime(1000, 1, 1))
    insert_blobs(conn, cur, exp_id, 12, [[3, 2, 5, 23.4], [2, 3, 7, 13.2]])
    conn.commit()
    conn.close()
    