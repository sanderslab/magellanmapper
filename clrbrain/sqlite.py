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
    cur.execute("CREATE TABLE rois (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                   "offset_x INTEGER, offset_y INTEGER, offset_z INTEGER, "
                                   "size_x INTEGER, size_y INTEGER, size_z INTEGER)")
    cur.execute("CREATE TABLE blobs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                    "experiment_id INTEGER, series INTEGER, roi_id INTEGER, "
                                    "x INTEGER, y INTEGER, z INTEGER, radius REAL, "
                                    "confirmed INTEGER)")
    
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

def insert_roi(conn, cur, offset, size):
    cur.execute("INSERT INTO rois (offset_x, offset_y, offset_z, size_x, size_y, size_z) "
                "VALUES (?, ?, ?, ?, ?, ?)", (*offset, *size))
    print("roi inserted with offset {} and size {}".format(offset, size))
    conn.commit()
    return cur.lastrowid

def insert_blobs(conn, cur, experiment_id, series, roi_id, blobs):
    """Inserts blobs into the database.
    
    Args:
        conn: The connection.
        cur: Connection's cursor.
        experiment_id: ID of the experiment.
        blobs: Array of blobs arrays, assumes to be in (x, y, z, radius, confirmed)
            format. "Confirmed" is given as -1 = unconfirmed, 0 = incorrect, 
            1 = correct.
    """
    blobs_list = []
    confirmed = 0
    for blob in blobs:
        blob_entry = [experiment_id, series, roi_id]
        blob_entry.extend(blob)
        blobs_list.append(blob_entry)
        if blob[4] == 1:
            confirmed = confirmed + 1
    cur.executemany("INSERT INTO blobs (experiment_id, series, roi_id, x, y, z, radius, confirmed) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", blobs_list)
    print("{} blobs inserted, {} confirmed".format(cur.rowcount, confirmed))
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
    