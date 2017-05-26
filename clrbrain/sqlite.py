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
import numpy as np

from clrbrain import cli

db_path = "clrbrain.db"

def _create_db():
    """Creates the database including initial schema insertion.
    
    Raises:
        FileExistsError: If file with the same path already exists.
    """
    # creates empty database in the current working directory if
    # not already there.
    if os.path.exists(db_path):
        raise FileExistsError("{} already exists; please rename"
                              " or remove it first".format(db_path))
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # experiments table
    cur.execute("CREATE TABLE experiments (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                          "name TEXT, date DATE)")
    
    # ROIs table
    cur.execute("CREATE TABLE rois (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                   "experiment_id INTEGER, series INTEGER, "
                                   "offset_x INTEGER, offset_y INTEGER, "
                                   "offset_z INTEGER, size_x INTEGER, "
                                   "size_y INTEGER, size_z INTEGER,"
                "UNIQUE (experiment_id, series, offset_x, offset_y, offset_z))")
    
    # blobs tabls
    cur.execute("CREATE TABLE blobs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                    "roi_id INTEGER, x INTEGER, y INTEGER, "
                                    "z INTEGER, radius REAL, "
                                    "confirmed INTEGER,"
                "UNIQUE (roi_id, x, y, z))")
    
    conn.commit()
    return conn, cur

def start_db(path=None):
    """Starts the database.
    
    Returns:
        conn: The connection.
        cur: Connection's cursor.
    """
    if path is None:
        path = db_path
    if not os.path.exists(path):
        conn, cur = _create_db()
    else:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        print("Loaded database from {}".format(path))
    return conn, cur

def insert_experiment(conn, cur, name, date):
    """Inserts an experiment into the database.
    
    Args:
        conn: The connection.
        cur: Connection's cursor.
        name: Name of the experiment.
        date: The date as a SQLite date object.
    """
    cur.execute("INSERT INTO experiments (name, date) VALUES (?, ?)", 
                (name, date))
    print("{} experiment inserted".format(name))
    conn.commit()

def select_experiment(cur, name):
    """Selects an experiment from the given name.
    
    Args:
        cur: Connection's cursor.
        name: Name of the experiment.
    
    Returns:
        All of the experiments with the given name, or an empty list 
            if none are found.
    """
    cur.execute("SELECT id, name, date FROM experiments WHERE name = ?", 
                (name, ))
    rows = cur.fetchall()
    return rows

def select_or_insert_experiment(conn, cur, exp_name, date):
    """Selects an experiment from the given name, or inserts the 
    experiment if not found.
    
    Args:
        conn: The connection.
        cur: Connection's cursor.
        exp_name: Name of the experiment, typically the filename.
        date: The date as a SQLite date object.
    
    Returns:
        The ID fo the selected or inserted experiment.
    """
    exps = select_experiment(cur, exp_name)
    if len(exps) >= 1:
        exp_id = exps[0][0]
    else:
        insert_experiment(conn, cur, exp_name, date)
        exp_id = cur.lastrowid
        #raise LookupError("could not find experiment {}".format(exp_name))
    return exp_id

def insert_roi(conn, cur, exp_id, series, offset, size):
    """Inserts an ROI into the database.
    
    Args:
        conn: The connection.
        cur: Connection's cursor.
        experiment_id: ID of the experiment.
        series: Series within the experiment.
        offset: ROI offset as (x, y, z).
        size: ROI size as (x, y, z)
    
    Returns:
        cur.lastrowid: The number of rows inserted, or -1 if none.
        feedback: Feedback string.
    """
    cur.execute("INSERT OR IGNORE INTO rois (experiment_id, series, "
                                             "offset_x, offset_y, "
                                             "offset_z, size_x, size_y, "
                                             "size_z) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
                (exp_id, series, *offset, *size))
    feedback = "ROI inserted with offset {} and size {}".format(offset, size)
    print(feedback)
    conn.commit()
    return cur.lastrowid, feedback

def select_or_insert_roi(conn, cur, exp_id, series, offset, size):
    """Selects an ROI from the given parameters, or inserts the 
    experiment if not found.
    
    Args:
        conn: The connection.
        cur: Connection's cursor.
        exp_id: ID of the experiment.
        series: Series within the experiment.
        offset: ROI offset as (x, y, z).
        size: ROI size as (x, y, z)
    
    Returns:
        row_id: ID of the selected or inserted row.
        feedback: User feedback on the result.
    """
    cur.execute("SELECT * FROM rois WHERE experiment_id = ? "
                "AND series = ? AND offset_x = ? AND "
                "offset_y = ? AND offset_z = ?", (exp_id, series, *offset))
    row = cur.fetchone()
    if row is not None and len(row) > 0:
        print("selected ROI {}".format(row[0]))
        return row[0], "Found ROI {}".format(row[0])
    else:
        return insert_roi(conn, cur, exp_id, series, offset, size)

def select_rois(cur, exp_id):
    """Selects ROIs from the given experiment
    
    Args:
        cur: Connection's cursor.
        experiment_id: ID of the experiment.
    """
    cur.execute("SELECT id, experiment_id, series, offset_x, offset_y, offset_z, size_x, size_y, "
                "size_z FROM rois WHERE experiment_id = ?", (exp_id, ))
    rows = cur.fetchall()
    return rows

def select_roi(cur, roi_id):
    """Selects an ROI from the ID.
    
    Params:
        cur: Connection's cursor.
        roi_id: The ID of the ROI.
    
    Returns:
        The ROI.
    """
    cur.execute("SELECT * FROM rois WHERE id = ?", (roi_id, ))
    row = cur.fetchone()
    return row

def insert_blobs(conn, cur, roi_id, blobs):
    """Inserts blobs into the database, replacing any duplicate blobs.
    
    Args:
        conn: The connection.
        cur: Connection's cursor.
        blobs: Array of blobs arrays, assumes to be in (x, y, z, radius, confirmed)
            format. "Confirmed" is given as -1 = unconfirmed, 0 = incorrect, 
            1 = correct.
    """
    blobs_list = []
    confirmed = 0
    for blob in blobs:
        blob_entry = [roi_id]
        blob_entry.extend(blob)
        blobs_list.append(blob_entry)
        if blob[4] == 1:
            confirmed = confirmed + 1
    cur.executemany("INSERT OR REPLACE INTO blobs (roi_id, x, y, z, radius, confirmed) "
                    "VALUES (?, ?, ?, ?, ?, ?)", blobs_list)
    print("{} blobs inserted, {} confirmed".format(cur.rowcount, confirmed))
    conn.commit()
    
def delete_blobs(conn, cur, roi_id, blobs):
    """Inserts blobs into the database, replacing any duplicate blobs.
    
    Args:
        conn: The connection.
        cur: Connection's cursor.
        blobs: Array of blobs arrays, assumes to be in (x, y, z, radius, confirmed)
            format. "Confirmed" is given as -1 = unconfirmed, 0 = incorrect, 
            1 = correct.
    """
    deleted = 0
    for blob in blobs:
        blob_entry = [roi_id]
        blob_entry.extend(blob[0:3])
        cur.execute("DELETE FROM blobs WHERE roi_id = ? AND x = ? AND y = ? "
                    "AND z = ?", blob_entry)
        deleted += 1
        print("deleted blob {}".format(blob))
    print("{} blob(s) deleted".format(deleted))
    conn.commit()
    return deleted

def _parse_blobs(rows):
    blobs = np.empty((len(rows), 5))
    rowi = 0
    for row in rows:
        blobs[rowi] = [row["z"], row["y"], row["x"], row["radius"], row["confirmed"]]
        rowi += 1
    return blobs

def select_blobs(cur, roi_id):
    """Selects ROIs from the given experiment
    
    Args:
        cur: Connection's cursor.
        experiment_id: ID of the experiment.
    
    Returns:
        Blobs in the given ROI.
    """
    cur.execute("SELECT * FROM blobs WHERE roi_id = ?", (roi_id, ))
    return _parse_blobs(cur.fetchall())

def select_blobs_confirmed(cur, confirmed):
    """Selects ROIs from the given experiment
    
    Args:
        cur: Connection's cursor.
        experiment_id: ID of the experiment.
    
    Returns:
        Blobs in the given ROI.
    """
    cur.execute("SELECT * FROM blobs WHERE confirmed = ?", (confirmed, ))
    return _parse_blobs(cur.fetchall())

def _test_db():
    # simple database test
    conn, cur = start_db()
    exp_name = "TextExp"
    exp_id = select_or_insert_experiment(conn, cur, exp_name, datetime.datetime(1000, 1, 1))
    insert_blobs(conn, cur, exp_id, 12, [[3, 2, 5, 23.4], [2, 3, 7, 13.2]])
    conn.commit()
    conn.close()

if __name__ == "__main__":
    print("Starting sqlite.py...")
    # parses arguments and sets up the DB
    cli.main(True)
    conn, cur = start_db()
    
    # selects experiment based on command-line arg and gathers all ROIs
    # and blobs within them
    exp = select_experiment(cur, os.path.basename(cli.filename))
    rois = select_rois(cur, exp[0][0])
    blobs = []
    for roi in rois:
        bb = select_blobs(cur, roi[0])
        blobs.extend(bb)
    blobs = np.array(blobs)
    
    # basic stats based on confirmation status, ignoring maybes
    blobs_true = blobs[blobs[:, 4] == 1] # all pos
    # radius = 0 indicates that the blob was manually added, not detected
    blobs_true_detected = blobs_true[np.nonzero(blobs_true[:, 3])] # true pos
    # not detected neg, so no "true neg" but only false pos
    blobs_false = blobs[blobs[:, 4] == 0] # false pos
    all_pos = blobs_true.shape[0]
    true_pos = blobs_true_detected.shape[0]
    false_pos = blobs_false.shape[0]
    false_neg = all_pos - true_pos # not detected but should have been
    sens = float(true_pos) / all_pos
    ppv = float(true_pos) / (true_pos + false_pos)
    
    # most conservative, where blobs tested pos that are only maybes are treated
    # as false pos, and missed blobs that are maybes are treated as pos
    blobs_maybe = blobs[blobs[:, 4] == 2] # all unknown
    # tested pos but only maybe in reality, so treated here as false pos
    blobs_maybe_from_detected = blobs_maybe[np.nonzero(blobs_maybe[:, 3])]
    false_pos_from_maybe = blobs_maybe_from_detected.shape[0]
    # adds the maybes that were undetected
    all_true_with_maybes = all_pos + blobs_maybe.shape[0] - false_pos_from_maybe
    false_pos_with_maybes = false_pos + false_pos_from_maybe
    false_neg_with_maybes = all_true_with_maybes - true_pos
    sens_maybe_missed = float(true_pos) / all_true_with_maybes
    ppv_maybe_missed = float(true_pos) / (true_pos + false_pos_with_maybes)
    
    # prints stats
    print("Ignoring maybes:\ncells = {}\ndetected cells = {}\nfalse pos cells = {}\n"
          "false neg cells = {}\nsensitivity = {}\nPPV = {}\n"
          .format(all_pos, true_pos, false_pos, false_neg, sens, ppv))
    print("Including maybes:\ncells = {}\ndetected cells = {}\nfalse pos cells = {}\n"
          "false neg cells = {}\nsensitivity = {}\nPPV = {}"
          .format(all_true_with_maybes, true_pos, false_pos_with_maybes, 
                  false_neg_with_maybes, sens_maybe_missed, ppv_maybe_missed))
