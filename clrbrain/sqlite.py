# SQLite database connection
# Author: David Young, 2017
"""Connects with a SQLite database for experiment and image
analysis storage.

Attributes:
    db_path: Path to the database.
"""

import os
import shutil
import datetime
import sqlite3
import numpy as np

from clrbrain import cli

DB_NAME = "clrbrain.db"
DB_NAME_VERIFIED = "clrbrain_verified.db"
DB_VERSION = 2

def _backup_db(path):
    i = 1
    backup_path = None
    while True:
        suffix = "({}).db".format(i)
        backup_path = path.replace(".db", suffix)
        if not os.path.exists(backup_path):
            shutil.move(path, backup_path)
            print("Backed up database to {}".format(backup_path))
            break
        i += 1

def _create_db(path):
    """Creates the database including initial schema insertion.
    
    Raises:
        FileExistsError: If file with the same path already exists.
    """
    # creates empty database in the current working directory if
    # not already there.
    if os.path.exists(path):
        _backup_db(path)
        '''
        raise FileExistsError("{} already exists; please rename"
                              " or remove it first".format(db_path))
        '''
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # create tables
    _create_table_about(cur)
    _create_table_experiments(cur)
    _create_table_rois(cur)
    _create_table_blobs(cur)
    
    # store DB version information
    insert_about(conn, cur, DB_VERSION, datetime.datetime.now())
    
    conn.commit()
    return conn, cur

def _create_table_about(cur):
    cur.execute("CREATE TABLE about (version INTEGER PRIMARY KEY, date DATE)")

def _create_table_experiments(cur):
    cur.execute("CREATE TABLE experiments (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                          "name TEXT, date DATE)")

def _create_table_rois(cur):
    cur.execute("CREATE TABLE rois (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                   "experiment_id INTEGER, series INTEGER, "
                                   "offset_x INTEGER, offset_y INTEGER, "
                                   "offset_z INTEGER, size_x INTEGER, "
                                   "size_y INTEGER, size_z INTEGER, "
                "UNIQUE (experiment_id, series, offset_x, offset_y, offset_z))")

def _create_table_blobs(cur):
    cur.execute("CREATE TABLE blobs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                    "roi_id INTEGER, x INTEGER, y INTEGER, "
                                    "z INTEGER, radius REAL, "
                                    "confirmed INTEGER, truth INTEGER, "
                "UNIQUE (roi_id, x, y, z, truth))")

def upgrade_db(conn, cur):
    db_ver = 0
    # about table does not exist until DB ver 2
    try:
        abouts = select_about(conn, cur)
        db_ver = abouts[len(abouts) - 1]["version"]
    except sqlite3.OperationalError as e:
        print(e)
        print("defaulting to upgrade from DB version {}".format(db_ver))
    
    # return if already at latest version
    if db_ver >= DB_VERSION:
        return
    
    # start upgrading DB for each version increment
    print("Starting database upgrade...")
    if db_ver < 2:
        print("upgrading DB version from {}".format(db_ver))
        
        # about table to track DB version numbers
        print("inserting new about table")
        _create_table_about(cur)
        insert_about(conn, cur, DB_VERSION, datetime.datetime.now())
        
        # new column with unique constraint on blobs table
        print("upgrading blobs table")
        cur.execute("ALTER TABLE blobs RENAME TO tmp_blobs")
        _create_table_blobs(cur)
        cols = "roi_id, z, y, x, radius, confirmed"
        cur.execute("INSERT INTO blobs (" + cols + ", truth) SELECT " + cols + ", -1 FROM tmp_blobs")
        cur.execute("DROP TABLE tmp_blobs")
    print("...finished database upgrade.")
    conn.commit()

def start_db(path=None, new_db=False):
    """Starts the database.
    
    Returns:
        conn: The connection.
        cur: Connection's cursor.
    """
    if path is None:
        path = DB_NAME
    if new_db or not os.path.exists(path):
        conn, cur = _create_db(path)
    else:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        print("Loaded database from {}".format(path))
    upgrade_db(conn, cur)
    return conn, cur

def insert_about(conn, cur, version, date):
    """Inserts an experiment into the database.
    
    Args:
        conn: The connection.
        cur: Connection's cursor.
        name: Name of the experiment.
        date: The date as a SQLite date object.
    """
    cur.execute("INSERT INTO about (version, date) VALUES (?, ?)", 
                (version, date))
    print("about table entry entered with version {}".format(version))
    conn.commit()
    return cur.lastrowid

def select_about(conn, cur):
    cur.execute("SELECT * FROM about")
    rows = cur.fetchall()
    return rows

def insert_experiment(conn, cur, name, date):
    """Inserts an experiment into the database.
    
    Args:
        conn: The connection.
        cur: Connection's cursor.
        name: Name of the experiment.
        date: The date as a SQLite date object.
    """
    if date is None:
        date = datetime.datetime.now()
    cur.execute("INSERT INTO experiments (name, date) VALUES (?, ?)", 
                (name, date))
    print("{} experiment inserted".format(name))
    conn.commit()
    return cur.lastrowid

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
        exp_id = insert_experiment(conn, cur, exp_name, date)
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
        cur.lastrowid: ID of the selected or inserted row.
        feedback: Feedback string.
    """
    cur.execute("INSERT OR REPLACE INTO rois (experiment_id, series, "
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
                "offset_y = ? AND offset_z = ? AND size_x = ? "
                "AND size_y = ? AND size_z = ?", 
                (exp_id, series, *offset, *size))
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

def update_rois(cur, offset, size):
    """Updates ROI positions and size.
    
    Args:
        cur: Connection's cursor.
        offset: Amount to subtract from the offset.
        size: Amount to add to the size.
    """
    cur.execute("SELECT * FROM rois")
    rows = cur.fetchall()
    for row in rows:
        # TODO: subtracts from offset but adds to size for now because of 
        # limitation in argparse accepting comma-delimited neg numbers
        cur.execute("UPDATE rois SET offset_x = ?, offset_y = ?, offset_z = ?, "
                                    "size_x = ?, size_y = ?, size_z = ? "
                                "WHERE id = ?", 
                    (row["offset_x"] - offset[0], row["offset_y"] - offset[1], 
                     row["offset_z"] - offset[2], row["size_x"] + size[0], 
                     row["size_y"] + size[1], row["size_z"] + size[2], 
                     row["id"]))
    conn.commit()

def select_roi(cur, roi_id):
    """Selects an ROI from the ID.
    
    Args:
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
        blobs: Array of blobs arrays, assumes to be in (z, y, x, radius, confirmed)
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
    cur.executemany("INSERT OR REPLACE INTO blobs (roi_id, z, y, x, radius, confirmed, truth) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", blobs_list)
    print("{} blobs inserted, {} confirmed".format(cur.rowcount, confirmed))
    conn.commit()
    
def delete_blobs(conn, cur, roi_id, blobs):
    """Inserts blobs into the database, replacing any duplicate blobs.
    
    Args:
        conn: The connection.
        cur: Connection's cursor.
        blobs: Array of blobs arrays, assumes to be in (z, y, x, radius, confirmed)
            format. "Confirmed" is given as -1 = unconfirmed, 0 = incorrect, 
            1 = correct.
    """
    deleted = 0
    for blob in blobs:
        blob_entry = [roi_id]
        blob_entry.extend(blob[0:3])
        cur.execute("DELETE FROM blobs WHERE roi_id = ? AND z = ? AND y = ? "
                    "AND x = ?", blob_entry)
        deleted += 1
        print("deleted blob {}".format(blob))
    print("{} blob(s) deleted".format(deleted))
    conn.commit()
    return deleted

def _parse_blobs(rows):
    blobs = np.empty((len(rows), 6))
    rowi = 0
    for row in rows:
        blobs[rowi] = [row["z"], row["y"], row["x"], row["radius"], row["confirmed"], row["truth"]]
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

def verification_stats(conn, cur):
    # selects experiment based on command-line arg and gathers all ROIs
    # and blobs within them
    exp = select_experiment(cur, os.path.basename(cli.filename))
    rois = select_rois(cur, exp[0][0])
    blobs = []
    for roi in rois:
        #print("got roi_id: {}".format(roi[0]))
        bb = select_blobs(cur, roi[0])
        blobs.extend(bb)
    blobs = np.array(blobs)
    
    if config.verified_db is None:
        # basic stats based on confirmation status, ignoring maybes
        blobs_true = blobs[blobs[:, 4] == 1] # all pos
        # radius = 0 indicates that the blob was manually added, not detected
        blobs_true_detected = blobs_true[np.nonzero(blobs_true[:, 3])] # true pos
        # not detected neg, so no "true neg" but only false pos
        blobs_false = blobs[blobs[:, 4] == 0] # false pos
    else:
        # basic stats based on confirmation status, ignoring maybes
        blobs_true = blobs[blobs[:, 5] >= 0] # all truth blobs
        blobs_detected = blobs[blobs[:, 5] == -1] # all non-truth blobs
        # radius = 0 indicates that the blob was manually added, not detected
        blobs_true_detected = blobs_detected[blobs_detected[:, 4] == 1] # true pos
        # not detected neg, so no "true neg" but only false pos
        blobs_false = blobs[blobs[:, 4] == 0] # false pos
    all_pos = blobs_true.shape[0]
    true_pos = blobs_true_detected.shape[0]
    false_pos = blobs_false.shape[0]
    false_neg = all_pos - true_pos # not detected but should have been
    sens = float(true_pos) / all_pos
    ppv = float(true_pos) / (true_pos + false_pos)
    print("Stats:\ncells = {}\ndetected cells = {}\nfalse pos cells = {}\n"
          "false neg cells = {}\nsensitivity = {}\nPPV = {}\n"
          .format(all_pos, true_pos, false_pos, false_neg, sens, ppv))
    
    if config.verified_db is None:
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
        print("Including maybes:\ncells = {}\ndetected cells = {}\nfalse pos cells = {}\n"
              "false neg cells = {}\nsensitivity = {}\nPPV = {}"
              .format(all_true_with_maybes, true_pos, false_pos_with_maybes, 
                      false_neg_with_maybes, sens_maybe_missed, ppv_maybe_missed))


def _test_db():
    # simple database test
    conn, cur = start_db()
    exp_name = "TextExp"
    exp_id = select_or_insert_experiment(conn, cur, exp_name, datetime.datetime(1000, 1, 1))
    insert_blobs(conn, cur, exp_id, 12, [[3, 2, 5, 23.4], [2, 3, 7, 13.2]])
    conn.commit()
    conn.close()

class ClrDB():
    conn = None
    cur = None
    blobs_truth = None
    
    def load_db(self, path, new_db):
        self.conn, self.cur = start_db(path, new_db)
    
    def load_truth_blobs(self):
        self.blobs_truth = select_blobs_confirmed(self.cur, 1)
        print("truth blobs:\n{}".format(self.blobs_truth))
    
    def get_rois(self, filename):
        exps = select_experiment(self.cur, filename)
        rois = None
        if len(exps) > 0:
            rois = select_rois(self.cur, exps[0]["id"])
        return rois

if __name__ == "__main__":
    print("Starting sqlite.py...")
    # parses arguments and sets up the DB
    cli.main(True)
    from clrbrain import config
    conn = config.db.conn
    cur = config.db.cur
    if config.verified_db is not None:
        conn = config.verified_db.conn
        cur = config.verified_db.cur
    verification_stats(conn, cur)
    #update_rois(cur, cli.offset, cli.roi_size)
    