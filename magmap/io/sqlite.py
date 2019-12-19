# SQLite database connection
# Author: David Young, 2017, 2018
"""Connects with a SQLite database for experiment and image
analysis storage.

Attributes:
    db_path: Path to the database.
"""

import os
import glob
import datetime
import sqlite3
import numpy as np

from magmap.settings import config
from magmap.cv import detector
from magmap.io import libmag

DB_NAME_VERIFIED = "clrbrain_verified.db"
DB_NAME_MERGED = "clrbrain_merged.db"
DB_SUFFIX_TRUTH = "_truth.db"
DB_VERSION = 3

_COLS_BLOBS = "roi_id, z, y, x, radius, confirmed, truth, channel"

def _create_db(path):
    """Creates the database including initial schema insertion.
    
    Raises:
        FileExistsError: If file with the same path already exists.
    """
    # creates empty database in the current working directory if
    # not already there.
    if os.path.exists(path):
        libmag.backup_file(path)
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
    print("created db at {}".format(path))
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
                                    "channel INTEGER, "
                "UNIQUE (roi_id, x, y, z, truth, channel))")

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
        print("upgrading DB version from {} to 2".format(db_ver))
        
        # about table to track DB version numbers
        print("inserting new about table")
        _create_table_about(cur)
        
        # new column for truth with updated unique constraint on blobs table
        print("upgrading blobs table")
        cur.execute("ALTER TABLE blobs RENAME TO tmp_blobs")
        _create_table_blobs(cur)
        cols = _COLS_BLOBS.rsplit(",", 2)[0]
        cur.execute("INSERT INTO blobs (" + cols + ", truth) SELECT " 
                    + cols + ", -1 FROM tmp_blobs")
        cur.execute("DROP TABLE tmp_blobs")
        
    if db_ver < 3:
        print("upgrading DB version from {} to 3".format(db_ver))
        
        # new column for channel and updated unique constraint on blobs table
        print("upgrading blobs table")
        cur.execute("ALTER TABLE blobs RENAME TO tmp_blobs")
        _create_table_blobs(cur)
        cols = _COLS_BLOBS.rsplit(",", 1)[0]
        cur.execute("INSERT INTO blobs (" + cols + ", channel) SELECT " 
                    + cols + ", 0 FROM tmp_blobs")
        cur.execute("DROP TABLE tmp_blobs")
    
    # record database upgrade version and time
    insert_about(conn, cur, DB_VERSION, datetime.datetime.now())
    
    print("...finished database upgrade.")
    conn.commit()

def start_db(path=None, new_db=False):
    """Starts the database.
    
    Args:
        path: Path where the new database resides; if None, defaults to 
            :attr:``DB_NAME``.
        new_db: If True or if ``path`` does not exist, a new database will 
            be created; defaults to False.
    
    Returns:
        conn: The connection.
        cur: Connection's cursor.
    """
    if path is None:
        path = config.db_name
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
        name: Name of the experiment. If None, all experiments will be 
            returned.
    
    Returns:
        All of the experiments with the given name, an empty list 
            if none are found, or all experiments if ``name`` is None.
    """
    cols = "id, name, date"
    #print("looking for exp: {}".format(name))
    if name is None:
        cur.execute("SELECT {} FROM experiments".format(cols))
    else:
        cur.execute("SELECT {} FROM experiments WHERE name = ?".format(cols), 
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

def _update_experiments(db_dir):
    """Updates experiment names by shifting the old .czi extension name 
    from midway through the name to its end
    
    Args:
        cur: Connection's cursor.
        db_dir: Directory that contains databases to update
    """
    ext = ".czi"
    db_paths = sorted(glob.glob(os.path.join(db_dir, "*.db")))
    for db_path in db_paths:
        db = ClrDB()
        db.load_db(db_path, False)
        rows = select_experiment(db.cur, None)
        for row in rows:
            name = row["name"]
            if not name.endswith(ext):
                name_updated = name.replace(ext, "_") + ext
                print("...replaced experiment {} with {}".format(name, name_updated))
                db.cur.execute("UPDATE experiments SET name = ? WHERE name = ?", 
                            (name_updated, name))
            else:
                print("...no update")
        db.conn.commit()

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
    ROI if not found.
    
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
    cur.execute("SELECT id, experiment_id, series, "
                "offset_x, offset_y, offset_z, size_x, size_y, "
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
        blobs: Array of blobs arrays, assumed to be formatted accorind to 
            :func:``detector.format_blob``. "Confirmed" is given as 
            -1 = unconfirmed, 0 = incorrect, 1 = correct.
    """
    blobs_list = []
    confirmed = 0
    for blob in blobs:
        blob_entry = [roi_id]
        blob_entry.extend(blob)
        #print("blob type:\n{}".format(blob.dtype))
        blobs_list.append(blob_entry)
        if detector.get_blob_confirmed(blob) == 1:
            confirmed = confirmed + 1
    #print(match_elements(_COLS_BLOBS, ", ", "?"))
    cur.executemany("INSERT OR REPLACE INTO blobs ({}) VALUES ({})"
                    .format(_COLS_BLOBS, match_elements(
                            _COLS_BLOBS, ", ", "?")), blobs_list)
    print("{} blobs inserted, {} confirmed".format(cur.rowcount, confirmed))
    conn.commit()
    
def delete_blobs(conn, cur, roi_id, blobs):
    """Deletes blobs matching the given blobs' ROI ID and coordinates.
    
    Args:
        conn: The connection.
        cur: Connection's cursor.
        blobs: Array of blobs arrays, assumed to be formatted accorind to 
            :func:``detector.format_blob``.
    
    Returns:
        The number of rows deleted.
    """
    deleted = 0
    for blob in blobs:
        blob_entry = [roi_id, *blob[:3], detector.get_blob_channel(blob)]
        print("attempting to delete blob {}".format(blob))
        cur.execute("DELETE FROM blobs WHERE roi_id = ? AND z = ? AND y = ? "
                    "AND x = ? AND channel = ?", blob_entry)
        count =  cur.rowcount
        if count > 0:
            deleted += count
            print("deleted blob {}".format(blob))
    print("{} blob(s) deleted".format(deleted))
    conn.commit()
    return deleted

def _parse_blobs(rows):
    blobs = np.empty((len(rows), 7))
    rowi = 0
    for row in rows:
        blobs[rowi] = [
            row["z"], row["y"], row["x"], row["radius"], row["confirmed"], 
            row["truth"], row["channel"]
        ]
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
    cur.execute(
        "SELECT {} FROM blobs WHERE roi_id = ?".format(_COLS_BLOBS), (roi_id, ))
    return _parse_blobs(cur.fetchall())

def select_blobs_confirmed(cur, confirmed):
    """Selects ROIs from the given experiment
    
    Args:
        cur: Connection's cursor.
        experiment_id: ID of the experiment.
    
    Returns:
        Blobs in the given ROI.
    """
    cur.execute(
        "SELECT {} FROM blobs WHERE confirmed = ?".format(_COLS_BLOBS), 
        (confirmed, ))
    return _parse_blobs(cur.fetchall())

def verification_stats(conn, cur):
    # selects experiment based on command-line arg and gathers all ROIs
    # and blobs within them
    exp = select_experiment(cur, os.path.basename(config.filename))
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
        # true pos, where radius <= 0 indicates that the blob was manually 
        # added, not detected
        blobs_true_detected = blobs_true[blobs_true[:, 3] < config.POS_THRESH]
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

def get_roi_offset(roi):
    return (roi["offset_x"], roi["offset_y"], roi["offset_z"])

def get_roi_size(roi):
    return (roi["size_x"], roi["size_y"], roi["size_z"])

def match_elements(src, delim, repeat):
    src_split = src.split(delim)
    return delim.join([repeat] * len(src_split))



def _merge_dbs(db_paths, db_merged=None):
    if db_merged is None:
        db_merged = ClrDB()
        db_merged.load_db(DB_NAME_MERGED, True)
    for db_path in db_paths:
        print("merging in database from {}".format(db_path))
        db = ClrDB()
        db.load_db(db_path, False)
        exps = select_experiment(db.cur, None)
        for exp in exps:
            exp_id = select_or_insert_experiment(
                db_merged.conn, db_merged.cur, exp["name"], exp["date"])
            rois = select_rois(db.cur, exp["id"])
            for roi in rois:
                roi_id, _ = insert_roi(
                    db_merged.conn, db_merged.cur, exp_id, roi["series"], 
                    get_roi_offset(roi), get_roi_size(roi))
                blobs = select_blobs(db.cur, roi["id"])
                insert_blobs(db_merged.conn, db_merged.cur, roi_id, blobs)
        exps_len = 0 if exps is None else len(exps)
        print("imported {} experiments from {}".format(exps_len, db_path))
    return db_merged

def merge_truth_dbs(img_paths):
    db_merged = None
    for img_path in img_paths:
        print(os.path.basename(img_path.rsplit(".", 1)[0]))
        db_paths = glob.glob("./{}*{}".format(
            os.path.basename(img_path.rsplit(".", 1)[0]), DB_SUFFIX_TRUTH))
        db_merged = _merge_dbs(db_paths, db_merged)

def clean_up_blobs(db):
    """Clean up blobs from pre-v.0.5.0, where user-added radii have neg 
    values rather than 0.0, and remove all unconfirmed blobs since 
    dragging/dropping decreases the number of delete/adds required for 
    proper blob placement.
    
    Args:
        db: Database to clean up, typically a truth database.
    """
    exps = select_experiment(db.cur, None)
    for exp in exps:
        exp_id = select_or_insert_experiment(
            db.conn, db.cur, exp["name"], exp["date"])
        rois = select_rois(db.cur, exp["id"])
        for roi in rois:
            roi_id = roi["id"]
            print("cleaning ROI {}".format(roi_id))
            blobs = select_blobs(db.cur, roi_id)
            #print("blobs:\n{}".format(blobs))
            del_mask = blobs[:, 4] != 1
            delete_blobs(db.conn, db.cur, roi_id, blobs[del_mask])
            blobs_confirmed = blobs[np.logical_not(del_mask)]
            blobs_confirmed[np.isclose(blobs_confirmed[:, 3], 0), 3] = -5
            insert_blobs(db.conn, db.cur, roi_id, blobs_confirmed)
        print("updated experiment {}".format(exp["name"]))

def _test_db():
    # simple database test
    conn, cur = start_db()
    exp_name = "TextExp"
    exp_id = select_or_insert_experiment(conn, cur, exp_name, datetime.datetime(1000, 1, 1))
    insert_blobs(conn, cur, exp_id, 12, [[3, 2, 5, 23.4], [2, 3, 7, 13.2]])
    conn.commit()
    conn.close()

def load_db(path):
    """Load a database from an existing path, raising an exception if 
    the path does not exist.
    
    Args:
        path: Path from which to load a database.
    
    Returns:
        The :class:``ClrDB`` database at the given location.
    
    Raises:
        :class:``FileNoutFoundError`` if ``path`` is not found.
    """
    # TODO: consider integrating with ClrDB directly
    if not os.path.exists(path):
        raise FileNotFoundError("{} not found for DB".format(path))
    print("loading DB from {}".format(path))
    db = ClrDB()
    db.load_db(path, False)
    return db

def load_truth_db(filename_base):
    """Convenience function to load a truth database associated with an 
    image.
    
    Args:
        filename_base: MagellanMapper-oriented base name associated with an 
            image path.
    
    Returns:
        The :class:``ClrDB`` truth database associated with the image.
    """
    path = filename_base
    if not filename_base.endswith(DB_SUFFIX_TRUTH):
        path = os.path.basename(filename_base + DB_SUFFIX_TRUTH)
    truth_db = load_db(path)
    truth_db.load_truth_blobs()
    config.truth_db = truth_db
    return truth_db

class ClrDB():
    conn = None
    cur = None
    blobs_truth = None
    
    def load_db(self, path, new_db):
        self.conn, self.cur = start_db(path, new_db)
    
    def load_truth_blobs(self):
        self.blobs_truth = select_blobs_confirmed(self.cur, 1)
        libmag.printv("truth blobs:\n{}".format(self.blobs_truth))
    
    def get_rois(self, filename):
        exps = select_experiment(self.cur, filename)
        rois = None
        if len(exps) > 0:
            rois = select_rois(self.cur, exps[0]["id"])
        return rois

def main():
    """Run main SQLite access commands after loading CLI."""
    # parses arguments and sets up the DB
    from magmap.io import cli
    cli.main(True)
    conn = config.db.conn
    cur = config.db.cur
    if config.verified_db is not None:
        conn = config.verified_db.conn
        cur = config.verified_db.cur
    
    #verification_stats(conn, cur)
    #update_rois(cur, cli.offset, cli.roi_size)
    #merge_truth_dbs(config.filenames)
    #clean_up_blobs(config.truth_db)
    #_update_experiments(config.filename)

if __name__ == "__main__":
    print("Starting sqlite.py...")
    main()
    