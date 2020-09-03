#!/usr/bin/env python
# SQLite database connection
# Author: David Young, 2017, 2020
"""Connects with a SQLite database for experiment and image
analysis storage.
"""

import os
import glob
import datetime
import sqlite3
import numpy as np

from magmap.settings import config
from magmap.cv import detector
from magmap.io import df_io
from magmap.io import libmag

DB_NAME_BASE = "magmap"
DB_NAME_VERIFIED = "{}_verified.db".format(DB_NAME_BASE)
DB_NAME_MERGED = "{}_merged.db".format(DB_NAME_BASE)
DB_SUFFIX_TRUTH = "_truth.db"
DB_VERSION = 4

_COLS_BLOBS = "roi_id, z, y, x, radius, confirmed, truth, channel"
_COLS_BLOB_MATCHES = "roi_id, blob1, blob2, dist"


class BlobMatch:
    """Blob match storage class."""
    
    def __init__(self, match_id=None, roi_id=None, blob1_id=None, blob1=None,
                 blob2_id=None, blob2=None, dist=None):
        """Initialize blob match object."""
        self.match_id = match_id
        self.roi_id = roi_id
        self.blob1_id = blob1_id
        self.blob1 = blob1
        self.blob2_id = blob2_id
        self.blob2 = blob2
        self.dist = dist
    
    def __repr__(self):
        """Output a string representation with major attributes."""
        return ("match_id {}, roi_id {}, blob1 {}, blob1_id {}, blob2 {},"
                "blob2_id {}, dist {}".format(
                    self.match_id, self.roi_id, self.blob1, self.blob1_id,
                    self.blob2, self.blob2_id, self.dist))
    
    def shift_blobs(self, offset):
        """Shift coordinates of blobs by offset.
        
        Args:
            offset (List[int]): Sequence of coordinates by which to shift
                the corresponding elements from the start of :attr:`blob1`
                and :attr:`blob2`.

        """
        self.blob1 = detector.shift_blob_rel_coords(self.blob1, offset)
        self.blob2 = detector.shift_blob_rel_coords(self.blob2, offset)


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
    _create_table_blob_matches(cur)
    
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


def _create_table_blob_matches(cur):
    cur.execute("CREATE TABLE blob_matches ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "roi_id INTEGER, blob1 INTEGER, blob2 INTEGER, dist REAL, "
                "FOREIGN KEY (roi_id) REFERENCES rois (id) "
                "ON UPDATE CASCADE ON DELETE CASCADE, "
                "FOREIGN KEY (blob1) REFERENCES blobs (id) "
                "ON UPDATE CASCADE ON DELETE CASCADE,"
                "FOREIGN KEY (blob2) REFERENCES blobs (id) "
                "ON UPDATE CASCADE ON DELETE CASCADE)")


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
    
    if db_ver < 4:
        print("upgrading DB version from {} to 4".format(db_ver))
        _create_table_blob_matches(cur)
    
    # record database upgrade version and time
    insert_about(conn, cur, DB_VERSION, datetime.datetime.now())
    
    print("...finished database upgrade.")
    conn.commit()


def start_db(path=None, new_db=False):
    """Starts the database.
    
    Args:
        path (str): Path where the new database resides; if None, defaults to
            :attr:``DB_NAME``.
        new_db (bool): If True or if ``path`` does not exist, a new database
            will  be created; defaults to False.
    
    Returns:
        conn: The connection.
        cur: Connection's cursor.
    """
    if path is None:
        path = config.db_name
    if new_db or not os.path.exists(path):
        conn, cur = _create_db(path)
        print("Created a new database at {}".format(path))
    else:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        print("Loaded database from {}".format(path))
    # add foreign key constraint support
    conn.execute("PRAGMA foreign_keys=ON")
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


def select_experiment(cur, name=None):
    """Selects an experiment from the given name.
    
    Args:
        cur (:obj:`sqlite3.Cursor): Connection's cursor.
        name (str): Name of the experiment. Defaults to None to get all
            experiments.
    
    Returns:
        List[:obj:`sqlite3.Row`]: Sequence of all the experiment rows with
        the given name.
    
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
        conn (:obj:`sqlite3.Connection): Connection object.
        cur (:obj:`sqlite3.Cursor): Cursor object.
        exp_id (int): ID of the experiment.
        series (int): Series within the experiment. Can be None, in which case
            the series will be set to 0.
        offset (List[int): ROI offset as (x, y, z).
        size (List[int): ROI size as (x, y, z)
    
    Returns:
        int, str: ID of the selected or inserted row and the feedback string.
    """
    if series is None:
        series = 0
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
        conn (:obj:`sqlite3.Connection): Connection object.
        cur (:obj:`sqlite3.Cursor): Cursor object.
        exp_id (int): ID of the experiment.
        series (int): Series within the experiment. Can be None, in which
            case all series will be allowed.
        offset (List[int): ROI offset as (x, y, z).
        size (List[int): ROI size as (x, y, z)
    
    Returns:
        int, str: ID of the selected or inserted row and feedback on the result.
    """
    stmnt = ("SELECT * FROM rois WHERE experiment_id = ? "
             "AND offset_x = ? AND offset_y = ? AND offset_z = ? AND "
             "size_x = ? AND size_y = ? AND size_z = ?")
    roi_args = [exp_id, *offset, *size]
    if series is not None:
        # specify series
        stmnt += " AND series = ?"
        roi_args.append(series)
    cur.execute(stmnt, roi_args)
    row = cur.fetchone()
    if row is not None and len(row) > 0:
        print("selected ROI {}".format(row[0]))
        return row[0], "Found ROI {}".format(row[0])
    else:
        return insert_roi(conn, cur, exp_id, series, offset, size)


def select_rois(cur, exp_id):
    """Selects ROIs from the given experiment.
    
    Args:
        cur (:obj:`sqlite3.Cursor): Connection's cursor.
        exp_id (int): ID of the experiment.
    
    Returns:
        List[:obj:`sqlite3.Row`]: Sequence of all ROIs for the experiment.
    
    """
    cur.execute("SELECT id, experiment_id, series, "
                "offset_x, offset_y, offset_z, size_x, size_y, "
                "size_z FROM rois WHERE experiment_id = ?", (exp_id, ))
    rows = cur.fetchall()
    return rows


def update_rois(conn, cur, offset, size):
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
        roi_id (int): ROI ID.
        blobs: Array of blobs arrays, assumed to be formatted according to
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
            confirmed += 1
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
        roi_id (int): ROI ID.
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
        count = cur.rowcount
        if count > 0:
            deleted += count
            print("deleted blob {}".format(blob))
    print("{} blob(s) deleted".format(deleted))
    conn.commit()
    return deleted


def _parse_blobs(rows):
    """Parse blobs to Numpy arrow.
    
    Args:
        rows (List[:obj:`sqlite3.Row`]): Sequence of rows.

    Returns:
        :obj:`np.ndarray`, List[int]: Blobs as a Numpy array. List of
        blob IDs if available.

    """
    blobs = np.empty((len(rows), 7))
    ids = []
    for i, row in enumerate(rows):
        blobs[i] = [
            row["z"], row["y"], row["x"], row["radius"], row["confirmed"], 
            row["truth"], row["channel"]
        ]
        if "id" in row.keys():
            ids.append(row["id"])
    return blobs, ids


def select_blobs(cur, roi_id):
    """Selects blobs from the given ROI.
    
    Args:
        cur: Connection's cursor.
        roi_id (int): ROI ID.
    
    Returns:
        Blobs in the given ROI.
    """
    cur.execute(
        "SELECT {} FROM blobs WHERE roi_id = ?".format(_COLS_BLOBS), (roi_id, ))
    return _parse_blobs(cur.fetchall())[0]


def select_blobs_confirmed(cur, confirmed):
    """Selects ROIs from the given experiment
    
    Args:
        cur: Connection's cursor.
        confirmed (int): Blob confirmation status.
    
    Returns:
        Blobs in the given ROI.
    """
    cur.execute(
        "SELECT {} FROM blobs WHERE confirmed = ?".format(_COLS_BLOBS), 
        (confirmed, ))
    return _parse_blobs(cur.fetchall())[0]


def verification_stats(cur, exp_name, treat_maybes=0):
    """Calculate accuracy metrics based on blob verification status in the
    database.

    Args:
        cur (:obj:`sqlite3.Cursor): Database cursor.
        exp_name (str): Experiment name in the database.
        treat_maybes (int): Pass to :meth:`detector.meas_detection_accurarcy`
            for how to treat maybe flags.

    Returns:
        float, float, str: Output from :meth:`detector.meas_detection_accurarcy`
        for all blobs in an experiment matching ``exp_name``.

    """
    # selects experiment based on command-line arg and gathers all ROIs
    # and blobs within them
    exp = select_experiment(cur, exp_name)
    rois = select_rois(cur, exp[0][0])
    blobs = []
    for roi in rois:
        #print("got roi_id: {}".format(roi[0]))
        bb = select_blobs(cur, roi[0])
        blobs.extend(bb)
    blobs = np.array(blobs)
    return detector.meas_detection_accuracy(
        blobs, config.verified_db is not None, treat_maybes)


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


def load_truth_db(filename_base):
    """Convenience function to load a truth database associated with an 
    image.
    
    Args:
        filename_base: MagellanMapper-oriented base name associated with an 
            image path. If the file is not found, :const:``DB_SUFFIX_TRUTH``
            will be appended and directories removed.
    
    Returns:
        The :class:``ClrDB`` truth database associated with the image.
    """
    path = filename_base
    if not os.path.exists(path):
        # convention has been to save truth databases with suffix in the
        # working directory; TODO: consider removing
        if not path.endswith(DB_SUFFIX_TRUTH):
            path += DB_SUFFIX_TRUTH
        path = os.path.basename(path)
    print("Loading truth DB...")
    config.truth_db = ClrDB()
    config.truth_db.load_db(path)
    config.truth_db.load_truth_blobs()
    return config.truth_db


class ClrDB:
    """MagellanMapper experiment database handler.

    Stores detection related data for ground truth sets, automated detections,
    and their comparisons.

    Attributes:
        conn (:obj:`sqlite3.Connection): Connection object.
        cur (:obj:`sqlite3.Cursor): Cursor object.
        blobs_truth (List[int]): Truth blobs list as returned by
            :meth:`select_blobs_confirmed`.
        path (str): Path from which the database was loaded.

    """
    conn = None
    cur = None
    blobs_truth = None

    def __init__(self):
        """Initialize a MagellanMapper experiment database."""
        self.path = None
    
    def load_db(self, path, new_db=False):
        """Load a database from an existing path, raising an exception if
        the path does not exist.

        Args:
            path (str): Path from which to load a database.
            new_db (bool): If True or if ``path`` does not exist, a new
                database will  be created; defaults to False.

        Returns:
            The :class:``ClrDB`` database at the given location.

        Raises:
            :class:``FileNoutFoundError`` if ``path`` is not found.
        """
        if not new_db and path and not os.path.exists(path):
            raise FileNotFoundError("{} not found for DB".format(path))
        self.path = path
        self.conn, self.cur = start_db(path, new_db)
    
    def load_truth_blobs(self):
        self.blobs_truth = select_blobs_confirmed(self.cur, 1)
        libmag.printv("truth blobs:\n{}".format(self.blobs_truth))
    
    def get_rois(self, exp_name, ignore_ext=True):
        """Get all ROIs for the given experiment.
        
        Args:
            exp_name (str): Name of experiment.
            ignore_ext (bool): True to ignore any extension in ``exp_name``
                and experiment names stored in the database; defaults to True.

        Returns:
            List[:obj:`sqlite3.Row`]: Sequence of all ROIs for the experiment.

        """
        # get all experiments
        exps = select_experiment(self.cur)
        if ignore_ext:
            exp_name = os.path.splitext(exp_name)[0]
        for exp in exps:
            # check for matching experiment by name
            name = exp["name"]
            if ignore_ext:
                name = os.path.splitext(name)[0]
            if name == exp_name:
                return select_rois(self.cur, exp["id"])
        return None
    
    @staticmethod
    def _get_blob(rows):
        """Get a single blob from a row.
        
        Args:
            rows (List[:obj:`sqlite3.Row`]): Sequence of rows.

        Returns:
            :obj:`np.ndarray`, int: The blob as a 1D array. The blob ID if
            available. If more than one row is given, only the first row's
            contents is returned.

        """
        blobs, ids = _parse_blobs(rows)
        if len(blobs) > 0:
            blob_id = ids[0] if ids else None
            return blobs[0], blob_id
        return None, None
    
    def select_blob(self, roi_id, blob):
        """Select a blob based on position and status.
        
        Args:
            roi_id (int): ROI ID.
            blob (:obj:`np.ndarray`): Blob as a 1D array of
                ``z, y, x, r, confirmed, truth, channel``.

        Returns:
            :obj:`np.ndarray`, int: The blob as a 1D array. The blob ID if
            available. If more than one blow is found, only the first blob
            is returned.

        """
        self.cur.execute(
            "SELECT {}, id FROM blobs WHERE roi_id = ? AND z = ? AND y = ? "
            "AND x = ? AND confirmed = ? AND truth = ? AND channel = ?"
            .format(_COLS_BLOBS), (roi_id, *blob[:3], *blob[4:7]))
        return self._get_blob(self.cur.fetchall())

    def select_blob_by_id(self, blob_id):
        """Select a blob by ID.
        
        Args:
            blob_id (int): Blob ID.

        Returns:
            :obj:`np.ndarray`, int: The blob as a 1D array.

        """
        self.cur.execute(
            "SELECT {}, id FROM blobs WHERE id = ?"
                .format(_COLS_BLOBS), (blob_id,))
        return self._get_blob(self.cur.fetchall())

    def insert_blob_matches(self, roi_id, matches):
        """Insert blob matches.
        
        Args:
            roi_id (int): ROI ID.
            matches: Blob matches.

        Returns:
            List[int]: List of blob match IDs.

        """
        ids = []
        for match in matches:
            blob1, blob1_id = self.select_blob(roi_id, match[0])
            blob2, blob2_id = self.select_blob(roi_id, match[1])
            if blob1_id and blob2_id:
                self.cur.execute(
                    "INSERT INTO blob_matches ({}) "
                    "VALUES (?, ?, ?, ?)".format(_COLS_BLOB_MATCHES),
                    (roi_id, blob1_id, blob2_id, match[2]))
                ids.append(self.cur.lastrowid)
                print("Blob match inserted for ROI ID {}, blob 1 ID {}, "
                      "blob 2 ID {}".format(roi_id, blob1_id, blob2_id))
            else:
                print("Could not find blobs for match:", match)
        self.conn.commit()
        return ids
    
    def select_blob_matches(self, roi_id):
        """Select blob matches for the given ROI.
        
        Args:
            roi_id (int): ROI ID.

        Returns:
            List[:obj:`BlobMatch`]: List of blob matches.

        """
        self.cur.execute(
            "SELECT {}, id FROM blob_matches WHERE roi_id = ?"
            .format(_COLS_BLOB_MATCHES), (roi_id,))
        matches = []
        for row in self.cur.fetchall():
            blob1_id = row["blob1"]
            blob2_id = row["blob2"]
            matches.append(BlobMatch(
                row["id"], row["roi_id"], blob1_id,
                self.select_blob_by_id(blob1_id)[0], blob2_id,
                self.select_blob_by_id(blob2_id)[0], row["dist"]))
        return matches
        

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
    
    #print(verification_stats(cur, os.path.basename(config.filename))[2])
    #update_rois(cur, cli.offset, cli.roi_size)
    #merge_truth_dbs(config.filenames)
    #clean_up_blobs(config.truth_db)
    #_update_experiments(config.filename)


if __name__ == "__main__":
    print("Starting sqlite.py...")
    main()
