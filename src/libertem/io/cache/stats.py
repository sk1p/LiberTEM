from typing import Union
import sqlite3
import time


class VerboseRow(sqlite3.Row):
    """sqlite3.Row with a __repr__"""
    def __repr__(self):
        return "<VerboseRow %r>" % (
            {
                k: self[k]
                for k in self.keys()
            },
        )


class CacheItem:
    """
    A CacheItem describes a single unit of data that is cached, in this case
    a partition of the CachedDataSet.
    """
    def __init__(self, dataset: str, partition: int, size: int, path: str):
        self.dataset = dataset  # dataset id string, for example the cache key
        self.partition = partition  # partition index as integer
        self.size = size  # partition size in bytes
        self.path = path  # full absolute path to the file for the partition
        self.is_orphan = False  # quack

    def __eq__(self, other):
        # dataset and partition are composite pk
        return self.dataset == other.dataset and self.partition == other.partition

    def __repr__(self):
        return "<CacheItem: %s/%d>" % (self.dataset, self.partition)

    @classmethod
    def from_row(cls, row):
        return cls(
            dataset=row["dataset"],
            partition=row["partition"],
            size=row["size"],
            path=row["path"]
        )


class OrphanItem:
    """
    An orphan, a file in the cache structure, which we don't know much about
    (only path and size)
    """
    def __init__(self, path, size):
        self.path = path
        self.size = size
        self.is_orphan = True

    def __eq__(self, other):
        return self.path == other.path

    def __repr__(self):
        return "<OrphanItem: %s>" % (self.path,)

    @classmethod
    def from_row(cls, row):
        return cls(
            size=row["size"],
            path=row["path"]
        )


class CacheStats:
    """
    A helper class for managing cache statistics. It uses sqlite under the hood.
    """
    def __init__(self, db_path):
        self._db_path = db_path
        self._conn = None

    def _connect(self):
        conn = sqlite3.connect(self._db_path, timeout=15000)
        conn.row_factory = VerboseRow
        self._conn = conn

    def close(self):
        self._conn.close()
        self._conn = None

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, *exc):
        self.close()
        self._conn = None

    def initialize_schema(self):
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS stats (
            dataset VARCHAR NOT NULL,           -- dataset id, for example the cache key
            partition INTEGER NOT NULL,         -- partition index as integer
            hits INTEGER NOT NULL,              -- access counter
            size INTEGER NOT NULL,              -- in bytes
            last_access REAL NOT NULL,          -- float timestamp, like time.time()
            path VARCHAR NOT NULL,              -- full path to the file for this partition
            PRIMARY KEY (dataset, partition)
        );""")
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS orphans (
            path VARCHAR NOT NULL,              -- full path to the orphaned file
            size INTEGER NOT NULL,              -- in bytes
            PRIMARY KEY (path)
        );""")
        self._conn.execute("PRAGMA user_version = 1;")
        self._conn.execute("PRAGMA journal_mode = WAL;")

    def _have_item(self, cache_item: CacheItem):
        rows = self._conn.execute("""
        SELECT hits FROM stats
        WHERE dataset = ? AND partition = ?
        """, [cache_item.dataset, cache_item.partition]).fetchall()
        return len(rows) > 0

    def record_hit(self, cache_item: CacheItem):
        now = time.time()
        cursor = self._conn.cursor()
        cursor.execute("BEGIN")
        if not self._have_item(cache_item):
            cursor.execute("""
            INSERT INTO stats (partition, dataset, hits, size, last_access, path)
            VALUES (?, ?, 1, ?, ?, ?)
            """, [cache_item.partition, cache_item.dataset,
                  cache_item.size, now, cache_item.path])
        else:
            cursor.execute("""
            UPDATE stats
            SET hits = MAX(hits + 1, 1), last_access = ?
            WHERE dataset = ? AND partition = ?
            """, [now, cache_item.dataset, cache_item.partition])
        cursor.execute("DELETE FROM orphans WHERE path = ?", [cache_item.path])
        self._conn.commit()
        cursor.close()

    def record_miss(self, cache_item: CacheItem):
        now = time.time()

        cursor = self._conn.cursor()
        cursor.execute("BEGIN")
        if not self._have_item(cache_item):
            cursor.execute("""
            INSERT INTO stats (partition, dataset, hits, size, last_access, path)
            VALUES (?, ?, 0, ?, ?, ?)
            """, [cache_item.partition, cache_item.dataset, cache_item.size,
                  now, cache_item.path])
        else:
            cursor.execute("""
            UPDATE stats
            SET hits = 0, last_access = ?
            WHERE dataset = ? AND partition = ?
            """, [now, cache_item.dataset, cache_item.partition])
        cursor.execute("DELETE FROM orphans WHERE path = ?", [cache_item.path])
        self._conn.commit()
        cursor.close()

    def record_eviction(self, cache_item: Union[CacheItem, OrphanItem]):
        if cache_item.is_orphan:
            self.remove_orphan(cache_item)
        else:
            self._conn.execute("""
            DELETE FROM stats
            WHERE partition = ? AND dataset = ?
            """, [cache_item.partition, cache_item.dataset])
            self._conn.commit()

    def maybe_orphan(self, orphan: OrphanItem):
        """
        Create an entry for a file we don't have any statistics about, after checking
        the stats table for the given path.
        Getting a conflict here means concurrently running maybe_orphan processes,
        so we can safely ignore it.
        """
        exists = len(self._conn.execute("""
        SELECT 1 FROM stats WHERE path = ?
        """, [orphan.path]).fetchall()) > 0
        if not exists:
            self._conn.execute("""
            INSERT OR IGNORE INTO orphans (path, size)
            VALUES (?, ?)
            """, [orphan.path, orphan.size])
            self._conn.commit()
            return orphan

    def get_orphans(self):
        cursor = self._conn.execute("SELECT path, size FROM orphans ORDER BY size DESC")
        return [
            OrphanItem.from_row(row)
            for row in cursor
        ]

    def remove_orphan(self, path: str):
        self._conn.execute("""
        DELETE FROM orphans WHERE path = ?
        """, [path])
        self._conn.commit()

    def get_stats_for_dataset(self, cache_key):
        """
        Return dataset cache stats as dict mapping partition ids to dicts of their
        properties (keys: size, last_access, hits)
        """
        cursor = self._conn.execute("""
        SELECT partition, path, size, hits, last_access
        FROM stats
        WHERE dataset = ?
        """, [cache_key])
        return {
            row["partition"]: self._format_row(row)
            for row in cursor.fetchall()
        }

    def query(self, sql, args=None):
        """
        Custom sqlite query, returns a sqlite3 Cursor object
        """
        if args is None:
            args = []
        return self._conn.execute(sql, args)

    def _format_row(self, row):
        return {
            "path": row["path"],
            "size": row["size"],
            "last_access": row["last_access"],
            "hits": row["hits"],
        }

    def get_used_capacity(self):
        size = self._conn.execute("""
        SELECT SUM(size) AS "total_size" FROM stats;
        """).fetchone()["total_size"]
        size_orphans = self._conn.execute("""
        SELECT SUM(size) AS "total_size" FROM orphans;
        """).fetchone()["total_size"]
        return (size or 0) + (size_orphans or 0)
