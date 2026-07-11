"""Tar-based storage helpers for Polars DataFrames.

Provides utilities to store multiple Polars DataFrames inside a single
uncompressed TAR archive (each entry is a Snappy-compressed Parquet file).
"""
import io
import tarfile
from typing import Dict, List, Optional, Union

import polars as pl
import pyarrow.parquet as pq


def append_to_tar(path: str, name: str, df: pl.DataFrame, mode: str = 'a'):
    """Append a Polars DataFrame as a Parquet entry inside a TAR file.

    Parameters
    ----------
    path : str
        Path to the TAR file.
    name : str
        Logical table name (stored as ``{name}.parquet`` inside the TAR).
    df : polars.DataFrame
        The table to store.
    mode : str, default 'a'
        Tar open mode. Use ``'w'`` to overwrite / create a new archive,
        ``'a'`` to append to an existing archive.
    """
    buf = io.BytesIO()
    pq.write_table(df.to_arrow(), buf, compression='snappy')
    data = buf.getvalue()

    with tarfile.open(path, mode) as tf:
        info = tarfile.TarInfo(name=f"{name}.parquet")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))


def read_from_tar(path: str, name: Optional[str] = None
                  ) -> Union[pl.DataFrame, Dict[str, pl.DataFrame]]:
    """Read one table or all tables from a TAR archive.

    Parameters
    ----------
    path : str
        Path to the TAR file.
    name : str, optional
        Logical table name to read (without the ``.parquet`` suffix).
        If ``None``, every table is returned in a dict keyed by name.

    Returns
    -------
    polars.DataFrame or dict[str, polars.DataFrame]
    """
    with tarfile.open(path, 'r') as tf:
        if name is not None:
            f = tf.extractfile(f"{name}.parquet")
            return pl.from_arrow(pq.read_table(io.BytesIO(f.read()))) # type: ignore
        tables: Dict[str, pl.DataFrame] = {}
        for m in tf.getmembers():
            if m.name.endswith('.parquet'):
                n = m.name[:-len('.parquet')]
                f = tf.extractfile(m)
                tables[n] = pl.from_arrow(pq.read_table(io.BytesIO(f.read())))
        return tables


def list_tar_tables(path: str) -> List[str]:
    """Return the list of logical table names stored inside the TAR file."""
    with tarfile.open(path, 'r') as tf:
        return [m.name[:-len('.parquet')] for m in tf.getmembers()
                if m.name.endswith('.parquet')]
