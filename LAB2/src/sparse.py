"""A minimal CSR (compressed sparse row) matrix, built from scratch with NumPy only.

We avoid scipy.sparse / sklearn entirely: this is just three flat numpy arrays
(data, indices, indptr) plus helpers to build one row-by-row and to slice
rows out for mini-batch training.
"""
import numpy as np


class CSRMatrix:
    def __init__(self, data, indices, indptr, n_cols):
        self.data = np.asarray(data, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.n_cols = n_cols

    @property
    def n_rows(self):
        return len(self.indptr) - 1

    @property
    def shape(self):
        return (self.n_rows, self.n_cols)

    @classmethod
    def from_rows(cls, rows, n_cols):
        """rows: list of dicts {col_index: value}"""
        data, indices, indptr = [], [], [0]
        for row in rows:
            items = sorted(row.items())
            for col, val in items:
                indices.append(col)
                data.append(val)
            indptr.append(len(indices))
        return cls(data, indices, indptr, n_cols)

    def get_row(self, i):
        s, e = self.indptr[i], self.indptr[i + 1]
        return self.indices[s:e], self.data[s:e]

    def row_nnz(self):
        return np.diff(self.indptr)

    def to_padded_batch(self, row_ids, pad_index, extra_index=None, extra_value=None):
        """Build a dense (B, maxnnz) pair of (indices, data) for the given rows,
        so a sparse-dense product can be done as one vectorized gather+multiply.

        If extra_index is given, an extra always-on feature (e.g. a bias term)
        with value extra_value is appended to every row before padding.
        """
        n_extra = 1 if extra_index is not None else 0
        lengths = (self.indptr[row_ids + 1] - self.indptr[row_ids]) + n_extra
        max_len = int(lengths.max()) if len(lengths) else 0
        B = len(row_ids)

        idx_out = np.full((B, max_len), pad_index, dtype=np.int32)
        val_out = np.zeros((B, max_len), dtype=np.float32)

        for k, r in enumerate(row_ids):
            s, e = self.indptr[r], self.indptr[r + 1]
            nnz = e - s
            idx_out[k, :nnz] = self.indices[s:e]
            val_out[k, :nnz] = self.data[s:e]
            if n_extra:
                idx_out[k, nnz] = extra_index
                val_out[k, nnz] = extra_value
        return idx_out, val_out
