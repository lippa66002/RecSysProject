
import numpy as np
import scipy.sparse as sps
from Recommenders.Recommender_utils import check_matrix
from sklearn.linear_model import ElasticNet
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender, create_shared_memory, _partial_fit
from Recommenders.Similarity.Compute_Similarity_Python import Incremental_Similarity_Builder
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
import time, sys
from tqdm import tqdm
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
class MultiThreadSLIM_SLIMElasticNetRecommender(SLIMElasticNetRecommender):
    cdef int

    def fit(self, alpha=1.0, l1_ratio=0.1, positive_only=True, topK=100,
            verbose=True, workers=int(cpu_count()*0.3)):

        assert l1_ratio>= 0 and l1_ratio<=1, \
            "ElasticNet: l1_ratio must be between 0 and 1, provided value was {}".format(l1_ratio)

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK

        self.workers = workers

        self.URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        indptr_shm = create_shared_memory(self.URM_train.indptr)
        indices_shm = create_shared_memory(self.URM_train.indices)
        data_shm = create_shared_memory(self.URM_train.data)

        _pfit = partial(_partial_fit, topK=self.topK, alpha=self.alpha, urm_shape=self.URM_train.shape,
                        l1_ratio=self.l1_ratio, positive_only=self.positive_only,
                        shm_names=[indptr_shm.name, indices_shm.name, data_shm.name],
                        shm_shapes=[self.URM_train.indptr.shape, self.URM_train.indices.shape, self.URM_train.data.shape],
                        shm_dtypes=[self.URM_train.indptr.dtype, self.URM_train.indices.dtype, self.URM_train.data.dtype])

        with Pool(processes=self.workers) as pool:

            pool_chunksize = 4
            item_chunksize = 8

            itemchunks = np.array_split(np.arange(self.n_items), int(self.n_items / item_chunksize))
            if verbose:
                pbar = tqdm(total=self.n_items)

            # res contains a vector of (values, rows, cols) tuples
            values, rows, cols = [], [], []
            for values_, rows_, cols_ in pool.imap_unordered(_pfit, itemchunks, pool_chunksize):
                values.extend(values_)
                rows.extend(rows_)
                cols.extend(cols_)
                if verbose:
                    pbar.update(item_chunksize)

        indptr_shm.close()
        indices_shm.close()
        data_shm.close()

        indptr_shm.unlink()
        indices_shm.unlink()
        data_shm.unlink()

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(self.n_items, self.n_items), dtype=np.float32)
        self.URM_train = self.URM_train.tocsr()
