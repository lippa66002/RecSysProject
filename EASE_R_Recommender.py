#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""
from scipy.sparse.linalg import spsolve

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Recommender_utils import similarityMatrixTopK, check_matrix
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
from sklearn.preprocessing import normalize
import numpy as np
import time
import scipy.sparse as sps
from scipy.sparse.linalg import inv
from scipy.sparse import diags
from Recommenders.Similarity.Compute_Similarity import Compute_Similarity


class EASE_R_Recommender(BaseItemSimilarityMatrixRecommender):
    """ EASE_R_Recommender

        https://arxiv.org/pdf/1905.03375.pdf

     @article{steck2019embarrassingly,
      title={Embarrassingly Shallow Autoencoders for Sparse Data},
      author={Steck, Harald},
      journal={arXiv preprint arXiv:1905.03375},
      year={2019}
    }

    """

    RECOMMENDER_NAME = "EASE_R_Recommender"


    def __init__(self, URM_train, sparse_threshold_quota = None):
        super(EASE_R_Recommender, self).__init__(URM_train)
        self.sparse_threshold_quota = sparse_threshold_quota

    from scipy.sparse.linalg import spsolve

    def fit(self, topK=None, l2_norm=1e3, normalize_matrix=False, verbose=True):

        self.verbose = verbose

        start_time = time.time()
        self._print("Fitting model... ")

        if normalize_matrix:
            # Normalize rows and then columns
            self.URM_train = normalize(self.URM_train, norm='l2', axis=1)
            self.URM_train = normalize(self.URM_train, norm='l2', axis=0)
            self.URM_train = sps.csr_matrix(self.URM_train)

        # Grahm matrix is X^t X, compute dot product
        similarity = Compute_Similarity(
            self.URM_train, shrink=0, topK=100, normalize=False, similarity="cosine"
        )

        # Compute similarity and apply Top-K in sparse format
        mat = similarity.compute_similarity()
        grahm_matrix_sparse = similarityMatrixTopK(mat, k=100, verbose=False)

        # Add diagonal terms (item popularity + l2_norm)
        diag_indices = np.arange(grahm_matrix_sparse.shape[0])
        item_popularity = np.ediff1d(self.URM_train.tocsc().indptr)
        grahm_matrix_sparse = grahm_matrix_sparse.tolil()  # Make it modifiable
        grahm_matrix_sparse[diag_indices, diag_indices] += item_popularity + l2_norm
        grahm_matrix_sparse = grahm_matrix_sparse.tocsr()  # Convert back to CSR format

        # Instead of computing the inverse, solve the linear system
        identity = sps.identity(grahm_matrix_sparse.shape[0], format='csr')
        P_sparse = spsolve(grahm_matrix_sparse, identity)  # Solving Ax = I instead of inverting

        # Normalize rows of P_sparse to get B_sparse
        B_sparse = sps.csr_matrix(P_sparse)
        diag_P = B_sparse.diagonal()  # Extract diagonal as array
        B_sparse = B_sparse.multiply(-1 / diag_P[:, None])  # Row-wise division
        B_sparse.setdiag(0.0)  # Set diagonal to zero

        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time() - start_time)
        self._print("Fitting model... done in {:.2f} {}".format(new_time_value, new_time_unit))

        # Apply Top-K to B_sparse if topK is specified
        if topK is not None:
            B_sparse = similarityMatrixTopK(B_sparse, k=topK, verbose=False)

        # Save the final sparse matrix
        self.W_sparse = B_sparse

    def _is_content_sparse_check(self, matrix):

        if self.sparse_threshold_quota is None:
            return False

        if sps.issparse(matrix):
            nonzero = matrix.nnz
        else:
            nonzero = np.count_nonzero(matrix)

        return nonzero / (matrix.shape[0]**2) <= self.sparse_threshold_quota



    def _compute_score_W_dense(self, user_id_array, items_to_compute = None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse)#.toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse)#.toarray()

        return item_scores





    def load_model(self, folder_path, file_name = None):
        super(EASE_R_Recommender, self).load_model(folder_path, file_name = file_name)

        if not sps.issparse(self.W_sparse):
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense