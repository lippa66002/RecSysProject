#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Recommender_utils import similarityMatrixTopK, check_matrix
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
from sklearn.preprocessing import normalize
import numpy as np
import time
import scipy.sparse as sps

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

    def fit(self, df, lambda_: float = 0.5, implicit=True, topK=None, normalize_matrix=False, verbose=True):
        """
        df: pandas.DataFrame with columns user_id, item_id and (rating)
        lambda_: L2 regularization term
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        topK: number of top-K items to use for similarity computation
        normalize_matrix: whether to normalize the matrix or not
        verbose: whether to print progress
        """
        self.verbose = verbose

        start_time = time.time()
        self._print("Fitting model... ")

        # Step 1: Normalize the URM if required
        if normalize_matrix:
            # Normalize rows and then columns (L2 normalization)
            self.URM_train = normalize(self.URM_train, norm='l2', axis=1)
            self.URM_train = normalize(self.URM_train, norm='l2', axis=0)

        # Step 2: Compute similarity matrix using the user-item matrix
        similarity = Compute_Similarity(self.URM_train, shrink=0, topK=topK, normalize=False, similarity="cosine")
        print(type(similarity))  # To check if it's a Compute_Similarity object

        mat = similarity.compute_similarity()

        # Step 3: Apply Top-K filtering to the similarity matrix (keep only top-K neighbors)
        mat = similarityMatrixTopK(mat, k=topK, verbose=False)
        print(f"Matrice sparsa (top-K): {mat.shape}")

        # Step 4: Operazioni sparse - No conversion to dense matrix
        # Compute the Graham matrix (dot product of the similarity matrix)
        grahm_matrix = mat.T.dot(mat)  # mat^T * mat to get the Graham matrix (item-item similarity)

        # Step 5: Modify diagonal of Graham matrix with item popularity + lambda_
        diag_indices = np.diag_indices(grahm_matrix.shape[0])
        item_popularity = np.ediff1d(self.URM_train.tocsc().indptr)
        grahm_matrix[diag_indices] = item_popularity + lambda_

        # Step 6: Calculate the inverse of the Graham matrix
        P = sps.linalg.inv(grahm_matrix)  # Use sparse matrix inverse for memory efficiency

        # Step 7: Calculate B matrix
        B = P / (-np.diag(P).reshape(-1, 1))  # Divide each row by its diagonal value
        B[diag_indices] = 0  # Set diagonal to zero to avoid self-interaction effects

        # Step 8: Save matrix B as an attribute
        self.B = B

        # Step 9: Compute the predictions matrix
        # Multiply the similarity matrix by B to get the predictions
        self.pred = mat.dot(B)

        # Print fitting time
        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time() - start_time)
        self._print("Fitting model... done in {:.2f} {}".format(new_time_value, new_time_unit))

        # Check if the matrix should be saved in a sparse or dense format
        # The matrix is sparse, regardless of the presence of the topK, if nonzero cells are less than sparse_threshold_quota %
        if topK is not None:
            B = similarityMatrixTopK(B, k = topK, verbose = False)


        if self._is_content_sparse_check(B):
            self._print("Detected model matrix to be sparse, changing format.")
            self.W_sparse = check_matrix(B, format='csr', dtype=np.float32)

        else:
            self.W_sparse = check_matrix(B, format='npy', dtype=np.float32)
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense
        #
        #
        # if topK is None:
        #     self.W_sparse = B
        #     self._W_sparse_format_checked = True
        #     self._compute_item_score = self._compute_score_W_dense
        #
        # else:
        #     self.W_sparse = similarityMatrixTopK(B, k = topK, verbose = False)
        #     self.W_sparse = sps.csr_matrix(self.W_sparse)


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