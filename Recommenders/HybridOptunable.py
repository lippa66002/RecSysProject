#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender, \
    BaseSimilarityMatrixRecommender


class HybridOptunable(BaseSimilarityMatrixRecommender):
    def __init__(self, URM_train, verbose=True):
        super(HybridOptunable, self).__init__(URM_train, verbose = verbose)

        self._URM_train_format_checked = False
        self._W_sparse_format_checked = False



    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse).toarray()

        return item_scores



    def fit(self, alpha,rec1, rec2):
        new_similarity = (1.0 - alpha) * rec1.W_sparse + alpha * rec2.W_sparse
        recommender_object = ItemKNNCustomSimilarityRecommender(self.URM_train)
        recommender_object.fit(new_similarity)



