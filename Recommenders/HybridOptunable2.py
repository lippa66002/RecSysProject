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


class HybridOptunable2(BaseItemSimilarityMatrixRecommender):
    def __init__(self, URM_train):
        super(HybridOptunable2, self).__init__(URM_train)
        self.W_sparse = None

    def fit(self, alpha, rec1, rec2):
        new_similarity = (1.0 - alpha) * rec1.W_sparse + alpha * rec2.W_sparse
        recommender_object = ItemKNNCustomSimilarityRecommender(self.URM_train)
        recommender_object.fit(new_similarity)
        self.W_sparse = recommender_object.W_sparse



