# ModelEnum.py
from enum import Enum

from Recommenders.HybridOptunable2 import HybridOptunable2


class ModelName(Enum):
    SLIM_ElasticNet = "SLIM_ElasticNet"
    ItemKNNCFRecommender = "ItemKNNCFRecommender"
    RP3betaRecommender = "RP3betaRecommender"
    ContentBasedRecommender = "ContentBasedRecommender"
    HybridOptunable2 = "HybridOptunable2"
    EASE_R_Recommender = "EASE_R_Recommender"
    MultVAERecommender_PyTorch = "MultVAERecommender_PyTorch"
    UserKNNCFRecommender = "UserKNNCFRecommender"
    Hybrid_ItemKNN_CF_CBF = "Hybrid_ItemKNN_CF_CBF"
    ItemKNNSimilarityHybridRecommender = "ItemKNNSimilarityHybridRecommender"