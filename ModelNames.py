# ModelEnum.py
from enum import Enum

from Recommenders.HybridOptunable2 import HybridOptunable2


class ModelName(Enum):
    SLIM_ElasticNet = "SLIM_ElasticNet" #teniamo
    ItemKNNCFRecommender = "ItemKNNCFRecommender" #teniamo
    RP3betaRecommender = "RP3betaRecommender" #teniamo
    ContentBasedRecommender = "ContentBasedRecommender"
    HybridOptunable2 = "HybridOptunable2"
    EASE_R_Recommender = "EASE_R_Recommender"
    MultVAERecommender_PyTorch = "MultVAERecommender_PyTorch"
    UserKNNCFRecommender = "UserKNNCFRecommender"
    Hybrid_ItemKNN_CF_CBF = "Hybrid_ItemKNN_CF_CBF"#teniamo
    ItemKNNSimilarityHybridRecommender = "ItemKNNSimilarityHybridRecommender"
    SLIM_BPR_Recommender = "SLIM_BPR_Recommender"
    P3alphaRecommender = "P3alphaRecommender"
    PureSVDRecommender = "PureSVDRecommender"
    ScoresHybridRecommender = "ScoresHybridRecommender"
    DifferentLossScoresHybridRecommender = "DifferentLossScoresHybridRecommender"
    CFW_D_Similarity_Cython = "CFW_D_Similarity_Cython"
    MatrixFactorization_Cython_Recommender = "MatrixFactorization_Cython_Recommender"
    IALSRecommender = "IALSRecommender"


