# ModelEnum.py
from enum import Enum

class ModelName(Enum):
    SLIM_ElasticNet = "SLIM_ElasticNet"
    ItemKNNCFRecommender = "ItemKNNCFRecommender"
    RP3betaRecommender = "RP3betaRecommender"
    ContentBasedRecommender = "ContentBasedRecommender"