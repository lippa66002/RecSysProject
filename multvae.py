import optuna
import pandas as pd
import scipy.sparse as sps
import zipfile

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Optimize.SaveResults import SaveResults
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

with zipfile.ZipFile("matrici_sparse.zip", "r") as zipf:
    zipf.extract("URM_trainval.npz")
    zipf.extract("URM_test.npz")
    zipf.extract("URM_train.npz")
    zipf.extract("URM_validation.npz")
    zipf.extract("ICM_all.npz")
    zipf.close()

# Carica la matrice sparse
URM_trainval = sps.load_npz("URM_trainval.npz")
URM_train = sps.load_npz("URM_train.npz")
URM_test = sps.load_npz("URM_test.npz")
URM_validation = sps.load_npz("URM_validation.npz")
ICM_all = sps.load_npz("ICM_all.npz")

mult = MultVAERecommender(URM_trainval)
mult.fit()