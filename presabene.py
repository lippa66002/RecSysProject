import optuna
from scipy.constants import alpha

from Optimize.SaveResults import SaveResults
import optuna
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender
import DataHandler
from ModelController import ModelController
import pandas as pd


URM_all_dataframe = pd.read_csv(filepath_or_buffer="Data/data_train.csv",
                                sep=",",
                                dtype={0:int, 1:int, 2:float},
                                engine='python')
users = pd.read_csv(filepath_or_buffer="Data/data_target_users_test.csv")

ICM = pd.read_csv(filepath_or_buffer="Data/data_ICM_metadata.csv",
                                sep=",",
                                dtype={0:int, 1:int, 2:float},
                                engine='python')

URM_all, ICM_all = DataHandler.create_urm_icm(URM_all_dataframe, ICM)
controller = ModelController(URM_all, ICM_all)
user = UserKNNCFRecommender(controller.URM_train)
user.load_model(folder_path="_saved_models", file_name="UserKNNCFTrain")
item = ItemKNNCFRecommender(controller.URM_train)
item.load_model(folder_path="_saved_models", file_name="ItemKNNCFTrain")
dd,_ = controller.evaluator_test.evaluateRecommender(user)
print(dd.loc[10]["MAP"])
ff, _= controller.evaluator_test.evaluateRecommender(item)
print(ff.loc[10]["MAP"])