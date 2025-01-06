import pandas as pd

import DataHandler
from ModelController import ModelController
from Recommenders.Neural.MultVAE_PyTorch_Recommender import MultVAERecommender_PyTorch

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
controller = ModelController()

mult = MultVAERecommender_PyTorch(controller.URM_train, controller.ICM_all)
mult.fit()
cc,_ = controller.evaluator_test.evaluateRecommender(mult)
print(cc.loc[10]["MAP"])
