import numpy as np

from ModelController import ModelController
from ModelNames import ModelName
import pandas as pd
import DataHandler
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

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
item = ItemKNNCFRecommender(controller.URM_train)
item.fit(similarity =  "cosine", topK =  8, shrink= 12)
item.save_model(folder_path="_saved_models", file_name = "uaua")


