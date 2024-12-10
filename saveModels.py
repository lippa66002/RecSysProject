import optuna


import DataHandler
from ModelController import ModelController
import pandas as pd
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender



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
evaluator_validation = EvaluatorHoldout(controller.URM_test, cutoff_list=[10])

URM_all, ICM_all = DataHandler.create_urm_icm(URM_all_dataframe, ICM)

controller = ModelController(URM_all, ICM_all)

'''itemKnnBoost = ItemKNNCFRecommender(controller.URM_train_boost)
itemKnnBoost.fit(topK= 10, shrink= 0, similarity= 'cosine')
itemKnnBoost.save_model(folder_path="_saved_models", file_name = "ItemKNNCFBoost")
itemknnTrain = ItemKNNCFRecommender(controller.URM_train)
itemKnnBoost.fit(topK= 10, shrink= 0, similarity= 'cosine')
itemKnnBoost.save_model(folder_path="_saved_models", file_name = "ItemKNNCFTrain")
itemknnAll = ItemKNNCFRecommender(controller.URM_all)
itemKnnBoost.fit(topK= 10, shrink= 0, similarity= 'cosine')
itemKnnBoost.save_model(folder_path="_saved_models", file_name = "ItemKNNCFAll")'''
itemknnTrain = ItemKNNCFRecommender(controller.URM_train)
itemknnTrain.load_model(folder_path="_saved_models", file_name="ItemKNNCFTrain")
result_df, _ = evaluator_validation.evaluateRecommender(itemknnTrain)
print(result_df.loc[10]["MAP"])
res,_ = controller.evaluator_test.evaluateRecommender(itemknnTrain)
print (res.loc[10]["MAP"])
