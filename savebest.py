from ModelController import ModelController
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import scipy.sparse as sps
controller = ModelController()
slim = SLIMElasticNetRecommender(controller.URM_boost)
slim.fit(alpha= 0.00022742003969239836, topK= 709, l1_ratio =0.1488442906776265)
slim.save_model(folder_path="_saved_models", file_name="slimboost")