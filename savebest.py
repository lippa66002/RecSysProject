from ModelController import ModelController
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import scipy.sparse as sps
controller = ModelController()
slim = SLIMElasticNetRecommender(controller.URM_train)
stacked = sps.vstack([0.8718332784366307 * controller.URM_train, (1 - 0.8718332784366307) * controller.ICM_all.T]).tocsr()
slim = SLIMElasticNetRecommender(stacked)
slim.fit(alpha =7.558004074517556e-05, topK=641, l1_ratio= 0.08844974678228634)
slim.load_model(folder_path="_saved_models", file_name="SLIMstackedTrainval2")