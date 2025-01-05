from ModelController import ModelController
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import scipy.sparse as sps
controller = ModelController()
stacked = sps.vstack([0.8392863849420211 * controller.URM_train, (1 - 0.8392863849420211) * controller.ICM_all.T]).tocsr()
slim = SLIMElasticNetRecommender(stacked)
slim.fit(alpha =5.632458754549518e-05, topK=619, l1_ratio= 0.053794482642909716)
slim.save_model(folder_path="_saved_models", file_name="slimbooststacked3")