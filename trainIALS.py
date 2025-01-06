from ModelController import ModelController
from ModelNames import ModelName
import scipy.sparse as sps

from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender

controller = ModelController()

ials = IALSRecommender(controller.URM_train)
ials.fit(confidence_scaling= 'linear', num_factors=173, alpha= 6.7979092186218715, epsilon= 1.3000803763047268, reg=0.03413944416805373, init_std= 0.06581036009302164)
ials.save_model(folder_path="_saved_models", file_name="IALS_train")