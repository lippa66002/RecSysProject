from ModelController import ModelController
from ModelNames import ModelName
import scipy.sparse as sps

controller = ModelController()

optuna = controller.optunizer(ModelName.RP3betaRecommender)