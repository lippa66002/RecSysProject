import os

# Torna alla directory principale del progetto
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

from ModelController import ModelController
from ModelNames import ModelName

controller = ModelController()
optuna = controller.optunizer(ModelName.SLIM_ElasticNet)