import os
import sys

# Aggiungi la directory principale al percorso dei moduli
sys.path.append(os.path.abspath(".."))

from ModelController import ModelController
from ModelNames import ModelName

controller = ModelController()
optuna = controller.optunizer(ModelName.SLIM_ElasticNet)