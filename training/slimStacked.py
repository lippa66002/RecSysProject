from Optimize.SaveResults import SaveResults
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from ModelController import ModelController
import scipy.sparse as sps
import optuna

controller = ModelController()

slim1 = SLIMElasticNetRecommender(controller.URM_train)
slim1.load_model(folder_path="../_saved_models", file_name="SLIMtrainval")
dd, _ = controller.evaluator_test.evaluateRecommender(slim1)
print(dd.loc[10]["MAP"])
def objective_function_SLIM(optuna_trial):
    # Aggiunta di gamma ai parametri del trial
    gamma = optuna_trial.suggest_float("gamma", 0.85, 1.0)  # Gamma può variare tra 0 e 1

    # Creazione di stacked con gamma
    stacked = sps.vstack([gamma * controller.URM_train, (1 - gamma) * controller.ICM_all.T]).tocsr()


    # Creazione dell'istanza del recommender
    recommender_instance = SLIMElasticNetRecommender(stacked)
    full_hyperp = {
        "alpha": optuna_trial.suggest_float("alpha", 0.00006, 0.00016, log=True),  # 1e-5, 1e-3 fino a 0.0003
        "topK": optuna_trial.suggest_int("topK", 600, 1000),  # 5, 1000
        "l1_ratio": optuna_trial.suggest_float("l1_ratio", 0.01, 0.09),  # 1e-3, 0.6 fino a 0.4
    }

    # Allenamento del recommender
    recommender_instance.fit(**full_hyperp)

    # Valutazione del recommender
    result_df, _ = controller.evaluator_test.evaluateRecommender(recommender_instance)

    # Restituzione del MAP

    return result_df.loc[10]["MAP"]



optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_SLIM,
                              callbacks=[save_results],
                              n_trials=50)

print(save_results.results_df)
print(optuna_study.best_trial.params)



