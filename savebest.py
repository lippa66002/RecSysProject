import optuna

from ModelController import ModelController
from Optimize.SaveResults import SaveResults
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import scipy.sparse as sps
controller = ModelController()
#stacked = sps.vstack([0.8392863849420211 * controller.URM_boost, (1 - 0.8392863849420211) * controller.ICM_all.T]).tocsr()
#slim = SLIMElasticNetRecommender(stacked)
#slim.fit(alpha =5.632458754549518e-05, topK=619, l1_ratio= 0.053794482642909716)
#slim.save_model(folder_path="_saved_models", file_name="slimbooststacked3")
stacked = sps.vstack([0.6814451172353111 * controller.URM_boost, (1 - 0.6814451172353111) * controller.ICM_all.T]).tocsr()
def objective_function_SLIM(optuna_trial):
    recommender_instance = SLIMElasticNetRecommender(stacked)
    full_hyperp = {
        "alpha": optuna_trial.suggest_float("alpha", 0.00003, 0.0006, log=True),  # 1e-5, 1e-3 fino a 0.0003
        "topK": optuna_trial.suggest_int("topK", 600, 1000),  # 5, 1000
        "l1_ratio": optuna_trial.suggest_float("l1_ratio", 0.01, 0.09),  # 1e-3, 0.6 fino a 0.4
    }

    recommender_instance.fit(**full_hyperp)
    result_df, _ = controller.evaluator_test.evaluateRecommender(recommender_instance)

    return result_df.loc[10]["MAP"]



optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_SLIM,
                              callbacks=[save_results],
                              n_trials=50)

print(save_results.results_df)
print(optuna_study.best_trial.params)