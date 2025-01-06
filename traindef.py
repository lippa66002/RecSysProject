import scipy.sparse as sps

from ModelController import ModelController
from Optimize.SaveResults import SaveResults
from Recommenders.HybridOptunable2 import HybridOptunable2
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import optuna

from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender

controller = ModelController()
stacked = sps.vstack([0.6814451172353111 * controller.URM_train, (1 - 0.6814451172353111) * controller.ICM_all.T]).tocsr()
slim1 = SLIMElasticNetRecommender(stacked)
slim1.load_model(folder_path="_saved_models", file_name="SLIMstackedTrainval1")
slim2 = SLIMElasticNetRecommender(controller.URM_train)
slim2.load_model(folder_path="_saved_models", file_name="SLIMtrain")
dd,_ = controller.evaluator_test.evaluateRecommender(slim1)
print(dd.loc[10]["MAP"])
dd,_ = controller.evaluator_test.evaluateRecommender(slim2)
print(dd.loc[10]["MAP"])
def objective_function_scores_hybrid_1( optuna_trial):

    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    recom1 = ScoresHybridRecommender(controller.URM_train, slim1, slim2, slim1, slim1, slim1)

    alpha = optuna_trial.suggest_float("alpha", 0.0, 1.0)







    recom1.fit(alpha, 1-alpha, 0, 0, 0)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]
optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_1,
                              callbacks=[save_results],
                              n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)


def objective_function_scores_hybrid_6( optuna_trial):
    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    recom1 = HybridOptunable2(controller.URM_train)

    alpha = optuna_trial.suggest_float("alpha", 0, 1)

    recom1.fit(alpha,slim1,slim2)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]


optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_6,
                      callbacks=[save_results],
                      n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)