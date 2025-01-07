import scipy.sparse as sps

from ModelController import ModelController
from Optimize.SaveResults import SaveResults
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.HybridOptunable2 import HybridOptunable2
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import optuna

from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender
controller = ModelController()

stacked = sps.vstack([0.6814451172353111 * controller.URM_train, (1 - 0.6814451172353111) * controller.ICM_all.T]).tocsr()
slim1 = SLIMElasticNetRecommender(stacked)
slim1.load_model(folder_path="_saved_models", file_name="SLIMstackedTrainval1")
slim2 = SLIMElasticNetRecommender(controller.URM_train)
slim2.load_model(folder_path="_saved_models", file_name="SLIMtrain")  #0.3588980025585651 slim1slim2 score,  0.7228086650480543 slim1slim2 optunable2
rp3 = RP3betaRecommender(controller.URM_train)
bestrp3 = RP3betaRecommender(controller.URM_train)
bestrp3.fit(topK= 18, beta= 0.2449115248846201, alpha= 0.34381573319072084)

def objective_function_scores_hybrid_1( optuna_trial):

    # bpr = SLIM_BPR_Cython(self.URM_train)
    # bpr.load_model(folder_path="_saved_models", file_name="SLIM_BPR_Recommender_train")
    print("slim1+slim2+rp3 score con valori liberi di tutti")
    recom1 = ScoresHybridRecommender(controller.URM_train, slim1, slim2, rp3, slim1, slim1)

    gamma = optuna_trial.suggest_float("gamma", 0.0, 1.0)
    alpha = optuna_trial.suggest_float("alpha", 0.0, 1.0)
    beta = optuna_trial.suggest_float("beta", 0.0, 1.0)







    recom1.fit(alpha, beta, gamma, 0, 0)

    result_df, _ = controller.evaluator_test.evaluateRecommender(recom1)
    return result_df.loc[10]["MAP"]
optuna_study = optuna.create_study(direction="maximize")
save_results = SaveResults()
optuna_study.optimize(objective_function_scores_hybrid_1,
                              callbacks=[save_results],
                              n_trials=50)
print(save_results.results_df)
print(optuna_study.best_trial.params)


'''def objective_function_scores_hybrid_6( optuna_trial):
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
'''