from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Optimize.SaveResults import SaveResults
from Recommenders.HybridOptunable2 import HybridOptunable2
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.Neural.MultVAE_PyTorch_Recommender import MultVAERecommender_PyTorch
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import optuna
from ModelNames import ModelName


class ModelController:

    def __init__(self, URM_all, ICM_all):
        self.URM_all = URM_all
        self.ICM_all = ICM_all
        self.URM_train, self.URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
        #self.URM_train, self.URM_validation = split_train_in_two_percentage_global_sample(self.URM_train_validation,
                       #                                                         train_percentage=0.8)
        #self.evaluator_validation = EvaluatorHoldout(self.URM_validation, cutoff_list=[10])
        self.evaluator_test = EvaluatorHoldout(self.URM_test, cutoff_list=[10])

    def generate_model(self, model_name, optuna_hpp):

        if model_name == ModelName.SLIM_ElasticNet:
            model = SLIMElasticNetRecommender(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.ItemKNNCFRecommender:
            model = ItemKNNCFRecommender(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.RP3betaRecommender:
            model = RP3betaRecommender(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.ContentBasedRecommender:
            model = ItemKNNCBFRecommender(self.URM_train, self.ICM_all)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.EASE_R_Recommender:
            model = EASE_R_Recommender(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.HybridOptunable2:
            model = HybridOptunable2(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.MultVAERecommender_PyTorch:
            model = MultVAERecommender_PyTorch(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.UserKNNCFRecommender:
            model = UserKNNCFRecommender(self.URM_train)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.Hybrid_ItemKNN_CF_CBF:
            model = ItemKNN_CFCBF_Hybrid_Recommender(self.URM_train, self.ICM_all)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.ItemKNNSimilarityHybridRecommender:
            model = ItemKNN_CFCBF_Hybrid_Recommender(self.URM_train, self.ICM_all)
            model.fit(**optuna_hpp)
        elif model_name == ModelName.SLIM_BPR_Recommender:
            model = SLIM_BPR_Cython(self.URM_train)
            model.fit(**optuna_hpp)
        else:
            raise ValueError("Model not found")

        return model


    def optunizer(self, model_name):

        if model_name == ModelName.SLIM_ElasticNet:
            obj_func = self.objective_function_SLIM
        elif model_name == ModelName.ItemKNNCFRecommender:
            obj_func = self.objective_function_KNN_similarities
        elif model_name == ModelName.RP3betaRecommender:
            obj_func = self.objective_function_graph
        elif model_name == ModelName.EASE_R_Recommender:
            obj_func = self.objective_function_easeR
        elif model_name == ModelName.HybridOptunable2:
            obj_func = self.objective_function_hybridOptunable2
        elif model_name == ModelName.MultVAERecommender_PyTorch:
            obj_func = self.objective_function_multVAE
        elif model_name == ModelName.UserKNNCFRecommender:
            obj_func = self.objective_function_userKNN
        elif model_name == ModelName.Hybrid_ItemKNN_CF_CBF:
            obj_func = self.objective_function_hybrid_ItemKNN_CF_CBF
        elif model_name == ModelName.ItemKNNSimilarityHybridRecommender:
            obj_func = self.objective_function_itemKNN_similarity_hybrid
        elif model_name == ModelName.SLIM_BPR_Recommender:
            obj_func = self.objective_function_SLIM_BPR_Cython
        else:
            raise ValueError("Model not found")

        optuna_study = optuna.create_study(direction="maximize")

        save_results = SaveResults()

        optuna_study.optimize(obj_func,
                              callbacks=[save_results],
                              n_trials=50)

        print(save_results.results_df)
        print(optuna_study.best_trial.params)

        return optuna_study.best_trial.params

    def objective_function_KNN_similarities(self, optuna_trial):

        recommender_instance = ItemKNNCFRecommender(self.URM_train)
        similarity = optuna_trial.suggest_categorical("similarity",
                                                      ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky',
                                                       'euclidean'])

        full_hyperp = {"similarity": similarity,
                       "topK": optuna_trial.suggest_int("topK", 5, 1000),
                       "shrink": optuna_trial.suggest_int("shrink", 0, 1000),
                       }

        if similarity == "asymmetric":
            full_hyperp["asymmetric_alpha"] = optuna_trial.suggest_float("asymmetric_alpha", 0, 2, log=False)
            full_hyperp["normalize"] = True

        elif similarity == "tversky":
            full_hyperp["tversky_alpha"] = optuna_trial.suggest_float("tversky_alpha", 0, 2, log=False)
            full_hyperp["tversky_beta"] = optuna_trial.suggest_float("tversky_beta", 0, 2, log=False)
            full_hyperp["normalize"] = True

        elif similarity == "euclidean":
            full_hyperp["normalize_avg_row"] = optuna_trial.suggest_categorical("normalize_avg_row", [True, False])
            full_hyperp["similarity_from_distance_mode"] = optuna_trial.suggest_categorical(
                "similarity_from_distance_mode", ["lin", "log", "exp"])
            full_hyperp["normalize"] = optuna_trial.suggest_categorical("normalize", [True, False])

        recommender_instance.fit(**full_hyperp)

        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)

        return result_df.loc[10]["MAP"]

    def objective_function_SLIM(self, optuna_trial):

        recommender_instance = SLIMElasticNetRecommender(self.URM_train)
        full_hyperp = {"alpha": optuna_trial.suggest_float("alpha", 1e-5, 1e-3),
                       "topK": optuna_trial.suggest_int("topK", 5, 1000),
                       "l1_ratio": optuna_trial.suggest_float("l1_ratio", 1e-3, 0.6),
                       }
        recommender_instance.fit(**full_hyperp)
        # epochs = recommender_instance.get_early_stopping_final_epochs_dict()["epochs"]
        # optuna_trial.set_user_attr("epochs", epochs)
        # optuna_trial.set_user_attr("train_time (min)", (time.time() - start_time) / 60)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)

        return result_df.loc[10]["MAP"]

    def objective_function_graph(self, optuna_trial):
        recomm = RP3betaRecommender(self.URM_train)
        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "beta": optuna_trial.suggest_float("beta", 0, 1),
            "alpha": optuna_trial.suggest_float("alpha", 0, 1.5)}
        recomm.fit(**full_hyperp)

        result_df, _ = self.evaluator_test.evaluateRecommender(recomm)

        return result_df.loc[10]["MAP"]

    def objective_function_easeR(self, optuna_trial):
        recommender_instance = EASE_R_Recommender(self.URM_train)
        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "l2_norm": optuna_trial.suggest_float("l2_norm", 1e-3, 1e3, log=True),
            "normalize_matrix": optuna_trial.suggest_categorical("normalize_matrix", [True, False])
        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_hybridOptunable2(self, optuna_trial, itemKnncf, slimrec):
        print("helloworld")
        alpha = optuna_trial.suggest_float("alpha", 0.1, 0.9)
        recommender_object = HybridOptunable2(self.URM_train)
        recommender_object.fit(alpha, itemKnncf, slimrec)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_object)
        return result_df.loc[10]["MAP"]

    def objective_function_multVAE(self, optuna_trial):
        recommender_instance = MultVAERecommender_PyTorch(self.URM_train, use_gpu=True)
        full_hyperp = {
            "epochs": optuna_trial.suggest_int("epochs", 10, 100),
            "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": optuna_trial.suggest_int("batch_size", 32, 512),
            "dropout": optuna_trial.suggest_float("dropout", 0.1, 0.5),
            "total_anneal_steps": optuna_trial.suggest_int("total_anneal_steps", 10000, 200000),
            "anneal_cap": optuna_trial.suggest_float("anneal_cap", 0.1, 0.5),
            "p_dims": [self.URM_train.shape[1], self.URM_train.shape[1]]
        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_userKNN(self, optuna_trial):
        recommender_instance = UserKNNCFRecommender(self.URM_train)
        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "shrink": optuna_trial.suggest_int("shrink", 0, 1000),
            "similarity": optuna_trial.suggest_categorical("similarity",
                                                           ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky',
                                                            'euclidean']),
            "normalize": optuna_trial.suggest_categorical("normalize", [True, False]),
            "feature_weighting": optuna_trial.suggest_categorical("feature_weighting", ["BM25", "TF-IDF", "none"])
        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_hybrid_ItemKNN_CF_CBF(self, optuna_trial):
        recommender_instance = ItemKNN_CFCBF_Hybrid_Recommender(self.URM_train, self.ICM_all)
        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "shrink": optuna_trial.suggest_int("shrink", 0, 1000),
            "similarity": optuna_trial.suggest_categorical("similarity",
                                                           ['cosine', 'dice', 'jaccard', 'asymmetric', 'tversky',
                                                            'euclidean']),
            "normalize": optuna_trial.suggest_categorical("normalize", [True, False]),
            "feature_weighting": optuna_trial.suggest_categorical("feature_weighting", ["BM25", "TF-IDF", "none"]),
            "ICM_weight": optuna_trial.suggest_float("ICM_weight", 0.1, 1.0)
        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_itemKNN_similarity_hybrid(self, optuna_trial):
        slim = SLIMElasticNetRecommender(self.URM_train)
        itemKNN = ItemKNNCFRecommender(self.URM_train)

        slim.load_model(folder_path="_saved_models", file_name="SLIMElasticNetRecommender")
        itemKNN.load_model(folder_path="_saved_models", file_name="ItemKNNCFRecommender")

        similarity_1 = slim.W_sparse
        similarity_2 = itemKNN.W_sparse

        recommender_instance = ItemKNNSimilarityHybridRecommender(self.URM_train, similarity_1, similarity_2)
        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "alpha": optuna_trial.suggest_float("alpha", 0.0, 1.0)
        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]

    def objective_function_SLIM_BPR_Cython(self, optuna_trial):
        recommender_instance = SLIM_BPR_Cython(self.URM_train)
        full_hyperp = {
            "topK": optuna_trial.suggest_int("topK", 5, 1000),
            "learning_rate": optuna_trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "lambda_i": optuna_trial.suggest_float("lambda_i", 1e-4, 1e-2, log=True),
            "lambda_j": optuna_trial.suggest_float("lambda_j", 1e-4, 1e-2, log=True),
            "batch_size": optuna_trial.suggest_int("batch_size", 32, 512),
            "symmetric": optuna_trial.suggest_categorical("symmetric", [True, False]),
            "sgd_mode": optuna_trial.suggest_categorical("sgd_mode", ["sgd", "adagrad", "adam"]),
        }
        recommender_instance.fit(**full_hyperp)
        result_df, _ = self.evaluator_test.evaluateRecommender(recommender_instance)
        return result_df.loc[10]["MAP"]










