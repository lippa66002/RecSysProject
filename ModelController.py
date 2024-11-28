from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Optimize.SaveResults import SaveResults
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
import optuna
from ModelNames import ModelName

def generate_model(model_name, optuna_hpp, URM_train, ICM_all=None):

    if model_name == ModelName.SLIM_ElasticNet:
        model = SLIMElasticNetRecommender(URM_train)
        model.fit(**optuna_hpp)
    elif model_name == ModelName.ItemKNNCFRecommender:
        model = ItemKNNCFRecommender(URM_train)
        model.fit(**optuna_hpp)
    elif model_name == ModelName.RP3betaRecommender:
        model = RP3betaRecommender(URM_train)
        model.fit(**optuna_hpp)
    elif model_name == ModelName.ContentBasedRecommender:
        model = ItemKNNCBFRecommender(URM_train, ICM_all)
        model.fit(**optuna_hpp)
    else:
        raise ValueError("Model not found")

    return model

def optunizer(model_name):

    if model_name == ModelName.SLIM_ElasticNet:
        obj_func = objective_function_SLIM
    elif model_name == ModelName.ItemKNNCFRecommender:
        obj_func = objective_function_KNN_similarities
    elif model_name == ModelName.RP3betaRecommender:
        obj_func = objective_function_graph
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

def objective_function_KNN_similarities(optuna_trial, URM_train, evaluator_validation):

    recommender_instance = ItemKNNCFRecommender(URM_train)
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

    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)

    return result_df.loc[10]["MAP"]

def objective_function_SLIM(optuna_trial, URM_train, evaluator_validation):

    recommender_instance = SLIMElasticNetRecommender(URM_train)
    full_hyperp = {"alpha": optuna_trial.suggest_float("alpha", 1e-5, 1e-3),
                   "topK": optuna_trial.suggest_int("topK", 5, 1000),
                   "l1_ratio": optuna_trial.suggest_float("l1_ratio", 1e-3, 0.6),
                   }
    recommender_instance.fit(**full_hyperp)
    # epochs = recommender_instance.get_early_stopping_final_epochs_dict()["epochs"]
    # optuna_trial.set_user_attr("epochs", epochs)
    # optuna_trial.set_user_attr("train_time (min)", (time.time() - start_time) / 60)
    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)

    return result_df.loc[10]["MAP"]

def objective_function_graph(optuna_trial, URM_train, evaluator_validation):
    recomm = RP3betaRecommender(URM_train)
    full_hyperp = {
        "topK": optuna_trial.suggest_int("topK", 5, 1000),
        "beta": optuna_trial.suggest_float("beta", 0, 1),
        "alpha": optuna_trial.suggest_float("alpha", 0, 1.5)}
    recomm.fit(**full_hyperp)

    result_df, _ = evaluator_validation.evaluateRecommender(recomm)

    return result_df.loc[10]["MAP"]

