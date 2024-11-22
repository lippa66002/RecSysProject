import pandas as pd
class SaveResults(object):

    def __init__(self):
        self.results_df = pd.DataFrame(columns=["result"])

    def __call__(self, optuna_study, optuna_trial):
        hyperparam_dict = optuna_trial.params.copy()
        hyperparam_dict["result"] = optuna_trial.values[0]

        self.results_df = self.results_df.append(hyperparam_dict, ignore_index=True)