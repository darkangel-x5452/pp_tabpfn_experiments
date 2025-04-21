# # TabPFN Community installs optional functionalities around the TabPFN model
# # These include post-hoc ensembles, interpretability tools, and more
# !git clone https://github.com/PriorLabs/tabpfn-extensions
# !pip install -e tabpfn-extensions[all]
#
# # Install Baselines
# !pip install catboost xgboost
#
# # Install example datasets
# !pip install datasets
import os

import numpy as np

from utils.advanced_classifier import AdvancedClassifier
from utils.tools import get_data
from utils.unsupervised_class import Unsupervised


def run_app():

    # https://colab.research.google.com/drive/1SHa43VuHASLjevzO7y3-wPCxHY18-2H6?usp=sharing#scrollTo=o03aOVAw0Etg
    data_fp = os.environ["DATA_FP"]
    # data_fp = r"openml"
    adv = AdvancedClassifier()
    X, y = get_data(data_type=data_fp)
    # X_train, X_test, y_train, y_test = adv.setup_train_data(x=X, y=y)
    # num_class = len(np.unique(y))
    # model = adv.train_model(
    #     X_train=X_train,
    #     y_train=y_train,
    # )
    # adv.evaluate_model(
    #     model=model,
    #     X_test=X_test,
    #     y_test=y_test,
    #     num_class=num_class
    # )
    # adv.save_model(model=model)
    model = adv.load_model()

    adv.compare_models(
        X=X,
        y=y,
    )

    # adv.interpret_features()

    # adv.playground_data_chart()
    # uns = Unsupervised()
    # uns.basic()




if __name__ == "__main__":
    run_app()