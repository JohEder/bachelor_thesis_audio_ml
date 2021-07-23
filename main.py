import config
from training_setup import TrainingSetup
import logging
from utils import plot_all_rocs

logging.basicConfig(level=logging.INFO)


classes = config.CLASSES

def generateTwoScenarios(classes, model_type):
    training_setups_first_scenario = []
    for category in classes:
        if category == 'None':
            continue

        normal_classes = ['None', category]
        anomalous_classes = list(filter(lambda x: x not in normal_classes, classes))
        training_setups_first_scenario.append(TrainingSetup(normal_classes, anomalous_classes, model_type))




    training_setups_second_scenario = []
    for category in classes:

        anomalous_classes = [category]
        normal_classes = list(filter(lambda x: x not in anomalous_classes, classes))
        training_setups_second_scenario.append(TrainingSetup(normal_classes, anomalous_classes, model_type))

    return training_setups_first_scenario, training_setups_second_scenario

first_scen, second_scen = generateTwoScenarios(classes, config.MODEL_TYPES.TRANSFORMER)


all_roc_auc_scores = {}
for train_scen in first_scen:
    roc_auc_scores = train_scen.run(3)
    all_roc_auc_scores[train_scen.setup_name] = roc_auc_scores

plot_all_rocs("All ROC_AUC Scores Scenario 1")

print(all_roc_auc_scores)
