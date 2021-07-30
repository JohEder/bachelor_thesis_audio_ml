from torch import utils
import config
from training_setup import TrainingSetup
import logging
from utils import plot_all_rocs, convert_to_df, plot_roc_curve, generateTwoScenarios, plot_all_results, plot_error_distribution
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import random

logging.basicConfig(level=logging.INFO)


classes = config.CLASSES
experiment_name = "standart"
start_time = datetime.datetime.now()


transformer_scenario, _ = generateTwoScenarios(classes, config.MODEL_TYPES.TRANSFORMER)
ae_1, _ = generateTwoScenarios(classes, config.MODEL_TYPES.AUTOENCODER)
idnn_1, _ = generateTwoScenarios(classes, config.MODEL_TYPES.IDNN)

all_roc_scores = []
all_model_types = []
all_setups = []

def train_and_plot(scenario, plot_roc_and_loss=False):
    global all_roc_scores
    global all_model_types
    global all_setups
    if plot_roc_and_loss:
        fig_losses, axes_loss = plt.subplots(1, len(scenario), figsize=(12, 4))
        fig_rocs, axes_rocs = plt.subplots(1, len(scenario), figsize=(12, 4))
        fig_error_dists, axes_errors = plt.subplots(1, len(scenario), figsize=(12, 4))
    for i in range(len(scenario)):
        print(f"Starting setup number {i} : {scenario[i].setup_name} : {scenario[i].model_type}")
        roc_auc_scores, losses, fp_rate, tp_rate, roc, scores_classes = scenario[i].run(config.NUMBER_REPEAT_EXPERIMENT)
        scores, classes = scores_classes
        print(f"Classes: {classes}")
        all_roc_scores += roc_auc_scores
        all_setups += [scenario[i].setup_name for j in range(len(roc_auc_scores))]
        all_model_types += [str(scenario[i].model_type)[12:] for j in range(len(roc_auc_scores))]

        if plot_roc_and_loss:
            plot_error_distribution(axes_errors[i], scores_classes, scenario[i].setup_name)
            axes_loss[i].set_title(f'Loss of Setup: {scenario[i].setup_name}')
            losses = convert_to_df(losses)
            sns.lineplot(data=losses, ax=axes_loss[i])
            plot_roc_curve(scenario[i].setup_name, fp_rate, tp_rate, roc, axes_rocs[i])
    if plot_roc_and_loss:
        fig_error_dists.savefig(config.RESULT_DIR + 'error_dist_' + str(scenario[0].model_type) + '.png')
        fig_losses.savefig(config.RESULT_DIR +'loss_' + str(scenario[0].model_type) + '.png')
        fig_rocs.savefig(config.RESULT_DIR + 'roc_' + str(scenario[0].model_type) + '.png')

fig_results, axe = plt.subplots(1, 1, figsize=(12, 4))
#train_and_plot(ae_1, True)
train_and_plot([transformer_scenario[0], transformer_scenario[1]], True)

results = {'Normal_Data' : all_setups, 'Model_Type' : all_model_types, 'ROC_AUC' : all_roc_scores}
plot_all_results(results, axe)
fig_results.savefig(config.RESULT_DIR + 'all_results.png')


end_time = datetime.datetime.now()
matplotlib.pyplot.show()
print(end_time-start_time)
exit(0)
