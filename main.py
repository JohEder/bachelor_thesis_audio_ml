from torch import utils
import config
from training_setup import TrainingSetup
import logging
from utils import plot_all_rocs, convert_to_df, plot_roc_curve, generateTwoScenarios, plot_all_results
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
    for i in range(len(scenario)):
        print(f"Starting setup number {i} : {scenario[i].setup_name} : {scenario[i].model_type}")
        roc_auc_scores, losses, fp_rate, tp_rate, roc = scenario[i].run(config.NUMBER_REPEAT_EXPERIMENT)
        all_roc_scores += roc_auc_scores
        all_setups += [scenario[i].setup_name for j in range(len(roc_auc_scores))]
        all_model_types += [str(scenario[i].model_type)[12:] for j in range(len(roc_auc_scores))]

        if plot_roc_and_loss:

            axes_loss[i].set_title(f'Loss of Setup: {scenario[i].setup_name}')
            losses = convert_to_df(losses)
            sns.lineplot(data=losses, ax=axes_loss[i])
            plot_roc_curve(scenario[i].setup_name, fp_rate, tp_rate, roc, axes_rocs[i])
    if plot_roc_and_loss:
        fig_losses.savefig(config.RESULT_DIR +'loss_' + str(scenario[0].model_type) + '.png')
        fig_rocs.savefig(config.RESULT_DIR + 'roc_' + str(scenario[0].model_type) + '.png')

fig_results, axe = plt.subplots(1, 1, figsize=(12, 4))
#train_and_plot(transformer_scenario)
#train_and_plot(autoencoder_scenario)
train_and_plot(idnn_1, True)

results = {'Normal_Data' : all_setups, 'Model_Type' : all_model_types, 'ROC_AUC' : all_roc_scores}
plot_all_results(results, axe)
fig_results.savefig(config.RESULT_DIR + 'all_results.png')



"""
for i in range(len(first_scen)):
    print(f"Starting Train Setup number {i} form first scenario")
    train_scen = first_scen[i]
    roc_auc_scores, losses, fp_rate, tp_rate, roc = train_scen.run(config.NUMBER_REPEAT_EXPERIMENT)
    all_roc_auc_scores_1['ROC_AUC'] = roc_auc_scores
    all_roc_auc_scores_1['Normal_Classes'] = [train_scen.setup_name for i in range(len(roc_auc_scores))]
    all_roc_auc_scores_1['Model_Type'] = [str()]
    axes_losses[i].set_title(f'Loss of Setup: {train_scen.setup_name}')
    losses = convert_to_df(losses)
    sns.lineplot(data=losses, ax=axes_losses[i])
    plot_roc_curve(train_scen.setup_name, fp_rate, tp_rate, roc, axes_rocs[i])

fig_losses.savefig(config.RESULT_DIR + 'losses_scenario_1.png')
fig_rocs.savefig(config.RESULT_DIR + 'rocs_scen_1.png')



print(all_roc_auc_scores_1)
plot_all_rocs("All ROC_AUC Scores Scenario 1", all_roc_auc_scores_1, axes_all_rocs[0])

print("\n\nSecond scenario")
all_roc_auc_scores_2 = {}

fig_losses_2, axes_losses_2 = plt.subplots(1, len(second_scen), figsize=(25, 5), sharey=True)
fig_losses_2.suptitle('Scenario 2: Loss Curves')

fig_rocs_2, axes_rocs_2 = plt.subplots(1, len(second_scen), figsize=(25, 5), sharex=True, sharey=True)
fig_rocs_2.suptitle('Scenario 2: ROC Curves')


for i in range(len(second_scen)):
    print(f"Starting Train Setup number {i} form second scenario")
    train_scen = second_scen[i]
    roc_auc_scores_2, losses_2, fp_rate_2, tp_rate_2, roc_2 = train_scen.run(config.NUMBER_REPEAT_EXPERIMENT)
    all_roc_auc_scores_2['ROC_AUC'] = roc_auc_scores_2
    axes_losses_2[i].set_title(f'Loss of Setup: {train_scen.setup_name}')
    losses_2 = convert_to_df(losses_2)
    sns.lineplot(data=losses_2, ax=axes_losses_2[i])
    plot_roc_curve(train_scen.setup_name, fp_rate_2, tp_rate_2, roc_2, axes_rocs_2[i])
    

"""
end_time = datetime.datetime.now()
matplotlib.pyplot.show()
"""
print(all_roc_auc_scores_2)
plot_all_rocs("All ROC_AUC Scores Scenario 2", all_roc_auc_scores_2, axes_all_rocs[1])
fig_all_rocs.savefig(config.RESULT_DIR + 'all_rocs.png')
fig_losses_2.savefig(config.RESULT_DIR + 'losses_scen_2.png')
fig_rocs_2.savefig(config.RESULT_DIR + 'rocs_scen_2.png')
print(f"Total Runtime: {end_time - start_time}")
matplotlib.pyplot.show()

#first_scen[0].run()

"""
exit(0)
