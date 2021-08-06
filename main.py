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
start_time = datetime.datetime.now()


transformer_scenario, _ = generateTwoScenarios(classes)

none_scenario_tf = TrainingSetup(['None'], ['C', 'T', 'B', 'M'])

c_scenario_tf = TrainingSetup(['C'], ['B', 'M'])
c_scenario_tf_with_none = TrainingSetup(['C'], ['B', 'M', 'None'])
#c_scenario_tf_with_t = TrainingSetup(['C'], ['T', 'B', 'M'], config.MODEL_TYPES.TRANSFORMER)
c_t_scenario_tf = TrainingSetup(['C', 'T'], ['B', 'M'])
c_t_scenario_tf_with_None = TrainingSetup(['C', 'T'], ['None', 'B', 'M'])
m_scenario_tf = TrainingSetup(['M'], ['C', 'T'])

c_scenario_idnn = TrainingSetup(['C'], ['T', 'B', 'M'])
c_scenario_ae = TrainingSetup(['C'], ['B', 'M'])

t_scenario_tf = TrainingSetup(['T'], ['B', 'M'])
t_scenario_idnn = TrainingSetup(['T'], ['C', 'B', 'M'])
t_scenario_ae = TrainingSetup(['T'], ['B', 'M'])

velocity_setup_70_30 = TrainingSetup(['70'], ['30'], config.SETUP_TYPES.VELOCITY)
velocity_setup_30 = TrainingSetup(['50'], ['30'], config.SETUP_TYPES.VELOCITY)
velocity_setup_70 = TrainingSetup(['50'], ['70'], config.SETUP_TYPES.VELOCITY)
velocity_setup_both = TrainingSetup(['50'], ['30', '70'], config.SETUP_TYPES.VELOCITY)
road_conditions_setups = TrainingSetup(['D'], ['W'], config.SETUP_TYPES.WEATHER)

all_roc_scores = []
all_model_types = []
all_setups = []

def train_and_plot(scenario, model_type, plot_roc_and_loss=False):
    global all_roc_scores
    global all_model_types
    global all_setups
    if plot_roc_and_loss:
        len_scen = len(scenario) if len(scenario) > 1 else 2 #just for indexing, that running only one scenario doesnt break the graphs
        if len_scen > 5:
            raise Exception("Too many scenarios at the same time. Graphs will be too large.")
        fig_losses, axes_loss = plt.subplots(1, len_scen, figsize=(6* len_scen, 4))
        fig_rocs, axes_rocs = plt.subplots(1, len_scen, figsize=(6 * len_scen, 4))
        fig_error_dists, axes_errors = plt.subplots(1, len_scen, figsize=(6 * len_scen, 4))
    for i in range(len(scenario)):
        print(f"Starting setup number {i} : {scenario[i].setup_name} : {model_type}")
        roc_auc_scores, losses, fp_rate, tp_rate, roc, scores_classes = scenario[i].run(model_type, config.NUMBER_REPEAT_EXPERIMENT)
        scores, classes = scores_classes
        print(f"Classes: {classes}")
        all_roc_scores += roc_auc_scores
        all_setups += [scenario[i].setup_name for j in range(len(roc_auc_scores))]
        all_model_types += [str(model_type)[12:] for j in range(len(roc_auc_scores))]

        if plot_roc_and_loss:
            plot_error_distribution(axes_errors[i], scores_classes, scenario[i].setup_name)
            axes_loss[i].set_title(f'Loss of Setup: {scenario[i].setup_name}')
            losses = convert_to_df(losses)
            sns.lineplot(data=losses, ax=axes_loss[i])
            plot_roc_curve(scenario[i].setup_name, fp_rate, tp_rate, roc, axes_rocs[i])
    if plot_roc_and_loss:
        fig_error_dists.savefig(config.RESULT_DIR + 'error_dist_' + str(model_type) +scenario[i].setup_name + '.png')
        fig_losses.savefig(config.RESULT_DIR +'loss_' + str(model_type) +scenario[i].setup_name+ '.png')
        fig_rocs.savefig(config.RESULT_DIR + 'roc_' + str(model_type)+ scenario[i].setup_name + '.png')

fig_results, axe = plt.subplots(1, 1, figsize=(12, 4))
#train_and_plot(ae_1, True)
train_and_plot([velocity_setup_70_30, velocity_setup_30], model_type=config.MODEL_TYPES.TRANSFORMER, plot_roc_and_loss=True)
#train_and_plot([velocity_setup_30, velocity_setup_70, velocity_setup_both], model_type=config.MODEL_TYPES.TRANSFORMER, plot_roc_and_loss=True)
#train_and_plot([none_scenario_tf, c_scenario_tf, t_scenario_tf, m_scenario_tf, c_t_scenario_tf],config.MODEL_TYPES.TRANSFORMER, True)
#train_and_plot([none_scenario_tf, c_scenario_tf, t_scenario_tf, m_scenario_tf, c_t_scenario_tf],config.MODEL_TYPES.AUTOENCODER, True)
#train_and_plot([none_scenario_tf, c_scenario_tf, t_scenario_tf, m_scenario_tf, c_t_scenario_tf],config.MODEL_TYPES.IDNN, True)
#train_and_plot([c_t_scenario_tf,c_scenario_tf_with_none, c_t_scenario_tf_with_None], True)
#train_and_plot([c_scenario_idnn, t_scenario_idnn], True)

results = {'Normal_Data' : all_setups, 'Model_Type' : all_model_types, 'ROC_AUC' : all_roc_scores}
plot_all_results(results, axe)
fig_results.savefig(config.RESULT_DIR + 'all_results.png')


end_time = datetime.datetime.now()
matplotlib.pyplot.show()
print(end_time-start_time)
exit(0)
