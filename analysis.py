import pandas as pd

results_csv = '/home/johannes/Desktop/Documents/Informatikstudium/Semester06_07/Bachelorarbeit/acoustic_ml/transformer_asd/results/velocites_results_30_ep/results/all_results.csv'

dataframe = pd.read_csv(results_csv)
dataframe.columns = ['index', 'setup', 'model_type', 'roc_auc', 'n_mels', 'seed', 'loss_func']
vehicle_setups = ['None_anom_C_T_B_M', 'C_anom_B_M', 'T_anom_B_M', 'C_T_anom_B_M']
velocity_setups = ['70_anom_30', '30_anom_50_70']

def get_mean_std(df, model, setup):
    df = df[df['model_type'] == model]
    df = df[df['setup'] == setup]
    mean = df['roc_auc'].mean()
    std = df['roc_auc'].std()
    return round(mean, 3), round(std, 3)

def get_mean_std_setup(df, model, setups):
    print(f"Model: {model}")
    for setup in setups:
        mean, std = get_mean_std(df, model, setup)
        print(f"{setup}::  mean: {mean}, std: {std}")

get_mean_std_setup(dataframe, 'AUTOENCODER', velocity_setups)
get_mean_std_setup(dataframe, 'TRANSFORMER', velocity_setups)
get_mean_std_setup(dataframe, 'IDNN', velocity_setups)