""" Import script for IDMT-Traffic dataset
Ref:
    J. Abeßer, S. Gourishetti, A. Kátai, T. Clauß, P. Sharma, J. Liebetrau: IDMT-Traffic: An Open Benchmark
    Dataset for Acoustic Traffic Monitoring Research, EUSIPCO, 2021
"""

import os
import glob
import pandas as pd

__author__ = 'Jakob Abeßer (jakob.abesser@idmt.fraunhofer.de)'


def import_idmt_traffic_dataset(fn_txt: str = "idmt_traffic_all") -> pd.DataFrame:
    """ Import IDMT-Traffic dataset
    Args:
        fn_txt (str): Text file with all WAV files
    Returns:
        df_dataset (pd.Dataframe): File-wise metadata
            Columns:
                'file': WAV filename,
                'is_background': True if recording contains background noise (no vehicle), False else
                'date_time': Recording time (YYYY-MM-DD-HH-mm)
                'location': Recording location
                'speed_kmh': Speed limit at recording site (km/h), UNK if unknown,
                'sample_pos': Sample position (centered) within the original audio recording,
                'daytime': M(orning) or (A)fternoon,
                'weather': (D)ry or (W)et road condition,
                'vehicle': (B)us, (C)ar, (M)otorcycle, or (T)ruck,
                'source_direction': Source direction of passing vehicle: from (L)eft or from (R)ight,
                'microphone': (SE)= (high-quality) sE8 microphones, (ME) = (low-quality) MEMS microphones (ICS-43434),
                'channel': Original stereo pair channel (12) or (34)
    """
    # load file list
    df_files = pd.read_csv(fn_txt, names=('file',))
    fn_file_list = df_files['file'].to_list()

    # load metadata from file names
    df_dataset = []

    for f, fn in enumerate(fn_file_list):
        fn = fn.replace('.wav', '')
        parts = fn.split('_')

        # background noise files
        if '-BG' in fn:
            date_time, location, speed_kmh, sample_pos, mic, channel = parts
            vehicle, source_direction, weather, daytime = 'None', 'None', 'None', 'None'
            is_background = True

        # files with vehicle passings
        else:
            date_time, location, speed_kmh, sample_pos, daytime, weather, vehicle_direction, mic, channel = parts
            vehicle, source_direction = vehicle_direction
            is_background = False

        channel = channel.replace('-BG', '')
        speed_kmh = speed_kmh.replace('unknownKmh', 'UNK')
        speed_kmh = speed_kmh.replace('Kmh', '')

        df_dataset.append({'file': fn,
                           'is_background': is_background,
                           'date_time': date_time,
                           'location': location,
                           'speed_kmh': speed_kmh,
                           'sample_pos': sample_pos,
                           'daytime': daytime,
                           'weather': weather,
                           'vehicle': vehicle,
                           'source_direction': source_direction,
                           'microphone': mic,
                           'channel': channel})

    df_dataset = pd.DataFrame(df_dataset, columns=('file', 'is_background', 'date_time', 'location', 'speed_kmh', 'sample_pos', 'daytime', 'weather', 'vehicle',
                                                   'source_direction', 'microphone', 'channel'))

    return df_dataset


if __name__ == '__main__':
    TEST_SET_NAME = 'eusipco_2021_test'
    TRAIN_SET_NAME = 'eusipco_2021_train'
    ALL_DATA_NAME = 'idmt_traffic_all'

    training_data = import_idmt_traffic_dataset('/home/johannes/Desktop/Documents/Informatikstudium/Semester06/Bachelorarbeit/acoustic_ml/datasets/IDMT_Traffic/annotation/' + TRAIN_SET_NAME + '.txt')
    print(len(training_data))
    training_data.to_csv('/home/johannes/Desktop/Documents/Informatikstudium/Semester06/Bachelorarbeit/acoustic_ml/datasets/IDMT_Traffic/annotation/' + TRAIN_SET_NAME + '.csv', sep=',', encoding='utf-8', header=None)

    test_data = import_idmt_traffic_dataset('/home/johannes/Desktop/Documents/Informatikstudium/Semester06/Bachelorarbeit/acoustic_ml/datasets/IDMT_Traffic/annotation/' + TEST_SET_NAME + '.txt')
    print(len(test_data))
    test_data.to_csv('/home/johannes/Desktop/Documents/Informatikstudium/Semester06/Bachelorarbeit/acoustic_ml/datasets/IDMT_Traffic/annotation/' + TEST_SET_NAME + '.csv', sep=',', encoding='utf-8', header=None)

    all_data = import_idmt_traffic_dataset('/home/johannes/Desktop/Documents/Informatikstudium/Semester06/Bachelorarbeit/acoustic_ml/datasets/IDMT_Traffic/annotation/' + ALL_DATA_NAME + '.txt')
    print(len(all_data))
    all_data.to_csv('/home/johannes/Desktop/Documents/Informatikstudium/Semester06/Bachelorarbeit/acoustic_ml/datasets/IDMT_Traffic/annotation/' + ALL_DATA_NAME + '.csv', sep=',', encoding='utf-8', header=None)
    all_vehicles = all_data[all_data.vehicle != "None"]
    all_vehicles.to_csv('/home/johannes/Desktop/Documents/Informatikstudium/Semester06/Bachelorarbeit/acoustic_ml/datasets/IDMT_Traffic/annotation/' + ALL_DATA_NAME + '_only_vehicles.csv', sep=',', encoding='utf-8', header=None)
    print(len(all_vehicles))
    """
    print(len(all_data[all_data.is_background])) #8144 -> #9362 labbelled background sounds
    print(len(all_data[all_data.vehicle == 'C'])) #7804
    print(len(all_data[all_data.vehicle == 'M'])) #430
    print(len(all_data[all_data.vehicle == 'T'])) #1022
    print(len(all_data[all_data.vehicle == 'B'])) #106
    """

    print(len(all_data[all_data.vehicle.isin(['C', 'M'])]))

    """
    # example use
    fn_txt_list = ["idmt_traffic_all.txt",    # complete IDMT-Traffic dataset
                   "eusipco_2021_train.txt",  # training set of EUSIPCO 2021 paper
                   "eusipco_2021_test.txt"]   # test set of EUSIPCO 2021 paper

    # import metadata
    for fn_txt in fn_txt_list:
        print('Metadata for {}:'.format(fn_txt))
        print(import_idmt_traffic_dataset(fn_txt))

    """
