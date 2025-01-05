import os.path
import json

from mt_libs.mt_keras import CreateEmotionsModel
from mt_libs.mt_mood_analyze import AnalyzeMood
from mt_libs.colloborativ_filtering_system_corrected import COLAB_MODEL
import argparse


PLOT_DIR = "C:/Data/DhritiData/JugendForchst/JF_Project/Prototype/input/Plots/Emotions/"
INPUT_EEG_CSV = "C:/Data/DhritiData/JugendForchst/JF_Project/Data/emotions.csv"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--mode", required=True)
    parser.add_argument("-mod", "--model", required=False)
    args = parser.parse_args()

    if args.mode == 'mood_model':
        print ("Creating mood_model...")
        CreateEmotionsModel(INPUT_EEG_CSV, PLOT_DIR,5)
        COLAB_MODEL('raw_cbf_data.csv','Genre_IDs.csv',30)
    elif args.mode == 'mood_analyze':
        print ("Analyzing mood based on Mood Model...")
        model_path = args.model
        if not os.path.exists(model_path):
            print("No Model Found...")
            exit(1)
        with open('GenreModel.json') as f:
            song_data_file = json.load(f)
            print(song_data_file)

        AnalyzeMood(model_path,song_data_file)

    else:
        print ("Exiting")

# Invoke Main
main()






