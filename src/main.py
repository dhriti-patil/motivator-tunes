import os.path
import json

from mt_libs.mt_keras import CreateEmotionsModel
from mt_libs.mt_mood_analyze import AnalyzeMood
from mt_libs.mt_collab_filter_create import create_collaboration_model
from mt_libs.mt_collab_filter_predict import predict_rating
import argparse
import pandas as pd

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
    elif args.mode == 'mood_analyze':
        print ("Analyzing mood based on Mood Model...")
        model_path = args.model
        if not os.path.exists(model_path):
            print("No Model Found...")
            exit(1)
        with open('GenreModel.json') as f:
            song_data_file = json.load(f)
            print(song_data_file)

        #AnalyzeMood(model_path,song_data_file)
        df = pd.DataFrame(columns=['User_ID', 'Genre_ID', 'rating', 'Genre'])
        df.loc[0] = [99, 0, 1, "Genre1:classical - vocals"]
        df.loc[1] = [99, 1, 2, "Genre1:classical - string inst."]

        model , data, user2user_encoded, genre2genre_encoded, genre_encoded2genre = (
            create_collaboration_model('InputRatingData.csv', 'Genre_IDs.csv', 30, df))

        recommendations = predict_rating(model, data, 'Genre_IDs.csv', 99, user2user_encoded, genre2genre_encoded, genre_encoded2genre)

    else:
        print ("Exiting")

# Invoke Main
main()






