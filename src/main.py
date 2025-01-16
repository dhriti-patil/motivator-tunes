import os.path
import json
import time
import argparse
import pandas as pd
from mt_libs.mt_mood_model_create import CreateEmotionsModel
from mt_libs.mt_mood_analyze import AnalyzeMood , play_song , close_chrome
from mt_libs.mt_collab_filter_create import create_collaboration_model
from mt_libs.mt_collab_filter_predict import predict_rating


PLOT_DIR = "C:/Data/DhritiData/JugendForchst/JF_Project/Prototype/input/Plots/Emotions/"
INPUT_EEG_CSV = "C:/Data/DhritiData/JugendForchst/JF_Project/Data/emotions.csv"
GENRE_MODEL_FILE = 'GenreModel.json'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--mode", required=True)
    parser.add_argument("-mod", "--model", required=False)
    args = parser.parse_args()

    if args.mode == 'mood_model':
        print ("Creating mood_model...")
        CreateEmotionsModel(INPUT_EEG_CSV, PLOT_DIR,30)
    elif args.mode == 'mood_analyze':
        print ("Analyzing mood based on Mood Model...")
        model_path = args.model
        if not os.path.exists(model_path):
            print("No Model Found...")
            exit(1)
        with open(GENRE_MODEL_FILE) as f:
            song_data = json.load(f)
            print(song_data)

        for genre in song_data['song_data']:
            genre_name = genre['Name']
            for sub_genre in genre['Sub-Genre']:
                sub_genre_name=sub_genre['Name']
                audio_url = sub_genre['URL']
                print("Processing : " + genre_name + " - " + sub_genre_name)
                print("Audio URL : " + audio_url)
                close_chrome()
                play_song(audio_url)
                input("Press Enter to continue...")
                print("Continuing...")
                time.sleep(11)
                rating = 1
                rating = AnalyzeMood(model_path)
                print("\n\n###### Obtained Rating : " + str(rating))

                df = pd.DataFrame(columns=['User_ID', 'Genre_ID', 'rating', 'Genre']) # created empty Dataframe with 4 different columns
                df.loc[0] = [99, 0, rating, "Genre1:classical - vocals"] #added following values into the Dataframe created above

                model , data, user2user_encoded, genre2genre_encoded, genre_encoded2genre = (
                    create_collaboration_model('InputRatingData.csv', 'Genre_IDs.csv', 100, df))
                # we specify the details of our model and input the above created Dataframe into the collaborative Filtering Function

                recommendations = predict_rating(model, data, 'Genre_IDs.csv', 99, user2user_encoded, genre2genre_encoded, genre_encoded2genre)
                print(recommendations)


    else:
        print ("Exiting")

# Invoke Main
main()






