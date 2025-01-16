import numpy as np
import pandas as pd
import os

def create_input(data, user_id, user2user_encoded, genre2genre_encoded, genre_encoded2genre):
    genre_id_const = "Genre_ID"
    user_id_const = "User_ID"

    new_user_ratings = data.loc[data[user_id_const] == user2user_encoded[user_id]]
    genre_ids_list = [genre2genre_encoded[i] for i in genre2genre_encoded]

    genres_rated_by_user = [entry for entry in new_user_ratings[genre_id_const]]
    encoded_rated_genres = [genre2genre_encoded[genre] for genre in genres_rated_by_user]
    genres_not_rated = [i for i in genre_ids_list if i not in encoded_rated_genres]
    user_genre_array = np.hstack(([[0]] * len(genres_not_rated), [[encoded_id] for encoded_id in genres_not_rated]))
    return user_genre_array, new_user_ratings

def predict(model, user_genre_array):
    predicted_rating = model.predict(user_genre_array).flatten()
    return predicted_rating


def predict_rating(model, data, ID_FILE, new_user_id, user2user_encoded, genre2genre_encoded, genre_encoded2genre):
    plotFileName_3 = "C:/Data/DhritiData/JugendForchst/JF_Project/git/motivator-tunes/src/Data"

    input_data, genres_listened_by_user = create_input(
        data,
        new_user_id,
        user2user_encoded,
        genre2genre_encoded,
        genre_encoded2genre)

    input_data_df = pd.DataFrame(input_data)
    os.makedirs(plotFileName_3, exist_ok=True)
    file_path = os.path.join(plotFileName_3, "collab_model_inp_data.csv")
    input_data_df.to_csv(file_path, index=False)

    Genre_ids = [i[1] for i in input_data]
    genre_data_frame = pd.read_csv(ID_FILE)
    genre_dict = genre_data_frame.set_index('GenreID')['Genre'].to_dict()
    genre_names = [genre_dict[genre_id] for genre_id in Genre_ids if genre_id in genre_dict]

    predicted_rating = predict(model, input_data)
    joined_predictions = pd.DataFrame()
    joined_predictions['Genre_IDs'] = np.array(Genre_ids)
    joined_predictions['Genre_Names'] = np.array(genre_names)
    joined_predictions['Predicted_Rating'] = np.array(predicted_rating)

    sorted_dataframe = joined_predictions.sort_values(by='Predicted_Rating', ascending=False)
    os.makedirs(plotFileName_3, exist_ok=True)
    file_path = os.path.join(plotFileName_3, "sorted_ratings.csv")
    sorted_dataframe.to_csv(file_path, index=False)
    return sorted_dataframe