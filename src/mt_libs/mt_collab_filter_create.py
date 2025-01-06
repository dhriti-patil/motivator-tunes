import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from .mt_reco import RecommenderNet
import os

# Globals
genre_id_const = "Genre_ID"
user_id_const = "User_ID"
rating_const = "rating"

# Defining Static Data Frame
rating_data_frame = None
flag_loaded_rating_file = False

def create_collaboration_model(RATING_FILE, ID_file, epochs, df):
    global rating_data_frame , flag_loaded_rating_file

    if(flag_loaded_rating_file == False):
        rating_data_frame = pd.read_csv(RATING_FILE)
        flag_loaded_rating_file = True

    rating_data_frame = pd.concat([rating_data_frame, df])

    user_ids = rating_data_frame[user_id_const].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}

    Genre_ids = rating_data_frame[genre_id_const].unique().tolist()
    genre2genre_encoded = {x: i for i, x in enumerate(Genre_ids)}
    genre_encoded2genre = {i: x for i, x in enumerate(Genre_ids)}


    rating_data_frame[user_id_const] = rating_data_frame[user_id_const].map(user2user_encoded)
    rating_data_frame[genre_id_const] = rating_data_frame[genre_id_const].map(genre2genre_encoded)
    num_users = len(user2user_encoded)
    num_genres = len(genre_encoded2genre)

    rating_data_frame[rating_const] = rating_data_frame[rating_const].values.astype(np.float32)

    # min and max ratings will be used to normalize the ratings later
    min_rating = min(rating_data_frame[rating_const])
    max_rating = max(rating_data_frame[rating_const])
    print(
        "Number of {}s: {}, Number of {}s: {}, Min {}: {}, Max rating: {}".format(
            user_id_const, num_users, genre_id_const, num_genres, rating_const, min_rating, max_rating
        ))

    rating_data_frame = rating_data_frame.sample(frac=1, random_state=42)

    x = rating_data_frame[[user_id_const, genre_id_const]].values
    y = rating_data_frame[rating_const].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    train_indices = int(0.9 * rating_data_frame.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:],
    )

    EMBEDDING_SIZE = 32
    model = RecommenderNet(num_users, num_genres, EMBEDDING_SIZE)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )

    history = model.fit(
        x = x_train,
        y = y_train,
        batch_size=64,
        epochs = epochs,
        verbose = 1,
        validation_data = (x_val, y_val),
    )
    model.summary()
    test_loss = model.evaluate(x_val, y_val)
    print('\\nTest Loss: {}'.format(test_loss))

    # Return the Model
    return model , rating_data_frame, user2user_encoded, genre2genre_encoded, genre_encoded2genre