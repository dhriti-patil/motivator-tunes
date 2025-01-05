import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def COLAB_MODEL(RATING_FILE,ID_file,epochs):

    genre_id_const = "Genre_ID"
    user_id_const = "User_ID"
    rating_const = "rating"

    data = pd.read_csv(RATING_FILE)

    user_ids = data[user_id_const].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}


    Genre_ids = data[genre_id_const].unique().tolist()
    genre2genre_encoded = {x: i for i, x in enumerate(Genre_ids)}
    genre_encoded2genre = {i: x for i, x in enumerate(Genre_ids)}
    print(genre_encoded2genre)


    data[user_id_const] = data[user_id_const].map(user2user_encoded)
    data[genre_id_const] = data[genre_id_const].map(genre2genre_encoded)
    num_users = len(user2user_encoded)
    num_genres = len(genre_encoded2genre)

    data[rating_const] = data[rating_const].values.astype(np.float32)

    # min and max ratings will be used to normalize the ratings later
    min_rating = min(data[rating_const])
    max_rating = max(data[rating_const])
    print(
        "Number of {}s: {}, Number of {}s: {}, Min {}: {}, Max rating: {}".format(
            user_id_const, num_users, genre_id_const, num_genres, rating_const, min_rating, max_rating
        ))

    data = data.sample(frac=1, random_state=42)

    x = data[[user_id_const, genre_id_const]].values
    y = data[rating_const].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    train_indices = int(0.9 * data.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:],
    )


    EMBEDDING_SIZE = 32
    class RecommenderNet(keras.Model):
        def __init__(self, num_users, num_genres, embedding_size, **kwargs):
            super(RecommenderNet, self).__init__(**kwargs)
            self.num_users = num_users
            self.num_genres = num_genres
            self.embedding_size = embedding_size
            self.user_embedding = layers.Embedding(
                num_users,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
                mask_zero=True
            )
            self.user_bias = layers.Embedding(num_users, 1)
            self.movie_embedding = layers.Embedding(
                num_genres,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
                mask_zero=True
            )
            self.movie_bias = layers.Embedding(num_genres, 1)
        def call(self, inputs):
            user_vector = self.user_embedding(inputs[:, 0])
            user_bias = self.user_bias(inputs[:, 0])
            genre_vector = self.movie_embedding(inputs[:, 1])
            genre_bias = self.movie_bias(inputs[:, 1])
            dot_user_genre = tf.tensordot(user_vector, genre_vector, 2)
            # Add all the components (including bias)
            x = dot_user_genre + user_bias + genre_bias
            # The sigmoid activation forces the rating to between 0 and 1
            return tf.nn.sigmoid(x)

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

    model.save('second.best.model.keras')

    ###########################################
    print("Testing Model with 1 user")

    genre_data_frame = pd.read_csv(ID_file)
    user_id = "new_user"
    genres_listened_by_user = data.sample(1)
    print(genres_listened_by_user)
    genres_not_listened = genre_data_frame[
        ~genre_data_frame["GenreID"].isin(genres_listened_by_user.Genre_ID.values)
    ]["GenreID"]
    genres_not_listened = list(
        set(genres_not_listened).intersection(set(genre2genre_encoded.keys()))
    )
    genres_not_listened = [[genre2genre_encoded.get(x)] for x in genres_not_listened]
    user_genre_array = np.hstack(
        ([[0]] * len(genres_not_listened), genres_not_listened)
    )
    predicted_rating = model.predict(user_genre_array).flatten()
    top_ratings_indices = predicted_rating.argsort()[::-1]
    recommended_genres_ids = [
        genre_encoded2genre.get(genres_not_listened[x][0]) for x in top_ratings_indices
    ]

    print("Showing recommendations for user: {}".format(user_id))
    print("====" * 9)
    print("Genres with high ratings from user")
    print("----" * 8)
    top_genres_user = (
        genres_listened_by_user.sort_values(by="rating", ascending=False)
        .head(5)
        .Genre_ID.values
    )
    genre_df_rows = genre_data_frame[genre_data_frame["GenreID"].isin(top_genres_user)]
    for row in genre_df_rows.itertuples():
        print(row.Genre, ":", row.Details)
        print("----" * 8)
        print("Top 10 movie recommendations")
        print("----" * 8)
        recommended_movies = genre_data_frame[genre_data_frame['GenreID'].isin(recommended_genres_ids)]
        for rows in recommended_movies.itertuples():
            print(rows.Genre, ":", rows.Details)
        print("===" * 9)
        print("Saving Model")
        print("===" * 9)


    ###################################################


