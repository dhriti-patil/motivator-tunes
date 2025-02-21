import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Model, regularizers

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



