#import os
import sys
from mt_libs.mt_utils import *
from mt_libs.mt_keras_bkp import *
from mt_libs.TempDict import *
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import keras
from keras import layers
import tensorflow as tf
from sklearn import preprocessing, model_selection
import random

##

SAMPLING_FREQUNCY = 256.00
PLOT_DIR = "C:/Data/DhritiData/JugendForchst/JF_Project/Prototype/input/Plots"
BASE_DIR = "C:/Data/DhritiData/JugendForchst/JF_Project/Prototype/input/"
psd_array = []
labels = []
labels_dict = GetLabels()
def main1():
    for file in os.listdir(BASE_DIR):
        if file.endswith(".csv"):
            filePath = os.path.join(BASE_DIR, file)
            labels.append(labels_dict[file])

            # Read CSV
            eeg_data = ReadCSV(filePath, "eeg_1","eeg_2","eeg_3","eeg_4")
            eeg_data_filtered = FilterData(eeg_data, 100, 0.01, 0.06)

            # Get FFT
            freqs, psd = GetFFT(eeg_data_filtered, SAMPLING_FREQUNCY)
            psd_array.append(psd)

            #plotFileName = PLOT_DIR + "/" + os.path.basename(filePath) + ".png"
            # Save Plots
            print("Generating Plot for " + filePath)
            #SaveImage(freqs, psd,plotFileName)

    # Call Keras Main
    KerasMain(psd_array, labels)



QUALITY_THRESHOLD = 128
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 2
def main():
    eeg = pd.read_csv("C:/Data/DhritiData/JugendForchst/JF_Project/Data/eeg-data.csv")
    eeg.drop(
        [
            "indra_time",
            "browser_latency",
            "reading_time",
            "attention_esense",
            "meditation_esense",
            "updatedAt",
            "createdAt",
        ],
        axis=1,
        inplace=True,
    )

    eeg.reset_index(drop=True, inplace=True)
    eeg.head()
    eeg = eeg.iloc[::2]
    eeg = eeg.iloc[::2]

    def convert_string_data_to_values(value_string):
        str_list = json.loads(value_string)
        return str_list

    eeg["raw_values"] = eeg["raw_values"].apply(convert_string_data_to_values)

    eeg = eeg.loc[eeg["signal_quality"] < QUALITY_THRESHOLD]
    eeg.head()

    print("Before replacing labels")
    print(eeg["label"].unique(), "\n")
    print(len(eeg["label"].unique()), "\n")

    eeg.replace(
        {
            "label": {
                "blink1": "blink",
                "blink2": "blink",
                "blink3": "blink",
                "blink4": "blink",
                "blink5": "blink",
                "math1": "math",
                "math2": "math",
                "math3": "math",
                "math4": "math",
                "math5": "math",
                "math6": "math",
                "math7": "math",
                "math8": "math",
                "math9": "math",
                "math10": "math",
                "math11": "math",
                "math12": "math",
                "thinkOfItems-ver1": "thinkOfItems",
                "thinkOfItems-ver2": "thinkOfItems",
                "video-ver1": "video",
                "video-ver2": "video",
                "thinkOfItemsInstruction-ver1": "thinkOfItemsInstruction",
                "thinkOfItemsInstruction-ver2": "thinkOfItemsInstruction",
                "colorRound1-1": "colorRound1",
                "colorRound1-2": "colorRound1",
                "colorRound1-3": "colorRound1",
                "colorRound1-4": "colorRound1",
                "colorRound1-5": "colorRound1",
                "colorRound1-6": "colorRound1",
                "colorRound2-1": "colorRound2",
                "colorRound2-2": "colorRound2",
                "colorRound2-3": "colorRound2",
                "colorRound2-4": "colorRound2",
                "colorRound2-5": "colorRound2",
                "colorRound2-6": "colorRound2",
                "colorRound3-1": "colorRound3",
                "colorRound3-2": "colorRound3",
                "colorRound3-3": "colorRound3",
                "colorRound3-4": "colorRound3",
                "colorRound3-5": "colorRound3",
                "colorRound3-6": "colorRound3",
                "colorRound4-1": "colorRound4",
                "colorRound4-2": "colorRound4",
                "colorRound4-3": "colorRound4",
                "colorRound4-4": "colorRound4",
                "colorRound4-5": "colorRound4",
                "colorRound4-6": "colorRound4",
                "colorRound5-1": "colorRound5",
                "colorRound5-2": "colorRound5",
                "colorRound5-3": "colorRound5",
                "colorRound5-4": "colorRound5",
                "colorRound5-5": "colorRound5",
                "colorRound5-6": "colorRound5",
                "colorInstruction1": "colorInstruction",
                "colorInstruction2": "colorInstruction",
                "readyRound1": "readyRound",
                "readyRound2": "readyRound",
                "readyRound3": "readyRound",
                "readyRound4": "readyRound",
                "readyRound5": "readyRound",
                "colorRound1": "colorRound",
                "colorRound2": "colorRound",
                "colorRound3": "colorRound",
                "colorRound4": "colorRound",
                "colorRound5": "colorRound",
            }
        },
        inplace=True,
    )


    print("After replacing labels")
    print(eeg["label"].unique())
    print(len(eeg["label"].unique()))

    # Call Keras Main
    KerasMain(eeg["raw_values"], eeg["label"])



# Call Main
main()
