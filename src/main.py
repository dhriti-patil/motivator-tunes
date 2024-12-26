import os
from mt_libs.mt_utils import *

##

SAMPLING_FREQUNCY = 256.00
PLOT_DIR = "C:/Data/DhritiData/JugendForchst/JF_Project/Prototype/input/Plots"
BASE_DIR = "C:/Data/DhritiData/JugendForchst/JF_Project/Prototype/input/"

def main():
    for file in os.listdir(BASE_DIR):
        if file.endswith(".csv"):
            filePath = os.path.join(BASE_DIR, file)

            # Read CSV
            eeg_data = ReadCSV(filePath, "eeg_1","eeg_2","eeg_3","eeg_4")
            eeg_data_filtered = FilterData(eeg_data, 100, 0.01, 0.06)

            # Get FFT
            freqs, psd = GetFFT(eeg_data_filtered, SAMPLING_FREQUNCY)

            plotFileName = PLOT_DIR + "/" + os.path.basename(filePath) + ".png"
            # Save Plots
            print("Generating Plot for " + plotFileName)
            SaveImage(freqs, psd,plotFileName)

# Call Main
main()
