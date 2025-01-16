import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import os

import builtins


##############################################################
# Read CSV
##############################################################
def ReadCSV(filePath, *headerNames):
    # Read CSV
    data = pd.read_csv(filePath)

    # Process Headers
    print("Num Headers : " + str(len(headerNames)))

    return_eeg = []
    for headerName in headerNames:
        data_eeg = data[headerName].tolist()
        data_eeg_cleaned = [x for x in data_eeg if str(x) != 'nan']
        for elem in data_eeg_cleaned:
            return_eeg.append(elem)
    return return_eeg


##############################################################
# Filter Data
##############################################################
def FilterData(data_eeg_cleaned, numTaps, band1, band2):
    plotFileName = "C:\Data\DhritiData\JugendForchst\JF_Project\git\motivator-tunes\src\Data"
    os.makedirs(plotFileName, exist_ok=True)
    plotFileName = os.path.join(plotFileName, "Filter_signal.png")

    filter = signal.firwin(numTaps, [band1, band2], pass_zero=False)
    filteredSignal = signal.convolve(data_eeg_cleaned, filter, mode='same')
    plt.figure(figsize=(8, 4))
    plt.plot(filteredSignal,color='k', lw=2)
    plt.savefig(plotFileName)
    plt.close()
    return filteredSignal


##############################################################
# Get FFT
##############################################################
def GetFFT(filteredSignal, frequency):
    plotFileName = "C:\Data\DhritiData\JugendForchst\JF_Project\git\motivator-tunes\src\Data"
    os.makedirs(plotFileName, exist_ok=True)
    plotFileName = os.path.join(plotFileName, "FFT.png")

    win = 4 * frequency
    freqs, psd = signal.welch(filteredSignal, frequency, nperseg=win)
    truncated_freqs = freqs[:512]
    truncated_psd = psd[:512]
    SaveImage(truncated_freqs, truncated_psd, plotFileName)
    return truncated_psd


##############################################################
# Save Image
##############################################################
def SaveImage(X_Values, Y_Values, plotPath):
    plt.figure(figsize=(8, 4))
    plt.plot(X_Values, Y_Values, color='k', lw=2)
    plt.xlabel('Frequency')
    plt.ylabel('Power spectral density')
    plt.ylim([0, Y_Values.max() * 1.1])
    plt.xlim([0, 30])
    plotFileName = plotPath
    plt.savefig(plotFileName)
    plt.close()

def WriteToCSV(fileName,*args):
    print(open is builtins.open)
    with open(fileName, 'a') as file1:
        strout = ""
        for arg in args:
            strout = strout + "," + str(arg)
        file1.write(strout)
    file1.close()



