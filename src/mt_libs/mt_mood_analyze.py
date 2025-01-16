import keras
import numpy as np
import pandas as pd
from .mt_muselsl import record_direct
from .mt_utils import GetFFT , FilterData
import time
import os


model = None
sampling_freq = 256.00
array_size = 512
muse_device = "Muse-15D3"
RECORDING_DURATION = 10



def AnalyzeMood(inp_model_file_path):
    global model

    def process(Data):
        filtered_signals = FilterData(Data, 100, 0.01, 0.06)
        fft_data = GetFFT(filtered_signals, sampling_freq)
        return fft_data

    def predict(model, eeg_data):
        data_array = [np.asarray(eeg_data).astype(np.float32).reshape(-1, array_size, 1)]
        predicted_labels = np.argmax(model.predict(data_array, verbose=0), axis=1)
        return predicted_labels

    print("Analyze Mood")
    model = keras.saving.load_model(inp_model_file_path, custom_objects=None, compile=True, safe_mode=True)

    # while True:
    eeg_from_muse = None
    merged_eeg = None
    plotFileName_2 = "C:/Data/DhritiData/JugendForchst/JF_Project/git/motivator-tunes/src/Data"
    try:
        eeg_from_muse = record_direct(RECORDING_DURATION, None, filename=None, backend='auto', interface=None, name=muse_device)
        data_tp9 = eeg_from_muse["TP9"].tolist()
        data_af7 = eeg_from_muse["AF7"].tolist()
        data_af8 = eeg_from_muse["AF8"].tolist()
        data_tp10 = eeg_from_muse["TP10"].tolist()
        merged_eeg = np.add(np.add(np.add(data_tp9, data_af7), data_af8), data_tp10)
        merged_eeg_df = pd.DataFrame(merged_eeg)
        os.makedirs(plotFileName_2, exist_ok=True)
        file_path = os.path.join(plotFileName_2, "merged_eeg.csv")
        merged_eeg_df.to_csv(file_path, index=False)
    except Exception as e:
        print (e)

    time.sleep(2)

    print(merged_eeg)
    fft_datas = process(merged_eeg)
    Predicted_Label = predict(model, fft_datas)



    if Predicted_Label == [0]:
        print("Predicted Emotion is: Negative")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    elif Predicted_Label == [1]:
        print("Predicted Emotion is: Neutral")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    else:
        print("Predicted Emotion is: Positive")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    return Predicted_Label[0]

#####################################################################################################################


def play_song(subgenre_url):
    # CHROME_COMMAND = "\"C:\Program Files\Google\Chrome\Application\chrome.exe\""
    BROWSER_COMMAND = "\"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe\""
    sys_commad = BROWSER_COMMAND + " " + subgenre_url + " &"
    print ("Executing : " + sys_commad)
    try:
        #os.system(sys_commad)
        print(sys_commad)
    except Exception as e:
        print(e)

def close_chrome():
    sys_commad = "taskkill /F /IM msedge.exe /T > nul"
    print ("Executing : " + sys_commad)
    try:
        os.system(sys_commad)
    except Exception as e:
        print(e)


###################################################################################################################

