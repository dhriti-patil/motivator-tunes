import keras
import numpy as np
from .mt_muselsl import record_direct
from .mt_utils import GetFFT , FilterData
import time

model = None
sampling_freq = 256.00
array_size = 512
muse_device = "Muse-15D3"

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
    header_names = ["TP9", "AF7", "AF8", "TP10"]
    while True:
        eeg_from_muse = None
        merged_eeg = None
        try:
            eeg_from_muse = record_direct(5, None, filename=None, backend='auto', interface=None, name=muse_device)
            data_tp9 = eeg_from_muse["TP9"].tolist()
            data_af7 = eeg_from_muse["AF7"].tolist()
            data_af8 = eeg_from_muse["AF8"].tolist()
            data_tp10 = eeg_from_muse["TP10"].tolist()
            merged_eeg = np.add(np.add(np.add(data_tp9, data_af7), data_af8), data_tp10)
        except Exception as e:
            print (e)

        time.sleep(2)

        print(merged_eeg)
        # fft_datas = process(eeg_from_muse)
        # Predicted_Label = predict(model, fft_datas)

        # print(Predicted_Label)




