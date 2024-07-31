import librosa 
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy

filename = "/home/rafael/Math_code/MOS/VoiceMOS_2023_track3/VoiceMOS2023Track3-clean_TMHINT_g4_27_01.wav"

y , sr = librosa.load(filename)

D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

print(sr)

plt.figure(figsize=(12, 8))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram (dB)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.show()

