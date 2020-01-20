# Voice-sentiment-analysis
About the data set:
The dataset used is RAVDESS in which there are two types of data: speech and song from 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. 
Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, and song contains calm, happy, sad, angry, and fearful emotions. 
Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. For creating the dataset, 2 classes are created: Positive (happy, calm) and Negative (angry, 
fearful, sad).
 
Audio Augmentation techniques used for increasing dataset size:
1. White Noise Addition
2. Pitch Tuning

Feature Extraction from audio (approx. 260 features per audio extracted) :
Using Mel Frequency Cepstral Coefficients
    Steps Involved:
    * Take the Fourier transform of (a windowed excerpt of) a signal.
    * Map the powers of the spectrum obtained above onto the mel scale, using triangular overlapping windows.
    * Take the logs of the powers at each of the mel frequencies.
    * Take the discrete cosine transform of the list of mel log powers, as if it were a signal.
    * The MFCCs are the amplitudes of the resulting spectrum.
Loading audio data and converting it to MFCC format can be easily done by the Python package librosa.

Our CNN model:
1. The CNN model is developed with Keras and constructed with 7 layers: 6 Conv1D layers followed by a Dense layer.
2. After creating the model, its weights are stored in model.h5 and provided 71.67% test accuracy.
3. This model is further converted into json file and used for predictions.

Consolidating speech with LinkedIn chat:
Based on the emotion of a person analysed through voice, we will provide some suitable suggestions to the sender for further conversation. Also, the sender's emotion will be sent at the receiver end
for the receiver to understand the mood of the sender and this will help both the sender as well as the receiver to respond further. 


COMMANDS TO RUN:
1. python app.py
2. tap start recording on the directed URL
3. filtered suggestions are displayed in accordance to the sentiment
