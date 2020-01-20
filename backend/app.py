from flask import Flask
from flask import request
from flask import render_template
#import requests
import numpy as np
from sklearn.externals import joblib
from keras.models import load_model
#loaded_model=load_model('model_aud.h5',compile=False)
from flask_socketio import SocketIO
import pyaudio
import wave
import speech_recognition as sr
import keras
import pandas as pd
import numpy as np
import os
import random
import sys
import glob 
import keras
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io.wavfile
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm
import tensorflow as tf
	
import nltk.classify.util #calculates accuracy
from nltk.classify import NaiveBayesClassifier #imports the classifier Naive Bayes
from nltk.corpus import movie_reviews #imports movie reviews from nltk
from nltk.corpus import stopwords #imports stopwords from nltk
from nltk.corpus import wordnet #imports wordnet(lexical database for the english language) from nltk
	
#import movie_reviews
from nltk.corpus import movie_reviews
graph = tf.get_default_graph()
#loading json model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


#hardcoded suggestions in accordance to LinkedIn suggestion model
sugg=['I ate dinner.',
'We had a three-course meal.',
'Brad did not came to dinner with us.',
'He loves fish tacos.',
'In the end, we all felt like we ate too much.',
'Would not it be lovely to enjoy a week soaking up the culture?',
'Oh, how I love to go!',
'Of all the places to travel, Mexico is at the top of my list.',
'There is so much to understand.',
'I love learning!',
'Sentences come in many shapes and sizes.',
'I do not like apple.']

#sentiment=[1,1,0,1,0,1,0,1,1,1,0,0,1,1,1]




from sklearn.externals import joblib
# run python script 
import nltk 
from nltk.corpus import stopwords 
# Loading the model
#model_sent=joblib.load('sentmodel.pkl')



file= open('sentmodel.pkl','rb')
model_sent=joblib.load(file)


def create_word_features(words):
	useful_words = [word for word in words if word not in stopwords.words("english")]
	my_dict = dict([(word, True) for word in useful_words])
	return my_dict
def preprocess(sentence):
 # nltk.download('punkt')
  sample_sent =sentence
  words = sentence.split(" ")
  words = create_word_features(words)
  return words
 # model.classify(words)



app=Flask(__name__)

app.config['SECRET_KEY'] = 'sukriti'
socketio = SocketIO(app)


@app.route('/')
def home():
	return render_template('home.html')


# @app.route('/record')
# def record():
#   berry.recording()
#   return

def messageReceived(methods=['GET', 'POST']):
	print('message was received!!!')

@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
	print('received my event: ' + str(json))
	socketio.emit('my response', json, callback=messageReceived)



@app.route('/record')
def record():
	output=""
	CHUNK = 1024
	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 44100
	RECORD_SECONDS = 5
	WAVE_OUTPUT_FILENAME = "output.wav"

	print('hi')

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
					channels=CHANNELS,
					rate=RATE,
					input=True,
					frames_per_buffer=CHUNK)

	#recording..


	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)

	#done recording..

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()
	print('bye')



	#speech to text

	r = sr.Recognizer()
	file_audio = sr.AudioFile('output.wav')


	try:
		 with file_audio as source:
			 audio_text = r.record(source)
			 output=r.recognize_google(audio_text)
	except OSError as e:
		 print('audio_file not found')
	except sr.UnknownValueError:
		 print("Google Speech Recognition could not understand audio")



	



	#preprocessing output.wav to get the input df
	#preprocessing: feature extraction
	
	input_duration=3
	#lb = LabelEncoder()



	data1 = pd.DataFrame(columns=['feature'])
	X, sample_rate = librosa.load("output.wav", res_type='kaiser_fast',duration=input_duration,sr=22050*2,offset=0.5)
	sample_rate = np.array(sample_rate)
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
	feature = mfccs
	data1.loc[0]= [feature]
	df13 = pd.DataFrame(data1['feature'].values.tolist()) 
	twodim= np.expand_dims(df13,axis=2)   

	#pred from model 
	with graph.as_default():
		preds= loaded_model.predict(twodim, batch_size=32,verbose=1)
	print(preds)
	preds1= preds.argmax(axis=1)
	live=preds1.astype(int).flatten()

	# print(live)
	# predictions = (lb.inverse_transform((live)))
	# label = predictions[0].split('_')[1]; 

	if live[0]==0:
		label='positive'
	else:
		label='negative'


	print(label);

	
	# text to LinkenIn suggestion model which will return a list of sugg.
	# sentiment = model_sent.predict(sugg)

	# with graph.as_default():
	# 	sentiment = model_sent.predict(sugg)


	outcome=[]
	for sentence in sugg:
	   out=preprocess(sentence)
	   out=model_sent.classify(out)
	   outcome.append(out)


	filtered_sugg=[] 
	for i in range(0,len(sugg)):
		if (outcome[i]==label):
			filtered_sugg.append(sugg[i])


	for i in filtered_sugg:
		print(i+"  ")

	return render_template('home.html',output=output,f_s=filtered_sugg,label=label)
	





 
# @app.route('/predict',methods=['GET','POST'])
# def predict():
#   out=""
#   if(request.method=='POST'):
#       x1=float(request.form["x1"])
#       x2=float(request.form["x2"])
#       x3=float(request.form["x3"])
#       x4=float(request.form["x4"])
#       data=[[x1,x2,x3,x4]]
#       out=model_1.predict(data)
#   return render_template('home.html',out=out)


@app.route('/send/<string:name>')
def send(name):
  return render_template('home.html',msg=name,emotion=1)



if __name__ == '__main__':
	 app.run(host="100.94.14.207",debug = True,port=3000)
