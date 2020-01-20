import speech_recognition as sr
import pyaudio

r1=sr.Recognizer()
r2=sr.Recognizer()
r3=sr.Recognizer()

with sr.Microphone() as source:
     print("start")
     audio =r1.adjust_for_ambient_noise(source)
     audio=r1.listen(source)
     print("end")
out=r2.recognize_google(audio)
print(out)
        
