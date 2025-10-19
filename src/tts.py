# src/tts.py
import pyttsx3
engine = pyttsx3.init()
_last = ""
def speak_if_changed(text: str):
    global _last
    if text and text != _last:
        engine.stop()
        engine.say(text)
        engine.runAndWait()
        _last = text
