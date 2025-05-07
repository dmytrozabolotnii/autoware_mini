import csv
import os
from gtts import gTTS
from pydub import AudioSegment # Needs ffmpeg installed!
import librosa
import soundfile as sf

CSV_PATH = '/home/rauno/autoware_mini_ws/src/autoware_mini/config/monitoring/lexus.csv'
OUTPUT_DIR = '/home/rauno/autoware_mini_ws/src/autoware_mini/nodes/platform/monitoring/sounds/generated'
FIXED_DURATION_MS = 2000  # 2 seconds per message

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Delete all .wav files in the output directory
print("Cleaning up old .wav files...")
for fname in os.listdir(OUTPUT_DIR):
    if fname.endswith('.wav'):
        os.remove(os.path.join(OUTPUT_DIR, fname))
print("Cleanup complete.")

def generate_wav(text, filename):
    tts = gTTS(text=text, lang='en')
    tts.save(f"{filename}.mp3") # Need to save as mp3 first (tts doesn't support wav)
    audio = AudioSegment.from_mp3(f"{filename}.mp3")
    audio.export(f"{filename}.wav", format="wav")
    os.remove(f"{filename}.mp3") # Remove the temporary mp3 file
    speed = len(audio) / FIXED_DURATION_MS
    
    # Fit the audio to the fixed duration, and make it louder
    y, sr = librosa.load(f"{filename}.wav", sr=None)
    y_stretched = librosa.effects.time_stretch(y, rate=speed)
    y_stretched = y_stretched * 10 # Louder volume
    y_stretched = librosa.util.normalize(y_stretched) # Normalize volume
    sf.write(f"{filename}.wav", y_stretched, sr)

with open(CSV_PATH, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        component = row['component'].strip()
        safe_component = component.replace(' ', '_').lower()
        messages = [
            (f"{component} frequency restored", f"{safe_component}_frequency_ok"),
            (f"{component} delay restored", f"{safe_component}_delay_ok"),
            (f"{component} warning. frequency too low", f"{safe_component}_frequency_warning"),
            (f"{component} warning. delay too high", f"{safe_component}_delay_warning"),
            (f"{component} error. frequency too low", f"{safe_component}_frequency_error"),
            (f"{component} error. delay too high", f"{safe_component}_delay_error"),
        ]
        for text, fname in messages:
            out_path = os.path.join(OUTPUT_DIR, fname)
            print(f"Generating: {out_path}")
            generate_wav(text, out_path)