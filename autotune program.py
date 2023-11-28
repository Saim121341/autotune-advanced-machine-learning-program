import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog
import librosa
import numpy as np
import psola
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech
import soundfile as sf
import torch

class AutoTuneApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle('Voice Matching Auto-Tune Application')
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        # Buttons and status label
        self.load_button = QPushButton('Load Audio', self)
        self.load_button.clicked.connect(self.load_audio)
        layout.addWidget(self.load_button)

        self.process_button = QPushButton('Process Audio', self)
        self.process_button.clicked.connect(self.process_audio)
        layout.addWidget(self.process_button)

        self.save_button = QPushButton('Save Audio', self)
        self.save_button.clicked.connect(self.save_audio)
        layout.addWidget(self.save_button)

        self.status_label = QLabel('Status: Waiting for input', self)
        layout.addWidget(self.status_label)

        # Variables for audio processing
        self.audio, self.sr, self.processed_audio = None, None, None

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.wav *.mp3)")
        if file_path:
            self.audio, self.sr = librosa.load(file_path, sr=None)
            self.status_label.setText('Audio loaded successfully')

    def save_audio(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Audio File", "", "Audio Files (*.wav *.mp3)")
        if file_path and self.processed_audio is not None:
            sf.write(file_path, self.processed_audio, self.sr)
            self.status_label.setText('Audio saved successfully')
        else:
            self.status_label.setText('No processed audio to save')

    def process_audio(self):
        if self.audio is not None:
            try:
                chroma = advanced_voice_analysis(self.audio, self.sr)
                f0, voiced_flag, voiced_probabilities = librosa.pyin(self.audio, sr=self.sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                corrected_f0 = np.nan_to_num(f0)
                vocoded_audio = apply_psola(self.audio, self.sr, corrected_f0)
                self.processed_audio = voice_transformation_with_speecht5(vocoded_audio, self.sr)
                self.status_label.setText("Audio processed successfully.")
            except Exception as e:
                self.status_label.setText(f"Error: {e}")
        else:
            self.status_label.setText('No audio file loaded')

def advanced_voice_analysis(audio, sr):
    D = librosa.stft(audio)
    s = np.abs(librosa.stft(audio)**2)
    chroma = librosa.feature.chroma_stft(S=s, sr=sr)
    return chroma

def voice_transformation_with_speecht5(audio, sr):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
    model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
    inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
    speaker_embeddings = None  # Placeholder for speaker embeddings
    with torch.no_grad():
        speech = model.generate_speech(inputs["input_values"], speaker_embeddings)
    return speech.numpy()

def apply_psola(audio, sr, target_pitch):
    return psola.vocode(audio=audio, sample_rate=sr, target_pitch=target_pitch)

# Main loop
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AutoTuneApp()
    ex.show()
    sys.exit(app.exec_())
