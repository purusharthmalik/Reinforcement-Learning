import os
import numpy as np
import streamlit as st
import torch
import torchaudio
import speech_recognition as sr
import pyaudio
import wave
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity

class SpeakerDiarizationApp:
    def __init__(self):
        # Load pre-trained Wav2Vec2 model
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Initialize speaker embeddings storage
        if 'speaker_embeddings' not in st.session_state:
            st.session_state.speaker_embeddings = []
        
        # Initialize REINFORCE parameters
        self.learning_rate = 0.01
        self.policy_network = self.create_policy_network()
    
    def create_policy_network(self):
        # Simple policy network for speaker assignment
        return torch.nn.Sequential(
            torch.nn.Linear(768, 64),  # Assuming 512-dim embeddings
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),  # Binary action (assign or not)
            torch.nn.Sigmoid()
        )
    
    def resample_audio(self, waveform, original_sample_rate):
        # Create resampler
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate, 
            new_freq=16000
        )
        
        # Resample the waveform
        resampled_waveform = resampler(waveform)
        
        return resampled_waveform
    
    def extract_embedding(self, audio_path):
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample audio to 16000 Hz
        waveform = self.resample_audio(waveform, sample_rate)
        
        # Preprocess audio
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(), 
            sampling_rate=16000,  # Explicitly set to 16000 
            return_tensors="pt"
        )
        
        # Extract embeddings
        with torch.no_grad():
            outputs = self.model(inputs.input_values)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        return embedding
    
    def record_audio(self, duration=5, sample_rate=44100):
        # Record audio using PyAudio
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, 
                            rate=sample_rate, input=True, 
                            frames_per_buffer=1024)
        
        frames = []
        for _ in range(0, int(sample_rate / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Save recorded audio
        output_path = "recorded_audio.wav"
        wf = wave.open(output_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return output_path
    
    def transcribe_audio(self, audio_path):
        # Transcribe audio using SpeechRecognition
        with sr.AudioFile(audio_path) as source:
            audio = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return "Could not understand audio"
            except sr.RequestError:
                return "Could not request results"
    
    def compare_embeddings(self, new_embedding):
        # Compare new embedding with existing embeddings
        similarities = []
        for existing_embedding in st.session_state.speaker_embeddings:
            sim = cosine_similarity(
                new_embedding.reshape(1, -1), 
                existing_embedding.reshape(1, -1)
            )[0][0]
            similarities.append(sim)
        
        # Threshold for speaker similarity
        threshold = 0.7
        matching_speakers = [
            i for i, sim in enumerate(similarities) 
            if sim > threshold
        ]
        
        return matching_speakers
    
    def reinforce_update(self, embedding, action, reward):
        # REINFORCE algorithm for speaker assignment
        log_prob = torch.log(self.policy_network(torch.tensor(embedding, dtype=torch.float32)))
        loss = -log_prob * reward
        
        # Perform gradient descent
        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def run(self):
        st.title("Speaker Diarization and Recognition App")
        
        # Audio input method selection
        input_method = st.radio("Select Audio Input Method", 
                                ["Upload Audio File", "Record Live Audio"])
        
        audio_path = None
        if input_method == "Upload Audio File":
            uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
            if uploaded_file is not None:
                with open("temp_audio.wav", "wb") as f:
                    f.write(uploaded_file.getvalue())
                audio_path = "temp_audio.wav"
        else:
            if st.button("Start Recording"):
                audio_path = self.record_audio()
        
        if audio_path:
            # Extract embedding
            embedding = self.extract_embedding(audio_path)
            
            # Transcribe audio
            transcription = self.transcribe_audio(audio_path)
            st.write("Transcription:", transcription)
            
            # Compare with existing speakers
            matching_speakers = self.compare_embeddings(embedding)
            
            if matching_speakers:
                st.write(f"Detected as similar to Speaker(s): {matching_speakers}")
                
                # Allow user confirmation
                user_confirmation = st.radio(
                    "Is the speaker assignment correct?", 
                    ["Yes", "No"]
                )
                
                # REINFORCE update based on user feedback
                reward = 1 if user_confirmation == "Yes" else -1
                self.reinforce_update(embedding, None, reward)
                
                # Store embedding if it's a new speaker
                if not matching_speakers:
                    st.session_state.speaker_embeddings.append(embedding)
            else:
                st.write("New speaker detected!")
                st.session_state.speaker_embeddings.append(embedding)

def main():
    app = SpeakerDiarizationApp()
    app.run()

if __name__ == "__main__":
    main()