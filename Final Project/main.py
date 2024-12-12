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
        
        # Initialize speaker embeddings storage and names
        if 'speaker_embeddings' not in st.session_state:
            st.session_state.speaker_embeddings = []
            st.session_state.speaker_names = []  # Store names of speakers
        
        # Initialize REINFORCE parameters
        self.learning_rate = 0.01
        self.policy_network = self.create_policy_network()
    
    def create_policy_network(self):
        return torch.nn.Sequential(
            torch.nn.Linear(768, 64),  # Assuming 768-dim embeddings
            torch.nn.ReLU(),
            torch.nn.Linear(64, len(st.session_state.speaker_names) + 1),  # +1 for new user action
            torch.nn.Softmax(dim=1)
        )
    
    def resample_audio(self, waveform, original_sample_rate):
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_sample_rate, 
            new_freq=16000
        )
        return resampler(waveform)
    
    def extract_embedding(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = self.resample_audio(waveform, sample_rate)
        
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(inputs.input_values)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        return embedding
    
    def record_audio(self, duration=5, sample_rate=44100):
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
        
        output_path = "recorded_audio.wav"
        wf = wave.open(output_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return output_path
    
    def transcribe_audio(self, audio_path):
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
        similarities = []
        for existing_embedding in st.session_state.speaker_embeddings:
            sim = cosine_similarity(
                new_embedding.reshape(1, -1), 
                existing_embedding.reshape(1, -1)
            )[0][0]
            similarities.append(sim)
        
        threshold = 0.7
        matching_speakers = [
            i for i, sim in enumerate(similarities) 
            if sim > threshold
        ]
        
        return matching_speakers
    
    def reinforce_update(self, embedding, action_index, reward):
        log_prob = torch.log(self.policy_network(torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)))
        
        loss = -log_prob[0][action_index] * reward
        
        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)

    def run(self):
        st.title("Speaker Diarization Using REINFORCE")

        # Sidebar for registered users
        st.sidebar.header("Registered Users")
        
        if 'speaker_names' in st.session_state and st.session_state.speaker_names:
            for idx, name in enumerate(st.session_state.speaker_names):
                st.sidebar.write(f"Speaker {idx + 1}: {name}")
        
        new_user_name = st.sidebar.text_input("Name for new user (if detected):")
        
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
                with st.spinner("Recording... Please speak!"):
                    audio_path = self.record_audio()

                    # After recording is done
                    st.success("Recording complete!")

        
        if audio_path:
            with st.spinner("Processing audio..."):
                embedding = self.extract_embedding(audio_path)

                transcription = self.transcribe_audio(audio_path)
                st.write("Transcription:", transcription)

                matching_speakers = self.compare_embeddings(embedding)

                cosine_sim_result = None

                if matching_speakers:
                    speaker_indices = ', '.join(str(i + 1) for i in matching_speakers)
                    cosine_sim_result = f"Detected as similar to Speaker(s): {speaker_indices}"
                    st.write(cosine_sim_result)

                    # Choose the first matching speaker for simplicity (could be improved with more options)
                    action_index_cosine_similarities = matching_speakers[0]

                    user_confirmation = st.radio(
                        "Is the speaker assignment correct?", 
                        ["Yes", "No"]
                    )

                    submit_feedback_button = st.button("Submit Feedback")

                    if submit_feedback_button:
                        # Determine reward based on user confirmation and action taken by RL model
                        if user_confirmation == "Yes":
                            reward_rl_model = 1  # Correct detection by RL model matches cosine similarity.
                        else:
                            reward_rl_model = -1  # Incorrect detection.
                        
                        self.reinforce_update(embedding, action_index_cosine_similarities, reward_rl_model)

                        # Print confirmation message after policy update
                        st.success("Model updated successfully!")
                
                        if user_confirmation == "No" and new_user_name:
                            # Add new user name to session state and embeddings list
                            st.session_state.speaker_names.append(new_user_name)
                            st.session_state.speaker_embeddings.append(embedding)

                else:
                    cosine_sim_result = "New speaker detected!"
                    st.write(cosine_sim_result)

                    if new_user_name:  # If a name is provided for the new user.
                        st.session_state.speaker_names.append(new_user_name)
                        st.session_state.speaker_embeddings.append(embedding)

def main():
    app = SpeakerDiarizationApp()
    app.run()

if __name__ == "__main__":
    main()