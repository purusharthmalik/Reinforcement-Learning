import torch
import numpy as np
import sounddevice as sd
import queue
import threading
import logging
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from typing import List, Dict, Optional

class AudioStreamProcessor:
    def __init__(
        self, 
        feature_extractor: Wav2Vec2FeatureExtractor, 
        wav2vec_model: Wav2Vec2Model, 
        diarization_agent,
        chunk_duration: float = 0.5,
        max_rewards_before_update: int = 10
    ):
        # Logging setup
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.feature_extractor = feature_extractor
        self.wav2vec_model = wav2vec_model
        self.diarization_agent = diarization_agent
        
        # Audio processing setup
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.chunk_duration = chunk_duration
        
        # Reward update configuration
        self.max_rewards_before_update = max_rewards_before_update
        
        # Tracking variables
        self.sample_rate = None
    
    def start_stream(self, sample_rate):
        """Start audio stream processing"""
        self.is_recording = True
        self.sample_rate = sample_rate
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Start audio capture thread
        threading.Thread(target=self._capture_audio, daemon=True).start()
        
        # Start audio processing thread
        threading.Thread(target=self._process_audio_chunks, daemon=True).start()
    
    def stop_stream(self):
        """Stop audio stream processing"""
        self.is_recording = False
    
    def _capture_audio(self):
        def audio_callback(indata, frames, time, status):
            if status:
                self.logger.warning(f"Audio capture status: {status}")
            try:
                self.audio_queue.put_nowait(indata.copy())
            except queue.Full:
                self.logger.warning("Audio queue full; dropping audio chunk.")
        # Ensure the thread terminates cleanly using an Event
        while not self.stop_event.is_set():
            audio_chunk = indata.copy()
            self.audio_queue.put(audio_chunk)
            
            try:
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=audio_callback
                ):
                    while self.is_recording:
                        sd.sleep(100)  # Short sleep to prevent CPU overload
            except Exception as e:
                self.logger.error(f"Error in audio capture: {e}")
    
    def _process_audio_chunks(self):
        """Process audio chunks for diarization"""
        while self.is_recording:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.5)
                
                # Ensure audio chunk is properly shaped
                audio_chunk = audio_chunk.squeeze()
                
                # Extract features
                features = self._extract_wav2vec_features(audio_chunk)
                
                # Speaker diarization
                speaker_id = self.diarization_agent.choose_speaker(features)
                
                # Update speaker segments
                self.diarization_agent.update_speaker_segment(speaker_id, audio_chunk)
                
                # Compute reward
                reward = self.diarization_agent.compute_reward(features, speaker_id)
                
                # Add reward to agent's rewards list
                self.diarization_agent.rewards.append(reward)
                
                # Periodically update policy
                if len(self.diarization_agent.rewards) >= self.max_rewards_before_update:
                    self.diarization_agent.update_policy()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing audio chunk: {e}")
    
    def _extract_wav2vec_features(self, audio_chunk):
        """Extract features using Wav2Vec2"""
        try:
            # Ensure audio chunk is numpy array and float32
            audio_chunk = audio_chunk.astype(np.float32)
            
            # Handle very short audio chunks
            if len(audio_chunk) < 100:  # Minimum meaningful chunk size
                padding = np.zeros(100 - len(audio_chunk), dtype=np.float32)
                audio_chunk = np.concatenate([audio_chunk, padding])
            
            inputs = self.feature_extractor(
                audio_chunk, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.wav2vec_model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
            
            return features
        
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            # Return a dummy feature tensor if extraction fails
            return torch.zeros((1, self.diarization_agent.feature_dim))