import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import List, Dict, Tuple

class SpeakerDiarizationAgent:
    def __init__(self, feature_dim, max_segments=100):
        self.max_segments = max_segments

class SpeakerDiarizationAgent:
    def __init__(self, feature_dim, num_speakers_max=5):
        # Logging setup
        self.logger = logging.getLogger(__name__)
        
        # Ensure feature_dim is appropriate
        self.feature_dim = feature_dim
        
        # Neural network for policy approximation
        self.policy_network = PolicyNetwork(feature_dim, num_speakers_max)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.num_speakers_max = num_speakers_max
        
        # Initialize rewards as an instance attribute
        self.rewards = []
        
        # Episode memory and tracking
        self.reset_episode_memory()
        
        # Speaker tracking
        self.speaker_segments = {}
        
    def reset_episode_memory(self):
        """Reset episode memory for policy learning"""
        self.log_probs = []
        # Note: Do NOT reset rewards here
        # Rewards should be managed separately
        
    def choose_speaker(self, audio_features: torch.Tensor) -> int:
        """
        Choose speaker based on current audio features using policy network
        """
        try:
            # Ensure input is 2D tensor with batch dimension
            if audio_features.dim() == 1:
                audio_features = audio_features.unsqueeze(0)
            
            # Validate input size
            if audio_features.size(1) != self.feature_dim:
                # Pad or truncate features if needed
                if audio_features.size(1) < self.feature_dim:
                    padding = torch.zeros(
                        audio_features.size(0), 
                        self.feature_dim - audio_features.size(1),
                        dtype=audio_features.dtype
                    )
                    audio_features = torch.cat([audio_features, padding], dim=1)
                else:
                    audio_features = audio_features[:, :self.feature_dim]
            
            action_probs = self.policy_network(audio_features)
            speaker_dist = torch.distributions.Categorical(action_probs)
            
            # Sample speaker assignment
            speaker_id = speaker_dist.sample()
            log_prob = speaker_dist.log_prob(speaker_id)
            
            self.log_probs.append(log_prob)
            return speaker_id.item()
        
        except Exception as e:
            self.logger.error(f"Error in choose_speaker: {e}")
            return 0  # Default to first speaker in case of error
    
    def update_speaker_segment(self, speaker_id, audio_chunk):
        """
        Update speaker segments
        """
        try:
            if speaker_id not in self.speaker_segments:
                self.speaker_segments[speaker_id] = []
            
            self.speaker_segments[speaker_id].append(audio_chunk)
            
            if len(self.speaker_segments[speaker_id]) > self.max_segments:
                self.speaker_segments[speaker_id].pop(0)
        except Exception as e:
            self.logger.error(f"Error updating speaker segment: {e}")
    
    def compute_reward(self, audio_features, assigned_speaker):
        """
        Compute reward based on speaker characteristics
        
        Args:
            audio_features (torch.Tensor): Features of the audio chunk
            assigned_speaker (int): ID of the assigned speaker
        
        Returns:
            torch.Tensor: Computed reward
        """
        try:
            # Base reward
            base_reward = 1.0
            
            # Reward based on segment characteristics
            if assigned_speaker in self.speaker_segments:
                # Number of segments for this speaker
                segment_count = len(self.speaker_segments[assigned_speaker])
                
                # Logarithmic reward for segment diversity
                base_reward += np.log1p(segment_count) * 0.1
                
                # Optional: Add feature coherence reward
                # This is a simplified version and should be replaced with 
                # more sophisticated feature comparison in a real-world scenario
                if len(self.speaker_segments[assigned_speaker]) > 1:
                    # Compare current features with previous segments
                    prev_segments = self.speaker_segments[assigned_speaker][:-1]
                    
                    # Simple feature similarity computation
                    similarity_scores = [
                        torch.nn.functional.cosine_similarity(
                            audio_features, 
                            torch.from_numpy(prev_segment).unsqueeze(0), 
                            dim=1
                        ).mean().item() 
                        for prev_segment in prev_segments
                    ]
                    
                    # Add reward for feature consistency
                    base_reward += np.mean(similarity_scores) * 0.2
            
            return torch.tensor(base_reward, dtype=torch.float32)
        
        except Exception as e:
            self.logger.error(f"Error computing reward: {e}")
            return torch.tensor(1.0, dtype=torch.float32)
    
    def update_policy(self):
        """
        REINFORCE policy gradient update
        """
        try:
            # Ensure we have log probabilities and rewards
            if not self.log_probs or not self.rewards:
                self.logger.warning("No log probs or rewards to update policy")
                return
            
            # Compute policy loss
            policy_loss = []
            for log_prob, reward in zip(self.log_probs, self.rewards):
                policy_loss.append(-log_prob * reward)
            
            # Perform optimization
            self.optimizer.zero_grad()
            policy_loss = torch.stack(policy_loss).mean()
            policy_loss.backward()
            self.optimizer.step()
            
            # Reset episode memory
            self.reset_episode_memory()
            
            # Optionally clear rewards after policy update
            # Depends on your specific implementation strategy
            self.rewards.clear()
        
        except Exception as e:
            self.logger.error(f"Error updating policy: {e}")

class PolicyNetwork(nn.Module):
    def __init__(self, feature_dim, num_speakers_max):
        super().__init__()
        # Adaptive network architecture
        self.feature_dim = feature_dim
        self.num_speakers_max = num_speakers_max
        
        # Adaptive layers based on feature dimension
        self.network = nn.Sequential(
            nn.Linear(feature_dim, max(64, feature_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(64, feature_dim // 2), num_speakers_max),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)