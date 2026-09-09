import numpy as np
import icassp.config as config

class EnergyBaselineVAD:
    """
    Standard Energy and Spectral-Flux Voice Activity Detector.
    Used as the Control Baseline representing current Siri / Gemini Live / Silero VAD behavior.
    """
    def __init__(self, energy_threshold=-6.0, silence_cutoff_ms=config.T_BASE_MS):
        self.energy_threshold = energy_threshold
        self.silence_cutoff_ms = silence_cutoff_ms
        self.hop_ms = config.HOP_SIZE_MS

    def process_stream(self, acoustic_features):
        """
        Processes a sequence of frame features and returns VAD decisions and interruption events.
        Args:
            acoustic_features: np.ndarray of shape (num_frames, ACOUSTIC_FEAT_DIM)
        Returns:
            vad_decisions: np.ndarray of shape (num_frames,) -> 1 (speech active), 0 (silence)
            cutoff_events: list of frame indices where VAD triggered Turn-End Interruption
        """
        num_frames = len(acoustic_features)
        vad_decisions = np.zeros(num_frames, dtype=np.int32)
        cutoff_events = []
        
        silence_counter_ms = 0.0
        in_speech = False
        
        for f in range(num_frames):
            energy = acoustic_features[f, 0]  # Log energy feature
            is_active = energy > self.energy_threshold
            
            if is_active:
                in_speech = True
                silence_counter_ms = 0.0
                vad_decisions[f] = 1
            else:
                vad_decisions[f] = 0
                if in_speech:
                    silence_counter_ms += self.hop_ms
                    if silence_counter_ms >= self.silence_cutoff_ms:
                        # VAD fires Premature Turn Completion!
                        cutoff_events.append(f)
                        in_speech = False
                        silence_counter_ms = 0.0
                        
        return vad_decisions, cutoff_events
