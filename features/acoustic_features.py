import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window
try:
    import config as config
except ImportError:
    import icassp.config as config

def compute_spectral_tilt(magnitude_spectrum, freqs):
    """
    Computes spectral tilt (slope of log magnitude vs log frequency).
    High glottal tension during blocks flattens or shifts spectral tilt.
    """
    valid_mask = (freqs >= config.SPECTRAL_TILT_BANDS[0]) & (freqs <= config.SPECTRAL_TILT_BANDS[1])
    if not np.any(valid_mask):
        return 0.0
    
    log_f = np.log(freqs[valid_mask] + 1e-6)
    log_m = np.log(magnitude_spectrum[valid_mask] + 1e-6)
    
    # Linear slope
    slope = np.polyfit(log_f, log_m, 1)[0]
    return float(slope)

def compute_spectral_entropy(magnitude_spectrum):
    """
    Computes normalized spectral entropy.
    Tense glottal airflow during silent blocks produces structured friction (lower entropy)
    compared to unvoiced random noise floor.
    """
    psd = magnitude_spectrum ** 2
    total_energy = np.sum(psd) + 1e-12
    p = psd / total_energy
    p = p[p > 1e-12]
    entropy = -np.sum(p * np.log2(p))
    max_entropy = np.log2(len(magnitude_spectrum))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0

def extract_frame_features(waveform, sample_rate=config.SAMPLE_RATE):
    """
    Extracts frame-by-frame micro-physiological acoustic features.
    Returns:
        features: np.ndarray of shape (num_frames, config.ACOUSTIC_FEAT_DIM)
    """
    if len(waveform) < config.FRAME_LENGTH:
        # Pad short audio
        waveform = np.pad(waveform, (0, config.FRAME_LENGTH - len(waveform)))
        
    num_samples = len(waveform)
    num_frames = max(1, 1 + (num_samples - config.FRAME_LENGTH) // config.HOP_LENGTH)
    
    features = np.zeros((num_frames, config.ACOUSTIC_FEAT_DIM), dtype=np.float32)
    window = get_window('hamming', config.FRAME_LENGTH)
    freqs = rfftfreq(config.N_FFT, 1.0 / sample_rate)
    
    subglottal_mask = (freqs >= config.SUBGLOTTAL_FREQ_MIN) & (freqs <= config.SUBGLOTTAL_FREQ_MAX)
    
    prev_spectrum = None
    
    for i in range(num_frames):
        start = i * config.HOP_LENGTH
        end = start + config.FRAME_LENGTH
        frame = waveform[start:end]
        
        if len(frame) < config.FRAME_LENGTH:
            frame = np.pad(frame, (0, config.FRAME_LENGTH - len(frame)))
            
        # Log frame energy
        energy = np.log(np.mean(frame ** 2) + 1e-8)
        
        # Zero Crossing Rate
        zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0
        
        # FFT Spectrum
        windowed_frame = frame * window
        spectrum = np.abs(rfft(windowed_frame, n=config.N_FFT))
        
        # Sub-glottal high-frequency micro-friction energy (>4.5kHz)
        subglottal_energy = np.log(np.mean(spectrum[subglottal_mask] ** 2) + 1e-8) if np.any(subglottal_mask) else 0.0
        
        # Spectral Tilt
        tilt = compute_spectral_tilt(spectrum, freqs)
        
        # Spectral Entropy
        entropy = compute_spectral_entropy(spectrum)
        
        # Spectral Centroid
        centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-8)
        
        # Spectral Flatness
        geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-8)))
        arithmetic_mean = np.mean(spectrum) + 1e-8
        flatness = geometric_mean / arithmetic_mean
        
        # Formant / Spectral Static Freeze Index (Cosine similarity vs previous frame)
        if prev_spectrum is not None:
            norm1 = np.linalg.norm(spectrum) + 1e-8
            norm2 = np.linalg.norm(prev_spectrum) + 1e-8
            freeze_idx = np.dot(spectrum, prev_spectrum) / (norm1 * norm2)
        else:
            freeze_idx = 1.0
            
        prev_spectrum = spectrum.copy()
        
        # Assemble feature vector: [Energy, Subglottal_Energy, Tilt, Entropy, Freeze_Index, ZCR, Centroid, Flatness]
        features[i] = [
            energy,
            subglottal_energy,
            tilt,
            entropy,
            freeze_idx,
            zcr,
            centroid / (sample_rate / 2.0), # Normalize centroid
            flatness
        ]
        
    return features
