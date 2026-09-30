import numpy as np
from scipy import signal

class AcousticThreatDetector:
    def __init__(self, npu_engine):
        self.engine = npu_engine

    def compute_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        # Generate 64-bin Log Mel-scaled spectrogram
        _, _, sxx = signal.spectrogram(audio_data, fs=16000, nperseg=512, noverlap=256)
        log_spec = np.log(sxx + 1e-6)
        # Pad/crop to fixed NPU tensor dimension (1, 64, 64)
        resized = np.resize(log_spec, (1, 1, 64, 64)).astype(np.float32)
        return resized

    def evaluate_threat(self, audio_frame: np.ndarray):
        features = self.compute_spectrogram(audio_frame)
        predictions = self.engine.infer(features)
        
        keystroke_risk = float(predictions[0][1])
        status = "NORMAL"
        if keystroke_risk > 0.75:
            status = "CRITICAL_SNIFFING_DETECTED"
        elif keystroke_risk > 0.45:
            status = "ELEVATED_ACOUSTIC_ACTIVITY"

        return {
            "status": status,
            "risk_score": round(keystroke_risk * 100, 2),
            "spectrogram_slice": features[0, 0, :16, :16]
        }
