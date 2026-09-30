import numpy as np
import sounddevice as sd
import queue

class AudioStreamManager:
    """Captures rolling audio buffers at 16kHz for continuous NPU inference."""
    def __init__(self, sample_rate=16000, buffer_duration=1.0):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_duration)
        self.audio_queue = queue.Queue(maxsize=10)
        self.stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            pass
        # Flatten to mono channel and normalize
        mono_audio = np.squeeze(indata[:, 0]) if indata.ndim > 1 else indata
        if not self.audio_queue.full():
            self.audio_queue.put(mono_audio.copy())

    def start(self):
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.buffer_size,
            callback=self._audio_callback
        )
        self.stream.start()

    def get_latest_frame(self):
        if not self.audio_queue.empty():
            return self.audio_queue.get()
        return None

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
