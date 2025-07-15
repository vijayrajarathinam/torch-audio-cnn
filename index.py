import requests
import torch
import torchaudio.transforms as T
from pydantic import BaseModel
import soundfile as sf
import numpy as np
import librosa

import base64
import io

# Infra As A Service [free GPU runtime]
import modal
from model import AudioCNN


app = modal.App("audio-cnn-inference")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt") # packages source
         .apt_install(["libsndfile1"])  # linux services to process audio
         .add_local_python_source("model"))

modal_volume = modal.Volume.from_name("esc-modal")  # for training modal
volumes = { '/models': modal_volume }
spectrogram_kwargs = {
    'sample_rate': 22050, 'n_fft': 1024,
    'hop_length': 512, 'n_mels': 128,
    'f_min': 0, 'f_max': 11025
}


class InferenceRequest(BaseModel):
    audio_data: str


class AudioProcessor:
    def __init__(self):
        self.transform = torch.nn.Sequential(
            T.MelSpectrogram(**spectrogram_kwargs),
            T.AmplitudeToDB(),
        )

    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()
        waveform = waveform.unsqueeze(0)
        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)


@app.cls(image=image, gpu="A10G", volumes=volumes, scaledown_window=15)
class AudioClassifier:
    @modal.enter()
    def load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load('/models/best_accuracy_model.pth', map_location=self.device)
        self.classes = checkpoint['classes']
        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device) # move the model to GPU
        self.model.eval()
        self.audio_processor = AudioProcessor()

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        audio_bytes = base64.b64decode(request.audio_data)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        if audio_data.ndim > 1: audio_data = np.mean(audio_data, axis=1)

        if sample_rate != 22050:
            audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=22050)

        spectrogram = self.audio_processor.process_audio_chunk(audio_data)
        spectrogram = spectrogram.to(self.device) # move to GPU

        with torch.no_grad():
            output = self.model(spectrogram)
            output = torch.nan_to_num(output) # convert Nan to Number(0)
            probabilities = torch.softmax(output, dim=1)
            top3_probs, top3_indices = torch.topk(probabilities[0], 3)
            predictions = [{
                "class": self.classes[idx.item()],
                "confidence": prob.item()
            } for prob, idx in zip(top3_probs, top3_indices)]

        return { "prediction": predictions }

@app.local_entrypoint()
def main():
    audio_data = sf.read("1.wav")
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, 22050, format="wav")
    audio_b64 = base64.b64decode(buffer.getvalue()).decode("utf-8")
    payload = {"audio_data": audio_b64}

    # create a [server] with endpoint to process the audio with pre-trained modal
    server = AudioClassifier()
    url = server.inference.get_web_url()
    response = requests.post(url,json=payload)
    return response.json()
