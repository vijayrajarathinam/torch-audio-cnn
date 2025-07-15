import torch
import torchaudio
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

# Infra As A Service [free GPU runtime]
import modal

# Project file
from train import training_loop


app = modal.App("audio-cnn")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")  # python packages source file
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])  # linux services to process audio
         .run_commands([
                "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -0 esc50.zip"
                "cd /tmp && unzip esc50.zip",
                "mkdir -p /opt/esc50-data",
                "cp -r /tmp/ESC-50-master/* /opt/esc50-data",
                "rm -rf /tmp/esc50.zip /tmpESC-50-master"
         ])  # startup script
         .add_local_python_source("model")  # model.py
         )
data_volume = modal.Volume.from_name("esc50-data", create_if_missing=True)  # for data storage
modal_volume = modal.Volume.from_name("esc-modal", create_if_missing=True)  # for training modal
volumes = {'/data': data_volume, '/models': modal_volume}


@app.function(image=image, gpu="A10G", volumes=volumes, timeout=60 * 60 * 3)
def train(): training_loop()

@app.local_entrypoint()
def main(): train.remote()