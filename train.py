import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import torchaudio.transforms as T
from pathlib import Path

# progressbar for cli
from tqdm import tqdm

# Project file
from model import AudioCNN
from dataset import ESC50Dataset
from utils import get_device, mix_up_data, mix_up_criterion


def training_loop():
    esc50_dir = Path("/opt/esc50-data")
    spectrogram_kwargs = {'sample_rate':22050, 'n_fft':1024, 'hop_length':512, 'n_mels':128, 'f_min':0, 'f_max':11025 }
    train_transform = nn.Sequential(
        T.MelSpectrogram(**spectrogram_kwargs),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)
    )
    test_transform = nn.Sequential(T.MelSpectrogram(**spectrogram_kwargs), T.AmplitudeToDB())
    train_dataset = ESC50Dataset(
        data_dir=esc50_dir, metadata_file= esc50_dir / "meta" / "esc50.csv", split="train", transform=train_transform
    )
    test_dataset = ESC50Dataset(
        data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="test", transform=test_transform
    )

    print(f"Train samples: {len(train_dataset)}") # 1600
    print(f"Test samples: {len(test_dataset)}") # 400

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # define the hardware
    device = get_device()
    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)

    # configure the training cycle
    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    sch_kwargs = {"max_lr":0.002, "epochs":num_epochs, "steps_per_epoch":len(train_dataloader), "pct_start":0.1}
    scheduler = OneCycleLR(optimizer, **sch_kwargs)
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            if np.random.random() > 0.7:
                data, target_a, target_b, lam = mix_up_data(data, target)
                output = model(data)
                loss = mix_up_criterion(criterion, output, target_a, target_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        model.eval()
        correct, total, test_loss = 0, 0, 0

        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(data)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_test_loss = test_loss/ len(train_dataloader)

        print(f"Epoch {epoch+1} Loss: {avg_epoch_loss:.4f} "+\
              f"Test Loss: {avg_test_loss:.4f} Accuracy: {accuracy:.2f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'classes': train_dataset.classes
            }, '/models/best_accuracy_model.pth')
            print(f"New best model saved: {accuracy:.2f}%")

    print(f"training completed.... best accuracy: {best_accuracy:.2f}%")
