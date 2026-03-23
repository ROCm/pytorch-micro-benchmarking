import torch
from audio.languagemodels import LanguageModel
import string
from audio.hubert_loss import hubert_loss
from audio.sdr import si_sdr_loss
from torch import nn
from audio.audio_model import *


def get_criterion(network_name):
    criterion = None
    if network_name in speech_representation_models:
        criterion = hubert_loss
    elif network_name in speech_recognition_models or network_name in acoustic_models:
        char_blank = "*"
        char_space = " "
        char_apostrophe = "'"
        labels = char_blank + char_space + char_apostrophe + string.ascii_lowercase
        language_model = LanguageModel(labels, char_blank, char_space)
        criterion = torch.nn.CTCLoss(blank=language_model.mapping[char_blank], zero_infinity=False)
    elif "wavernn" in network_name:
        criterion = nn.CrossEntropyLoss()
    elif "conv_tasnet" in network_name:
        criterion = si_sdr_loss
    elif "tacotron2" in network_name:
        criterion = nn.MSELoss()
    elif "hdemucs" in network_name:
        criterion = nn.L1Loss(reduction='none')
    elif "squim" in network_name:
        criterion = nn.L1Loss()
    else:
        print (f"Criterion for network name {network_name} not defined")
        sys.exit(1)
    return criterion

    
def calculate_loss(network_name, criterion, output, target, batch_size, input):
    loss = 0
    if network_name in speech_representation_models:
        logit_m, logit_u, feature_penalty = output
        loss = criterion(logit_m, logit_u, feature_penalty)
    elif network_name in speech_recognition_models or network_name in acoustic_models:
        output = output.transpose(-1, -2).transpose(0, 1)
        T, N, C = output.shape
        target, target_lengths = target
        tensors_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        loss = criterion(output, target, tensors_lengths, target_lengths)
    elif "wavernn" in network_name:
        output = output.squeeze(1)
        output = output.transpose(1, 2)
        loss = criterion(output, target)
    elif "conv_tasnet" in network_name:
        target, mask = target
        loss = criterion(output, target, mask)
    elif "tacotron2" in network_name or "subjective" in network_name:
        loss = criterion(output, target)
    elif "objective" in network_name:
        loss = 0
        weights = [1, 2, 0.5, 2]
        for index in range(len(output)):
            if index == 0:
                loss = criterion(output[index], target[index])
            else:
                loss += criterion(output[index], target[index])
        loss += criterion(input["x"], target[3])
    elif "hdemucs" in network_name:
        dims = tuple(range(2, target.dim()))
        loss = criterion(output, target)
        loss = loss.mean(dims).mean(0)
    else:
        print (f"Loss function for {network_name} not defined")
        sys.exit(1)
    return loss