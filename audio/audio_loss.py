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
    elif "hdemucs" in network_name or "squim" in network_name:
        criterion = nn.L1Loss()
    return criterion

    
def calculate_loss(network_name, criterion, output):
    if criterion is None:
        target = torch.randn_like(output)
        return torch.nn.functional.mse_loss(output, target)
    if network_name in speech_representation_models:
        logit_m, logit_u, feature_penalty = output
        loss = criterion(logit_m, logit_u, feature_penalty)
    elif network_name in speech_recognition_models or network_name in acoustic_models:
        output = output.transpose(-1, -2).transpose(0, 1)
        T, N, C = output.shape
        target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
        target = torch.randint(
            low=1,
            high=C,
            size=(sum(target_lengths),),
            dtype=torch.long,
        )
        tensors_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        loss = criterion(output, target, tensors_lengths, target_lengths)
    elif "wavernn" in network_name:
        target = torch.randn_like(output)
        output, target = output.squeeze(1), target.squeeze(1)
        output = output.transpose(1, 2)
        target = target.transpose(1, 2)
        loss = criterion(output, target)
    elif "conv_tasnet" in network_name:
        batch, _, time = output.shape
        mask = torch.randint(low=0, high=1, size=(batch,1,time), dtype=torch.long).cuda()
        target = torch.randn_like(output)
        loss = criterion(output, target, mask)
    elif "tacotron2" in network_name:
        target = torch.randn_like(output)
        loss = criterion(output, target)
    elif "hdemucs" in network_name or "subjective" in network_name:
        target = torch.randn_like(output)
        loss = criterion(output, target)
    elif "objective" in network_name:
        for index in range(len(output)):
            target = torch.randn_like(output[index])
            if index == 0:
                loss = criterion(output[index], target)
            else:
                loss += criterion(output[index], target)
    return loss