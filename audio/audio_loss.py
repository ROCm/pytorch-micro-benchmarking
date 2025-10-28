import torch
from audio.languagemodels import LanguageModel
import string
from audio.hubert_loss import hubert_loss
from audio.sdr import si_sdr_loss
from torch import nn


def get_criterion(network_name):
    criterion = None
    if network_name in ["wav2letter", "conformer", "deepspeech"] or "wav2vec2" in network_name:
        char_blank = "*"
        char_space = " "
        char_apostrophe = "'"
        labels = char_blank + char_space + char_apostrophe + string.ascii_lowercase
        language_model = LanguageModel(labels, char_blank, char_space)
        criterion = torch.nn.CTCLoss(blank=language_model.mapping[char_blank], zero_infinity=False)
    elif "hubert_pretrain" in network_name:
        criterion = hubert_loss
    elif "wavernn" in network_name:
        criterion = nn.CrossEntropyLoss()
    elif "conv_tasnet" in network_name:
        criterion = si_sdr_loss
    elif "tacotron2" in network_name:
        criterion = nn.MSELoss()
    return criterion

    
def calculate_loss(network_name, criterion, output):
    if criterion is None:
        target = torch.randn_like(output)
        return torch.nn.functional.mse_loss(output, target)
    if network_name in ["wav2letter", "conformer", "deepspeech"] or "wav2vec2" in network_name:
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
    elif "hubert_pretrain" in network_name:
        print ("hubert", len(output), output[0].shape, output[1])
        logit_m, logit_u, feature_penalty = output
        loss = criterion(logit_m, logit_u, feature_penalty)
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
    return loss