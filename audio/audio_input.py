import torch
from audio.audio_model import *


def get_input_type(network_name):
    if network_name in acoustic_models or network_name in source_separation_models or network_name in speech_quality_models:
        return "waveform"
    elif network_name in speech_recognition_models:
        return "acoustic features"
    elif network_name in speech_synthesis_models:
        if "wavernn" in network_name:
            return "waveform"
        else:
            return "tokens"


def get_input(network_name, network, batch_size):
    if network_name in acoustic_models:
        inp = {"waveforms": torch.randn(batch_size, FRAME_COUNT, device="cuda")}
    elif network_name in source_separation_models:
        if "hdemucs" in network_name:
            inp = {"input" : torch.randn(batch_size, 2, FRAME_COUNT, device="cuda")}
        else:
            inp = {"input" : torch.randn(batch_size, 1, FRAME_COUNT, device="cuda")}
    elif network_name in speech_recognition_models:
        if "deepspeech" in network_name:
            #number of channels must be specified for deepspeech
            inp = {"x" : torch.randn(batch_size, 1, FRAME_COUNT, ACOUSTIC_FEATURES_SIZE, device="cuda")}
        elif "wav2letter" in network_name:
            inp = {"x" : torch.randn(batch_size, ACOUSTIC_FEATURES_SIZE, FRAME_COUNT, device="cuda")}
        elif "emformer" in network_name:
            inp = {"input" : torch.randn(batch_size, FRAME_COUNT, ACOUSTIC_FEATURES_SIZE, device="cuda"),
                   "lengths" : torch.randint(1, FRAME_COUNT, (batch_size,)).to(device="cuda")}
        elif "conformer" in network_name:
            lengths = torch.randint(1, FRAME_COUNT, (batch_size,), device="cuda")
            inp = {"input" : torch.rand(batch_size, int(lengths.max()), 80, device="cuda"),
                   "lengths" : lengths}
    elif network_name in speech_quality_models:
        if "subjective" in network_name:
            inp = {"waveform" : torch.randn(batch_size, FRAME_COUNT, device="cuda"),
                   "reference" : torch.randn(batch_size, FRAME_COUNT, device="cuda")}
        else:
            inp = {"x" : torch.randn(batch_size, FRAME_COUNT, device="cuda")}
    elif network_name in speech_synthesis_models:
        if "wavernn" in network_name:
            spec_frames = 64
            waveform_length = HOP_LENGTH * (spec_frames - 4)
            
            inp = {"waveform" : torch.rand(batch_size, 1, waveform_length, device="cuda"),
                   "specgram": torch.rand(batch_size, 1, N_FREQ, spec_frames, device="cuda")}
        elif "tacotron2" in network_name:
            n_mels = 80
            max_mel_specgram_length = 300
            max_text_length = 100
            inp = {"tokens" : torch.randint(0, 148, (batch_size, max_text_length), dtype=torch.int32, device="cuda"),
                   "token_lengths" : max_text_length * torch.ones((batch_size,), device="cuda"),
                   "mel_specgram": torch.rand(batch_size, n_mels, max_mel_specgram_length, device="cuda"),
                   "mel_specgram_lengths" : max_mel_specgram_length * torch.ones((batch_size,), dtype=torch.int32, device="cuda")}
    elif network_name in speech_representation_models:
        inp = {"waveforms" : torch.rand(batch_size, FRAME_COUNT, device="cuda"),
               "labels" : torch.randint(0, 100, (batch_size, 2), dtype=torch.int32, device="cuda")}
    return inp