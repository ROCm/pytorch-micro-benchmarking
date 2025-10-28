import torch
import torchaudio
import sys

ACOUSTIC_FEATURES_SIZE = 32
FRAME_COUNT = 1024
HOP_LENGTH = 36
N_FREQ = 128
CLASSES_COUNT = 29

#different audio tasks related models
wav2vec_models = {
    "wav2vec2_base" : torchaudio.models.wav2vec2_base,
    "wav2vec2_large" : torchaudio.models.wav2vec2_large,
    "wav2vec2_large_lv60k" : torchaudio.models.wav2vec2_large_lv60k,
    "wav2vec2_xlsr_300m" : torchaudio.models.wav2vec2_xlsr_300m,
    "wav2vec2_xlsr_1b" : torchaudio.models.wav2vec2_xlsr_1b,
    "wav2vec2_xlsr_2b" : torchaudio.models.wav2vec2_xlsr_2b,
    "hubert_base" :  torchaudio.models.hubert_base,
    "hubert_large" : torchaudio.models.hubert_large,
    "hubert_xlarge" : torchaudio.models.hubert_xlarge,
    "wavlm_base" : torchaudio.models.wavlm_base,
    "wavlm_large" : torchaudio.models.wavlm_large,
}

speech_recognition_models = {
    "conformer" : torchaudio.models.Conformer,
    "deepspeech" : torchaudio.models.DeepSpeech,
    "emformer" : torchaudio.models.Emformer,
    "wav2letter" : torchaudio.models.Wav2Letter
}

source_separation_models = {
    "conv_tasnet_base" : torchaudio.models.conv_tasnet_base,
    "hdemucs_low" : torchaudio.models.hdemucs_low,
    "hdemucs_medium" : torchaudio.models.hdemucs_medium,
    "hdemucs_high" : torchaudio.models.hdemucs_high,
}

speech_quality_models = {
    "squim_objective_base" : torchaudio.models.squim_objective_base,
    "squim_subjective_base" : torchaudio.models.squim_subjective_base
}

speech_synthesis_models = {
    "tacotron2" : torchaudio.models.Tacotron2,
    "wavernn" : torchaudio.models.WaveRNN
}


hubert_pretrain_models = {
    "hubert_pretrain_base" : torchaudio.models.hubert_pretrain_base,
}

def get_network_names():
    return sorted(list(wav2vec_models.keys()) +
                  list(speech_recognition_models.keys()) +
                  list(source_separation_models.keys()) +
                  list(speech_quality_models.keys()) + 
                  list(speech_synthesis_models.keys()) +
                  list(hubert_pretrain_models.keys()))


def get_network(network_name):
    if network_name in wav2vec_models:
        return wav2vec_models[network_name](aux_num_out=CLASSES_COUNT).to(device="cuda")
    elif network_name in source_separation_models:
        if "hdemucs" in network_name:
            return source_separation_models[network_name](sources = ["vocals"]).to(device="cuda")
        else:
            return source_separation_models[network_name]().to(device="cuda")
    elif network_name in speech_recognition_models:
        if "deepspeech" in network_name:
            return speech_recognition_models[network_name](n_feature = ACOUSTIC_FEATURES_SIZE).to(device="cuda")
        elif "wav2letter" in network_name:
            return speech_recognition_models[network_name](num_features = ACOUSTIC_FEATURES_SIZE).to(device="cuda")
        elif "emformer" in network_name:
            return speech_recognition_models[network_name](input_dim = ACOUSTIC_FEATURES_SIZE,
                                                           num_heads=8, 
                                                           ffn_dim=1024, 
                                                           num_layers=20,
                                                           segment_length=4).to(device="cuda")
        elif "conformer" in network_name:
            return speech_recognition_models[network_name](input_dim = 80,
                                                           num_heads=4, 
                                                           ffn_dim=128, 
                                                           num_layers=4,
                                                           depthwise_conv_kernel_size=31).to(device="cuda")
    elif network_name in speech_quality_models:
        return speech_quality_models[network_name]().to(device="cuda")
    elif network_name in speech_synthesis_models:
        if "wavernn" in network_name:
            return speech_synthesis_models[network_name](upsample_scales = [3, 3, 4], n_classes = 10, 
                                                         hop_length = HOP_LENGTH, n_freq = 128).to(device="cuda")
        else:
            return speech_synthesis_models[network_name]().to(device="cuda")    
    elif network_name in hubert_pretrain_models:
        return hubert_pretrain_models[network_name]().to(device="cuda")                                       
    else:
        print ("ERROR: not a supported model '%s'" % network_name)
        sys.exit(1)


def get_input_type(network_name):
    if network_name in wav2vec_models or network_name in source_separation_models or network_name in speech_quality_models:
        return "waveform"
    elif network_name in speech_recognition_models:
        return "acoustic features"
    elif network_name in speech_synthesis_models:
        if "wavernn" in network_name:
            return "waveform"
        else:
            return "tokens"


def get_input(network_name, network, batch_size):
    if network_name in wav2vec_models:
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
    elif network_name in hubert_pretrain_models:
        
        inp = {"waveforms" : torch.rand(batch_size, FRAME_COUNT, device="cuda"),
               "labels" : torch.randint(0, 100, (batch_size, FRAME_COUNT), dtype=torch.int32, device="cuda"),
               "audio_lengths" : torch.randint(1, FRAME_COUNT, (batch_size,), device="cuda")}
    return inp


def get_output_selection(network_name):
    if "wav2vec2" in network_name:
        return 0
    elif "conformer" in network_name or "emformer" in network_name:
        return 0
    elif "objective" in network_name:
        return 0
    elif "tacotron2" in network_name:
        return 1
    return None