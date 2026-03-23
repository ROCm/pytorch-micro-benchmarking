import torch
import torchaudio
import sys

ACOUSTIC_FEATURES_SIZE = 32
FRAME_COUNT = 1024
HOP_LENGTH = 36
N_FREQ = 128
CLASSES_COUNT = 29

#different audio tasks related models
acoustic_models = {
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


speech_representation_models = {
    "hubert_pretrain_base" : torchaudio.models.hubert_pretrain_base,
    "hubert_pretrain_large" : torchaudio.models.hubert_pretrain_large,
    "hubert_pretrain_xlarge" : torchaudio.models.hubert_pretrain_xlarge
}

def get_network_names():
    return sorted(list(acoustic_models.keys()) +
                  list(speech_recognition_models.keys()) +
                  list(source_separation_models.keys()) +
                  list(speech_quality_models.keys()) + 
                  list(speech_synthesis_models.keys()) +
                  list(speech_representation_models.keys()))


def get_network(network_name):
    if network_name in acoustic_models:
        return acoustic_models[network_name](aux_num_out=CLASSES_COUNT).to(device="cuda")
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
    elif network_name in speech_representation_models:
        return speech_representation_models[network_name]().to(device="cuda")                                       
    else:
        print ("ERROR: not a supported model '%s'" % network_name)
        sys.exit(1)