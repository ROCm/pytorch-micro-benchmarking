from audio.audio_model import *


def get_output_selection(network_name):
    if network_name in acoustic_models:
        return 0
    elif "conformer" in network_name or "emformer" in network_name:
        return 0
    elif "tacotron2" in network_name:
        return 1
    return None