from audio.audio_model import *


def get_output_selection(network_name):
    if network_name in acoustic_models:
        return 0
    elif "conformer" in network_name or "emformer" in network_name:
        return 0
    elif "tacotron2" in network_name:
        return 1
    return None

def create_target(network_name, network, input, batch_size):
    
    #get output
    output = network(**input)
    output_index = get_output_selection(network_name) 
    print("output", output.shape)
    if output_index is not None:
        output = output[output_index]

    target = None
    if network_name in speech_recognition_models or network_name in acoustic_models:
        output = output.transpose(-1, -2).transpose(0, 1)
        T, N, C = output.shape
        target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
        target = torch.randint(
            low=1,
            high=C,
            size=(sum(target_lengths),),
            dtype=torch.long,
        )
        target = [target, target_lengths]
    elif "wavernn" in network_name:
        target = torch.randn_like(output)
        target = target.squeeze(1)
        target = target.transpose(1, 2)
    elif "conv_tasnet" in network_name:
        batch, _, time = output.shape
        mask = torch.randint(low=0, high=1, size=(batch,1,time), dtype=torch.long).cuda()
        target = torch.randn_like(output)
        target = [target, mask]
    elif "tacotron2" in network_name:
        target = torch.randn_like(output)
    elif "hdemucs" in network_name or "subjective" in network_name:
        target = torch.randn_like(output)
    elif "objective" in network_name:
        target = []
        for index in range(len(output)):
            target.append(torch.randn_like(output[index]))
        target.append(torch.randn_like(input["x"]))
    return target