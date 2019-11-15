import matplotlib
import matplotlib.pylab as plt
import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from scipy.io.wavfile import write
from train import load_model
from text import text_to_sequence
import os
# from denoiser import Denoiser


def plot_data(data, figsize=(16, 4)):
    print("disable plot_data!")
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')
    if not os.path.exists("img"):
        os.mkdir("img")
    plt.savefig(os.path.join("img", "model_test.jpg"))


def get_WaveGlow():
    waveglow_path = 'checkout'
    print("load waveglow model !!")
    waveglow_path = os.path.join(waveglow_path, "waveglow_256channels.pt")
    wave_glow = torch.load(waveglow_path)['model']
    wave_glow = wave_glow.remove_weightnorm(wave_glow)
    wave_glow.cuda().eval()
    for m in wave_glow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')

    return wave_glow


def get_Tacotron2(hparams):

    checkpoint_path = "checkout"
    checkpoint_path = os.path.join(checkpoint_path, "tacotron2_statedict.pt")
    print("load tacotron2 model !!")
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval()

    return model


def main():
    hparams = create_hparams()
    hparams.sampling_rate = 22050

    model = get_Tacotron2(hparams);
    waveglow = get_WaveGlow();

    # text = "Waveglow is really awesome!"
    text = "L_B EH1_I T_I S_E G_B OW1_E AW2_B T_E T_B OW0_E TH_S EH1_B R_I P_I AO2_I R_I T_E SIL TH_S P_B L_I EY1_I N_E L_B AE1_I N_I D_I IH0_I D_E T_B EH1_I N_E M_B IH1_I N_I AH0_I T_I S_E AH0_B G_I OW2_E SIL"

    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()


    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
               mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T))

    print("mel_out:", mel_outputs)
    print("mel_out_postnet:", mel_outputs_postnet)
    print("alignments:", alignments)

    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        audio = audio * hparams.max_wav_value;
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    if not os.path.exists("results"):
        os.mkdir("results")
    write("results/{}_synthesis.wav".format(text), hparams.sampling_rate, audio)


    # audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    # ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate)

if __name__ == "__main__":
    main();
