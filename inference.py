import matplotlib
matplotlib.use('agg')
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


def plot_data(data, prename="0", figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')
    if not os.path.exists("img"):
        os.mkdir("img")
    plt.savefig(os.path.join("img", str(prename) + "_model_test.jpg"))


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
    texts = [
            "PRIH1NTIH0NG , IH0N TH AO1NLIY0 SEH1NS WIH1TH HHWIH1CH W AA1R AE1T PRIY0ZEH1NT KAH0NSER1ND , DIH1FER0Z FRAH1M MOW2ST IH1F NAA1T FRAH1M AH0L TH AA1RTS AE1ND KRAE1FTS REH2PRIH0ZEH1NTIH0D IH0N TH EH2KSAH0BIH1SHAH0N",
            "AE1ND DIH0TEY1LIH0NG PAH0LIY1S IH0N SAH0VIH1LYAH0N KLOW1DHZ TOW0 B SKAE1TER0D THRUW0AW1T TH SAY1ZAH0BAH0L KRAW1D .",
            "AY1 LAH1V YUW1 VEH1RIY0 MAH1CH",
            "SAY1AH0NTIH0STS AE1T TH SER1N LAE1BRAH0TAO2RIY0 SEY1 DHEY1 HHAE1V DIH0SKAH1VER0D AH0 NUW1 PAA1RTAH0KAH0L .",
            "PREH1ZIH0DAH0NT TRAH1MP MEH1T WIH1TH AH1DHER0 LIY1DER0Z AE1T TH GRUW1P AH1V TWEH1NTIY0 KAA1NFER0AH0NS .",
            "LEH1TS GOW1 AW2T TOW0 TH EH1RPAO2RT . TH PLEY1N LAE1NDAH0D TEH1N MIH1NAH0TS AH0GOW2 .",
            "IH0N BIY1IH0NG KAH0MPEH1RAH0TIH0VLIY0 MAA1DER0N .",
            "VIH1PKIH0D",
            "VIH1P KIH0D"
            ]
    
    if not os.path.exists("results"):
        os.mkdir("results")

    for text in texts:
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()


        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        plot_data((mel_outputs.float().data.cpu().numpy()[0],
               mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T), text[:10])

    #print("mel_out:", mel_outputs)
    #print("mel_out_postnet:", mel_outputs_postnet)
    #print("alignments:", alignments)

        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
            audio = audio * hparams.max_wav_value;
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        write("results/{}_synthesis.wav".format(text), hparams.sampling_rate, audio)
        print("complete:",text)


    # audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    # ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate)

if __name__ == "__main__":
    main();
