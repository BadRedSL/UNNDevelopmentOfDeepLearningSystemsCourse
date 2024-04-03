import numpy as np
import torch
from vg_beat_detectors import FastWHVG, FastNVG
import matplotlib.pyplot as plt
from scipy.datasets import electrocardiogram
from ecg_generation_models.gan_model import DenoisedGAN
import datetime

if __name__ == "__main__":
    start = datetime.datetime.now()
    seed = 7
    torch.manual_seed(seed)

    ecg = electrocardiogram()
    fs = 360
    time = np.arange(ecg.size) / fs

    detector_nvg = FastNVG(sampling_frequency=fs)
    rpeaks_nvg = detector_nvg.find_peaks(ecg)

    detector_whvg = FastWHVG(sampling_frequency=fs)
    rpeaks_whvg = detector_whvg.find_peaks(ecg)

    denoised_generator = DenoisedGAN.load_from_checkpoint("../ecg_generation_models/models_weights/"
                                                          "epoch=1999_g_loss=6.17_d_loss=0.05.ckpt").cpu()
    denoised_generator.eval()
    num_samples = 10
    noise = torch.normal(0, 1, size=(num_samples, 100))
    denoised_generated_ecg = denoised_generator(noise)[0].detach().cpu().numpy()
    generated_fs = 480
    generated_time = np.arange(denoised_generated_ecg.size) / generated_fs

    detector_nvg = FastNVG(sampling_frequency=generated_fs)
    detector_whvg = FastWHVG(sampling_frequency=generated_fs)

    generated_rpeaks_nvg = detector_nvg.find_peaks(denoised_generated_ecg)
    generated_rpeaks_whvg = detector_whvg.find_peaks(denoised_generated_ecg)

    finish = datetime.datetime.now()
    print(f"Elapsed time: {finish-start}")

    plt.rcParams["figure.figsize"] = (20, 10)
    fig, axes = plt.subplots(2, 2)

    axes[0, 0].plot(time, ecg)
    axes[0, 0].plot(time[rpeaks_nvg], ecg[rpeaks_nvg], 'X')
    axes[0, 0].set_xlabel("time in s")
    axes[0, 0].set_xlim(0, 1.6)
    axes[0, 0].set_ylim(-1.2, 2.0)
    axes[0, 0].set_title("ScipyECG-NVGDetector")

    axes[0, 1].plot(time, ecg)
    axes[0, 1].plot(time[rpeaks_whvg], ecg[rpeaks_whvg], 'X')
    axes[0, 1].set_xlabel("time in s")
    axes[0, 1].set_xlim(0, 1.6)
    axes[0, 1].set_ylim(-1.2, 2.0)
    axes[0, 1].set_title("ScipyECG-WHVGDetector")

    axes[1, 0].plot(generated_time, denoised_generated_ecg)
    axes[1, 0].plot(generated_time[generated_rpeaks_nvg], denoised_generated_ecg[generated_rpeaks_nvg], 'X')
    axes[1, 0].set_xlabel("time in s")
    axes[1, 0].set_xlim(0, 1.6)
    axes[1, 0].set_ylim(-0.5, 0.9)
    axes[1, 0].set_title("Generated_ECG-NVGDetector")

    axes[1, 1].plot(generated_time, denoised_generated_ecg)
    axes[1, 1].plot(generated_time[generated_rpeaks_whvg], denoised_generated_ecg[generated_rpeaks_whvg], 'X')
    axes[1, 1].set_xlabel("time in s")
    axes[1, 1].set_xlim(0, 1.6)
    axes[1, 1].set_ylim(-0.5, 0.9)
    axes[1, 1].set_title("Generated_ECG-WHVGDetector")
    plt.show()
