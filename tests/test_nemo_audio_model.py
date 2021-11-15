import unittest
import os
from typing import Optional, Callable

import numpy as np
import onnxruntime as rt
import math
import librosa
import soundfile
import torch
from torch import Tensor
from nemo.collections.asr.models import EncDecClassificationModel
from torchaudio import transforms


class FullyExportableSTFT(torch.nn.Module):
    def __init__(self, win_length, hop_length, window_fn, n_fft, window_periodic = True, stft_mode = None):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.freq_cutoff = self.n_fft // 2 + 1
        self.register_buffer('window', window_fn(self.win_length, periodic=window_periodic).float())

        if stft_mode == 'conv':
            fourier_basis = torch.view_as_real(torch.fft.fft(torch.eye(self.n_fft), dim=1))
            forward_basis = fourier_basis[:self.freq_cutoff].permute(2, 0, 1).reshape(-1, 1, fourier_basis.shape[1])
            forward_basis = forward_basis * torch.as_tensor(
                librosa.util.pad_center(self.window, self.n_fft), dtype=forward_basis.dtype
            )
            self.stft = torch.nn.Conv1d(
                forward_basis.shape[1],
                forward_basis.shape[0],
                forward_basis.shape[2],
                bias=False,
                stride=self.hop_length
            ).requires_grad_(False)
            self.stft.weight.copy_(forward_basis)
        else:
            raise NotImplementedError()

    def forward(self, signal):
        # These need to be commented out to work on Jetson TensorRT
        pad = self.freq_cutoff - 1
        padded_signal = torch.nn.functional.pad(signal.unsqueeze(1), (pad, pad), mode = 'reflect').squeeze(1)

        #padded_signal = signal

        real, imag = self.stft(padded_signal.unsqueeze(dim = 1)).split(self.freq_cutoff, dim = 1)

        spec = real.pow(2) + imag.pow(2)
        return torch.sqrt(spec)


class FullyExportableSpectrogram(torch.nn.Module):
    __constants__ = ['n_fft', 'win_length', 'hop_length', 'pad', 'power', 'normalized']

    def __init__(self,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 pad: int = 0,
                 window_fn: Callable[..., Tensor] = torch.hann_window,
                 power: Optional[float] = 2.,
                 normalized: bool = False,
                 wkwargs: Optional[dict] = None,
                 center: bool = True,
                 pad_mode: str = "reflect",
                 onesided: bool = True,
                 return_complex: bool = True) -> None:
        super(FullyExportableSpectrogram, self).__init__()
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequencies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.return_complex = return_complex

        self.stft_module = FullyExportableSTFT(win_length, hop_length,
                                               window_fn=window_fn,
                                               n_fft=n_fft,
                                               stft_mode="conv")

    def forward(self, waveform: Tensor) -> Tensor:
        if self.pad > 0:
            # TODO add "with torch.no_grad():" back when JIT supports it
            waveform = torch.nn.functional.pad(waveform, (self.pad, self.pad), "constant")

        # pack batch
        shape = waveform.size()
        waveform = waveform.reshape(-1, shape[-1])

        # default values are consistent with librosa.core.spectrum._spectrogram
        # spec_f = torch.stft(
        #     input=waveform,
        #     n_fft=n_fft,
        #     hop_length=hop_length,
        #     win_length=win_length,
        #     window=window,
        #     center=center,
        #     pad_mode=pad_mode,
        #     normalized=False,
        #     onesided=onesided,
        #     return_complex=True,
        # )
        spec_f = self.stft_module(waveform)

        # TODO: Verify Jake's edit Take only the real part
        #spec_f, imag_f = spec_f

        # unpack batch
        spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-2:])

        if self.normalized:
            spec_f /= window.pow(2.).sum().sqrt()
        if self.power is not None:
            if self.power == 1.0:
                return spec_f.abs()
            return spec_f.abs().pow(self.power)
        if not self.return_complex:
            return torch.view_as_real(spec_f)
        return spec_f


class TestNemoAudioModel(unittest.TestCase):
    sample_rate = 16_000

    def test_model_export(self):
        asr_model = EncDecClassificationModel.from_pretrained("commandrecognition_en_matchboxnet3x2x64_v2")

        # Export the model
        asr_model.export("asr_command_recognition.onnx",
                         use_dynamic_axes=False)

    def test_featurizer_export(self):
        asr_model = EncDecClassificationModel.from_pretrained("commandrecognition_en_matchboxnet3x2x64_v2")
        print(asr_model)

        # Export the featurizer
        orig_spectrogram = asr_model.preprocessor.featurizer.MelSpectrogram.spectrogram
        new_spectrogram = FullyExportableSpectrogram(n_fft=orig_spectrogram.n_fft,
                                                     win_length=orig_spectrogram.win_length,
                                                     hop_length=orig_spectrogram.hop_length)

        new_spectrogram = new_spectrogram.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        asr_model.preprocessor.featurizer.MelSpectrogram.spectrogram = new_spectrogram

        torch.onnx.export(asr_model.preprocessor.featurizer,
                          torch.rand(1, 48000 * 4).cuda(),
                          "asr_featurizer.onnx",
                          input_names=["audio_signal"],
                          output_names=["audio_features"],
                          opset_version=12)

        # Test that the result matches between onnx and a native run
        random_input = torch.rand(1, 48000 * 4).cuda()
        orig_result = asr_model.preprocessor.featurizer(random_input)

        onnx_sess = rt.InferenceSession("asr_featurizer.onnx")
        onnx_result = onnx_sess.run(["audio_features"], {
            "audio_signal": random_input.cpu().detach().numpy()
        })

        np.testing.assert_almost_equal(orig_result[0].cpu().numpy(), onnx_result[0][0], decimal=3)

    def test_export_sample_data(self):
        wav, sr = soundfile.read(os.path.join(os.path.dirname(__file__), "test_data", "jake_clean_forward.wav"))
        asr_model = EncDecClassificationModel.from_pretrained("commandrecognition_en_matchboxnet3x2x64_v2")

        wav = torch.from_numpy(wav).float()
        len = torch.from_numpy(np.array(wav.shape))

        resample_transform = transforms.Resample(sr, self.sample_rate)
        wav = resample_transform(wav)

        asr_model = asr_model.eval().cpu()

        features, features_len = asr_model.preprocessor(input_signal=torch.unsqueeze(wav, 0), length=len)

        np.save("clean_forward_input.npy", wav)
        np.save("clean_forward_features.npy", features[0])
        print(features)

    def test_nemo_sample_data(self):
        wav, sr = soundfile.read(os.path.join(os.path.dirname(__file__), "test_data", "jake_clean_forward.wav"))
        asr_model = EncDecClassificationModel.from_pretrained("commandrecognition_en_matchboxnet3x2x64_v2")

        wav = torch.from_numpy(wav).float()
        len = torch.from_numpy(np.array(wav.shape))

        resample_transform = transforms.Resample(sr, self.sample_rate)
        wav = resample_transform(wav)

        asr_model = asr_model.eval().cpu()
        result = asr_model.forward(input_signal=torch.unsqueeze(wav, 0), input_signal_length=len)

        result = torch.softmax(result, dim=-1)

        self.assertEqual(asr_model.cfg.labels[24], "forward")
        self.assertEqual(np.argmax(result.detach().numpy()), 24)

