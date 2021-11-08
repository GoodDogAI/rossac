import unittest
import os
import numpy as np
import nemo
import soundfile
import torch
from nemo.collections.asr.models import EncDecClassificationModel
from torchaudio import transforms


class TestNemoAudioModel(unittest.TestCase):
    sample_rate = 16_000

    def test_nemo_export(self):
        asr_model = EncDecClassificationModel.from_pretrained("commandrecognition_en_matchboxnet3x2x64_v2")

        # Export the model
        asr_model.export("asr_command_recognition.onnx", onnx_opset_version=12)

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

        print(result)

