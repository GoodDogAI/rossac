import unittest
import os
import numpy as np
import nemo
import soundfile
import torch
from nemo.collections.asr.models import EncDecClassificationModel
from torchaudio import transforms


class FullyExportableEncDecClassificationModel(EncDecClassificationModel):
    def _get_input_example(self):
        return tuple([torch.rand(1, 48000 * 4).to(next(self.parameters()).device),
                      torch.tensor([48000 * 4])])

    def forward_for_export(self, input, length=None):
        processed_signal, processed_signal_len = self.preprocessor(
            input_signal=input, length=length,
        )
        # Crop or pad is always applied
        if self.crop_or_pad is not None:
            processed_signal, processed_signal_len = self.crop_or_pad(
                input_signal=processed_signal, length=processed_signal_len
            )
        encoder_output = self.input_module(processed_signal, processed_signal_len)
        if isinstance(encoder_output, tuple):
            return self.output_module(encoder_output[0])
        else:
            return self.output_module(encoder_output)


class TestNemoAudioModel(unittest.TestCase):
    sample_rate = 16_000

    def test_nemo_export(self):
        asr_model = FullyExportableEncDecClassificationModel.from_pretrained("commandrecognition_en_matchboxnet3x2x64_v2")

        # Export the model
        asr_model.export("asr_command_recognition.onnx",
                         use_dynamic_axes=False)

    def test_nemo_sample_data(self):
        wav, sr = soundfile.read(os.path.join(os.path.dirname(__file__), "test_data", "victor_forward_48k.wav"))
        asr_model = EncDecClassificationModel.from_pretrained("commandrecognition_en_matchboxnet3x2x64_v2")

        wav = torch.from_numpy(wav).float()
        len = torch.from_numpy(np.array(wav.shape))

        resample_transform = transforms.Resample(sr, self.sample_rate)
        wav = resample_transform(wav)

        asr_model = asr_model.eval().cpu()
        result = asr_model.forward(input_signal=torch.unsqueeze(wav, 0), input_signal_length=len)

        result = torch.softmax(result, dim=-1)

        print(result)

