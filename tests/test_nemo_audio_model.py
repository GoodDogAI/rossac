import unittest
import os
import nemo
import soundfile
from nemo.collections.asr.models import EncDecClassificationModel


class TestNemoAudioModel(unittest.TestCase):
    sample_rate = 16_000

    def test_nemo_export(self):
        asr_model = EncDecClassificationModel.from_pretrained("commandrecognition_en_matchboxnet3x2x64_v2")

        # Export the model
        asr_model.export("asr_command_recognition.onnx", onnx_opset_version=12)

    def test_nemo_sample_data(self):
        wav, sr = soundfile.read(os.path.join(os.path.dirname(__file__), "test_data", "victor_forward_48k.wav"))

