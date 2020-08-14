# Jake's code to create an actor critic and export it to ONNX format
from bot_env import RobotEnvironment
from core import MLPActorCritic
import numpy as np
import torch

actor = MLPActorCritic(RobotEnvironment.observation_space, RobotEnvironment.action_space, )

sample_input = RobotEnvironment.observation_space.sample()
sample_input = np.expand_dims(sample_input, 0)
sample_input = torch.from_numpy(sample_input)

print("TORCH Version: ", torch.__version__)

#torch.onnx.export(mlp([observation_space.shape[0], 512, 512], nn.ReLU), sample_input, "mlp.onnx", verbose=True, opset_version=12)
torch.onnx.export(actor.pi, (sample_input,), "mlp.onnx", verbose=True, opset_version=12,
                  input_names=["yolo_intermediate"], output_names=["actions"])