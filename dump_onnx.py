# Jake's code to create an actor critic and export it to ONNX format
from bot_env import RobotEnvironment
import numpy as np
import torch

def export(sac, file_name):
    sample_input = RobotEnvironment.observation_space.sample()
    sample_input = np.expand_dims(sample_input, 0)
    sample_input = torch.from_numpy(sample_input)

    torch.onnx.export(sac.pi, (sample_input,), file_name, verbose=True, opset_version=12,
                      input_names=["yolo_intermediate"], output_names=["actions"])

if __name__ == '__main__':
    from actor_critic.core import MLPActorCritic
    
    actor = MLPActorCritic(RobotEnvironment.observation_space, RobotEnvironment.action_space, )
    export(actor, 'sac.onnx')