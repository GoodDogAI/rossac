# Jake's code to create an actor critic and export it to ONNX format
import numpy as np
import torch

from bot_env import RobotEnvironment


def export(sac, device, file_name):
    sample_input = RobotEnvironment.observation_space.sample()
    sample_input = np.expand_dims(sample_input, 0)
    sample_input = torch.from_numpy(sample_input).to(device=device)

    # Temporarily set the parameters needed for deterministic exports
    orig_det, orig_logprob = sac.pi.deterministic, sac.pi.with_logprob
    sac.pi.deterministic = True
    sac.pi.with_logprob = True

    torch.onnx.export(sac.pi, (sample_input,), file_name, verbose=False, opset_version=12,
                      input_names=["yolo_intermediate"], output_names=["actions", "log_prob"])

    sac.pi.deterministic, sac.pi.with_logprob = orig_det, orig_logprob


if __name__ == '__main__':
    from actor_critic.core import MLPActorCritic
    
    actor = MLPActorCritic(RobotEnvironment.observation_space, RobotEnvironment.action_space, )
    export(actor, torch.device("cpu"), 'sac.onnx')