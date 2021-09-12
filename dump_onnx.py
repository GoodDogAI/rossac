# Jake's code to create an actor critic and export it to ONNX format
import numpy as np
import onnx
import torch

from bot_env import RobotEnvironment


def export(sac, device, file_name, env):
    sample_input = env.observation_space.sample()

    # Expand to put the batch dimension in
    sample_input = np.expand_dims(sample_input, 0)

    # In LSTM Mode, expand the time dimension also, to be a dynamic dimension
    sample_input = np.expand_dims(sample_input, 0)
    sample_input = np.repeat(sample_input, repeats=max(abs(x) for x in sac.pi.history_indexes) + 1, axis=1)

    sample_input = torch.from_numpy(sample_input).to(device=device)

    # Temporarily set the parameters needed for deterministic exports
    orig_det, orig_logprob = sac.pi.deterministic, sac.pi.with_logprob
    orig_stddev = sac.pi.with_stddev
    sac.pi.deterministic = True
    sac.pi.with_logprob = False
    sac.pi.with_stddev = True

    torch.onnx.export(sac.pi, (sample_input,), file_name, verbose=False, opset_version=12,
                      dynamic_axes={
                          "yolo_intermediate": {
                              1: "sequence"
                          },
                      },
                      input_names=["yolo_intermediate"], output_names=["actions", "stddev"])

    onnx_model = onnx.load(file_name)
    onnx.checker.check_model(onnx_model)

    sac.pi.deterministic, sac.pi.with_logprob = orig_det, orig_logprob
    sac.pi.with_stddev = orig_stddev


if __name__ == '__main__':
    from actor_critic.core import MLPActorCritic
    
    actor = MLPActorCritic(RobotEnvironment.observation_space, RobotEnvironment.action_space, )
    export(actor, torch.device("cpu"), 'sac.onnx', RobotEnvironment)