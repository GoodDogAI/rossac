import numpy as np

def normalize_pantilt(env, pantilt):
    return env.normalize_pan(pantilt[0, np.newaxis]), env.normalize_tilt(pantilt[1, np.newaxis])

def make_observation(env, interpolated_entry, backbone_slice):
    pan_curr, tilt_curr = normalize_pantilt(env, interpolated_entry.dynamixel_cur_state)
    return np.concatenate([pan_curr, tilt_curr,
                            interpolated_entry.head_gyro / 10.0,  # Divide radians/sec by ten to center around 0 closer
                            interpolated_entry.head_accel / 10.0,  # Divide m/s by 10, and center the y axis
                            interpolated_entry.odrive_feedback[0:2],  # Only the actual vel, not the commanded vel
                            interpolated_entry.vbus - 27.0,  # Volts different from ~50% charge
                            interpolated_entry.yolo_intermediate[::backbone_slice]])

def load_entries(entries, replay_buffer, env, backbone_slice):
    num_samples = min(len(entries)-1, replay_buffer.max_size)

    used = [False for _ in range(num_samples + 1)]

    nans = 0
    oobs = 0
    dones = 0

    def ts_from_seconds(seconds):
        return int(seconds * 1000000000)

    MIN_TS_DIFF = ts_from_seconds(0.47)
    MAX_TS_DIFF = ts_from_seconds(0.75)

    t = tqdm(total=num_samples)
    loaded = 0

    threads = 0
    while not all(used[:-1]):
        threads += 1
        last_terminated = False
        lstm_history_count = 0
        last_ts = None
        i = 0
        while i < num_samples:
            if used[i]:
                i += 1
                continue

            used[i] = True
            loaded += 1
            t.update()

            entry = entries.iloc[i]
            ts = entry.name
            i += 1
            while i < num_samples and (entries.iloc[i].name < ts + MIN_TS_DIFF or used[i]):
                i += 1
            if i >= num_samples:
                continue
            next_entry = entries.iloc[i]
            if next_entry.name >= ts + MAX_TS_DIFF:
                lstm_history_count = 0
                continue

            if lstm_history_count >= opt.lstm_history:
                lstm_history_count -= 1

            pan_command, tilt_command = normalize_pantilt(env, entry.dynamixel_command_state)
            pan_curr, tilt_curr = normalize_pantilt(env, entry.dynamixel_cur_state)

            move_penalty = abs(entry.cmd_vel).mean() * 0.002
            pantilt_penalty = float((abs(pan_command - pan_curr) + abs(tilt_command - tilt_curr)) * 0.001)
            if move_penalty + pantilt_penalty > 10:
                print("WARNING: high move penalty!")
            reward = entry.reward
            reward -= move_penalty + pantilt_penalty
            reward += next_entry.punishment * DEFAULT_PUNISHMENT_MULTIPLIER

            obs = make_observation(env, entry, backbone_slice)
            future_obs = make_observation(env, next_entry, backbone_slice)
            lstm_history_count += 1

            if np.isnan(obs).any() or np.isnan(future_obs).any() or np.isnan(reward).any():
                nans += 1
                continue

            if obs.max() > 1000 or future_obs.max() > 1000:
                oobs += 1
                continue

            terminated = next_entry.punishment < -0.0
            if terminated and last_terminated:
                continue
            last_terminated = terminated
            if terminated:
                dones += 1

            replay_buffer.store(obs=obs,
                act=np.concatenate([entry.cmd_vel, pan_command, tilt_command]),
                rew=reward,
                next_obs=future_obs,
                lstm_history_count=lstm_history_count,
                done=terminated)

    t.close()

    print(f"NaNs in {nans} of {num_samples} samples, large obs in {oobs}, threads: {threads}")
    print(f"avg. episode len: {(num_samples + 1) / (dones + 1)}")