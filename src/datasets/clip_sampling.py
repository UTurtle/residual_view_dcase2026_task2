
import numpy as np

TRANSITION_CROP_POLICIES = {
    "fixed10s",
    "fixed_front_10s",
    "fixed_center_10s",
    "fixed_back_10s",
    "fixed_start_0s_10s",
    "fixed_start_3s_13s",
    "fixed_start_6s_16s",
    "early_transition_active_stable_10s",
    "transition_after_10s",
}

FIXED_START_SECONDS = {
    "fixed10s": 0,
    "fixed_front_10s": 0,
    "fixed_start_0s_10s": 0,
    "fixed_start_3s_13s": 3,
    "fixed_start_6s_16s": 6,
}


def raw_sample():
    pass


def fixed_10s_sample(wave, audio_length=160000):
    return wave[:audio_length]


def _pad_to_length(wave, audio_length):
    if len(wave) >= audio_length:
        return wave
    pad_width = [(0, audio_length - len(wave))]
    pad_width.extend([(0, 0) for _ in range(wave.ndim - 1)])
    return np.pad(wave, pad_width, "constant")


def _fixed_start_10s_sample(wave, start_sample, audio_length):
    end_sample = start_sample + audio_length
    wave = _pad_to_length(wave, end_sample)
    return wave[start_sample:end_sample]


def window_10s_sample(wave, audio_length=160000, stride=16000):
    # 16000 samples = 1 second at 16 kHz.
    wave = _pad_to_length(wave, audio_length)
    starts = range(0, len(wave) - audio_length + 1, stride)
    return np.stack([wave[s:s + audio_length] for s in starts])


def robust_z(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    center = np.median(values)
    mad = np.median(np.abs(values - center))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = float(np.std(values))
    if scale < 1e-9:
        return np.zeros_like(values)
    return (values - center) / scale


def _second_transition_features(wave, sample_rate):
    n_seconds = int(np.ceil(len(wave) / sample_rate))
    padded_len = max(sample_rate, n_seconds * sample_rate)
    wave = _pad_to_length(wave, padded_len)
    if wave.ndim == 1:
        wave_for_fft = wave[:, None]
    else:
        wave_for_fft = wave

    log_energy = []
    spectra = []
    hann = np.hanning(sample_rate)[:, None]
    eps = 1e-12
    for second in range(n_seconds):
        start = second * sample_rate
        end = start + sample_rate
        chunk = wave_for_fft[start:end]
        energy = float(np.mean(np.square(chunk)))
        log_energy.append(float(np.log(energy + eps)))
        magnitude = np.abs(np.fft.rfft(chunk * hann, axis=0)).mean(axis=1)
        magnitude = magnitude / (float(magnitude.sum()) + eps)
        spectra.append(magnitude)

    log_energy = np.asarray(log_energy, dtype=np.float64)
    energy_change = np.zeros(n_seconds, dtype=np.float64)
    spectral_flux = np.zeros(n_seconds, dtype=np.float64)
    for second in range(1, n_seconds):
        energy_change[second] = abs(log_energy[second] - log_energy[second - 1])
        spectral_flux[second] = float(
            np.linalg.norm(spectra[second] - spectra[second - 1])
        )
    transition = robust_z(energy_change) + robust_z(spectral_flux)
    return log_energy, energy_change, transition


def _transition_window_scores(wave, audio_length, sample_rate):
    window_seconds = int(round(audio_length / sample_rate))
    log_energy, energy_change, transition = _second_transition_features(
        wave,
        sample_rate,
    )
    n_seconds = len(log_energy)
    max_start = max(0, n_seconds - window_seconds)
    rows = []
    for start in range(max_start + 1):
        end = start + window_seconds
        window_log_energy = log_energy[start:end]
        window_change = energy_change[start + 1:end]
        post_settle_change = energy_change[min(start + 2, end):end]
        start_transition = transition[start] if start > 0 else 0.0
        early_transition = transition[start + 1:min(start + 3, n_seconds)]
        rows.append(
            {
                "start": start,
                "active": float(np.mean(window_log_energy)),
                "inside_change": (
                    float(np.mean(window_change)) if window_change.size else 0.0
                ),
                "post_change": (
                    float(np.mean(post_settle_change))
                    if post_settle_change.size
                    else 0.0
                ),
                "start_transition": float(start_transition),
                "early_transition": (
                    float(np.max(early_transition)) if early_transition.size else 0.0
                ),
            }
        )
    return rows


def transition_10s_sample(
    wave,
    audio_length=160000,
    sample_rate=16000,
    policy="fixed10s",
):
    if policy not in TRANSITION_CROP_POLICIES:
        raise ValueError(
            f"Unknown crop policy {policy}. "
            f"Expected one of {sorted(TRANSITION_CROP_POLICIES)}."
        )
    wave = _pad_to_length(wave, audio_length)
    if policy in FIXED_START_SECONDS:
        start_sample = FIXED_START_SECONDS[policy] * sample_rate
        return _fixed_start_10s_sample(wave, start_sample, audio_length)
    if policy == "fixed_center_10s":
        start_sample = max(0, (len(wave) - audio_length) // 2)
        return _fixed_start_10s_sample(wave, start_sample, audio_length)
    if policy == "fixed_back_10s":
        start_sample = max(0, len(wave) - audio_length)
        return _fixed_start_10s_sample(wave, start_sample, audio_length)
    if len(wave) == audio_length:
        return wave[:audio_length]

    rows = _transition_window_scores(wave, audio_length, sample_rate)
    active_z = robust_z([row["active"] for row in rows])
    post_stability_z = robust_z([-row["post_change"] for row in rows])
    start_transition_z = robust_z([row["start_transition"] for row in rows])
    early_transition_z = robust_z([row["early_transition"] for row in rows])

    best_start = 0
    best_score = -np.inf
    for idx, row in enumerate(rows):
        if policy == "transition_after_10s":
            score = start_transition_z[idx] + active_z[idx] + post_stability_z[idx]
        elif policy == "early_transition_active_stable_10s":
            score = early_transition_z[idx] + active_z[idx] + post_stability_z[idx]
        else:
            raise AssertionError("unreachable")
        if score > best_score:
            best_score = float(score)
            best_start = int(row["start"])

    start_sample = best_start * sample_rate
    return wave[start_sample:start_sample + audio_length]
