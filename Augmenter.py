import numpy as np
from scipy.interpolate import interp1d


class Augmenter:
    def __init__(self, fps, random_state, target_len=16, max_shift=6, speed_range=(0.75, 1.25), pad_value=0.0):
        self.fps = fps
        self.target_len = target_len
        self.max_shift = max_shift
        self.speed_range = speed_range
        self.pad_value = pad_value
        np.random.seed(random_state)

    def augment_window(self, window):
        window = np.asarray(window)
        assert window.shape[0] == self.target_len, f"expected T={self.target_len}, got {window.shape[0]}"

        out = []

        out.append(self.switch_left_and_right_window(window))
        out.append(self.random_time_shift_pad(window, max_shift=self.max_shift, pad_value=self.pad_value))
        # out.append(self.random_time_shift_pad(window, max_shift=self.max_shift, pad_value=self.pad_value))
        # out.append(self.random_time_shift_pad(window, max_shift=self.max_shift, pad_value=self.pad_value))
        time_scaled = self.stretch_squeeze_interp_then_pad_crop(window, speed_range=self.speed_range, pad_value=self.pad_value)
        out.append(time_scaled)
        # out.append(self.stretch_squeeze_interp_then_pad_crop(window, speed_range=self.speed_range, pad_value=self.pad_value))
        out.append(self.jitter_pose_noise(window, sigma=0.015))
        switched_and_time_stretched = self.switch_left_and_right_window(time_scaled.copy())
        out.append(switched_and_time_stretched)
        return out

    # -------------------------
    # 1) Shift: nur pad + crop (keine Interpolation)
    # -------------------------
    def random_time_shift_pad(self, window, max_shift=5, pad_value=0.0):
        window = np.asarray(window)
        T = window.shape[0]
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return window

        pad = np.full((abs(shift), *window.shape[1:]), pad_value, dtype=window.dtype)
        if shift > 0:
            return np.concatenate([pad, window], axis=0)[:T]
        else:
            return np.concatenate([window, pad], axis=0)[-T:]

    # -------------------------
    # 2) Stretch/Squeeze: mit Interpolation -> Länge ändert sich -> am Ende pad/crop auf T
    # -------------------------
    def stretch_squeeze_interp_then_pad_crop(self, window, speed_range=(0.85, 1.15), pad_value=0.0):
        window = np.asarray(window).astype(np.float32)
        T, K, F = window.shape

        speed = np.random.uniform(speed_range[0], speed_range[1])
        new_len = max(2, int(round(T * speed)))  # nach stretch/squeeze

        # Zeitachsen (normiert)
        t_orig = np.linspace(0.0, 1.0, T)
        t_new  = np.linspace(0.0, 1.0, new_len)

        warped = np.zeros((new_len, K, F), dtype=np.float32)

        kind = "cubic" if T >= 4 else "linear"
        for k in range(K):
            for f in range(F):
                seq = window[:, k, f]
                fn = interp1d(t_orig, seq, kind=kind, fill_value="extrapolate")
                warped[:, k, f] = fn(t_new).astype(np.float32)

        # NUR pad/crop zurück auf T (keine zweite Interpolation)
        return self._pad_or_crop_to_T(warped, T, pad_value=pad_value)

    def _pad_or_crop_to_T(self, seq, T, pad_value=0.0):
        L = seq.shape[0]
        if L == T:
            return seq
        if L > T:
            return seq[:T]
        pad = np.full((T - L, *seq.shape[1:]), pad_value, dtype=seq.dtype)
        return np.concatenate([seq, pad], axis=0)

    # -------------------------
    # 3) Left/Right Switch (wie gehabt)
    # -------------------------
    def switch_left_and_right_window(self, window):
        window = np.asarray(window)
        out = np.empty_like(window)
        for t in range(window.shape[0]):
            out[t] = self.switch_left_and_right_one_frame(window[t])
        return out

    def switch_left_and_right_one_frame(self, data):
        data = np.asarray(data).copy()

        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                data[[i, i + 1]] = data[[i + 1, i]]

        data[:, 0] = 1.0 - data[:, 0]  # mirror x
        return data

    def jitter_pose_noise(self, window, sigma=0.01):
        """
        Jittering für bounding-box-normalisierte Skeletons (x,y in [0,1]).

        - Fügt gaußsches Rauschen auf x,y hinzu
        - Punkte die (0,0) sind (Padding/Missing) bleiben unverändert
        - Clipping auf [0,1]
        """

        window = np.asarray(window).astype(np.float32)
        out = window.copy()

        T, K, F = out.shape

        # gültige Punkte (nicht beide 0)
        valid = np.any(out[..., :2] != 0.0, axis=2)  # (T, K)

        if not np.any(valid):
            return out

        # Noise nur für x,y
        noise = np.random.normal(loc=0.0, scale=sigma, size=(T, K, 2)).astype(np.float32)

        # Anwenden nur auf valide Punkte
        mask = valid[..., None]
        out[..., :2] = np.where(mask, out[..., :2] + noise, out[..., :2])

        # Clipping auf [0,1]
        out[..., 0] = np.clip(out[..., 0], 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1], 0.0, 1.0)

        return out
