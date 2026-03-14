from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import numpy as np
import sounddevice as sd


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
PALM_POINTS = (0, 5, 9, 13, 17)
TIP_POINTS = (4, 8, 12, 16, 20)
LONG_FINGERS = (8, 12, 16, 20)


@dataclass(frozen=True)
class ThereminConfig:
    sample_rate: int = 44100
    audio_block_size: int = 512
    pitch_zone: tuple[float, float] = (0.52, 0.94)
    volume_zone: tuple[float, float] = (0.06, 0.48)
    frequency_min: float = 110.0
    frequency_max: float = 987.77
    fine_tune_semitones: float = 1.35
    volume_floor: float = 0.01
    master_gain: float = 0.18
    pitch_smoothing: float = 0.08
    amplitude_smoothing: float = 0.12
    brightness_smoothing: float = 0.08
    vibrato_smoothing: float = 0.12
    vibrato_rate_hz: float = 5.6
    vibrato_max_cents: float = 28.0
    release_hold_frames: int = 4
    release_openness_threshold: float = 1.55
    release_drop_threshold: float = 0.022
    release_velocity_threshold: float = 0.014
    release_horizontal_tolerance: float = 0.05
    hand_depth_threshold: float = 0.18
    model_path: str = "hand_landmarker.task"
    window_title: str = "Air Theremin"


@dataclass
class HandData:
    role: str
    landmarks: list


@dataclass(frozen=True)
class HandFeatures:
    center_x: float
    center_y: float
    wrist_x: float
    wrist_y: float
    openness: float
    finger_spread: float
    thumb_index_distance: float


@dataclass
class ThereminState:
    frequency: float = 220.0
    amplitude: float = 0.0
    brightness: float = 0.2
    vibrato: float = 0.0
    voice_state: str = "silent"


@dataclass
class ThereminAudioEngine:
    config: ThereminConfig
    current_frequency: float = field(init=False)
    target_frequency: float = field(init=False)
    current_amplitude: float = field(init=False, default=0.0)
    target_amplitude: float = field(init=False, default=0.0)
    current_brightness: float = field(init=False, default=0.2)
    target_brightness: float = field(init=False, default=0.2)
    current_vibrato: float = field(init=False, default=0.0)
    target_vibrato: float = field(init=False, default=0.0)
    phase: float = field(init=False, default=0.0)
    lfo_phase: float = field(init=False, default=0.0)
    lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    stream: sd.OutputStream | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.current_frequency = self.config.frequency_min * 2.0
        self.target_frequency = self.current_frequency

    def start(self) -> None:
        self.stream = sd.OutputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.audio_block_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is None:
            return
        self.stream.stop()
        self.stream.close()
        self.stream = None

    def update_voice(self, frequency: float, amplitude: float, brightness: float, vibrato: float) -> None:
        with self.lock:
            self.target_frequency = float(np.clip(frequency, self.config.frequency_min, self.config.frequency_max))
            self.target_amplitude = float(np.clip(amplitude, 0.0, 1.0))
            self.target_brightness = float(np.clip(brightness, 0.0, 1.0))
            self.target_vibrato = float(np.clip(vibrato, 0.0, 1.0))

    def _callback(self, outdata: np.ndarray, frames: int, _time: object, status: sd.CallbackFlags) -> None:
        if status:
            pass

        with self.lock:
            target_frequency = self.target_frequency
            target_amplitude = self.target_amplitude
            target_brightness = self.target_brightness
            target_vibrato = self.target_vibrato

        freq_line = np.empty(frames, dtype=np.float64)
        amp_line = np.empty(frames, dtype=np.float64)
        bright_line = np.empty(frames, dtype=np.float64)
        vib_line = np.empty(frames, dtype=np.float64)

        current_frequency = self.current_frequency
        current_amplitude = self.current_amplitude
        current_brightness = self.current_brightness
        current_vibrato = self.current_vibrato

        for idx in range(frames):
            current_frequency += (target_frequency - current_frequency) * self.config.pitch_smoothing
            current_amplitude += (target_amplitude - current_amplitude) * self.config.amplitude_smoothing
            current_brightness += (target_brightness - current_brightness) * self.config.brightness_smoothing
            current_vibrato += (target_vibrato - current_vibrato) * self.config.vibrato_smoothing
            freq_line[idx] = current_frequency
            amp_line[idx] = current_amplitude
            bright_line[idx] = current_brightness
            vib_line[idx] = current_vibrato

        lfo_increments = (2.0 * np.pi * self.config.vibrato_rate_hz) / self.config.sample_rate
        lfo_phase_line = self.lfo_phase + np.arange(1, frames + 1, dtype=np.float64) * lfo_increments
        vibrato_ratio = 2.0 ** ((np.sin(lfo_phase_line) * vib_line * self.config.vibrato_max_cents) / 1200.0)
        phase_increments = (2.0 * np.pi * (freq_line * vibrato_ratio)) / self.config.sample_rate
        phase_line = self.phase + np.cumsum(phase_increments)

        second_harmonic = (0.08 + 0.18 * bright_line) * np.sin(2.0 * phase_line + 0.12)
        third_harmonic = (0.03 + 0.09 * bright_line) * np.sin(3.0 * phase_line + 0.21)
        airy_layer = (0.02 + 0.04 * bright_line) * np.sin(5.0 * phase_line + 0.37)
        wave = 0.82 * np.sin(phase_line) + second_harmonic + third_harmonic + airy_layer
        wave *= amp_line

        outdata[:, 0] = (wave * self.config.master_gain).astype(np.float32)

        self.phase = float(phase_line[-1] % (2.0 * np.pi))
        self.lfo_phase = float(lfo_phase_line[-1] % (2.0 * np.pi))
        self.current_frequency = float(freq_line[-1])
        self.current_amplitude = float(amp_line[-1])
        self.current_brightness = float(bright_line[-1])
        self.current_vibrato = float(vib_line[-1])


def create_hand_landmarker(base_dir: Path, config: ThereminConfig) -> vision.HandLandmarker:
    model_path = base_dir / config.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {model_path}")

    base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    return vision.HandLandmarker.create_from_options(options)


def hand_center_x(hand: HandData) -> float:
    return sum(landmark.x for landmark in hand.landmarks) / len(hand.landmarks)


def assign_theremin_hands(result: vision.HandLandmarkerResult, config: ThereminConfig) -> dict[str, HandData]:
    hands: list[HandData] = []
    for index, landmarks in enumerate(result.hand_landmarks):
        label = ""
        if index < len(result.handedness) and result.handedness[index]:
            label = result.handedness[index][0].category_name
        hands.append(HandData(role=label, landmarks=landmarks))

    if not hands:
        return {}

    split_x = 0.5 * (config.volume_zone[1] + config.pitch_zone[0])
    assigned: dict[str, HandData] = {}
    volume_candidates = [hand for hand in hands if hand_center_x(hand) <= split_x]
    pitch_candidates = [hand for hand in hands if hand_center_x(hand) >= split_x]

    if volume_candidates:
        assigned["volume"] = min(volume_candidates, key=hand_center_x)
    if pitch_candidates:
        assigned["pitch"] = max(pitch_candidates, key=hand_center_x)

    if "volume" not in assigned or "pitch" not in assigned:
        hands_by_x = sorted(hands, key=hand_center_x)
        assigned.setdefault("volume", hands_by_x[0])
        assigned.setdefault("pitch", hands_by_x[-1])

    return assigned


def _mean_point(landmarks: list, indices: tuple[int, ...]) -> tuple[float, float]:
    xs = [landmarks[idx].x for idx in indices]
    ys = [landmarks[idx].y for idx in indices]
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


def _mean_distance(landmarks: list, origin_index: int, indices: tuple[int, ...]) -> float:
    origin = landmarks[origin_index]
    distances = []
    for idx in indices:
        point = landmarks[idx]
        distances.append(float(np.hypot(point.x - origin.x, point.y - origin.y)))
    return float(sum(distances) / len(distances))


def extract_hand_features(landmarks: list, config: ThereminConfig) -> HandFeatures | None:
    if any(landmarks[idx].z > config.hand_depth_threshold for idx in LONG_FINGERS):
        return None

    center_x, center_y = _mean_point(landmarks, PALM_POINTS)
    wrist = landmarks[0]
    palm_size = max(_mean_distance(landmarks, 0, (5, 9, 13, 17)), 1e-4)
    openness = _mean_distance(landmarks, 0, LONG_FINGERS) / palm_size

    spread_segments = (
        np.hypot(landmarks[8].x - landmarks[12].x, landmarks[8].y - landmarks[12].y),
        np.hypot(landmarks[12].x - landmarks[16].x, landmarks[12].y - landmarks[16].y),
        np.hypot(landmarks[16].x - landmarks[20].x, landmarks[16].y - landmarks[20].y),
    )
    finger_spread = float(sum(spread_segments) / len(spread_segments)) / palm_size
    thumb_index_distance = float(np.hypot(landmarks[4].x - landmarks[8].x, landmarks[4].y - landmarks[8].y)) / palm_size

    return HandFeatures(
        center_x=center_x,
        center_y=center_y,
        wrist_x=float(wrist.x),
        wrist_y=float(wrist.y),
        openness=openness,
        finger_spread=finger_spread,
        thumb_index_distance=thumb_index_distance,
    )


def normalized_to_frequency(x: float, config: ThereminConfig) -> float:
    normalized = np.clip(
        (x - config.pitch_zone[0]) / max(config.pitch_zone[1] - config.pitch_zone[0], 1e-6),
        0.0,
        1.0,
    )
    return float(config.frequency_min * ((config.frequency_max / config.frequency_min) ** normalized))


def fine_tune_multiplier(thumb_index_distance: float, config: ThereminConfig) -> float:
    normalized = np.clip((thumb_index_distance - 0.5) / 1.3, 0.0, 1.0)
    semitone_offset = (normalized - 0.5) * 2.0 * config.fine_tune_semitones
    return float(2 ** (semitone_offset / 12.0))


def volume_for_y(y: float, config: ThereminConfig) -> float:
    normalized = 1.0 - np.clip(y, 0.0, 1.0)
    shaped = normalized ** 1.6
    return float(config.volume_floor + (1.0 - config.volume_floor) * shaped)


def brightness_for_openness(openness: float) -> float:
    normalized = np.clip((openness - 1.1) / 1.0, 0.0, 1.0)
    return float(0.15 + 0.85 * (normalized ** 0.9))


def vibrato_for_motion(
    previous_features: HandFeatures | None,
    current_features: HandFeatures | None,
) -> float:
    if previous_features is None or current_features is None:
        return 0.0

    wrist_dx = current_features.wrist_x - previous_features.wrist_x
    centered_openness = np.clip((current_features.openness - 1.35) / 0.7, 0.0, 1.0)
    motion = np.clip(abs(wrist_dx) / 0.018, 0.0, 1.0)
    return float(motion * centered_openness)


def update_open_hold_frames(current_frames: int, features: HandFeatures | None, config: ThereminConfig) -> int:
    if features is None:
        return 0
    if features.openness >= config.release_openness_threshold:
        return current_frames + 1
    return 0


def detect_cutoff_gesture(
    previous_features: HandFeatures | None,
    current_features: HandFeatures | None,
    open_hold_frames: int,
    config: ThereminConfig,
) -> bool:
    if previous_features is None or current_features is None:
        return False
    if open_hold_frames < config.release_hold_frames:
        return False

    delta_y = current_features.center_y - previous_features.center_y
    delta_x = abs(current_features.center_x - previous_features.center_x)
    fingers_still_open = current_features.openness >= config.release_openness_threshold * 0.88

    return bool(
        fingers_still_open
        and delta_y >= config.release_drop_threshold
        and delta_y >= config.release_velocity_threshold
        and delta_x <= config.release_horizontal_tolerance
    )


def create_theremin_scene(
    frame_shape: tuple[int, int, int],
    config: ThereminConfig,
    state: ThereminState,
    pitch_features: HandFeatures | None,
    volume_features: HandFeatures | None,
) -> np.ndarray:
    height, width, _ = frame_shape
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    for row in range(height):
        blend = row / max(height - 1, 1)
        frame[row, :] = (
            int(18 + 12 * blend),
            int(20 + 36 * blend),
            int(42 + 78 * blend),
        )

    volume_start_x = int(config.volume_zone[0] * width)
    volume_end_x = int(config.volume_zone[1] * width)
    pitch_start_x = int(config.pitch_zone[0] * width)
    pitch_end_x = int(config.pitch_zone[1] * width)

    cv2.rectangle(frame, (volume_start_x, int(height * 0.12)), (volume_end_x, int(height * 0.88)), (44, 84, 126), 2)
    cv2.rectangle(frame, (pitch_start_x, int(height * 0.12)), (pitch_end_x, int(height * 0.88)), (126, 92, 54), 2)

    if volume_features is not None:
        volume_level = 1.0 - np.clip(volume_features.center_y, 0.0, 1.0)
        fill_top = int((1.0 - volume_level) * height * 0.76 + height * 0.12)
        cv2.rectangle(frame, (volume_start_x + 8, fill_top), (volume_end_x - 8, int(height * 0.88) - 8), (80, 160, 214), -1)
        marker_y = int(volume_features.center_y * height)
        cv2.circle(frame, (int(volume_features.center_x * width), marker_y), 14, (130, 240, 255), -1)

    if pitch_features is not None:
        pitch_marker_x = int(pitch_features.center_x * width)
        cv2.line(frame, (pitch_marker_x, int(height * 0.15)), (pitch_marker_x, int(height * 0.85)), (245, 215, 122), 2)
        cv2.circle(frame, (pitch_marker_x, int(height * 0.5)), 16, (255, 210, 120), -1)

    cv2.putText(frame, "VOLUME / RELEASE", (volume_start_x + 10, int(height * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 240, 250), 2, cv2.LINE_AA)
    cv2.putText(frame, "PITCH / VIBRATO", (pitch_start_x + 10, int(height * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (250, 230, 205), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Freq: {state.frequency:6.1f} Hz", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Level: {state.amplitude:0.2f}", (24, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"State: {state.voice_state}", (24, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Right hand controls pitch. Left hand controls volume and cutoff. Press Q to quit.", (24, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (236, 240, 245), 2, cv2.LINE_AA)

    return frame


def draw_hand_landmarks(frame: np.ndarray, hand: HandData, color: tuple[int, int, int]) -> None:
    height, width, _ = frame.shape
    for start_idx, end_idx in HAND_CONNECTIONS:
        start = hand.landmarks[start_idx]
        end = hand.landmarks[end_idx]
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv2.line(frame, start_point, end_point, color, 3, cv2.LINE_AA)

    for landmark in hand.landmarks:
        point = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, point, 6, (14, 20, 30), -1)
        cv2.circle(frame, point, 4, color, -1)


def run_app(config: ThereminConfig) -> None:
    base_dir = Path(__file__).resolve().parent
    hand_landmarker = create_hand_landmarker(base_dir, config)
    audio_engine = ThereminAudioEngine(config)
    audio_engine.start()

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        audio_engine.stop()
        hand_landmarker.close()
        raise RuntimeError("Could not open the default camera.")

    previous_pitch_features: HandFeatures | None = None
    previous_volume_features: HandFeatures | None = None
    open_hold_frames = 0

    try:
        while camera.isOpened():
            frame_ok, frame = camera.read()
            if not frame_ok:
                raise RuntimeError("Could not read a frame from the camera.")

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = hand_landmarker.detect(mp_image)
            hands = assign_theremin_hands(result, config)
            pitch_hand = hands.get("pitch")
            volume_hand = hands.get("volume")

            pitch_features = extract_hand_features(pitch_hand.landmarks, config) if pitch_hand is not None else None
            volume_features = extract_hand_features(volume_hand.landmarks, config) if volume_hand is not None else None

            open_hold_frames = update_open_hold_frames(open_hold_frames, volume_features, config)
            release_detected = detect_cutoff_gesture(previous_volume_features, volume_features, open_hold_frames, config)

            state = ThereminState()
            state.voice_state = "silent"
            if pitch_features is not None:
                state.frequency = normalized_to_frequency(pitch_features.center_x, config) * fine_tune_multiplier(
                    pitch_features.thumb_index_distance,
                    config,
                )
                state.vibrato = vibrato_for_motion(previous_pitch_features, pitch_features)

            if volume_features is not None and pitch_features is not None:
                state.brightness = brightness_for_openness(volume_features.openness)
                if release_detected:
                    state.amplitude = 0.0
                    state.voice_state = "release"
                    open_hold_frames = 0
                else:
                    state.amplitude = volume_for_y(volume_features.center_y, config)
                    if state.amplitude <= config.volume_floor + 0.02:
                        state.voice_state = "silent"
                        state.amplitude = 0.0
                    elif state.amplitude >= 0.74:
                        state.voice_state = "crescendo"
                    else:
                        state.voice_state = "sustain"
            else:
                state.amplitude = 0.0
                state.brightness = 0.2
                state.vibrato = 0.0

            audio_engine.update_voice(state.frequency, state.amplitude, state.brightness, state.vibrato)

            display_frame = create_theremin_scene(frame.shape, config, state, pitch_features, volume_features)
            if volume_hand is not None:
                draw_hand_landmarks(display_frame, volume_hand, (120, 228, 255))
            if pitch_hand is not None:
                draw_hand_landmarks(display_frame, pitch_hand, (255, 190, 120))
            cv2.imshow(config.window_title, display_frame)

            previous_pitch_features = pitch_features
            previous_volume_features = volume_features

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        audio_engine.stop()
        camera.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()


def main() -> int:
    config = ThereminConfig()
    try:
        run_app(config)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
