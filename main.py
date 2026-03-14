from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import numpy as np
import sounddevice as sd


FRET_FINGERS = (8, 12, 16, 20)
PLUCK_FINGERS = (4, 8, 12)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True)
class GuitarConfig:
    sample_rate: int = 44100
    audio_block_size: int = 512
    string_threshold: float = 0.045
    fret_threshold: float = 0.06
    pluck_min_delta: float = 0.012
    pluck_cooldown_seconds: float = 0.09
    neck_x_range: tuple[float, float] = (0.1, 0.68)
    pluck_x_range: tuple[float, float] = (0.72, 0.93)
    string_ys: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    open_string_frequencies: tuple[float, ...] = (82.41, 110.0, 146.83, 196.0, 246.94, 329.63)
    fret_count: int = 8
    pitch_smoothing: float = 0.05
    master_gain: float = 0.16
    min_pluck_strength: float = 0.1
    min_pluck_speed: float = 0.01
    max_pluck_speed: float = 0.22
    min_string_decay_seconds: float = 0.8
    max_string_decay_seconds: float = 4.2
    pluck_attack_seconds: float = 0.012
    brightness_decay_seconds: float = 0.18
    model_path: str = "hand_landmarker.task"
    window_title: str = "Air Guitar"


@dataclass
class HandData:
    role: str
    landmarks: list


@dataclass
class AudioEngine:
    config: GuitarConfig
    phases: np.ndarray = field(init=False)
    current_frequencies: np.ndarray = field(init=False)
    target_frequencies: np.ndarray = field(init=False)
    envelopes: np.ndarray = field(init=False)
    pending_plucks: np.ndarray = field(init=False)
    current_decay_factors: np.ndarray = field(init=False)
    pending_decay_factors: np.ndarray = field(init=False)
    attack_samples_remaining: np.ndarray = field(init=False)
    attack_step_sizes: np.ndarray = field(init=False)
    brightness: np.ndarray = field(init=False)
    lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    stream: sd.OutputStream | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        string_count = len(self.config.string_ys)
        self.phases = np.zeros(string_count, dtype=np.float64)
        self.current_frequencies = np.array(self.config.open_string_frequencies, dtype=np.float64)
        self.target_frequencies = np.array(self.config.open_string_frequencies, dtype=np.float64)
        self.envelopes = np.zeros(string_count, dtype=np.float64)
        self.pending_plucks = np.zeros(string_count, dtype=np.float64)
        default_decay_factor = decay_factor_for_seconds(self.config.min_string_decay_seconds, self.config)
        self.current_decay_factors = np.full(string_count, default_decay_factor, dtype=np.float64)
        self.pending_decay_factors = np.full(string_count, default_decay_factor, dtype=np.float64)
        self.attack_samples_remaining = np.zeros(string_count, dtype=np.int32)
        self.attack_step_sizes = np.zeros(string_count, dtype=np.float64)
        self.brightness = np.zeros(string_count, dtype=np.float64)

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

    def pluck_strings(self, pluck_events: dict[int, tuple[float, float, float]]) -> None:
        with self.lock:
            for string_idx, (frequency, strength, decay_factor) in pluck_events.items():
                self.target_frequencies[string_idx] = frequency
                self.pending_plucks[string_idx] = max(self.pending_plucks[string_idx], strength)
                self.pending_decay_factors[string_idx] = decay_factor

    def _callback(self, outdata: np.ndarray, frames: int, _time: object, status: sd.CallbackFlags) -> None:
        if status:
            pass

        with self.lock:
            target_frequencies = self.target_frequencies.copy()
            pending_plucks = self.pending_plucks.copy()
            pending_decay_factors = self.pending_decay_factors.copy()
            self.pending_plucks.fill(0.0)

        mix = np.zeros(frames, dtype=np.float64)
        attack_samples = attack_samples_for_seconds(self.config.pluck_attack_seconds, self.config)
        brightness_decay_factor = decay_factor_for_seconds(self.config.brightness_decay_seconds, self.config)
        for string_idx in range(len(self.config.string_ys)):
            current_frequency = self.current_frequencies[string_idx]
            envelope = self.envelopes[string_idx]
            decay_factor = self.current_decay_factors[string_idx]
            attack_remaining = int(self.attack_samples_remaining[string_idx])
            attack_step = self.attack_step_sizes[string_idx]
            brightness = self.brightness[string_idx]

            if pending_plucks[string_idx] > 0.0:
                target_envelope = max(pending_plucks[string_idx], envelope)
                attack_remaining = attack_samples
                attack_step = (target_envelope - envelope) / attack_remaining
                decay_factor = pending_decay_factors[string_idx]
                brightness = max(brightness, brightness_for_strength(pending_plucks[string_idx], self.config))

            freq_line = np.empty(frames, dtype=np.float64)
            env_line = np.empty(frames, dtype=np.float64)
            brightness_line = np.empty(frames, dtype=np.float64)
            for frame_idx in range(frames):
                current_frequency += (target_frequencies[string_idx] - current_frequency) * self.config.pitch_smoothing
                if attack_remaining > 0:
                    envelope += attack_step
                    attack_remaining -= 1
                else:
                    envelope *= decay_factor
                brightness *= brightness_decay_factor
                freq_line[frame_idx] = current_frequency
                env_line[frame_idx] = max(envelope, 0.0)
                brightness_line[frame_idx] = brightness

            if np.max(env_line) < 1e-4:
                self.current_frequencies[string_idx] = current_frequency
                self.envelopes[string_idx] = envelope
                self.attack_samples_remaining[string_idx] = attack_remaining
                self.attack_step_sizes[string_idx] = attack_step
                self.brightness[string_idx] = brightness
                continue

            increments = (2.0 * np.pi * freq_line) / self.config.sample_rate
            phase_line = self.phases[string_idx] + np.cumsum(increments)
            second_harmonic = (0.11 + 0.16 * brightness_line) * np.sin(2.0 * phase_line + 0.18)
            third_harmonic = (0.03 + 0.09 * (brightness_line ** 1.4)) * np.sin(3.0 * phase_line + 0.32)
            fourth_harmonic = 0.025 * brightness_line * np.sin(4.0 * phase_line + 0.45)
            wave = 0.92 * np.sin(phase_line) + second_harmonic + third_harmonic + fourth_harmonic
            mix += wave * env_line

            self.phases[string_idx] = float(phase_line[-1] % (2.0 * np.pi))
            self.current_frequencies[string_idx] = current_frequency
            self.envelopes[string_idx] = envelope
            self.current_decay_factors[string_idx] = decay_factor
            self.attack_samples_remaining[string_idx] = attack_remaining
            self.attack_step_sizes[string_idx] = attack_step
            self.brightness[string_idx] = brightness

        outdata[:, 0] = (mix * self.config.master_gain).astype(np.float32)


def create_hand_landmarker(base_dir: Path, config: GuitarConfig) -> vision.HandLandmarker:
    model_path = base_dir / config.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {model_path}")

    base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    return vision.HandLandmarker.create_from_options(options)


def pluck_speed_to_strength(speed: float, config: GuitarConfig) -> float:
    normalized = np.clip(
        (speed - config.min_pluck_speed) / max(config.max_pluck_speed - config.min_pluck_speed, 1e-6),
        0.0,
        1.0,
    )
    shaped = normalized ** 1.8
    return config.min_pluck_strength + (1.0 - config.min_pluck_strength) * shaped


def decay_seconds_for_strength(strength: float, config: GuitarConfig) -> float:
    normalized = np.clip(
        (strength - config.min_pluck_strength) / max(1.0 - config.min_pluck_strength, 1e-6),
        0.0,
        1.0,
    )
    shaped = normalized ** 0.7
    return config.min_string_decay_seconds + (
        config.max_string_decay_seconds - config.min_string_decay_seconds
    ) * shaped


def decay_factor_for_seconds(decay_seconds: float, config: GuitarConfig) -> float:
    return float(np.exp(-1.0 / (decay_seconds * config.sample_rate)))


def attack_samples_for_seconds(attack_seconds: float, config: GuitarConfig) -> int:
    return max(1, int(round(attack_seconds * config.sample_rate)))


def brightness_for_strength(strength: float, config: GuitarConfig) -> float:
    normalized = np.clip(
        (strength - config.min_pluck_strength) / max(1.0 - config.min_pluck_strength, 1e-6),
        0.0,
        1.0,
    )
    return float(0.25 + 0.75 * (normalized ** 0.85))


def line_crosses_string(previous_y: float, current_y: float, string_y: float, threshold: float) -> bool:
    previous_delta = previous_y - string_y
    current_delta = current_y - string_y

    if abs(previous_delta) <= threshold or abs(current_delta) <= threshold:
        return True
    return previous_delta * current_delta < 0


def note_name_for_frequency(frequency: float) -> str:
    midi_note = int(round(69 + 12 * np.log2(frequency / 440.0)))
    return NOTE_NAMES[midi_note % 12]


def fret_for_x(x: float, config: GuitarConfig) -> int | None:
    min_x, max_x = config.neck_x_range
    if not (min_x <= x <= max_x):
        return None

    normalized_x = np.clip((x - min_x) / max(max_x - min_x, 1e-6), 0.0, 1.0)
    return 1 + int(round(normalized_x * (config.fret_count - 1)))


def frequency_for_string_and_fret(string_idx: int, fret: int, config: GuitarConfig) -> float:
    return config.open_string_frequencies[string_idx] * (2 ** (fret / 12.0))


def get_string_contact(x: float, y: float, config: GuitarConfig) -> tuple[int, float] | None:
    if not (config.neck_x_range[0] <= x <= config.pluck_x_range[1]):
        return None

    best_string_idx = None
    best_distance = None
    for string_idx, string_y in enumerate(config.string_ys):
        distance = abs(y - string_y)
        if distance > config.string_threshold:
            continue
        if best_distance is None or distance < best_distance:
            best_string_idx = string_idx
            best_distance = distance

    if best_string_idx is None or best_distance is None:
        return None
    return best_string_idx, best_distance


def hand_center_x(hand: HandData) -> float:
    return sum(landmark.x for landmark in hand.landmarks) / len(hand.landmarks)


def assign_hands(result: vision.HandLandmarkerResult, config: GuitarConfig) -> dict[str, HandData]:
    hands: list[HandData] = []
    for index, landmarks in enumerate(result.hand_landmarks):
        label = ""
        if index < len(result.handedness) and result.handedness[index]:
            label = result.handedness[index][0].category_name
        hands.append(HandData(role=label, landmarks=landmarks))

    if not hands:
        return {}

    # Prefer stable screen-space zones over MediaPipe handedness labels.
    fretting_candidates = [
        hand for hand in hands
        if hand_center_x(hand) <= 0.5 * (config.neck_x_range[1] + config.pluck_x_range[0])
    ]
    plucking_candidates = [
        hand for hand in hands
        if hand_center_x(hand) >= 0.5 * (config.neck_x_range[1] + config.pluck_x_range[0])
    ]

    assigned: dict[str, HandData] = {}
    if fretting_candidates:
        assigned["fretting"] = min(fretting_candidates, key=hand_center_x)
    if plucking_candidates:
        assigned["plucking"] = max(plucking_candidates, key=hand_center_x)

    if "fretting" not in assigned or "plucking" not in assigned:
        hands_by_x = sorted(hands, key=hand_center_x)
        assigned.setdefault("fretting", hands_by_x[0])
        assigned.setdefault("plucking", hands_by_x[-1])

    return assigned


def extract_fingertip_positions(landmarks: list, tip_indices: tuple[int, ...]) -> dict[int, tuple[float, float, float]]:
    positions: dict[int, tuple[float, float, float]] = {}
    for tip_idx in tip_indices:
        tip = landmarks[tip_idx]
        if tip.z > 0.15:
            continue
        positions[tip_idx] = (tip.x, tip.y, tip.z)
    return positions


def detect_fretted_strings(
    fingertip_positions: dict[int, tuple[float, float, float]],
    config: GuitarConfig,
) -> dict[int, int]:
    fretted_strings: dict[int, tuple[int, float]] = {}

    for x, y, _ in fingertip_positions.values():
        contact = get_string_contact(x, y, config)
        fret = fret_for_x(x, config)
        if contact is None or fret is None:
            continue

        string_idx, distance = contact
        if distance > config.fret_threshold:
            continue

        best = fretted_strings.get(string_idx)
        if best is None or fret > best[0] or (fret == best[0] and distance < best[1]):
            fretted_strings[string_idx] = (fret, distance)

    return {string_idx: fret for string_idx, (fret, _) in fretted_strings.items()}


def detect_pluck_events(
    previous_positions: dict[int, tuple[float, float, float]],
    current_positions: dict[int, tuple[float, float, float]],
    config: GuitarConfig,
) -> dict[int, tuple[float, float]]:
    plucked_strings: dict[int, tuple[float, float]] = {}

    for tip_idx, (current_x, current_y, current_z) in current_positions.items():
        previous_position = previous_positions.get(tip_idx)
        if previous_position is None:
            continue

        previous_x, previous_y, previous_z = previous_position
        avg_x = (previous_x + current_x) / 2.0
        if not (config.pluck_x_range[0] <= avg_x <= config.pluck_x_range[1]):
            continue
        delta_y = current_y - previous_y
        if abs(delta_y) < config.pluck_min_delta:
            continue

        delta_x = current_x - previous_x
        delta_z = current_z - previous_z
        impact_speed = float(np.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z))

        for string_idx, string_y in enumerate(config.string_ys):
            if line_crosses_string(previous_y, current_y, string_y, config.string_threshold):
                strength = pluck_speed_to_strength(impact_speed, config)
                decay_seconds = decay_seconds_for_strength(strength, config)
                existing = plucked_strings.get(string_idx)
                if existing is None or strength > existing[0]:
                    plucked_strings[string_idx] = (strength, decay_seconds)

    return plucked_strings


def create_guitar_scene(
    frame_shape: tuple[int, int, int],
    config: GuitarConfig,
    fretted_strings: dict[int, int],
) -> np.ndarray:
    height, width, _ = frame_shape
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    neck_start_x = int(config.neck_x_range[0] * width)
    neck_end_x = int(config.neck_x_range[1] * width)
    body_start_x = int(config.pluck_x_range[0] * width)

    for row in range(height):
        blend = row / max(height - 1, 1)
        frame[row, :] = (
            int(22 + 18 * blend),
            int(48 + 62 * blend),
            int(78 + 110 * blend),
        )

    cv2.rectangle(frame, (neck_start_x, int(height * 0.14)), (neck_end_x, int(height * 0.76)), (72, 118, 170), -1)
    cv2.rectangle(frame, (neck_start_x, int(height * 0.14)), (neck_end_x, int(height * 0.76)), (20, 28, 42), 2)
    cv2.ellipse(
        frame,
        (int(width * 0.84), int(height * 0.48)),
        (int(width * 0.13), int(height * 0.24)),
        0,
        0,
        360,
        (52, 104, 164),
        -1,
    )
    cv2.ellipse(
        frame,
        (int(width * 0.84), int(height * 0.48)),
        (int(width * 0.13), int(height * 0.24)),
        0,
        0,
        360,
        (18, 26, 40),
        3,
    )

    for fret_idx in range(config.fret_count):
        x = int(
            config.neck_x_range[0] * width
            + (config.neck_x_range[1] - config.neck_x_range[0]) * width * (fret_idx / max(config.fret_count - 1, 1))
        )
        cv2.line(frame, (x, int(height * 0.14)), (x, int(height * 0.76)), (210, 220, 230), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            str(fret_idx + 1),
            (x - 8, int(height * 0.12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 240, 250),
            1,
            cv2.LINE_AA,
        )

    for string_idx, string_y in enumerate(config.string_ys):
        y = int(string_y * height)
        cv2.line(frame, (neck_start_x, y), (int(width * 0.93), y), (240, 245, 255), 2)
        if string_idx in fretted_strings:
            fret = fretted_strings[string_idx]
            fret_x = fret_position_x(fret, width, config)
            note = note_name_for_frequency(frequency_for_string_and_fret(string_idx, fret, config))
            cv2.circle(frame, (fret_x, y), 10, (110, 225, 255), -1)
            cv2.putText(
                frame,
                note,
                (fret_x - 12, y - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.rectangle(
        frame,
        (body_start_x, int(height * 0.14)),
        (int(width * 0.93), int(height * 0.76)),
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "Left hand frets on the neck. Right hand plucks on the body. Press Q to quit.",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Neck",
        (neck_start_x + 8, int(height * 0.84)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (235, 240, 250),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Pluck zone",
        (body_start_x + 8, int(height * 0.84)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (235, 240, 250),
        1,
        cv2.LINE_AA,
    )

    return frame


def fret_position_x(fret: int, width: int, config: GuitarConfig) -> int:
    if config.fret_count <= 1:
        return int(sum(config.neck_x_range) * width / 2)
    x = config.neck_x_range[0] + (config.neck_x_range[1] - config.neck_x_range[0]) * ((fret - 1) / (config.fret_count - 1))
    return int(x * width)


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
        cv2.circle(frame, point, 6, (20, 30, 45), -1)
        cv2.circle(frame, point, 4, color, -1)


def run_app(config: GuitarConfig) -> None:
    base_dir = Path(__file__).resolve().parent
    hand_landmarker = create_hand_landmarker(base_dir, config)
    audio_engine = AudioEngine(config)
    audio_engine.start()

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        audio_engine.stop()
        hand_landmarker.close()
        raise RuntimeError("Could not open the default camera.")

    previous_pluck_positions: dict[int, tuple[float, float, float]] = {}
    last_plucked_at = {string_idx: 0.0 for string_idx in range(len(config.string_ys))}

    try:
        while camera.isOpened():
            frame_ok, frame = camera.read()
            if not frame_ok:
                raise RuntimeError("Could not read a frame from the camera.")

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = hand_landmarker.detect(mp_image)
            hands = assign_hands(result, config)
            fretting_hand = hands.get("fretting")
            plucking_hand = hands.get("plucking")

            fretted_strings: dict[int, int] = {}
            pluck_positions: dict[int, tuple[float, float, float]] = {}
            if fretting_hand is not None:
                fret_positions = extract_fingertip_positions(fretting_hand.landmarks, FRET_FINGERS)
                fretted_strings = detect_fretted_strings(fret_positions, config)
            if plucking_hand is not None:
                pluck_positions = extract_fingertip_positions(plucking_hand.landmarks, PLUCK_FINGERS)

            raw_plucks = detect_pluck_events(previous_pluck_positions, pluck_positions, config)
            pluck_events: dict[int, tuple[float, float, float]] = {}
            now = time.monotonic()
            for string_idx, (strength, decay_seconds) in raw_plucks.items():
                if now - last_plucked_at[string_idx] < config.pluck_cooldown_seconds:
                    continue
                fret = fretted_strings.get(string_idx, 0)
                frequency = frequency_for_string_and_fret(string_idx, fret, config)
                pluck_events[string_idx] = (
                    frequency,
                    strength,
                    decay_factor_for_seconds(decay_seconds, config),
                )
                last_plucked_at[string_idx] = now

            if pluck_events:
                audio_engine.pluck_strings(pluck_events)

            display_frame = create_guitar_scene(frame.shape, config, fretted_strings)
            if fretting_hand is not None:
                draw_hand_landmarks(display_frame, fretting_hand, (110, 255, 190))
            if plucking_hand is not None:
                draw_hand_landmarks(display_frame, plucking_hand, (255, 185, 90))
            cv2.imshow(config.window_title, display_frame)

            previous_pluck_positions = pluck_positions

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        audio_engine.stop()
        camera.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()


def main() -> int:
    config = GuitarConfig()
    try:
        run_app(config)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
