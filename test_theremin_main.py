import unittest
from types import SimpleNamespace

from theremin_main import (
    HandData,
    HandFeatures,
    ThereminConfig,
    assign_theremin_hands,
    detect_cutoff_gesture,
    fine_tune_multiplier,
    normalized_to_frequency,
    update_open_hold_frames,
    vibrato_for_motion,
    volume_for_y,
)


class ThereminLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ThereminConfig()

    def make_hand(self, x: float) -> HandData:
        landmarks = [SimpleNamespace(x=x, y=0.5, z=0.0) for _ in range(21)]
        return HandData(role="", landmarks=landmarks)

    def make_features(
        self,
        *,
        center_x: float = 0.3,
        center_y: float = 0.4,
        wrist_x: float = 0.3,
        wrist_y: float = 0.4,
        openness: float = 1.8,
        finger_spread: float = 0.7,
        thumb_index_distance: float = 0.9,
    ) -> HandFeatures:
        return HandFeatures(
            center_x=center_x,
            center_y=center_y,
            wrist_x=wrist_x,
            wrist_y=wrist_y,
            openness=openness,
            finger_spread=finger_spread,
            thumb_index_distance=thumb_index_distance,
        )

    def test_assign_theremin_hands_prefers_screen_zones(self) -> None:
        result = SimpleNamespace(
            hand_landmarks=[self.make_hand(0.18).landmarks, self.make_hand(0.82).landmarks],
            handedness=[[SimpleNamespace(category_name="Right")], [SimpleNamespace(category_name="Left")]],
        )

        hands = assign_theremin_hands(result, self.config)
        self.assertLess(hands["volume"].landmarks[0].x, hands["pitch"].landmarks[0].x)

    def test_normalized_to_frequency_stays_in_range(self) -> None:
        left = normalized_to_frequency(self.config.pitch_zone[0], self.config)
        right = normalized_to_frequency(self.config.pitch_zone[1], self.config)

        self.assertAlmostEqual(left, self.config.frequency_min, places=3)
        self.assertAlmostEqual(right, self.config.frequency_max, places=3)

    def test_volume_for_y_is_louder_near_top(self) -> None:
        quiet = volume_for_y(0.85, self.config)
        loud = volume_for_y(0.15, self.config)
        self.assertLess(quiet, loud)

    def test_fine_tune_multiplier_changes_with_pinch(self) -> None:
        tight = fine_tune_multiplier(0.55, self.config)
        wide = fine_tune_multiplier(1.45, self.config)
        self.assertLess(tight, wide)

    def test_vibrato_for_motion_requires_previous_features(self) -> None:
        current = self.make_features(wrist_x=0.48, openness=1.9)
        self.assertEqual(vibrato_for_motion(None, current), 0.0)

    def test_vibrato_for_motion_grows_with_wrist_motion(self) -> None:
        previous = self.make_features(wrist_x=0.40, openness=1.9)
        current = self.make_features(wrist_x=0.425, openness=1.9)
        self.assertGreater(vibrato_for_motion(previous, current), 0.0)

    def test_update_open_hold_frames_resets_for_closed_hand(self) -> None:
        frames = update_open_hold_frames(3, self.make_features(openness=1.2), self.config)
        self.assertEqual(frames, 0)

    def test_detect_cutoff_gesture_requires_open_hold_and_downward_motion(self) -> None:
        previous = self.make_features(center_y=0.32, openness=1.8)
        current = self.make_features(center_y=0.36, openness=1.75)
        self.assertTrue(
            detect_cutoff_gesture(previous, current, self.config.release_hold_frames, self.config)
        )

    def test_detect_cutoff_gesture_ignores_small_motion(self) -> None:
        previous = self.make_features(center_y=0.32, openness=1.8)
        current = self.make_features(center_y=0.335, openness=1.8)
        self.assertFalse(
            detect_cutoff_gesture(previous, current, self.config.release_hold_frames, self.config)
        )


if __name__ == "__main__":
    unittest.main()
