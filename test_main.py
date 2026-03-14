import unittest
from types import SimpleNamespace

from main import (
    GuitarConfig,
    HandData,
    attack_samples_for_seconds,
    assign_hands,
    brightness_for_strength,
    decay_seconds_for_strength,
    detect_fretted_strings,
    detect_pluck_events,
    frequency_for_string_and_fret,
    fret_for_x,
    get_string_contact,
    pluck_speed_to_strength,
)


class GuitarLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GuitarConfig()

    def make_hand(self, x: float) -> HandData:
        landmarks = [SimpleNamespace(x=x, y=0.5, z=0.0) for _ in range(21)]
        return HandData(role="", landmarks=landmarks)

    def test_get_string_contact_returns_middle_string(self) -> None:
        contact = get_string_contact(0.4, 0.41, self.config)
        self.assertIsNotNone(contact)
        self.assertEqual(contact[0], 2)

    def test_fret_for_x_maps_neck_to_fret_number(self) -> None:
        fret = fret_for_x(self.config.neck_x_range[0], self.config)
        self.assertEqual(fret, 1)

    def test_frequency_for_string_and_fret_raises_pitch(self) -> None:
        open_frequency = frequency_for_string_and_fret(0, 0, self.config)
        fretted_frequency = frequency_for_string_and_fret(0, 5, self.config)
        self.assertLess(open_frequency, fretted_frequency)

    def test_detect_fretted_strings_keeps_highest_fret_for_same_string(self) -> None:
        fingertip_positions = {
            8: (0.2, 0.4, -0.05),
            12: (0.5, 0.401, -0.03),
        }

        fretted_strings = detect_fretted_strings(fingertip_positions, self.config)
        self.assertEqual(fretted_strings[2], fret_for_x(0.5, self.config))

    def test_detect_pluck_events_requires_crossing_motion(self) -> None:
        previous_positions = {8: (0.8, 0.34, -0.02)}
        current_positions = {8: (0.8, 0.46, -0.02)}

        plucks = detect_pluck_events(previous_positions, current_positions, self.config)
        self.assertIn(2, plucks)

    def test_detect_pluck_events_ignores_motion_outside_pluck_zone(self) -> None:
        previous_positions = {8: (0.5, 0.34, -0.02)}
        current_positions = {8: (0.5, 0.46, -0.02)}

        plucks = detect_pluck_events(previous_positions, current_positions, self.config)
        self.assertEqual(plucks, {})

    def test_pluck_speed_to_strength_increases_with_faster_motion(self) -> None:
        soft_strength = pluck_speed_to_strength(self.config.min_pluck_speed, self.config)
        hard_strength = pluck_speed_to_strength(self.config.max_pluck_speed, self.config)
        self.assertLess(soft_strength, hard_strength)

    def test_detect_pluck_events_gives_stronger_value_for_faster_motion(self) -> None:
        slow_previous = {8: (0.8, 0.34, -0.02)}
        slow_current = {8: (0.8, 0.39, -0.02)}
        fast_previous = {8: (0.8, 0.34, -0.02)}
        fast_current = {8: (0.8, 0.52, -0.02)}

        slow_plucks = detect_pluck_events(slow_previous, slow_current, self.config)
        fast_plucks = detect_pluck_events(fast_previous, fast_current, self.config)

        self.assertLess(slow_plucks[2][0], fast_plucks[2][0])
        self.assertLess(slow_plucks[2][1], fast_plucks[2][1])

    def test_decay_seconds_for_strength_grows_with_stronger_pluck(self) -> None:
        short_decay = decay_seconds_for_strength(self.config.min_pluck_strength, self.config)
        long_decay = decay_seconds_for_strength(1.0, self.config)
        self.assertLess(short_decay, long_decay)

    def test_attack_samples_for_seconds_is_positive(self) -> None:
        self.assertGreaterEqual(attack_samples_for_seconds(self.config.pluck_attack_seconds, self.config), 1)

    def test_brightness_for_strength_grows_with_stronger_pluck(self) -> None:
        soft_brightness = brightness_for_strength(self.config.min_pluck_strength, self.config)
        hard_brightness = brightness_for_strength(1.0, self.config)
        self.assertLess(soft_brightness, hard_brightness)

    def test_assign_hands_prefers_screen_zones_over_labels(self) -> None:
        result = SimpleNamespace(
            hand_landmarks=[self.make_hand(0.22).landmarks, self.make_hand(0.82).landmarks],
            handedness=[[SimpleNamespace(category_name="Right")], [SimpleNamespace(category_name="Left")]],
        )

        hands = assign_hands(result, self.config)
        self.assertLess(hands["fretting"].landmarks[0].x, hands["plucking"].landmarks[0].x)


if __name__ == "__main__":
    unittest.main()
