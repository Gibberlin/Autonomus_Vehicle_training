import unittest
from types import SimpleNamespace
from unittest.mock import patch

from navigation.fixed_route import (
    blend_model_and_navigation_steering,
    build_route_from_locations,
)
from live_demo import parse_args


class FakeLocation:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class FakePlanner:
    def __init__(self, empty=False):
        self.calls = []
        self.empty = empty

    def trace_route(self, start, destination):
        self.calls.append((start, destination))
        if self.empty:
            return []
        return [
            (SimpleNamespace(transform=SimpleNamespace(location=start)), "start"),
            (SimpleNamespace(transform=SimpleNamespace(location=destination)), "end"),
        ]


class TestModelNavigation(unittest.TestCase):
    def test_route_segments_are_sequential_and_joined(self):
        start = FakeLocation(0, 0)
        checkpoint_one = FakeLocation(3, 4)
        checkpoint_two = FakeLocation(6, 8)
        planner = FakePlanner()

        route, distance = build_route_from_locations(
            planner,
            start,
            [("P1", checkpoint_one), ("P2", checkpoint_two)],
        )

        self.assertEqual(planner.calls, [
            (start, checkpoint_one),
            (checkpoint_one, checkpoint_two),
        ])
        self.assertEqual(len(route), 3)
        self.assertAlmostEqual(distance, 10.0)

    def test_empty_route_segment_fails_with_target(self):
        planner = FakePlanner(empty=True)

        with self.assertRaisesRegex(RuntimeError, "P1"):
            build_route_from_locations(
                planner,
                FakeLocation(0, 0),
                [("P1", FakeLocation(3, 4))],
            )

    def test_navigation_has_primary_steering_weight(self):
        self.assertAlmostEqual(
            blend_model_and_navigation_steering(1.0, -1.0),
            -0.5,
        )
        self.assertEqual(
            blend_model_and_navigation_steering(1.0, 1.0),
            1.0,
        )

    def test_live_demo_accepts_navigation_option(self):
        with patch("sys.argv", ["live_demo.py", "--navigation", "--start-model"]):
            args = parse_args()

        self.assertTrue(args.navigation)
        self.assertFalse(args.start_autopilot)


if __name__ == "__main__":
    unittest.main()
