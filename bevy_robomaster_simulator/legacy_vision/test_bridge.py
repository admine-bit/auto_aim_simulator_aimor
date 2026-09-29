"""Protocol regression checks for the simulator-side legacy serial bridge."""

import math
import struct
import unittest

import bridge


class BridgeFeedbackTest(unittest.TestCase):
    def test_feedback_applies_configured_height_offset(self):
        meta = bytearray(bridge.META_SIZE)
        tilt = math.radians(25.0) / 2.0
        for channel, position, quaternion in (
            (0, (0.0, 0.0, 0.0), (math.cos(tilt), 0.0, -math.sin(tilt), 0.0)),
            (1, (1.2, -0.4, 0.85), (1.0, 0.0, 0.0, 0.0)),
        ):
            base = bridge.POSE_BASE + channel * 256
            meta[base] = 0x80
            struct.pack_into("<Q3f4f4xQ", meta, base + 64, 1, *position, *quaternion, 123)

        for offset, expected_z in ((0.0, 0.85), (-0.5, 0.35), (-1.0, -0.15)):
            with self.subTest(offset=offset):
                feedback = struct.unpack(
                    bridge.RX_FORMAT, bridge.receive_frame(meta, offset))
                self.assertEqual(feedback[0], 0x5A)
                self.assertEqual(feedback[-1], 0xA5)
                self.assertAlmostEqual(feedback[8], -2.0 * tilt, places=5)
                self.assertAlmostEqual(feedback[11], expected_z, places=5)


if __name__ == "__main__":
    unittest.main()
