"""Protocol regression checks for the simulator-side legacy serial bridge."""

import math
import struct
import unittest

import bridge


class BridgeFeedbackTest(unittest.TestCase):
    def test_yaw_velocity_feedback_uses_pose_timestamps_and_wraps(self):
        meta = bytearray(bridge.META_SIZE)
        feedback = bridge.GimbalFeedback()
        base = bridge.POSE_BASE
        meta[base] = 0x80
        for seq, timestamp_ns, yaw_degrees in (
            (1, 1_000_000_000, 179.0),
            (2, 1_020_000_000, -179.0),
        ):
            half_yaw = math.radians(yaw_degrees) / 2.0
            struct.pack_into("<Q3f4f4xQ", meta, base + 64, seq, 0, 0, 0,
                             math.cos(half_yaw), 0, 0, math.sin(half_yaw), timestamp_ns)
            frame = struct.unpack(bridge.RX_FORMAT, bridge.receive_frame(meta, 0.0, feedback))
        self.assertAlmostEqual(frame[7], math.radians(2.0) / 0.02, places=4)
        repeated = struct.unpack(bridge.RX_FORMAT, bridge.receive_frame(meta, 0.0, feedback))
        self.assertAlmostEqual(repeated[7], frame[7], places=5)

    def test_feedback_forwards_vision_mode(self):
        meta = bytearray(bridge.META_SIZE)
        for mode in range(6):
            with self.subTest(mode=mode):
                meta[bridge.VISION_MODE_OFFSET] = mode
                feedback = struct.unpack(
                    bridge.RX_FORMAT, bridge.receive_frame(meta, 0.0))
                self.assertEqual(feedback[1], mode)

        meta[bridge.VISION_MODE_OFFSET] = 255
        feedback = struct.unpack(bridge.RX_FORMAT, bridge.receive_frame(meta, 0.0))
        self.assertEqual(feedback[1], 0)

    def test_idle_mode_blocks_stale_vision_command(self):
        frame = struct.pack(bridge.TX_FORMAT, 0x5A, 2, 0.2, 0, 0, -0.1, 0, 0, 0xA5)
        for vision_mode, expected_distance, expected_fire in ((0, -1.0, 0), (2, 1.0, 1)):
            with self.subTest(vision_mode=vision_mode):
                meta = bytearray(bridge.META_SIZE)
                meta[bridge.VISION_MODE_OFFSET] = vision_mode
                bridge.publish_command(meta, frame)
                command = struct.unpack_from("<QfffB11x", meta, bridge.GIMBAL_CMD_BASE + 64)
                self.assertEqual(command[3], expected_distance)
                self.assertEqual(command[4], expected_fire)

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
