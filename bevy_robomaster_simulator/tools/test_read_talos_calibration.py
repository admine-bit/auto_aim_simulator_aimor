"""Pure-memory regression tests; no simulator shared memory is created/consumed."""

import math
import struct
import time
import unittest

from read_talos_calibration import META_SIZE, decode_snapshot, read_measurement, rotation


def bundle():
    data = bytearray(META_SIZE)
    struct.pack_into("<II", data, 0, 0x54414C05, 2)
    struct.pack_into("<QQIIBB", data, 128, 17, 1234, 1440, 1080, 0, 0)
    yaw, elevation = -2.8, 0.2
    # Rz(yaw) Ry(-elevation); negative yaw must not become yaw+180.
    q = (math.cos(yaw / 2) * math.cos(elevation / 2),
         math.sin(yaw / 2) * math.sin(elevation / 2),
         -math.cos(yaw / 2) * math.sin(elevation / 2),
         math.sin(yaw / 2) * math.cos(elevation / 2))
    optical_q = (0.5, -0.5, 0.5, -0.5)
    for channel in range(4):
        position = (0.1, 0.02, 0.15) if channel == 3 else (1.0, 2.0, 0.3)
        orientation = q if channel == 0 else optical_q if channel == 3 else (1, 0, 0, 0)
        struct.pack_into("<Q3f4f4xQ", data, 320 + channel * 256,
                         17, *position, *orientation, 1234)
    struct.pack_into("<Q9dII", data, 1728,
                     1234, 1303.0, 1303.0, 719.5, 539.5, *([0.0] * 5), 1440, 1080)
    return data


class CalibrationTests(unittest.TestCase):
    def test_reader_checks_header_heartbeat_without_consuming_buffers(self):
        data = bundle()
        struct.pack_into("<Q", data, 16, time.time_ns())
        struct.pack_into("<II", data, 24, 1440, 1080)
        before = bytes(data)
        result = read_measurement(data, 0.1)
        self.assertEqual(result["frame_seq"], 17)
        self.assertEqual(bytes(data), before)
        struct.pack_into("<Q", data, 16, time.time_ns() - 4_000_000_000)
        with self.assertRaisesRegex(ValueError, "heartbeat expired"):
            read_measurement(data, 0.1)

    def test_complete_bundle_preserves_nonzero_offset_and_physical_angles(self):
        data = bundle()
        before = bytes(data)
        result = decode_snapshot(data)
        self.assertEqual(result["frame_seq"], 17)
        self.assertAlmostEqual(result["gimbal_yaw_deg"], math.degrees(-2.8), places=5)
        self.assertAlmostEqual(result["gimbal_elevation_deg"], math.degrees(0.2), places=5)
        self.assertAlmostEqual(result["t_camera2gimbal"][2], 0.15, places=6)
        self.assertEqual(result["camera_matrix"][2], 719.5)
        self.assertEqual(result["R_camera2gimbal"], [0, 0, 1, -1, 0, 0, 0, -1, 0])
        self.assertEqual(bytes(data), before)

    def test_mixed_pose_frame_is_rejected(self):
        data = bundle()
        struct.pack_into("<Q", data, 320 + 3 * 256, 18)
        self.assertIsNone(decode_snapshot(data))

    def test_mixed_camera_info_timestamp_is_rejected(self):
        data = bundle()
        struct.pack_into("<Q", data, 1728, 1235)
        self.assertIsNone(decode_snapshot(data))

    def test_invalid_header_and_quaternion_are_rejected(self):
        data = bundle()
        struct.pack_into("<I", data, 4, 99)
        with self.assertRaises(ValueError):
            decode_snapshot(data)
        with self.assertRaises(ValueError):
            rotation([0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
