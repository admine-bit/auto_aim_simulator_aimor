#!/usr/bin/env python3
"""Read-only Talos v2 measurements. Uses only the Python standard library.

This diagnostic does NOT consume triple buffers or interfere with the vision
consumer. Matching sequence/timestamp checks reject mixed-frame snapshots.
"""

import argparse
import json
import math
import mmap
import struct
import sys
import time

META_SIZE = 3712
META_PATH = "/tmp/talos_ipc_meta"
CAMERA_INFO_OFFSET = 1728


def rotation(wxyz):
    if not all(math.isfinite(v) for v in wxyz):
        raise ValueError("non-finite quaternion")
    norm = math.sqrt(sum(v * v for v in wxyz))
    if norm < 1e-6 or abs(norm - 1.0) > 1e-3:
        raise ValueError("invalid quaternion norm")
    w, x, y, z = (v / norm for v in wxyz)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def decode_snapshot(data):
    """Return a complete image/pose/calibration bundle, or None if incomplete."""
    if len(data) != META_SIZE:
        raise ValueError("expected 3712-byte Talos v2 metadata")
    magic, version = struct.unpack_from("<II", data)
    if (magic, version) != (0x54414C05, 2):
        raise ValueError("unsupported Talos header")
    info = struct.unpack_from("<Q9dII", data, CAMERA_INFO_OFFSET)
    info_stamp, fx, fy, cx, cy = info[:5]
    distortion, width, height = list(info[5:10]), info[10], info[11]
    if width != 1440 or height != 1080 or fx <= 0 or fy <= 0:
        return None
    if not all(math.isfinite(v) for v in info[1:10]):
        raise ValueError("non-finite camera intrinsics")
    images = [struct.unpack_from("<QQIIBB", data, 128 + i * 32) for i in range(3)]
    pose_slots = [
        [struct.unpack_from("<Q3f4f4xQ", data, 256 + channel * 256 + 64 + i * 64)
         for i in range(3)]
        for channel in range(4)
    ]
    for image in sorted(images, reverse=True):
        seq, stamp, image_w, image_h, buffer_id, image_format = image
        if (seq <= 0 or stamp != info_stamp or buffer_id >= 3 or image_format != 0
                or (image_w, image_h) != (width, height)):
            continue
        poses = [next((p for p in slots if p[0] == seq and p[8] == stamp), None)
                 for slots in pose_slots]
        if any(p is None for p in poses):
            continue
        if not all(math.isfinite(v) for p in poses for v in p[1:8]):
            raise ValueError("non-finite pose")
        gimbal, odom, muzzle, camera = poses
        rg = rotation(gimbal[4:8])
        rc = rotation(camera[4:8])
        forward = [rg[i][0] for i in range(3)]
        yaw = math.atan2(forward[1], forward[0])
        pitch = math.atan2(-forward[2], math.hypot(forward[0], forward[1]))
        return {
            "frame_seq": seq,
            "timestamp_ns": stamp,
            "age_ms": (time.time_ns() - stamp) / 1e6,
            "gimbal_quaternion_wxyz": list(gimbal[4:8]),
            "gimbal_origin_world_m": list(odom[1:4]),
            "gimbal_yaw_deg": math.degrees(yaw),
            "gimbal_pitch_ros_deg": math.degrees(pitch),
            "gimbal_elevation_deg": -math.degrees(pitch),
            "muzzle_offset_gimbal_m": list(muzzle[1:4]),
            "camera_quaternion_wxyz": list(camera[4:8]),
            "R_camera2gimbal": [v for row in rc for v in row],
            "t_camera2gimbal": list(camera[1:4]),
            "camera_matrix": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
            "distort_coeffs": distortion,
            "image_size": [width, height],
        }
    return None


def read_measurement(meta, timeout, after_seq=-1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        first = meta[:]
        second = meta[:]
        # Ignore the independently changing heartbeat, commands and ground truth.
        # Double-copy checks are best-effort diagnostics, not a replacement for
        # the atomic consume API in a real-time vision receiver.
        if (first[64:1536] == second[64:1536]
                and first[1728:1856] == second[1728:1856]):
            heartbeat = struct.unpack_from("<Q", second, 16)[0]
            if time.time_ns() - heartbeat > 3_000_000_000:
                raise ValueError("simulator heartbeat expired; restart the simulator")
            result = decode_snapshot(second)
            if result is not None and result["frame_seq"] > after_seq:
                return result
        time.sleep(0.002)
    raise ValueError("no complete fresh bundle; run simulator AND its vision consumer")


def yaml_text(result):
    comments = [
        "# Measured Talos parameters; does not move the camera or change the PID.",
        f"# frame_seq={result['frame_seq']}; timestamp_ns={result['timestamp_ns']}",
        "# OpenCV camera -> barrel-aligned gimbal: p_g = R * p_cv + t (metres).",
        "# R is row-major; quaternions in IPC use [w,x,y,z].",
        "# First-person only for static extrinsics; other views change the camera pose.",
        "# Current IMCA simulator Solver overrides YAML t to ZERO and drops camera pose!",
        "# Copying this YAML alone will NOT take effect until the vision receiver uses it.",
        "# gimbal origin world (metres): " + json.dumps(result["gimbal_origin_world_m"]),
        f"# actual yaw={result['gimbal_yaw_deg']:.6f} deg; "
        f"pitch(ROS, up negative)={result['gimbal_pitch_ros_deg']:.6f} deg",
    ]
    keys = ["camera_matrix", "distort_coeffs", "R_camera2gimbal", "t_camera2gimbal"]
    return "\n".join(comments + [key + ": " + json.dumps(result[key]) for key in keys])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", default=META_PATH)
    parser.add_argument("--json", action="store_true", help="JSON lines including actual gimbal pose")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.samples < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("samples and timeout must be positive")
    if args.samples > 1 and not args.json:
        parser.error("multiple samples require --json")
    try:
        with open(args.meta, "rb") as source:
            if source.seek(0, 2) != META_SIZE:
                raise ValueError("metadata size does not match Talos v2")
            with mmap.mmap(source.fileno(), META_SIZE, access=mmap.ACCESS_READ) as meta:
                seq = -1
                for _ in range(args.samples):
                    result = read_measurement(meta, args.timeout, seq)
                    seq = result["frame_seq"]
                    print(json.dumps(result) if args.json else yaml_text(result), flush=True)
        return 0
    except (OSError, ValueError) as error:
        print(f"Talos measurement error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
