from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass
from pathlib import Path
import time

import mujoco
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField


DEFAULT_SCAN_PATTERN_PATH = (
    Path(__file__).resolve().parent / "scan_mode" / "mid360.csv"
)
SCAN_PATTERN_HEADER = "Time/s,Azimuth/deg,Zenith/deg"

LIVOX_POINT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("intensity", "<f4"),
        ("tag", "u1"),
        ("line", "u1"),
        ("timestamp", "<f8"),
    ],
    align=False,
)


@dataclass(frozen=True)
class LidarConfig:
    site_name: str = "front_lidar_site"
    frame_id: str = "front_lidar_link"
    base_body_name: str = "base"
    scan_pattern_path: Path = DEFAULT_SCAN_PATTERN_PATH
    samples_per_scan: int = 24000
    downsample: int = 2
    ray_worker_count: int = 12
    horizontal_min_deg: float = -135.0
    horizontal_max_deg: float = 135.0
    min_range: float = 0.1
    max_range: float = 40.0
    noise_stddev: float = 0.005
    random_seed: int | None = None


class MujocoLidar:
    def __init__(self, model: mujoco.MjModel, config: LidarConfig = LidarConfig()):
        self.model = model
        self.config = config
        self._validate_config(config)
        self.site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, config.site_name
        )
        if self.site_id < 0:
            raise ValueError(f"Cannot find lidar site '{config.site_name}'")

        self.output_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, config.frame_id
        )
        if self.output_body_id < 0:
            raise ValueError(f"Cannot find lidar output frame body '{config.frame_id}'")
        self.base_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, config.base_body_name
        )
        if self.base_body_id < 0:
            raise ValueError(
                f"Cannot find lidar ray-exclusion body '{config.base_body_name}'"
            )

        self.scan_pattern_directions = self._load_scan_pattern(
            config.scan_pattern_path
        )
        self.pattern_size = len(self.scan_pattern_directions)
        self.sample_offsets = np.arange(
            0, config.samples_per_scan, config.downsample, dtype=np.int64
        )
        self.max_ray_count = len(self.sample_offsets)
        self.current_scan_index = 0
        self.geom_ids = np.empty(self.max_ray_count, dtype=np.int32)
        self.distances = np.empty(self.max_ray_count, dtype=np.float64)
        self.geom_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.random = np.random.default_rng(config.random_seed)
        self.ray_executor = ThreadPoolExecutor(
            max_workers=config.ray_worker_count,
            thread_name_prefix="front_lidar_ray",
        )
        self.ray_data = [
            mujoco.MjData(model) for _ in range(config.ray_worker_count)
        ]

    def close(self) -> None:
        self.ray_executor.shutdown(wait=True)

    def warm_up(self, data: mujoco.MjData) -> None:
        """Initialize ray workers without consuming scan-pattern or noise state."""
        scan_index = self.current_scan_index
        random_state = copy.deepcopy(self.random.bit_generator.state)
        self.scan(data)
        self.current_scan_index = scan_index
        self.random.bit_generator.state = random_state

    @staticmethod
    def _validate_config(config: LidarConfig) -> None:
        if config.samples_per_scan <= 0:
            raise ValueError("Lidar samples_per_scan must be positive")
        if config.downsample <= 0:
            raise ValueError("Lidar downsample must be positive")
        if config.ray_worker_count <= 0:
            raise ValueError("Lidar ray_worker_count must be positive")
        if not (
            -180.0 <= config.horizontal_min_deg
            < config.horizontal_max_deg
            <= 180.0
        ):
            raise ValueError(
                "Lidar horizontal angles must satisfy "
                "-180 <= min < max <= 180"
            )
        if not 0.0 <= config.min_range < config.max_range:
            raise ValueError("Lidar range must satisfy 0 <= min_range < max_range")
        if config.noise_stddev < 0.0:
            raise ValueError("Lidar noise_stddev must be non-negative")

    @staticmethod
    def _load_scan_pattern(scan_pattern_path: Path) -> np.ndarray:
        path = Path(scan_pattern_path)
        if not path.is_file():
            raise FileNotFoundError(f"Cannot find lidar scan pattern: {path}")

        with path.open("r", encoding="ascii") as pattern_file:
            header = pattern_file.readline().rstrip("\r\n")
        if header != SCAN_PATTERN_HEADER:
            raise ValueError(
                f"Unexpected lidar scan pattern header in {path}: {header!r}"
            )

        try:
            pattern = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
        except ValueError as error:
            raise ValueError(f"Invalid lidar scan pattern data in {path}") from error
        if pattern.ndim != 2 or pattern.shape[1] != 3 or pattern.shape[0] == 0:
            raise ValueError(
                f"Expected a non-empty three-column lidar scan pattern in {path}"
            )
        if not np.isfinite(pattern).all():
            raise ValueError(f"Lidar scan pattern contains non-finite values: {path}")

        azimuth = np.deg2rad(pattern[:, 1])
        pitch = np.deg2rad(pattern[:, 2]) - np.pi / 2.0
        cos_pitch = np.cos(pitch)
        directions = np.column_stack(
            (
                cos_pitch * np.cos(azimuth),
                cos_pitch * np.sin(azimuth),
                -np.sin(pitch),
            )
        )
        return np.ascontiguousarray(directions, dtype=np.float64)

    def _next_local_directions(self) -> np.ndarray:
        indices = (self.current_scan_index + self.sample_offsets) % self.pattern_size
        directions = self.scan_pattern_directions[indices]
        self.current_scan_index = (
            self.current_scan_index + self.config.samples_per_scan
        ) % self.pattern_size
        azimuth_deg = np.rad2deg(np.arctan2(directions[:, 1], directions[:, 0]))
        directions = directions[
            (azimuth_deg >= self.config.horizontal_min_deg)
            & (azimuth_deg <= self.config.horizontal_max_deg)
        ]
        return np.ascontiguousarray(directions, dtype=np.float64)

    def scan(self, data: mujoco.MjData) -> np.ndarray:
        site_position = data.site_xpos[self.site_id]
        site_rotation = data.site_xmat[self.site_id].reshape(3, 3)
        local_directions = self._next_local_directions()
        ray_count = len(local_directions)
        world_directions = np.ascontiguousarray(
            local_directions @ site_rotation.T, dtype=np.float64
        )

        chunk_size = (ray_count + self.config.ray_worker_count - 1) // (
            self.config.ray_worker_count
        )
        chunks = []
        for start in range(0, ray_count, chunk_size):
            end = min(start + chunk_size, ray_count)
            chunks.append((start, end))

        for worker_index in range(len(chunks)):
            mujoco.mj_copyData(self.ray_data[worker_index], self.model, data)

        futures = []
        for worker_index, (start, end) in enumerate(chunks):
            futures.append(
                self.ray_executor.submit(
                    self._cast_rays,
                    self.ray_data[worker_index],
                    site_position,
                    world_directions[start:end],
                    start,
                    end,
                )
            )
        for future in futures:
            future.result()

        measured_distances = self.distances[:ray_count].copy()
        hit = measured_distances >= 0.0
        if self.config.noise_stddev > 0.0 and np.any(hit):
            measured_distances[hit] += self.random.normal(
                0.0, self.config.noise_stddev, np.count_nonzero(hit)
            )
        valid = hit & (measured_distances > self.config.min_range) & (
            measured_distances < self.config.max_range
        )

        points = np.full((ray_count, 3), np.nan, dtype=np.float32)
        if np.any(valid):
            world_points = (
                site_position
                + world_directions[valid]
                * measured_distances[valid, np.newaxis]
            )
            output_position = data.xpos[self.output_body_id]
            output_rotation = data.xmat[self.output_body_id].reshape(3, 3)
            points[valid] = ((world_points - output_position) @ output_rotation).astype(
                np.float32
            )

        # 留一份"原始测量"给定位用：雷达本体系的方向 + 测距 + 有效掩膜。
        # 定位要拿这些去和已知地图配准，不能只有 PointCloud2 —— 从点云反推
        # 方向会丢掉打空的那些射线，而"这个方向上 40m 内没东西"同样是信息。
        self.last_local_dirs = local_directions
        self.last_ranges = measured_distances
        self.last_valid = valid
        # 高程图累积要的是命中点本身（雷达本体系，已剔除打空的）。
        # 从 PointCloud2 里再解一遍是白费功夫，这里直接留一份。
        self.last_points = points[valid] if np.any(valid) else points[:0]

        return points

    def _cast_rays(
        self,
        data: mujoco.MjData,
        site_position: np.ndarray,
        directions: np.ndarray,
        start: int,
        end: int,
    ) -> None:
        ray_count = end - start
        mujoco.mj_multiRay(
            self.model,
            data,
            site_position,
            directions.ravel(),
            self.geom_groups,
            True,
            self.base_body_id,
            self.geom_ids[start:end],
            self.distances[start:end],
            None,
            ray_count,
            self.config.max_range,
        )

    def create_point_cloud(self, points: np.ndarray, stamp) -> PointCloud2:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Expected lidar points with shape (N, 3), got {points.shape}"
            )

        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = self.config.frame_id
        cloud.height = 1
        cloud.width = len(points)
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="intensity", offset=12, datatype=PointField.FLOAT32, count=1
            ),
            PointField(name="tag", offset=16, datatype=PointField.UINT8, count=1),
            PointField(name="line", offset=17, datatype=PointField.UINT8, count=1),
            PointField(
                name="timestamp", offset=18, datatype=PointField.FLOAT64, count=1
            ),
        ]
        cloud.is_bigendian = False
        cloud.point_step = LIVOX_POINT_DTYPE.itemsize
        cloud.row_step = cloud.point_step * cloud.width
        encoded_points = np.zeros(len(points), dtype=LIVOX_POINT_DTYPE)
        encoded_points["x"], encoded_points["y"], encoded_points["z"] = (
            np.asarray(points, dtype=np.float32).T
        )
        encoded_points["intensity"] = 1.0
        encoded_points["timestamp"] = 0.0
        cloud.data = encoded_points.tobytes()
        cloud.is_dense = False
        return cloud


@dataclass(frozen=True)
class LidarScanResult:
    cloud: PointCloud2
    elapsed_sec: float
    deadline_sec: float

    @property
    def overrun_sec(self) -> float:
        return max(0.0, self.elapsed_sec - self.deadline_sec)


class AsyncMujocoLidar:
    """Runs one lidar scan at a time against an immutable data snapshot."""

    def __init__(
        self,
        model: mujoco.MjModel,
        lidar: MujocoLidar,
        scan_deadline_sec: float,
    ):
        if scan_deadline_sec <= 0.0:
            raise ValueError("Lidar scan deadline must be positive")
        self.model = model
        self.lidar = lidar
        self.scan_deadline_sec = scan_deadline_sec
        self.snapshot = mujoco.MjData(model)
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="front_lidar_scan",
        )
        self.future = None
        self.started_at = None
        self.closed = False

    @property
    def busy(self) -> bool:
        return self.future is not None

    def start_scan(self, data: mujoco.MjData, stamp) -> None:
        if self.closed:
            raise RuntimeError("Cannot start a scan after lidar shutdown")
        if self.future is not None:
            raise RuntimeError("Previous lidar scan has not been collected")

        self.started_at = time.perf_counter()
        mujoco.mj_copyData(self.snapshot, self.model, data)
        self.future = self.executor.submit(self._scan_snapshot, stamp)

    def active_elapsed_sec(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.perf_counter() - self.started_at

    def take_result(self) -> LidarScanResult | None:
        if self.future is None or not self.future.done():
            return None

        future = self.future
        self.future = None
        result = future.result()
        self.started_at = None
        return result

    def _scan_snapshot(self, stamp) -> LidarScanResult:
        started_at = time.perf_counter()
        points = self.lidar.scan(self.snapshot)
        cloud = self.lidar.create_point_cloud(points, stamp)
        return LidarScanResult(
            cloud=cloud,
            elapsed_sec=time.perf_counter() - started_at,
            deadline_sec=self.scan_deadline_sec,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.executor.shutdown(wait=True)
        self.lidar.close()
