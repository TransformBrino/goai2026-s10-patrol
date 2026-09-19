from dataclasses import dataclass

import mujoco
from sensor_msgs.msg import Imu


@dataclass(frozen=True)
class ImuConfig:
    frame_id: str
    orientation_sensor: str
    accelerometer_sensor: str
    gyro_sensor: str


class MujocoImu:
    def __init__(self, model: mujoco.MjModel, config: ImuConfig):
        self.model = model
        self.config = config
        self.orientation_address = self._sensor_address(
            config.orientation_sensor, 4
        )
        self.accelerometer_address = self._sensor_address(
            config.accelerometer_sensor, 3
        )
        self.gyro_address = self._sensor_address(config.gyro_sensor, 3)

    def _sensor_address(self, name: str, expected_dimension: int) -> int:
        sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, name
        )
        if sensor_id < 0:
            raise ValueError(f"Cannot find IMU sensor '{name}'")
        dimension = int(self.model.sensor_dim[sensor_id])
        if dimension != expected_dimension:
            raise ValueError(
                f"IMU sensor '{name}' has dimension {dimension}; "
                f"expected {expected_dimension}"
            )
        return int(self.model.sensor_adr[sensor_id])

    def create_message(self, data: mujoco.MjData, stamp) -> Imu:
        orientation = data.sensordata[
            self.orientation_address:self.orientation_address + 4
        ]
        acceleration = data.sensordata[
            self.accelerometer_address:self.accelerometer_address + 3
        ]
        angular_velocity = data.sensordata[
            self.gyro_address:self.gyro_address + 3
        ]

        message = Imu()
        message.header.stamp = stamp
        message.header.frame_id = self.config.frame_id
        message.orientation.w = float(orientation[0])
        message.orientation.x = float(orientation[1])
        message.orientation.y = float(orientation[2])
        message.orientation.z = float(orientation[3])
        message.angular_velocity.x = float(angular_velocity[0])
        message.angular_velocity.y = float(angular_velocity[1])
        message.angular_velocity.z = float(angular_velocity[2])
        message.linear_acceleration.x = float(acceleration[0])
        message.linear_acceleration.y = float(acceleration[1])
        message.linear_acceleration.z = float(acceleration[2])
        return message
