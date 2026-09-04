import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/wrist_image_left": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "observation/mic1_log_mel": np.random.randn(80, 198).astype(np.float32),
        "observation/mic2_log_mel": np.random.randn(80, 198).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _parse_sound(data: dict) -> np.ndarray:
    if "observation/sound" in data:
        sound = np.asarray(data["observation/sound"], dtype=np.float32)
    elif "observation/mic1_log_mel" in data and "observation/mic2_log_mel" in data:
        sound = np.stack(
            [
                np.asarray(data["observation/mic1_log_mel"], dtype=np.float32),
                np.asarray(data["observation/mic2_log_mel"], dtype=np.float32),
            ],
            axis=0,
        )
    else:
        raise KeyError(
            "Expected observation/sound or both observation/mic1_log_mel and observation/mic2_log_mel."
        )

    if sound.ndim == 2:
        raise ValueError("Expected two microphone spectrograms, got a single 2D tensor.")
    if sound.ndim == 3 and sound.shape[-1] == 2 and sound.shape[0] != 2:
        sound = np.moveaxis(sound, -1, 0)
    if sound.shape[0] != 2:
        raise ValueError(f"Expected sound shape [2, mel, time], got {sound.shape}.")
    return sound.astype(np.float32, copy=False)


def _parse_pointcloud(data: dict) -> np.ndarray:
    if "observation/pointcloud" in data:
        pointcloud = np.asarray(data["observation/pointcloud"])
    elif (
        "observation/pointcloud_coord" in data
        and "observation/pointcloud_color" in data
    ):
        coord = np.asarray(data["observation/pointcloud_coord"])
        color = np.asarray(data["observation/pointcloud_color"])
        pointcloud = np.concatenate([coord, color], axis=-1)
    else:
        raise KeyError(
            "Expected either observation/pointcloud or both "
            "observation/pointcloud_coord and observation/pointcloud_color."
        )

    if pointcloud.shape[-1] < 6:
        raise ValueError(
            f"Expected pointcloud with coord+color in the last dimension, got shape {pointcloud.shape}."
        )

    return pointcloud


def _parse_camera_pointcloud(data: dict, camera_key: str) -> np.ndarray:
    stems = {
        "base_0_pointcloud": "pointcloud",
        "left_wrist_0_pointcloud": "left_wrist_pointcloud",
        "right_wrist_0_pointcloud": "right_wrist_pointcloud",
    }
    try:
        stem = stems[camera_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported point-cloud camera key: {camera_key!r}.") from exc

    combined_key = f"observation/{stem}"
    coord_key = f"observation/{stem}_coord"
    color_key = f"observation/{stem}_color"
    if combined_key in data:
        pointcloud = np.asarray(data[combined_key])
    elif coord_key in data and color_key in data:
        pointcloud = np.concatenate(
            [np.asarray(data[coord_key]), np.asarray(data[color_key])], axis=-1
        )
    else:
        raise KeyError(
            f"Expected {combined_key} or both {coord_key} and {color_key}."
        )
    if pointcloud.shape[-1] < 6:
        raise ValueError(f"Expected XYZRGB for {camera_key}, got {pointcloud.shape}.")
    return pointcloud.astype(np.float32, copy=False)


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    use_pointcloud: bool = False
    pointcloud_keys: tuple[str, ...] = ()
    use_sound: bool = False
    use_right_wrist_image: bool = False

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])

        pointcloud = None
        if self.use_pointcloud and self.pointcloud_keys:
            pointcloud = {
                key: _parse_camera_pointcloud(data, key) for key in self.pointcloud_keys
            }
        elif (
            "observation/pointcloud" in data
            or "observation/pointcloud_coord" in data
            or "observation/pointcloud_color" in data
        ):
            pointcloud = _parse_pointcloud(data)

        match self.model_type:
            case (
                _model.ModelType.PI0
                | _model.ModelType.PI05
                | _model.ModelType.PROXY
                | _model.ModelType.PROXY_SCORE
                | _model.ModelType.PROXY_SOUND
                | _model.ModelType.RESIDUAL
            ):
                # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
                # stores as float32 (C,H,W), gets skipped for policy inference.
                base_image = _parse_image(data["observation/exterior_image_1_left"])
                wrist_image = _parse_image(data["observation/wrist_image_left"])
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                if self.use_right_wrist_image:
                    right_wrist_image = _parse_image(
                        data["observation/wrist_image_right"]
                    )
                    images = (base_image, wrist_image, right_wrist_image)
                    image_masks = (np.True_, np.True_, np.True_)
                else:
                    images = (base_image, wrist_image, np.zeros_like(base_image))
                    image_masks = (np.True_, np.True_, np.False_)
                inputs = {
                    "state": state,
                    "image": dict(zip(names, images, strict=True)),
                    "image_mask": dict(zip(names, image_masks, strict=True)),
                }
                if self.model_type == _model.ModelType.PROXY_SOUND:
                    inputs["sound"] = _parse_sound(data)
                elif (
                    self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05)
                    and self.use_sound
                    and (
                        "observation/sound" in data
                        or "observation/mic1_log_mel" in data
                        or "observation/mic2_log_mel" in data
                    )
                ):
                    inputs["sound"] = _parse_sound(data)
                if (
                    self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05, _model.ModelType.PROXY)
                    and self.use_pointcloud
                    and pointcloud is not None
                ):
                    inputs["pointcloud"] = pointcloud
            case _model.ModelType.PI0_FAST:
                # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
                # stores as float32 (C,H,W), gets skipped for policy inference.
                base_image = _parse_image(data["observation/exterior_image_1_left"])
                wrist_image = _parse_image(data["observation/wrist_image_left"])
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
                inputs = {
                    "state": state,
                    "image": dict(zip(names, images, strict=True)),
                    "image_mask": dict(zip(names, image_masks, strict=True)),
                }
            case _model.ModelType.PROXY_POINTCLOUD | _model.ModelType.PROXY_DP3:
                if pointcloud is None:
                    raise ValueError(
                        "Proxy pointcloud model requires pointcloud coord+color input."
                    )
                inputs = {
                    "state": state,
                    "pointcloud": pointcloud,
                }
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            # print(f"prompt: {data['prompt']}")
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][:, :8])}
