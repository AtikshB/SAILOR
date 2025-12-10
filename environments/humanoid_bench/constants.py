"""Constants for humanoid_bench integration with SAILOR.

Camera keys are from humanoid_bench.wrappers.ObservationWrapper.get_camera_obs()
which renders two eye cameras: left_eye_camera and right_eye_camera.
"""

IMAGE_OBS_KEYS = ["image_left_eye", "image_right_eye"]

# Minimal state shape metadata - empty by default. If needed, you can extend
# this dict to include specific low-dimensional observation keys and shapes.
STATE_SHAPE_META = {}
