"""Constants for humanoid_bench integration with SAILOR.

Camera keys are from the new cameras added to MuJoCo XML files:
- agentview: overhead/third-person scene camera
- robot0_eye_in_hand: centered eye camera (between left and right eyes)
"""

IMAGE_OBS_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]

# Minimal state shape metadata - empty by default. If needed, you can extend
# this dict to include specific low-dimensional observation keys and shapes.
STATE_SHAPE_META = {}
