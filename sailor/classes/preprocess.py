import torch
import torch.nn.functional as F
from einops import rearrange

from sailor.diffusion.data4robotics import transforms


class Preprocessor:
    DATA_KEYS = ["agentview_image", "state", "robot0_eye_in_right_hand_image", "robot0_eye_in_left_hand_image", "action"]

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.gpu_transform = transforms.get_gpu_transform_by_name("gpu_medium")
        self.inference_transform = transforms.get_transform_by_name("preproc")

        # Filter DATA_KEYS based on num_cams
        if self.config.dp["num_cams"] == 1:
            self.DATA_KEYS = [key for key in self.DATA_KEYS if "eye_in" not in key]
        elif self.config.dp["num_cams"] == 2:
            self.DATA_KEYS = [key for key in self.DATA_KEYS if key != "robot0_eye_in_left_hand_image"]

        if config.state_only:
            # Remove all in DATA_KEYS that contains "image"
            self.DATA_KEYS = [key for key in self.DATA_KEYS if "image" not in key]

    @staticmethod
    def convert_hwc_to_chw_224(img):
        """
        Input can be B x H x W x C or B x T x H x W x C
        Output will be B x C x H x W or B x T x C x H x W
        Resize to 224 if H, W are not 224
        """
        assert len(img.shape) in [4, 5], f"Invalid shape: {img.shape}"
        img = rearrange(img, "... h w c -> ... c h w")

        if img.shape[-1] != 224:
            # Resize each image to 224
            B = img.shape[0]

            # Convert to (B*T, C, H, W) if data has time dimension
            timestep = len(img.shape) == 5
            if timestep:
                img = rearrange(img, "b t c h w -> (b t) c h w")

            img = F.interpolate(
                img, size=(224, 224), mode="bilinear", align_corners=False
            )

            if timestep:
                # Convert back to (B, T, C, H, W)
                img = rearrange(img, "(b t) c h w -> b t c h w", b=B)

        return img

    def preprocess_batch(self, batch, training=True):
        """
        Input shape: BS x some_dim(Optional) x ...
        """
        batch = batch.copy()

        # Normalize images to [0, 1]
        for key, value in batch.items():
            if "image" in key:
                batch[key] = torch.Tensor(value) / 255.0

        batch["cont"] = 1 - torch.Tensor(batch["is_terminal"]).unsqueeze(-1)
        batch = {k: torch.Tensor(v).to(self.device) for k, v in batch.items()}

        # Make images from (H, W, C) to (C, H, W) and resize to 224
        if not self.config.state_only:
            # Process all image keys dynamically
            image_keys = [k for k in batch.keys() if "image" in k]
            for img_key in image_keys:
                batch[img_key] = self.convert_hwc_to_chw_224(batch[img_key])

            # Apply GPU Transform to images if training
            if training:
                for img_key in image_keys:
                    batch[img_key] = self.gpu_transform(batch[img_key])
            else:
                for img_key in image_keys:
                    batch[img_key] = self.inference_transform(batch[img_key])

        return batch

    def dreamer_preprocess_batch(self, batch):
        """
        Input: BS x BL x ...
        """
        if "discount" in batch:
            batch["discount"] *= self.config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            batch["discount"] = torch.Tensor(batch["discount"]).unsqueeze(-1)

        return self.preprocess_batch(batch, training=False)

    def encoder_preprocess_batch(self, batch, training=True):
        """
        Input is BS x BL x ... x stack_dim
        Applies preprocessing
        Assumes BL = 2 and stack_dim = 2 for DATA_KEYS
        Returns in format (s, a, s_next, s_priv)
        """
        # Modify from (BS, BL, ..., stack_dim) to (BS*BL, stack_dim, ...) for DATA_KEYS
        # We assume the l = 1 here so do not reshape back to (BS, BL, ...)
        BS = batch["state"].shape[0]
        for key in self.DATA_KEYS:
            batch[key] = rearrange(batch[key], "b l ... s -> (b l) s ...")

        batch = self.preprocess_batch(batch, training=training)

        for key in self.DATA_KEYS:
            batch[key] = rearrange(batch[key], "(b l) ... -> b l ...", b=BS)

        """
        The sampled data is of form
        For BL = 0:
            S_{t-1}, s_t
                     a_t, a_{t+1},...
                 
        For BL = 1:
            s_t, s_{t+1}
                 a_{t+1}, a_{t+2},...
        
        Extracting- (s_t, a_t, s_{t+1}) for training the encoder
        """

        # Assert if BL=2 and obs_stack is 2
        # for key in self.DATA_KEYS:
        #     assert batch[key].shape[1] == 2, f"Shape: {key} - {batch[key].shape}"
        #     assert batch[key].shape[2] == 2, f"Shape: {key} - {batch[key].shape}"

        # Build camera dicts dynamically based on available images
        s = {}
        s_next = {}
        cam_idx = 0
        image_keys = [k for k in batch.keys() if "image" in k]
        for img_key in image_keys:
            s[f"cam{cam_idx}"] = batch[img_key][:, 1, 0, ...]
            s_next[f"cam{cam_idx}"] = batch[img_key][:, 1, 1, ...]
            cam_idx += 1

        a = batch["action"][:, 0, 0, ...]

        s_priv = batch["privileged_state"][:, 0, ...]
        return s, a.to(torch.float32), s_next, s_priv.to(torch.float32)

    def d4r_preprocess_batch(self, batch, training=True):
        """
        Input is BS x BL x ... x stack_dim
        Applies preprocessing
        Returns with shape BS*BL x stack_dim x ... and format (imgs, obs), actions, mask
        """
        # Modify from (BS, BL, ..., stack_dim) to (BS*BL, stack_dim, ...) for DATA_KEYS
        # We assume the l = 1 here so do not reshape back to (BS, BL, ...)
        for key in self.DATA_KEYS:
            batch[key] = rearrange(batch[key], "b l ... s -> (b l) s ...")

        batch = self.preprocess_batch(batch, training=training)

        # Build images dict dynamically based on available cameras
        if not self.config.state_only:
            imgs = {}
            cam_idx = 0
            image_keys = [k for k in batch.keys() if "image" in k]
            for img_key in image_keys:
                imgs[f"cam{cam_idx}"] = batch[img_key]
                cam_idx += 1
        else:
            imgs = None

        obs = batch["state"][:, -1, :].to(torch.float32)  # Take the last state
        actions = batch["action"].to(torch.float32)
        mask = torch.ones_like(actions).to(torch.float32)  # No mask for now
        return (imgs, obs), actions, mask
