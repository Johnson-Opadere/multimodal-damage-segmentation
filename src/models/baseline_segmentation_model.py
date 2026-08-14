"""
Dual-Branch PRE/POST ResNet50 Segmentation Model.

Overview:
---------
This model performs semantic segmentation for disaster damage assessment
using multimodal inputs (RGB + SAR) across temporal states (pre- and post-disaster).

Input Tensor:
-------------
Shape: (512, 512, 22)

Channel Layout:
---------------
RGB_pre   : 3 channels  → [0:3]
RGB_post  : 3 channels  → [3:6]
SAR_pre   : 8 channels  → [6:14]
SAR_post  : 8 channels  → [14:22]

Architecture:
-------------
- Dual-branch design:
    • PRE branch  → RGB_pre + SAR_pre  (11 channels)
    • POST branch → RGB_post + SAR_post (11 channels)

- Backbone:
    • ResNet50 (ImageNet-pretrained weights adapted for multi-channel input)

- Feature fusion:
    • Concatenation of PRE and POST features at multiple scales

- Decoder:
    • UNet-style upsampling with skip connections

- Output:
    • Pixel-wise softmax segmentation (4 classes)

Classes:
--------
0 → Background  
1 → Minor damage  
2 → Major damage  
3 → Destroyed  

Key Features:
-------------
- Temporal modeling via PRE/POST separation
- Multimodal fusion (RGB + SAR)
- Expanded first convolution to support >3 channels
- Feature-level fusion (not early pixel fusion)
- Optional freezing of early backbone layers

Run Example:
------------
from src.models.segmentation_model import build_segmentation_model

model = build_segmentation_model(
    input_shape=(512, 512, 22),
    num_classes=4,
    freeze_early=False
)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

NUM_CLASSES = 4


# ---------------------------------------------------------
# Decoder Block
# ---------------------------------------------------------

def decoder_block(x, skip, filters, name):
    """
    UNet-style decoder block.

    Args:
        x (tf.Tensor): Input feature map (from previous decoder stage)
        skip (tf.Tensor): Skip connection from encoder
        filters (int): Number of convolution filters
        name (str): Block name prefix

    Returns:
        tf.Tensor: Refined feature map

    Processing:
    -----------
    1. Upsample input feature map (bilinear interpolation)
    2. Concatenate with skip connection
    3. Apply two Conv2D + ReLU layers

    Notes:
        - Maintains spatial alignment with encoder features
        - Forms part of UNet decoder structure
    """
    x = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        name=f"{name}_upsample"
    )(x)

    x = layers.Concatenate(name=f"{name}_concat")([x, skip])

    x = layers.Conv2D(
        filters, 3, padding="same",
        activation="relu",
        name=f"{name}_conv1"
    )(x)

    x = layers.Conv2D(
        filters, 3, padding="same",
        activation="relu",
        name=f"{name}_conv2"
    )(x)

    return x


# ---------------------------------------------------------
# Expand First Conv Weights
# ---------------------------------------------------------

def expand_first_conv_weights(pretrained_weights, new_channels):
    """
    Expand pretrained ResNet first convolution weights to support multi-channel input.

    Args:
        pretrained_weights (List[np.ndarray]):
            Original ResNet weights for conv1 (7x7x3x64 kernel + bias)
        new_channels (int): Target number of input channels

    Returns:
        List[np.ndarray]: Expanded kernel weights and original bias

    Method:
    -------
    - Compute mean across RGB channels (axis=2)
    - Tile mean kernel to match new channel dimension
    - Preserve original bias

    Notes:
        - Enables transfer learning for non-RGB inputs (e.g., RGB+SAR)
        - Avoids random initialization for additional channels
    """
    kernel, bias = pretrained_weights  # (7,7,3,64)

    mean_kernel = tf.reduce_mean(kernel, axis=2, keepdims=True)
    new_kernel = tf.tile(mean_kernel, [1, 1, new_channels, 1])

    return [new_kernel.numpy(), bias]


# ---------------------------------------------------------
# Build ResNet Branch (No name arg!)
# ---------------------------------------------------------

def build_resnet_branch(input_tensor, input_channels, prefix):
    """
    Build a ResNet50 backbone adapted for multi-channel input.

    Args:
        input_tensor (tf.Tensor): Input tensor for the branch
        input_channels (int): Number of input channels (e.g., 11)
        prefix (str): Prefix for layer naming (pre/post)

    Returns:
        tf.keras.Model: ResNet backbone with adapted input

    Workflow:
    ---------
    1. Load pretrained ResNet50 (3-channel ImageNet weights)
    2. Build new ResNet50 with custom input tensor
    3. Rename layers to avoid naming conflicts
    4. Expand first convolution weights to match new_channels
    5. Copy pretrained weights for remaining layers

    Notes:
        - First conv is adapted using channel expansion
        - Remaining layers retain pretrained weights
        - Prefix ensures unique layer names across branches
    """

    # Load pretrained 3-channel model
    pretrained = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(512, 512, 3)
    )

    first_conv = pretrained.get_layer("conv1_conv")
    pretrained_weights = first_conv.get_weights()

    # Build new model WITHOUT name argument
    branch = ResNet50(
        include_top=False,
        weights=None,
        input_tensor=input_tensor
    )

    # Rename layers to avoid duplicates
    for layer in branch.layers:
        layer._name = prefix + "_" + layer.name

    # Expand first conv
    new_first_conv = branch.get_layer(prefix + "_conv1_conv")
    expanded = expand_first_conv_weights(
        pretrained_weights,
        input_channels
    )
    new_first_conv.set_weights(expanded)

    # Copy remaining weights
    for layer in pretrained.layers:
        if layer.name == "conv1_conv":
            continue

        try:
            branch_layer = branch.get_layer(prefix + "_" + layer.name)
            branch_layer.set_weights(layer.get_weights())
        except:
            pass

    return branch


# ---------------------------------------------------------
# Full Dual-Branch Model
# ---------------------------------------------------------

def build_segmentation_model(
        input_shape=(512, 512, 22),
        num_classes=4,
        freeze_early=False):
	"""
    Build the full dual-branch segmentation model.

    Args:
        input_shape (Tuple[int, int, int]):
            Input tensor shape (default: 512x512x22)
        num_classes (int):
            Number of output segmentation classes (default: 4)
        freeze_early (bool):
            If True, freezes early ResNet layers (conv1–conv3)

    Returns:
        tf.keras.Model: Compiled segmentation model

    Pipeline:
    ---------
    1. Split input tensor into:
        - RGB_pre, RGB_post
        - SAR_pre, SAR_post

    2. Construct two branches:
        - PRE branch  → RGB_pre + SAR_pre
        - POST branch → RGB_post + SAR_post

    3. Pass each branch through ResNet50 backbone

    4. Extract multi-scale features:
        - conv1 → conv5 feature maps

    5. Fuse PRE and POST features:
        - Concatenation at each scale

    6. Decode fused features:
        - UNet-style decoder blocks

    7. Produce final segmentation output:
        - 1x1 convolution + softmax

    Feature Fusion:
    ---------------
    - Fusion occurs at multiple levels:
        skip1, skip2, skip3, skip4, bottleneck

    Freezing Strategy:
    ------------------
    If freeze_early=True:
        - conv1, conv2, conv3 layers are frozen
        - conv4, conv5 remain trainable

    Output:
    -------
    Shape: (512, 512, num_classes)
    Activation: softmax

    Notes:
        - Model supports multimodal temporal learning
        - Designed for disaster damage segmentation
    """

    inputs = layers.Input(shape=input_shape, dtype="float32")

    # -----------------------------------------------------
    # Channel Splits
    # -----------------------------------------------------

    rgb_pre  = layers.Lambda(lambda x: x[:, :, :, 0:3],  name="rgb_pre")(inputs)
    rgb_post = layers.Lambda(lambda x: x[:, :, :, 3:6],  name="rgb_post")(inputs)

    sar_pre  = layers.Lambda(lambda x: x[:, :, :, 6:14], name="sar_pre")(inputs)
    sar_post = layers.Lambda(lambda x: x[:, :, :, 14:22], name="sar_post")(inputs)

    # PRE branch = RGB_pre + SAR_pre (11 channels)
    pre_input = layers.Concatenate(name="pre_concat")([rgb_pre, sar_pre])

    # POST branch = RGB_post + SAR_post (11 channels)
    post_input = layers.Concatenate(name="post_concat")([rgb_post, sar_post])

    # -----------------------------------------------------
    # Build Branches
    # -----------------------------------------------------

    pre_backbone = build_resnet_branch(
        pre_input,
        input_channels=11,
        prefix="pre"
    )

    post_backbone = build_resnet_branch(
        post_input,
        input_channels=11,
        prefix="post"
    )

    # Optional freezing
    if freeze_early:
        for layer in pre_backbone.layers:
            if any(stage in layer.name for stage in ["conv1", "conv2", "conv3"]):
                layer.trainable = False

        for layer in post_backbone.layers:
            if any(stage in layer.name for stage in ["conv1", "conv2", "conv3"]):
                layer.trainable = False

    # -----------------------------------------------------
    # Extract Features
    # -----------------------------------------------------

    def extract(backbone, prefix):
        skip1 = backbone.get_layer(prefix + "_conv1_relu").output
        skip2 = backbone.get_layer(prefix + "_conv2_block3_out").output
        skip3 = backbone.get_layer(prefix + "_conv3_block4_out").output
        skip4 = backbone.get_layer(prefix + "_conv4_block6_out").output
        x     = backbone.get_layer(prefix + "_conv5_block3_out").output
        return skip1, skip2, skip3, skip4, x

    pre_skip1, pre_skip2, pre_skip3, pre_skip4, pre_x = extract(pre_backbone, "pre")
    post_skip1, post_skip2, post_skip3, post_skip4, post_x = extract(post_backbone, "post")

    # -----------------------------------------------------
    # Feature Fusion
    # -----------------------------------------------------

    skip1 = layers.Concatenate(name="fuse_skip1")([pre_skip1, post_skip1])
    skip2 = layers.Concatenate(name="fuse_skip2")([pre_skip2, post_skip2])
    skip3 = layers.Concatenate(name="fuse_skip3")([pre_skip3, post_skip3])
    skip4 = layers.Concatenate(name="fuse_skip4")([pre_skip4, post_skip4])
    x     = layers.Concatenate(name="fuse_bottleneck")([pre_x, post_x])

    # -----------------------------------------------------
    # Decoder
    # -----------------------------------------------------

    x = decoder_block(x, skip4, 512, "decoder4")
    x = decoder_block(x, skip3, 256, "decoder3")
    x = decoder_block(x, skip2, 128, "decoder2")
    x = decoder_block(x, skip1, 64,  "decoder1")

    x = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        name="final_upsample"
    )(x)

    outputs = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation="softmax",
        name="segmentation_head"
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name="DualBranch_PRE_POST_ResNet50_UNet"
    )

    model.summary(line_length=140)

    return model
