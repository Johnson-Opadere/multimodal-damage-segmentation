"""
Dual-Branch PRE/POST + DIFF ResNet50 Segmentation Model.

Overview:
---------
This model extends the baseline dual-branch architecture by explicitly modeling
temporal changes between pre- and post-disaster inputs using difference (DIFF) features.

Instead of relying on the network to implicitly learn change, this architecture
provides explicit temporal signals at both input and feature levels.

Input Tensor:
-------------
Shape: (512, 512, 22)

Channel Layout:
---------------
RGB_pre   : 3 channels  → [0:3]
RGB_post  : 3 channels  → [3:6]
SAR_pre   : 8 channels  → [6:14]
SAR_post  : 8 channels  → [14:22]

Computed Channels:
------------------
RGB_diff = RGB_post - RGB_pre   → 3 channels  
SAR_diff = SAR_post - SAR_pre   → 8 channels  

DIFF (concatenated):
    → 11 channels total

Branch Inputs:
--------------
PRE branch  = RGB_pre  + SAR_pre  + DIFF → 22 channels  
POST branch = RGB_post + SAR_post + DIFF → 22 channels  

Architecture:
-------------
- Dual ResNet50 encoders (PRE / POST)
- Expanded first convolution (supports 22-channel input)
- Feature-level fusion across PRE and POST branches
- Optional feature-level DIFF fusion
- UNet-style decoder

Output:
-------
- Pixel-wise segmentation map
- Shape: (512, 512, num_classes)
- Activation: softmax

Classes:
--------
0 → Background  
1 → Minor damage  
2 → Major damage  
3 → Destroyed  

Key Innovations:
----------------
- Explicit temporal difference modeling (input-level)
- Optional feature-level difference fusion
- Multimodal fusion (RGB + SAR)
- Dual-branch temporal encoding

Run Example:
------------
from src.models.segmentation_model_diff import build_segmentation_model

model = build_segmentation_model(
    input_shape=(512, 512, 22),
    num_classes=4,
    freeze_early=False,
    use_feature_diff=True
)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50


# ---------------------------------------------------------
# Decoder Block
# ---------------------------------------------------------

def decoder_block(x, skip, filters, name):
    """
    UNet-style decoder block.

    Args:
        x (tf.Tensor): Input feature map from previous decoder stage
        skip (tf.Tensor): Corresponding encoder skip connection
        filters (int): Number of convolution filters
        name (str): Block name prefix

    Returns:
        tf.Tensor: Upsampled and refined feature map

    Processing:
    -----------
    1. Bilinear upsampling (2× spatial resolution)
    2. Concatenate with encoder skip features
    3. Apply two Conv2D + ReLU layers

    Notes:
        - Preserves spatial detail via skip connections
        - Forms hierarchical reconstruction in decoder
    """
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"{name}_upsample")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv1")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv2")(x)
    return x


# ---------------------------------------------------------
# Expand First Conv Weights
# ---------------------------------------------------------

def expand_first_conv_weights(pretrained_weights, new_channels):
    """
    Expand pretrained ResNet first convolution weights for multi-channel input.

    Args:
        pretrained_weights (List[np.ndarray]):
            Original weights from conv1 layer (7x7x3x64 kernel + bias)
        new_channels (int): Desired number of input channels

    Returns:
        List[np.ndarray]: Expanded kernel weights and original bias

    Method:
    -------
    - Compute mean across RGB channels (axis=2)
    - Tile mean kernel to match new_channels
    - Preserve bias unchanged

    Notes:
        - Enables transfer learning for non-RGB inputs (e.g., RGB+SAR+DIFF)
        - Avoids random initialization instability
    """
    kernel, bias = pretrained_weights  # (7,7,3,64)

    mean_kernel = tf.reduce_mean(kernel, axis=2, keepdims=True)
    new_kernel = tf.tile(mean_kernel, [1, 1, new_channels, 1])

    return [new_kernel.numpy(), bias]


# ---------------------------------------------------------
# Build ResNet Branch
# ---------------------------------------------------------

def build_resnet_branch(input_tensor, input_channels, prefix):
    """
    Build a ResNet50 encoder adapted for multi-channel input.

    Args:
        input_tensor (tf.Tensor): Input tensor for this branch
        input_channels (int): Number of input channels (e.g., 22)
        prefix (str): Prefix for layer naming ("pre" or "post")

    Returns:
        tf.keras.Model: ResNet50 backbone with adapted input

    Workflow:
    ---------
    1. Load pretrained ResNet50 (ImageNet, 3-channel)
    2. Build new ResNet50 with custom input tensor
    3. Rename layers to avoid conflicts between branches
    4. Expand first convolution weights to match input_channels
    5. Copy pretrained weights for remaining layers

    Notes:
        - First layer is adapted using channel expansion
        - Remaining layers retain pretrained initialization
        - Prefix ensures unique layer naming
    """

    pretrained = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(512, 512, 3)
    )

    first_conv = pretrained.get_layer("conv1_conv")
    pretrained_weights = first_conv.get_weights()

    branch = ResNet50(
        include_top=False,
        weights=None,
        input_tensor=input_tensor
    )

    # Rename layers
    for layer in branch.layers:
        layer._name = prefix + "_" + layer.name

    # Expand first conv correctly
    new_first_conv = branch.get_layer(prefix + "_conv1_conv")
    expanded = expand_first_conv_weights(pretrained_weights, input_channels)
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
# Full Model
# ---------------------------------------------------------

def build_segmentation_model(
        input_shape=(512, 512, 22),
        num_classes=4,
        freeze_early=False,
        use_feature_diff=True):
    """
    Build the dual-branch PRE/POST + DIFF segmentation model.

    Args:
        input_shape (Tuple[int, int, int]):
            Input tensor shape (default: 512x512x22)
        num_classes (int):
            Number of output segmentation classes (default: 4)
        freeze_early (bool):
            If True, freezes early ResNet layers (conv1–conv3)
        use_feature_diff (bool):
            If True, enables feature-level DIFF fusion

    Returns:
        tf.keras.Model: Segmentation model

    Pipeline:
    ---------
    1. Split input tensor into:
        - RGB_pre, RGB_post
        - SAR_pre, SAR_post

    2. Compute temporal differences:
        - RGB_diff = RGB_post - RGB_pre
        - SAR_diff = SAR_post - SAR_pre

    3. Concatenate DIFF features:
        diff = [RGB_diff, SAR_diff]

    4. Construct branch inputs:
        PRE  = RGB_pre  + SAR_pre  + diff
        POST = RGB_post + SAR_post + diff

    5. Pass each branch through ResNet50 encoders

    6. Extract multi-scale features:
        conv1 → conv5 outputs

    7. Feature fusion:
        If use_feature_diff:
            fused = [pre_feat, post_feat, (post - pre)]
        Else:
            fused = [pre_feat, post_feat]

    8. Decode fused features:
        UNet-style decoder with skip connections

    9. Output segmentation map:
        Conv2D (1×1) + softmax

    Feature Fusion:
    ---------------
    Applied at:
        - skip1, skip2, skip3, skip4
        - bottleneck

    Freezing Strategy:
    ------------------
    If freeze_early=True:
        - conv1, conv2, conv3 layers are frozen
        - deeper layers remain trainable

    Output:
    -------
    Shape: (512, 512, num_classes)
    Activation: softmax

    Notes:
        - Explicit DIFF modeling improves temporal awareness
        - Feature-level DIFF provides additional refinement
        - Designed for multimodal disaster damage segmentation
    """

    inputs = layers.Input(shape=input_shape, dtype="float32")

    # -----------------------------------------------------
    # Split channels
    # -----------------------------------------------------

    rgb_pre  = layers.Lambda(lambda x: x[:, :, :, 0:3])(inputs)
    rgb_post = layers.Lambda(lambda x: x[:, :, :, 3:6])(inputs)

    sar_pre  = layers.Lambda(lambda x: x[:, :, :, 6:14])(inputs)
    sar_post = layers.Lambda(lambda x: x[:, :, :, 14:22])(inputs)

    # -----------------------------------------------------
    # DIFF channels
    # -----------------------------------------------------

    rgb_diff = layers.Subtract()([rgb_post, rgb_pre])
    sar_diff = layers.Subtract()([sar_post, sar_pre])

    diff = layers.Concatenate()([rgb_diff, sar_diff])  # 11 channels

    # -----------------------------------------------------
    # Branch inputs (22 channels each)
    # -----------------------------------------------------

    pre_input = layers.Concatenate()([rgb_pre, sar_pre, diff])
    post_input = layers.Concatenate()([rgb_post, sar_post, diff])

    # -----------------------------------------------------
    # Build encoders (FIXED CHANNEL COUNT)
    # -----------------------------------------------------

    pre_backbone = build_resnet_branch(
        pre_input,
        input_channels=22,   
        prefix="pre"
    )

    post_backbone = build_resnet_branch(
        post_input,
        input_channels=22,   
        prefix="post"
    )

    # Optional freezing
    if freeze_early:
        for layer in pre_backbone.layers:
            if any(s in layer.name for s in ["conv1", "conv2", "conv3"]):
                layer.trainable = False

        for layer in post_backbone.layers:
            if any(s in layer.name for s in ["conv1", "conv2", "conv3"]):
                layer.trainable = False

    # -----------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------

    def extract(backbone, prefix):
        s1 = backbone.get_layer(prefix + "_conv1_relu").output
        s2 = backbone.get_layer(prefix + "_conv2_block3_out").output
        s3 = backbone.get_layer(prefix + "_conv3_block4_out").output
        s4 = backbone.get_layer(prefix + "_conv4_block6_out").output
        x  = backbone.get_layer(prefix + "_conv5_block3_out").output
        return s1, s2, s3, s4, x

    pre_s1, pre_s2, pre_s3, pre_s4, pre_x = extract(pre_backbone, "pre")
    post_s1, post_s2, post_s3, post_s4, post_x = extract(post_backbone, "post")

    # -----------------------------------------------------
    # Feature fusion
    # -----------------------------------------------------

    def fuse(pre_feat, post_feat, name):
        if use_feature_diff:
            diff_feat = layers.Subtract(name=f"{name}_diff")([post_feat, pre_feat])
            return layers.Concatenate(name=name)([pre_feat, post_feat, diff_feat])
        else:
            return layers.Concatenate(name=name)([pre_feat, post_feat])

    skip1 = fuse(pre_s1, post_s1, "fuse_skip1")
    skip2 = fuse(pre_s2, post_s2, "fuse_skip2")
    skip3 = fuse(pre_s3, post_s3, "fuse_skip3")
    skip4 = fuse(pre_s4, post_s4, "fuse_skip4")
    x     = fuse(pre_x,  post_x,  "fuse_bottleneck")

    # -----------------------------------------------------
    # Decoder
    # -----------------------------------------------------

    x = decoder_block(x, skip4, 512, "decoder4")
    x = decoder_block(x, skip3, 256, "decoder3")
    x = decoder_block(x, skip2, 128, "decoder2")
    x = decoder_block(x, skip1, 64,  "decoder1")

    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)

    outputs = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name="DualBranch_PRE_POST_DIFF_ResNet50_UNet"
    )

    model.summary(line_length=140)

    return model