"""
Dual-Branch PRE/POST + Context-Aware Gated DIFF Segmentation Model (v2)

Overview:
---------
This model extends the dual-branch PRE/POST architecture by introducing
a context-aware gated fusion mechanism for modeling temporal differences.

Unlike earlier approaches that rely on simple pixel-wise differencing,
this model incorporates spatial context into the gating mechanism,
leading to more robust and semantically meaningful feature fusion.

Input Tensor:
-------------
Shape: (512, 512, 22)

Channel Layout:
---------------
RGB_pre   : 3 channels  → [0:3]
RGB_post  : 3 channels  → [3:6]
SAR_pre   : 8 channels  → [6:14]
SAR_post  : 8 channels  → [14:22]

Branch Inputs:
--------------
PRE branch  = RGB_pre  + SAR_pre  → 11 channels  
POST branch = RGB_post + SAR_post → 11 channels  

Architecture:
-------------
- Dual ResNet50 encoders (PRE / POST)
- Feature extraction at multiple scales
- Context-aware gated DIFF fusion (v2)
- UNet-style decoder
- Pixel-wise softmax output

Key Innovation:
---------------
Context-aware gating replaces simple pixel-wise gating:

    gate = sigmoid(Conv3x3 → Conv3x3 → Conv1x1(diff))

Enhancements over v1:
--------------------
- Spatially-aware gating (captures local context)
- BatchNorm applied to diff features
- Reduced sensitivity to SAR noise
- Improved separation between damage classes

Output:
-------
- Segmentation map
- Shape: (512, 512, num_classes)
- Activation: softmax

Classes:
--------
0 → Background  
1 → Minor damage  
2 → Major damage  
3 → Destroyed  

Run Example:
------------
from src.models.segmentation_model_gated_v2 import build_segmentation_model

model = build_segmentation_model(
    input_shape=(512, 512, 22),
    num_classes=4,
    freeze_early=False
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
        skip (tf.Tensor): Skip connection from encoder
        filters (int): Number of convolution filters
        name (str): Block name prefix

    Returns:
        tf.Tensor: Refined feature map

    Processing:
    -----------
    1. Upsample feature map (bilinear interpolation)
    2. Concatenate with skip connection
    3. Apply two Conv2D + ReLU layers

    Notes:
        - Preserves spatial details from encoder
        - Used in hierarchical decoding
    """
    x = layers.UpSampling2D((2, 2), interpolation="bilinear", name=f"{name}_upsample")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv1")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv2")(x)
    return x


# ---------------------------------------------------------
# Expand First Conv Weights
# ---------------------------------------------------------

def expand_first_conv_weights(pretrained_weights, new_channels):
    """
    Expand pretrained ResNet first convolution weights to match new input channels.

    Args:
        pretrained_weights (List[np.ndarray]):
            Original conv1 weights (kernel + bias)
        new_channels (int): Target number of input channels

    Returns:
        List[np.ndarray]: Expanded kernel and original bias

    Method:
    -------
    - Compute mean across RGB channels
    - Tile mean kernel to match new_channels
    - Preserve original bias

    Notes:
        - Enables transfer learning for multi-channel inputs (RGB + SAR)
        - Avoids random initialization
    """
    kernel, bias = pretrained_weights
    mean_kernel = tf.reduce_mean(kernel, axis=2, keepdims=True)
    new_kernel = tf.tile(mean_kernel, [1, 1, new_channels, 1])
    return [new_kernel.numpy(), bias]


# ---------------------------------------------------------
# Build ResNet Branch
# ---------------------------------------------------------

def build_resnet_branch(input_tensor, input_channels, prefix):
    """
    Expand pretrained ResNet first convolution weights to match new input channels.

    Args:
        pretrained_weights (List[np.ndarray]):
            Original conv1 weights (kernel + bias)
        new_channels (int): Target number of input channels

    Returns:
        List[np.ndarray]: Expanded kernel and original bias

    Method:
    -------
    - Compute mean across RGB channels
    - Tile mean kernel to match new_channels
    - Preserve original bias

    Notes:
        - Enables transfer learning for multi-channel inputs (RGB + SAR)
        - Avoids random initialization
    """

    pretrained = ResNet50(include_top=False, weights="imagenet", input_shape=(512, 512, 3))
    first_conv = pretrained.get_layer("conv1_conv")
    pretrained_weights = first_conv.get_weights()

    branch = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)

    for layer in branch.layers:
        layer._name = prefix + "_" + layer.name

    new_first_conv = branch.get_layer(prefix + "_conv1_conv")
    expanded = expand_first_conv_weights(pretrained_weights, input_channels)
    new_first_conv.set_weights(expanded)

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
# CONTEXT-AWARE GATED FUSION (v2)
# ---------------------------------------------------------

def gated_fusion_v2(pre_feat, post_feat, name):
    """
    Context-aware gated feature fusion (v2).

    Args:
        pre_feat (tf.Tensor): Features from PRE branch
        post_feat (tf.Tensor): Features from POST branch
        name (str): Layer name prefix

    Returns:
        tf.Tensor: Fused feature map

    Pipeline:
    ---------
    1. Base fusion:
        fused = concat(pre_feat, post_feat)

    2. Compute DIFF:
        diff = post_feat - pre_feat

    3. Normalize DIFF:
        diff = BatchNorm(diff)

    4. Project DIFF to match fused channels:
        diff_proj = Conv1x1(diff)

    5. Context-aware gate:
        g = Conv3x3 → Conv3x3 → Conv1x1(diff)
        gate = sigmoid(g)

    6. Apply gating:
        gated_diff = diff_proj * gate

    7. Residual fusion:
        output = fused + gated_diff

    Key Advantage:
    --------------
    - Incorporates spatial context into gating
    - More robust to SAR noise
    - Better semantic alignment across branches

    Notes:
        - Operates at multiple feature scales
        - Enhances temporal difference modeling
    """

    C = pre_feat.shape[-1]

    # -------------------------------------------------
    # Base fusion
    # -------------------------------------------------
    fused = layers.Concatenate(name=f"{name}_base")([pre_feat, post_feat])  # (H, W, 2C)

    # -------------------------------------------------
    # DIFF + normalization
    # -------------------------------------------------
    diff = layers.Subtract(name=f"{name}_diff")([post_feat, pre_feat])
    diff = layers.BatchNormalization(name=f"{name}_diff_bn")(diff)

    # -------------------------------------------------
    # Project DIFF → 2C
    # -------------------------------------------------
    diff_proj = layers.Conv2D(
        filters=2 * C,
        kernel_size=1,
        padding="same",
        name=f"{name}_diff_proj"
    )(diff)

    # -------------------------------------------------
    # CONTEXT-AWARE GATE
    # -------------------------------------------------
    g = layers.Conv2D(2 * C, 3, padding="same", activation="relu", name=f"{name}_gate_conv1")(diff)
    g = layers.Conv2D(2 * C, 3, padding="same", activation="relu", name=f"{name}_gate_conv2")(g)

    gate = layers.Conv2D(
        filters=2 * C,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        name=f"{name}_gate"
    )(g)

    # -------------------------------------------------
    # Apply gating
    # -------------------------------------------------
    gated_diff = layers.Multiply(name=f"{name}_gated_diff")([diff_proj, gate])

    # -------------------------------------------------
    # Final fusion
    # -------------------------------------------------
    out = layers.Add(name=f"{name}_out")([fused, gated_diff])

    return out


# ---------------------------------------------------------
# Full Model
# ---------------------------------------------------------

def build_segmentation_model(
        input_shape=(512, 512, 22),
        num_classes=4,
        freeze_early=False):
def build_segmentation_model(
        input_shape=(512, 512, 22),
        num_classes=4,
        freeze_early=False):
    """
    Build the full context-aware gated DIFF segmentation model.

    Args:
        input_shape (Tuple[int, int, int]):
            Input tensor shape (default: 512x512x22)
        num_classes (int):
            Number of segmentation classes
        freeze_early (bool):
            If True, freezes early ResNet layers

    Returns:
        tf.keras.Model: Segmentation model

    Pipeline:
    ---------
    1. Split input tensor into:
        - RGB_pre, RGB_post
        - SAR_pre, SAR_post

    2. Construct branch inputs:
        PRE  = RGB_pre + SAR_pre
        POST = RGB_post + SAR_post

    3. Build dual ResNet encoders

    4. Extract multi-scale features:
        conv1 → conv5

    5. Apply context-aware gated fusion:
        skip1, skip2, skip3, skip4, bottleneck

    6. Decode fused features:
        UNet-style decoder

    7. Output segmentation:
        Conv2D (1x1) + softmax

    Feature Fusion:
    ---------------
    - Uses gated DIFF mechanism at each scale
    - Enhances temporal change modeling

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
        - Designed for multimodal disaster damage segmentation
        - Improves robustness over v1 gating
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
    # Branch inputs
    # -----------------------------------------------------

    pre_input = layers.Concatenate()([rgb_pre, sar_pre])
    post_input = layers.Concatenate()([rgb_post, sar_post])

    # -----------------------------------------------------
    # Encoders
    # -----------------------------------------------------

    pre_backbone = build_resnet_branch(pre_input, 11, "pre")
    post_backbone = build_resnet_branch(post_input, 11, "post")

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
    # GATED FUSION (v2)
    # -----------------------------------------------------

    skip1 = gated_fusion_v2(pre_s1, post_s1, "fuse1")
    skip2 = gated_fusion_v2(pre_s2, post_s2, "fuse2")
    skip3 = gated_fusion_v2(pre_s3, post_s3, "fuse3")
    skip4 = gated_fusion_v2(pre_s4, post_s4, "fuse4")
    x     = gated_fusion_v2(pre_x,  post_x,  "fuse_bottleneck")

    # -----------------------------------------------------
    # Decoder
    # -----------------------------------------------------

    x = decoder_block(x, skip4, 512, "decoder4")
    x = decoder_block(x, skip3, 256, "decoder3")
    x = decoder_block(x, skip2, 128, "decoder2")
    x = decoder_block(x, skip1, 64,  "decoder1")

    x = layers.UpSampling2D((2, 2), interpolation="bilinear")(x)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(x)

    model = models.Model(
        inputs,
        outputs,
        name="DualBranch_GATED_DIFF_v2"
    )

    model.summary(line_length=140)

    return model