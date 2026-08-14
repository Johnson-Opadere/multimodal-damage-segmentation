"""
Dual-Branch Gated Temporal-Difference Segmentation Model.

This module implements a multimodal semantic segmentation architecture for
disaster damage assessment using paired pre-event and post-event RGB + SAR
imagery.

Architecture overview
---------------------
Input:
    22-channel tensor:
        RGB pre-event   : 3 channels
        RGB post-event  : 3 channels
        SAR pre-event   : 8 channels
        SAR post-event  : 8 channels

Model:
    1. Split the multimodal tensor into pre-event and post-event inputs.
    2. Encode each temporal view with an independent ResNet50 backbone.
    3. Compute temporal feature differences at multiple encoder scales.
    4. Apply learnable sigmoid gates to control the contribution of the
       temporal-difference features.
    5. Decode the fused multiscale representations using a U-Net-style decoder.
    6. Produce per-pixel probabilities for four damage classes.

Output classes:
    0 - Background
    1 - Minor damage
    2 - Major damage
    3 - Destroyed

The key architectural idea is gated temporal-difference fusion. Rather than
directly concatenating temporal differences with the pre/post features, the
model learns how strongly the difference signal should contribute at each
spatial location and feature channel.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def decoder_block(x, skip, filters, name):
    """
    Apply one U-Net-style decoder stage.

    The decoder upsamples the current feature map, concatenates it with the
    corresponding fused encoder skip connection, and refines the result using
    two 3x3 convolutions.

    Args:
        x: Input decoder feature tensor.
        skip: Skip-connection tensor from the corresponding encoder scale.
        filters: Number of convolution filters used in this decoder stage.
        name: Prefix used to construct layer names.

    Returns:
        Tensor containing the refined decoder features.
    """
    x = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        name=f"{name}_upsample",
    )(x)

    x = layers.Concatenate(
        name=f"{name}_concat"
    )([x, skip])

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        name=f"{name}_conv1",
    )(x)

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        name=f"{name}_conv2",
    )(x)

    return x


# ---------------------------------------------------------------------------
# ResNet50 initialization utilities
# ---------------------------------------------------------------------------

def expand_first_conv_weights(pretrained_weights, new_channels):
    """
    Adapt ImageNet ResNet50 first-layer weights to a larger input channel count.

    ImageNet-pretrained ResNet50 expects a three-channel RGB input. Each
    temporal branch in this model receives 11 channels (3 RGB + 8 SAR).
    The pretrained first-layer kernel is therefore averaged across its RGB
    channel dimension and tiled across the required number of input channels.

    Args:
        pretrained_weights: List containing the original Conv2D kernel and bias.
        new_channels: Number of channels expected by the new input tensor.

    Returns:
        List containing the expanded convolution kernel and original bias.
    """
    kernel, bias = pretrained_weights

    # Collapse the RGB channel dimension into a single representative kernel.
    mean_kernel = tf.reduce_mean(kernel, axis=2, keepdims=True)

    # Replicate the averaged kernel across all multimodal input channels.
    new_kernel = tf.tile(mean_kernel, [1, 1, new_channels, 1])

    return [new_kernel.numpy(), bias]


def build_resnet_branch(input_tensor, input_channels, prefix):
    """
    Construct one multimodal ResNet50 encoder branch.

    A standard ImageNet-pretrained ResNet50 is first instantiated to obtain
    pretrained weights. A second ResNet50 is then constructed for the
    multimodal input tensor. The first convolution is adapted to the requested
    number of input channels, while compatible pretrained weights are copied
    into the remaining layers.

    Args:
        input_tensor: Input tensor for this temporal branch.
        input_channels: Number of channels in the branch input.
        prefix: Prefix used to uniquely name layers in the branch.

    Returns:
        Keras ResNet50 model representing one temporal encoder.
    """
    # Reference ImageNet model used only as the source of pretrained weights.
    pretrained = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(512, 512, 3),
    )

    first_conv = pretrained.get_layer("conv1_conv")
    pretrained_weights = first_conv.get_weights()

    # Build the branch using the multimodal input tensor.
    branch = ResNet50(
        include_top=False,
        weights=None,
        input_tensor=input_tensor,
    )

    # Prefix layer names so the PRE and POST encoders remain distinguishable.
    for layer in branch.layers:
        layer._name = prefix + "_" + layer.name

    # Adapt the ImageNet first convolution from RGB to multimodal input.
    new_first_conv = branch.get_layer(prefix + "_conv1_conv")
    expanded = expand_first_conv_weights(
        pretrained_weights,
        input_channels,
    )
    new_first_conv.set_weights(expanded)

    # Transfer all remaining compatible ImageNet weights.
    for layer in pretrained.layers:
        if layer.name == "conv1_conv":
            continue

        try:
            branch_layer = branch.get_layer(prefix + "_" + layer.name)
            branch_layer.set_weights(layer.get_weights())
        except (ValueError, AttributeError):
            # Some layers may not have compatible or transferable weights.
            pass

    return branch


# ---------------------------------------------------------------------------
# Gated temporal-difference fusion
# ---------------------------------------------------------------------------

def gated_fusion(pre_feat, post_feat, name):
    """
    Fuse pre-event and post-event features using gated temporal differences.

    For feature tensors with C channels, the base representation concatenates
    the pre-event and post-event features to produce 2C channels:

        base = concat(pre, post)

    The temporal change signal is:

        diff = post - pre

    Because ``diff`` contains C channels while ``base`` contains 2C channels,
    the difference tensor is projected to 2C channels with a 1x1 convolution.

    A second 1x1 convolution followed by a sigmoid learns a gate:

        gate = sigmoid(Conv1x1(diff))

    The final fused representation is:

        output = base + gate * projected_diff

    This allows the network to adaptively suppress or emphasize temporal
    changes rather than forcing the raw difference signal into every feature.

    Args:
        pre_feat: Pre-event feature tensor of shape (B, H, W, C).
        post_feat: Post-event feature tensor of shape (B, H, W, C).
        name: Prefix used to construct fusion-layer names.

    Returns:
        Fused feature tensor of shape (B, H, W, 2C).
    """
    channels = pre_feat.shape[-1]

    # Preserve both temporal representations as the base fused feature.
    base = layers.Concatenate(
        name=f"{name}_base"
    )([pre_feat, post_feat])

    # Explicit temporal change representation.
    diff = layers.Subtract(
        name=f"{name}_diff"
    )([post_feat, pre_feat])

    # Project C-channel temporal differences to the 2C-dimensional base space.
    diff_proj = layers.Conv2D(
        filters=2 * channels,
        kernel_size=1,
        padding="same",
        name=f"{name}_diff_proj",
    )(diff)

    # Learn a spatially and channel-adaptive weight for the change signal.
    gate = layers.Conv2D(
        filters=2 * channels,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        name=f"{name}_gate",
    )(diff)

    gated_diff = layers.Multiply(
        name=f"{name}_gated_diff"
    )([diff_proj, gate])

    # Residually inject the gated temporal signal into the base representation.
    output = layers.Add(
        name=f"{name}_out"
    )([base, gated_diff])

    return output


# ---------------------------------------------------------------------------
# Complete segmentation model
# ---------------------------------------------------------------------------

def build_segmentation_model(
    input_shape=(512, 512, 22),
    num_classes=4,
    freeze_early=False,
):
    """
    Build the dual-branch gated-difference segmentation network.

    Args:
        input_shape:
            Input tensor shape. The default 22 channels correspond to
            RGB_pre (3), RGB_post (3), SAR_pre (8), and SAR_post (8).
        num_classes:
            Number of semantic segmentation classes. Default is four.
        freeze_early:
            Reserved configuration argument retained for compatibility with
            the training pipeline.

    Returns:
        A ``tf.keras.Model`` mapping a multimodal image tensor of shape
        ``(H, W, 22)`` to per-pixel class probabilities of shape
        ``(H, W, num_classes)``.
    """
    inputs = layers.Input(
        shape=input_shape,
        dtype="float32",
        name="multimodal_input",
    )

    # -----------------------------------------------------------------------
    # Separate the 22-channel tensor by modality and acquisition time.
    # -----------------------------------------------------------------------

    rgb_pre = layers.Lambda(
        lambda x: x[:, :, :, 0:3],
        name="rgb_pre",
    )(inputs)

    rgb_post = layers.Lambda(
        lambda x: x[:, :, :, 3:6],
        name="rgb_post",
    )(inputs)

    sar_pre = layers.Lambda(
        lambda x: x[:, :, :, 6:14],
        name="sar_pre",
    )(inputs)

    sar_post = layers.Lambda(
        lambda x: x[:, :, :, 14:22],
        name="sar_post",
    )(inputs)

    # These explicit input-space temporal differences are retained as part of
    # the model definition for architectural traceability. Gated fusion below
    # operates on multiscale encoder features.
    rgb_diff = layers.Subtract(
        name="rgb_input_diff"
    )([rgb_post, rgb_pre])

    sar_diff = layers.Subtract(
        name="sar_input_diff"
    )([sar_post, sar_pre])

    _ = layers.Concatenate(
        name="multimodal_input_diff"
    )([rgb_diff, sar_diff])

    # -----------------------------------------------------------------------
    # Temporal encoder inputs
    # -----------------------------------------------------------------------

    # Each encoder receives RGB + SAR from one acquisition time.
    pre_input = layers.Concatenate(
        name="pre_multimodal_input"
    )([rgb_pre, sar_pre])

    post_input = layers.Concatenate(
        name="post_multimodal_input"
    )([rgb_post, sar_post])

    # Independent PRE and POST ResNet50 encoders.
    pre_backbone = build_resnet_branch(
        pre_input,
        input_channels=11,
        prefix="pre",
    )

    post_backbone = build_resnet_branch(
        post_input,
        input_channels=11,
        prefix="post",
    )

    def extract(backbone, prefix):
        """
        Extract multiscale ResNet50 features for U-Net skip connections.
        """
        s1 = backbone.get_layer(prefix + "_conv1_relu").output
        s2 = backbone.get_layer(prefix + "_conv2_block3_out").output
        s3 = backbone.get_layer(prefix + "_conv3_block4_out").output
        s4 = backbone.get_layer(prefix + "_conv4_block6_out").output
        bottleneck = backbone.get_layer(
            prefix + "_conv5_block3_out"
        ).output

        return s1, s2, s3, s4, bottleneck

    pre_s1, pre_s2, pre_s3, pre_s4, pre_x = extract(
        pre_backbone,
        "pre",
    )

    post_s1, post_s2, post_s3, post_s4, post_x = extract(
        post_backbone,
        "post",
    )

    # -----------------------------------------------------------------------
    # Multiscale gated temporal fusion
    # -----------------------------------------------------------------------

    # Temporal fusion is applied independently at every encoder resolution.
    skip1 = gated_fusion(pre_s1, post_s1, "fuse1")
    skip2 = gated_fusion(pre_s2, post_s2, "fuse2")
    skip3 = gated_fusion(pre_s3, post_s3, "fuse3")
    skip4 = gated_fusion(pre_s4, post_s4, "fuse4")

    x = gated_fusion(
        pre_x,
        post_x,
        "fuse_bottleneck",
    )

    # -----------------------------------------------------------------------
    # U-Net-style decoder
    # -----------------------------------------------------------------------

    x = decoder_block(x, skip4, 512, "decoder4")
    x = decoder_block(x, skip3, 256, "decoder3")
    x = decoder_block(x, skip2, 128, "decoder2")
    x = decoder_block(x, skip1, 64, "decoder1")

    # Restore the original 512x512 spatial resolution.
    x = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        name="final_upsample",
    )(x)

    # 1x1 classification convolution followed by per-pixel softmax.
    outputs = layers.Conv2D(
        filters=num_classes,
        kernel_size=1,
        activation="softmax",
        name="segmentation_output",
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="DualBranch_GATED_DIFF_ResNet50_UNet",
    )

    model.summary(line_length=140)

    return model