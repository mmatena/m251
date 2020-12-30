"""TODO: Add title.

NOTE: Everything here is from SimCLRv2. I'm using simclr to refer
to it for the sake of brevity.

NOTE: I am also only considering models without the selective kernels.
"""
import tensorflow as tf


BATCH_NORM_EPSILON = 1e-5


class BatchNormRelu(tf.keras.layers.Layer):
    def __init__(
        self,
        relu=True,
        init_zero=False,
        center=True,
        scale=True,
        data_format="channels_last",
        # TODO: Make this settable.
        batch_norm_decay=0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.relu = relu

        if init_zero:
            gamma_initializer = tf.zeros_initializer()
        else:
            gamma_initializer = tf.ones_initializer()

        if data_format == "channels_first":
            axis = 1
        else:
            axis = -1

        self.bn = tf.keras.layers.BatchNormalization(
            axis=axis,
            momentum=batch_norm_decay,
            epsilon=BATCH_NORM_EPSILON,
            center=center,
            scale=scale,
            fused=False,
            gamma_initializer=gamma_initializer,
        )

    def call(self, inputs, training):
        inputs = self.bn(inputs, training=training)
        if self.relu:
            inputs = tf.nn.relu(inputs)
        return inputs


class DropBlock(tf.keras.layers.Layer):
    def __init__(
        self, keep_prob, dropblock_size, data_format="channels_last", **kwargs
    ):
        self.keep_prob = keep_prob
        self.dropblock_size = dropblock_size
        self.data_format = data_format
        super().__init__(**kwargs)

    def call(self, net, training):
        keep_prob = self.keep_prob
        dropblock_size = self.dropblock_size
        data_format = self.data_format
        if not training or keep_prob is None:
            return net

        tf.logging.info(
            "Applying DropBlock: dropblock_size {}, net.shape {}".format(
                dropblock_size, net.shape
            )
        )

        if data_format == "channels_last":
            _, width, height, _ = net.get_shape().as_list()
        else:
            _, _, width, height = net.get_shape().as_list()
        if width != height:
            raise ValueError("Input tensor with width!=height is not supported.")

        dropblock_size = min(dropblock_size, width)
        # seed_drop_rate is the gamma parameter of DropBlcok.
        seed_drop_rate = (
            (1.0 - keep_prob)
            * width ** 2
            / dropblock_size ** 2
            / (width - dropblock_size + 1) ** 2
        )

        # Forces the block to be inside the feature map.
        w_i, h_i = tf.meshgrid(tf.range(width), tf.range(width))
        valid_block_center = tf.logical_and(
            tf.logical_and(
                w_i >= int(dropblock_size // 2), w_i < width - (dropblock_size - 1) // 2
            ),
            tf.logical_and(
                h_i >= int(dropblock_size // 2), h_i < width - (dropblock_size - 1) // 2
            ),
        )

        valid_block_center = tf.expand_dims(valid_block_center, 0)
        valid_block_center = tf.expand_dims(
            valid_block_center, -1 if data_format == "channels_last" else 0
        )

        randnoise = tf.random_uniform(net.shape, dtype=tf.float32)
        block_pattern = (
            1
            - tf.cast(valid_block_center, dtype=tf.float32)
            + tf.cast((1 - seed_drop_rate), dtype=tf.float32)
            + randnoise
        ) >= 1
        block_pattern = tf.cast(block_pattern, dtype=tf.float32)

        if dropblock_size == width:
            block_pattern = tf.reduce_min(
                block_pattern,
                axis=[1, 2] if data_format == "channels_last" else [2, 3],
                keepdims=True,
            )
        else:
            if data_format == "channels_last":
                ksize = [1, dropblock_size, dropblock_size, 1]
            else:
                ksize = [1, 1, dropblock_size, dropblock_size]
            block_pattern = -tf.nn.max_pool(
                -block_pattern,
                ksize=ksize,
                strides=[1, 1, 1, 1],
                padding="SAME",
                data_format="NHWC" if data_format == "channels_last" else "NCHW",
            )

        percent_ones = tf.cast(tf.reduce_sum((block_pattern)), tf.float32) / tf.cast(
            tf.size(block_pattern), tf.float32
        )

        net = net / tf.cast(percent_ones, net.dtype) * tf.cast(block_pattern, net.dtype)
        return net


class FixedPadding(tf.keras.layers.Layer):
    def __init__(self, kernel_size, data_format="channels_last", **kwargs):
        super(FixedPadding, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.data_format = data_format

    def call(self, inputs, training):
        kernel_size = self.kernel_size
        data_format = self.data_format
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        if data_format == "channels_first":
            padded_inputs = tf.pad(
                inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]
            )
        else:
            padded_inputs = tf.pad(
                inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
            )

        return padded_inputs


class Conv2dFixedPadding(tf.keras.layers.Layer):
    def __init__(
        self, filters, kernel_size, strides, data_format="channels_last", **kwargs
    ):
        super(Conv2dFixedPadding, self).__init__(**kwargs)
        if strides > 1:
            self.fixed_padding = FixedPadding(kernel_size, data_format=data_format)
        else:
            self.fixed_padding = None
        self.conv2d = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=("SAME" if strides == 1 else "VALID"),
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            data_format=data_format,
        )

    def call(self, inputs, training):
        if self.fixed_padding:
            inputs = self.fixed_padding(inputs, training=training)
        return self.conv2d(inputs, training=training)


class IdentityLayer(tf.keras.layers.Layer):
    def call(self, inputs, training):
        return tf.identity(inputs)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        strides,
        use_projection=False,
        data_format="channels_last",
        dropblock_keep_prob=None,
        dropblock_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        del dropblock_keep_prob
        del dropblock_size
        if use_projection:
            self.shortcut_layers = [
                Conv2dFixedPadding(
                    filters=filters,
                    kernel_size=1,
                    strides=strides,
                    data_format=data_format,
                ),
                BatchNormRelu(relu=False, data_format=data_format),
            ]
        else:
            self.shortcut_layers = []

        self.conv2d_bn_layers = [
            Conv2dFixedPadding(
                filters=filters, kernel_size=3, strides=strides, data_format=data_format
            ),
            BatchNormRelu(data_format=data_format),
            Conv2dFixedPadding(
                filters=filters, kernel_size=3, strides=1, data_format=data_format
            ),
            BatchNormRelu(relu=False, init_zero=True, data_format=data_format),
        ]

    def call(self, inputs, training):
        shortcut = inputs
        for layer in self.shortcut_layers:
            # Projection shortcut in first layer to match filters and strides
            shortcut = layer(shortcut, training=training)

        for layer in self.conv2d_bn_layers:
            inputs = layer(inputs, training=training)

        return tf.nn.relu(inputs + shortcut)


class BottleneckBlock(tf.keras.layers.Layer):
    """BottleneckBlock."""

    def __init__(
        self,
        filters,
        strides,
        use_projection=False,
        data_format="channels_last",
        dropblock_keep_prob=None,
        dropblock_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projection_layers = []
        if use_projection:
            filters_out = 4 * filters
            self.projection_layers.append(
                Conv2dFixedPadding(
                    filters=filters_out,
                    kernel_size=1,
                    strides=strides,
                    data_format=data_format,
                )
            )
            self.projection_layers.append(
                BatchNormRelu(relu=False, data_format=data_format)
            )
        self.shortcut_dropblock = DropBlock(
            data_format=data_format,
            keep_prob=dropblock_keep_prob,
            dropblock_size=dropblock_size,
        )

        self.conv_relu_dropblock_layers = [
            Conv2dFixedPadding(
                filters=filters, kernel_size=1, strides=1, data_format=data_format
            ),
            BatchNormRelu(data_format=data_format),
            DropBlock(
                data_format=data_format,
                keep_prob=dropblock_keep_prob,
                dropblock_size=dropblock_size,
            ),
            Conv2dFixedPadding(
                filters=filters,
                kernel_size=3,
                strides=strides,
                data_format=data_format,
            ),
            BatchNormRelu(data_format=data_format),
            DropBlock(
                data_format=data_format,
                keep_prob=dropblock_keep_prob,
                dropblock_size=dropblock_size,
            ),
            Conv2dFixedPadding(
                filters=4 * filters, kernel_size=1, strides=1, data_format=data_format
            ),
            BatchNormRelu(relu=False, init_zero=True, data_format=data_format),
            DropBlock(
                data_format=data_format,
                keep_prob=dropblock_keep_prob,
                dropblock_size=dropblock_size,
            ),
        ]

    def call(self, inputs, training):
        shortcut = inputs
        for layer in self.projection_layers:
            shortcut = layer(shortcut, training=training)
        shortcut = self.shortcut_dropblock(shortcut, training=training)

        for layer in self.conv_relu_dropblock_layers:
            inputs = layer(inputs, training=training)

        return tf.nn.relu(inputs + shortcut)


class BlockGroup(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        block_fn,
        blocks,
        strides,
        data_format="channels_last",
        dropblock_keep_prob=None,
        dropblock_size=None,
        **kwargs,
    ):
        self._name = kwargs.get("name")
        super().__init__(**kwargs)

        self.layers = []
        self.layers.append(
            block_fn(
                filters,
                strides,
                use_projection=True,
                data_format=data_format,
                dropblock_keep_prob=dropblock_keep_prob,
                dropblock_size=dropblock_size,
            )
        )

        for _ in range(1, blocks):
            self.layers.append(
                block_fn(
                    filters,
                    1,
                    data_format=data_format,
                    dropblock_keep_prob=dropblock_keep_prob,
                    dropblock_size=dropblock_size,
                )
            )

    def call(self, inputs, training):
        for layer in self.layers:
            inputs = layer(inputs, training=training)
        return tf.identity(inputs, self._name)


class Resnet(tf.keras.layers.Layer):
    def __init__(
        self,
        block_fn,
        layers,
        width_multiplier,
        data_format="channels_last",
        dropblock_keep_probs=None,
        dropblock_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_format = data_format
        if dropblock_keep_probs is None:
            dropblock_keep_probs = [None] * 4
        if not isinstance(dropblock_keep_probs, list) or len(dropblock_keep_probs) != 4:
            raise ValueError("dropblock_keep_probs is not valid:", dropblock_keep_probs)

        self.initial_conv_relu_max_pool = [
            Conv2dFixedPadding(
                filters=64 * width_multiplier,
                kernel_size=7,
                strides=2,
                data_format=data_format,
                trainable=True,
            ),
            IdentityLayer(name="initial_conv", trainable=True),
            BatchNormRelu(data_format=data_format, trainable=True),
            tf.keras.layers.MaxPooling2D(
                pool_size=3,
                strides=2,
                padding="SAME",
                data_format=data_format,
                trainable=True,
            ),
            IdentityLayer(name="initial_max_pool", trainable=True),
        ]

        self.block_groups = []
        for i in range(4):
            bg = BlockGroup(
                filters=64 * width_multiplier * 2 ** i,
                block_fn=block_fn,
                blocks=layers[i],
                strides=1 + bool(i),
                name=f"block_group{i+1}",
                data_format=data_format,
                dropblock_keep_prob=dropblock_keep_probs[i],
                dropblock_size=dropblock_size,
                trainable=True,
            )
            self.block_groups.append(bg)

    def call(self, inputs, training):
        for layer in self.initial_conv_relu_max_pool:
            inputs = layer(inputs, training=training)

        for i, layer in enumerate(self.block_groups):
            inputs = layer(inputs, training=training)

        if self.data_format == "channels_last":
            inputs = tf.reduce_mean(inputs, [1, 2])
        else:
            inputs = tf.reduce_mean(inputs, [2, 3])

        inputs = tf.identity(inputs, "final_avg_pool")
        return inputs


def resnet(
    resnet_depth,
    width_multiplier,
    data_format="channels_last",
    dropblock_keep_probs=None,
    dropblock_size=None,
):
    """Returns the ResNet model for a given size and number of output classes."""
    model_params = {
        18: {"block": ResidualBlock, "layers": [2, 2, 2, 2]},
        34: {"block": ResidualBlock, "layers": [3, 4, 6, 3]},
        50: {"block": BottleneckBlock, "layers": [3, 4, 6, 3]},
        101: {"block": BottleneckBlock, "layers": [3, 4, 23, 3]},
        152: {"block": BottleneckBlock, "layers": [3, 8, 36, 3]},
        200: {"block": BottleneckBlock, "layers": [3, 24, 36, 3]},
    }

    if resnet_depth not in model_params:
        raise ValueError("Not a valid resnet_depth:", resnet_depth)

    params = model_params[resnet_depth]
    return Resnet(
        params["block"],
        params["layers"],
        width_multiplier,
        dropblock_keep_probs=dropblock_keep_probs,
        dropblock_size=dropblock_size,
        data_format=data_format,
    )


###############################################################################
###############################################################################


class LinearLayer(tf.keras.layers.Layer):
    def __init__(
        self, num_classes, use_bias=True, use_bn=False, name="linear_layer", **kwargs
    ):
        # Note: use_bias is ignored for the dense layer when use_bn=True.
        # However, it is still used for batch norm.
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.use_bn = use_bn
        self._name = name
        if callable(self.num_classes):
            num_classes = -1
        else:
            num_classes = self.num_classes
        self.dense = tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            use_bias=use_bias and not self.use_bn,
        )
        if self.use_bn:
            self.bn_relu = BatchNormRelu(relu=False, center=use_bias)

    def build(self, input_shape):
        # TODO(srbs): Add a new SquareDense layer.
        if callable(self.num_classes):
            self.dense.units = self.num_classes(input_shape)
        super(LinearLayer, self).build(input_shape)

    def call(self, inputs, training):
        assert inputs.shape.ndims == 2, inputs.shape
        inputs = self.dense(inputs)
        if self.use_bn:
            inputs = self.bn_relu(inputs, training=training)
        return inputs


class ProjectionHead(tf.keras.layers.Layer):
    def __init__(self, finetune_layer=1, **kwargs):
        super().__init__(**kwargs)
        assert finetune_layer < 3
        self.linear_layers = [
            LinearLayer(
                num_classes=lambda input_shape: int(input_shape[-1]),
                use_bias=True,
                use_bn=True,
                name="nl_%d" % i,
            )
            for i in range(finetune_layer)
        ]

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.linear_layers:
            x = layer(x, training=training)
            x = tf.nn.relu(x)
        return x


class SimClrBaseModel(tf.keras.models.Model):
    """Resnet model with projection or supervised layer."""

    def __init__(self, depth, width_multiplier, **kwargs):
        super().__init__(**kwargs)
        self.resnet_model = resnet(
            resnet_depth=depth,
            width_multiplier=width_multiplier,
        )
        self._projection_head = ProjectionHead()

    def call(self, inputs, training=None):
        features = inputs

        # Base network forward pass.
        hiddens = self.resnet_model(features, training=training)

        # Add heads.
        supervised_head_inputs = self._projection_head(hiddens, training)

        return supervised_head_inputs
