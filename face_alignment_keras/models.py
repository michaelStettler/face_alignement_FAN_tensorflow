import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


def conv_block(x, out_planes):
    residual = x

    out1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    out1 = layers.Activation('relu')(out1)
    out1 = layers.Conv2D(int(out_planes/2), kernel_size=3, strides=1, padding='same', use_bias=False)(out1)

    out2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(out1)
    out2 = layers.Activation('relu')(out2)
    out2 = layers.Conv2D(int(out_planes/4), kernel_size=3, strides=1, padding='same', use_bias=False)(out2)

    out3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(out2)
    out3 = layers.Activation('relu')(out3)
    out3 = layers.Conv2D(int(out_planes/4), kernel_size=3, strides=1, padding='same', use_bias=False)(out3)

    out3 = layers.Concatenate()([out1, out2, out3])

    in_planes = x.shape[-1]
    if in_planes != out_planes:
        residual = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(residual)
        residual = layers.Activation('relu')(residual)
        residual = layers.Conv2D(out_planes, kernel_size=1, strides=1, padding='same', use_bias=False)(residual)

    out3 = layers.Add()([out3, residual])

    return out3


def hour_glass(inp, depth, num_features):
    def generate_network(level, inp):
        print("level", level, inp.shape)
        # Upper branch
        up1 = inp
        up1 = conv_block(up1, num_features)

        # lower branch
        low1 = layers.AveragePooling2D(pool_size=2, strides=2)(inp)
        low1 = conv_block(low1, num_features)

        if level > 1:
            low2 = generate_network(level - 1, low1)
        else:
            low2 = low1
            low2 = conv_block(low2, num_features)

        low3 = low2
        low3 = conv_block(low3, num_features)

        up2 = layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(low3)

        return layers.Add()([up1, up2])

    return generate_network(depth, inp)


def FAN(num_modules=1, training=False, custom_loss=True, lr=[0.001], boundaries=[0, 1000]):
    inputs = tf.keras.Input(shape=(256, 256, 3), name='inputs')

    # Base part
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.AveragePooling2D(pool_size=2, strides=2)(x)

    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding='same', use_bias=False)(x)

    previous = x

    outputs = []
    # Stacking part
    for i in range(num_modules):
        hg = hour_glass(previous, 4, 256)

        ll = hg
        ll = conv_block(ll, 256)

        ll = layers.Conv2D(256, kernel_size=1, strides=1, padding='same', use_bias=False)(ll)
        ll = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(ll)
        ll = layers.Activation('relu')(ll)

        # Predict heatmaps
        tmp_out = layers.Conv2D(68, kernel_size=1, strides=1, padding='same', use_bias=False)(ll)
        outputs.append(tmp_out)
        # if i == 0:
        #     outputs = tmp_out
        # else:
        #     outputs = Add()([outputs, tmp_out])

        if i < num_modules - 1:
            ll = layers.Conv2D(256, kernel_size=1, strides=1, padding='same', use_bias=False)(ll)
            tmp_out_ = layers.Conv2D(256, kernel_size=1, strides=1, padding='same', use_bias=False)(tmp_out)
            previous = layers.Add()([previous, ll, tmp_out_])

    if custom_loss:  # https://stackoverflow.com/questions/50124158/keras-loss-function-with-additional-dynamic-parameter
        def normalized_mean_loss(y_true, y_pred, d):
            l2 = tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred)))
            return K.mean(l2 / d)

        d = tf.keras.Input(shape=(1,), name='d')
        y_true = tf.keras.Input(shape=(64, 64, 68), name='y_true')

        learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, lr)
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_fn, rho=0.9)
        model = tf.keras.Model(inputs=[inputs, y_true, d], outputs=outputs)
        model.add_loss(normalized_mean_loss(y_true, outputs, d))
        model.compile(optimizer=opt, loss=None, metrics=[normalized_mean_loss])
        return model
    else:
        return tf.keras.Model(inputs=inputs, outputs=outputs)


def depths_network(config, num_images):
    # built model
    input = tf.keras.layers.Input(shape=(64, 64, 3 + 68))
    resnet = tf.keras.applications.resnet_v2.ResNet152V2(include_top=False, weights=None, input_tensor=input)
    x = tf.keras.layers.Flatten()(resnet.output)
    x = tf.keras.layers.Dense(68, activation='linear')(x)
    model = tf.keras.Model(inputs=resnet.input, outputs=x)

    # get boundaries for learning schedule
    boundaries = []
    for bound in config['schedule']:
        boundaries.append(num_images / config['batch_size'] * bound)
    # set learning rate decay
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, config['lr'])
    # set optimizer
    opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_fn, rho=0.9)
    # compile model
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error', 'accuracy'])
    return model


if __name__ == '__main__':
    print("coucou :)")
    model = FAN(4)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])