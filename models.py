from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              activation='relu',
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """

    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation==None):
        return x

    x = Activation(activation, name=name)(x)
    return x


def trans_conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(2, 2),
              activation='relu',
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """

    
    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    return x


def inceptionBlock(layers, inp):

    layers *= 1.67

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(layers*0.167) + int(layers*0.333) + int(layers*0.5), 1, 1, activation=None, padding='same')


    conv3x3 = conv2d_bn(inp, int(layers*0.167), 3, 3, activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(layers*0.333), 3, 3, activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(layers*0.5), 3, 3, activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)
    
    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out

def residualConnection(layers, depth, inp):

    shortcut = inp
    shortcut = conv2d_bn(shortcut, layers , 1, 1, activation=None, padding='same')

    out = conv2d_bn(inp, layers, 3,3 , activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for deep in range(depth-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, layers , 1, 1, activation=None, padding='same')

        out = conv2d_bn(out, layers, 3,3 , activation='relu', padding='same')        

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)


    return out

def MultiResUNet(height,width):

    unet_unit = 32

    inputs = Input((height, width, 5))
    inputs_norm = BatchNormalization(axis=3, scale=False)(inputs)
    incption1 = inceptionBlock(unet_unit, inputs_norm) 
    incption1 = BatchNormalization(axis=3, scale=False)(incption1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(incption1)
    pool1 = BatchNormalization(axis=3, scale=False)(pool1)
    incption1 = residualConnection(unet_unit, 4,incption1) 
    incption1 = BatchNormalization(axis=3, scale=False)(incption1)

    incption2 = inceptionBlock(unet_unit*2, pool1)
    incption2 = BatchNormalization(axis=3, scale=False)(incption2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(incption2)
    pool2 = BatchNormalization(axis=3, scale=False)(pool2)
    incption2 = residualConnection(unet_unit*2, 3,incption2) 
    incption2 = BatchNormalization(axis=3, scale=False)(incption2)

    incption3 = inceptionBlock(unet_unit*4, pool2)
    incption3 = BatchNormalization(axis=3, scale=False)(incption3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(incption3)
    pool3 = BatchNormalization(axis=3, scale=False)(pool3)
    incption3 = residualConnection(unet_unit*4, 2,incption3) 
    incption3 = BatchNormalization(axis=3, scale=False)(incption3)

    incption4 = inceptionBlock(unet_unit*8, pool3)
    incption4 = BatchNormalization(axis=3, scale=False)(incption4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(incption4)
    pool4 = BatchNormalization(axis=3, scale=False)(pool4)
    incption4 = residualConnection(unet_unit*8, 1,incption4) 
    incption4 = BatchNormalization(axis=3, scale=False)(incption4)

    incption5 = inceptionBlock(unet_unit*16, pool4)
    incption5 = BatchNormalization(axis=3, scale=False)(incption5)


    level4 = Conv2D(1, (1,1), name="level4", activation='sigmoid')(incption5)



    up6 = concatenate([Conv2DTranspose(unet_unit*8, (2, 2), strides=(2, 2), padding='same')(incption5), incption4], axis=3)
    up6 = BatchNormalization(axis=3, scale=False)(up6)
    inception6 = inceptionBlock(unet_unit*8,up6)
    inception6 = BatchNormalization(axis=3, scale=False)(inception6)


    level3 = Conv2D(1, (1,1), name="level3", activation='sigmoid')(inception6)
    
    up7 = concatenate([Conv2DTranspose(unet_unit*4, (2, 2), strides=(2, 2), padding='same')(inception6), incption3], axis=3)
    up7 = BatchNormalization(axis=3, scale=False)(up7)
    inception7 = inceptionBlock(unet_unit*4,up7)
    inception7 = BatchNormalization(axis=3, scale=False)(inception7)


    level2 = Conv2D(1, (1,1), name="level2", activation='sigmoid')(inception7)

    up8 = concatenate([Conv2DTranspose(unet_unit*2, (2, 2), strides=(2, 2), padding='same')(inception7), incption2], axis=3)
    up8 = BatchNormalization(axis=3, scale=False)(up8)
    inception8 = inceptionBlock(unet_unit*2,up8)
    inception8 = BatchNormalization(axis=3, scale=False)(inception8)

    level1 = Conv2D(1, (1,1), name="level1", activation='sigmoid')(inception8)

    up9 = concatenate([Conv2DTranspose(unet_unit, (2, 2), strides=(2, 2), padding='same')(inception8), incption1], axis=3)
    up9 = BatchNormalization(axis=3, scale=False)(up9)
    inception9 = inceptionBlock(unet_unit,up9)
    inception9 = BatchNormalization(axis=3, scale=False)(inception9)

    out = Conv2D(1, (1,1), name="out", activation='sigmoid')(inception9)

    #conv10 = conv2d_bn(inception9 , 1, 1, 1, activation='sigmoid')

    #model = Model(inputs=[inputs], outputs=[conv10])

    model = Model(inputs=[inputs], outputs=[out, level1, level2, level3, level4])

    return model


def UNetDS32(length, n_channel=5):
    
    x = 32
    inputs = Input((length,length, n_channel))
    inputs_norm = BatchNormalization(axis=3, scale=False)(inputs)
    conv1 = Conv2D(x,(3,3), activation='relu', padding='same')(inputs_norm)
    conv1 = BatchNormalization(axis=3, scale=False)(conv1)
    conv1 = Conv2D(x,(3,3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization(axis=3, scale=False)(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    #pool1 = BatchNormalization(axis=3, scale=False)(pool1)


    conv2 = Conv2D(x*2,(3,3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization(axis=3, scale=False)(conv2)
    conv2 = Conv2D(x*2,(3,3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization(axis=3, scale=False)(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    #pool2 = BatchNormalization(axis=3, scale=False)(pool2)

    conv3 = Conv2D(x*4,(3,3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization(axis=3, scale=False)(conv3)
    conv3 = Conv2D(x*4,(3,3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization(axis=3, scale=False)(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    #pool3 = BatchNormalization(axis=3, scale=False)(pool3)



    conv4 = Conv2D(x*8,(3,3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization(axis=3, scale=False)(conv4)
    conv4 = Conv2D(x*8,(3,3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization(axis=3, scale=False)(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)
    #pool4 = BatchNormalization(axis=3, scale=False)(pool4)

    conv5 = Conv2D(x*16, (3,3), activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization(axis=3, scale=False)(conv5)
    conv5 = Conv2D(x*16, (3,3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization(axis=3, scale=False)(conv5)
    
    level4 = Conv2D(1, (1,1), name="level4", activation='sigmoid')(conv5)

    print("Shape of conv4: ",conv4.shape)
    print("Shape of conv5: ",conv5.shape)


    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    up6 = BatchNormalization(axis=3, scale=False)(up6)
    conv6 = Conv2D(x*8, (3,3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization(axis=3, scale=False)(conv6)
    conv6 = Conv2D(x*8, (3,3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization(axis=3, scale=False)(conv6)
    
    level3 = Conv2D(1, (1,1), name="level3", activation='sigmoid')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    up7 = BatchNormalization(axis=3, scale=False)(up7)
    conv7 = Conv2D(x*4, (3,3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization(axis=3, scale=False)(conv7)
    conv7 = Conv2D(x*4,(3,3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization(axis=3, scale=False)(conv7)
    
    level2 = Conv2D(1, (1,1), name="level2", activation='sigmoid')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    up8 = BatchNormalization(axis=3, scale=False)(up8)
    conv8 = Conv2D(x*2, (3,3), activation='relu', padding='same')(up8)
    conv8 = BatchNormalization(axis=3, scale=False)(conv8)
    conv8 = Conv2D(x*2, (3,3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization(axis=3, scale=False)(conv8)
    
    level1 = Conv2D(1, (1,1), name="level1", activation='sigmoid')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    up9 = BatchNormalization(axis=3, scale=False)(up9)
    conv9 = Conv2D(x, (3,3), activation='relu', padding='same')(up9)
    conv9 = BatchNormalization(axis=3, scale=False)(conv9)
    conv9 = Conv2D(x, (3,3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization(axis=3, scale=False)(conv9)

    out = Conv2D(1, (1,1), name="out", activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[out, level1, level2, level3, level4])
    
    

    return model
