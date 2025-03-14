import tensorflow as tf
import numpy as np
tf.keras.utils.set_random_seed(1234)

class Mish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.tanh(tf.nn.softplus(inputs))
    
def get_activation(name):
    if name == "mish":
        return Mish()
    elif name == "relu":
        return tf.keras.layers.ReLU()
    elif name == "gelu":
        return tf.keras.layers.Activation(tf.nn.gelu)
    else:
        raise ValueError(f"Activation '{name}' is not supported.")

class Att_Res_UNet():
    def __init__(self, list_predictors, list_targets, dim, cropped_dim, batch_size, n_filters, activation, kernel_initializer, batch_norm, pooling_type, dropout):
        self.list_predictors = list_predictors
        self.list_targets = list_targets
        self.patch_dim = tuple([d for d in cropped_dim])
        self.batch_size = batch_size
        self.n_filters = n_filters
        self.activation = get_activation(activation)
        self.kernel_initializer = kernel_initializer
        self.batch_norm = batch_norm
        self.pooling_type = pooling_type
        self.dropout = dropout
        if list_predictors[0] == 'dp_constrained':
            self.n_predictors = 51
            self.n_targets = 50
        else:
            self.n_predictors = len(list_predictors)
            self.n_targets = len(list_targets)
        self.weight_regularizer = None ## A implementer eventuellement ...
        self.repeat_elem_counter = -1

    def repeat_elem(self, tensor, rep, name=None):
        self.repeat_elem_counter += 1
        # Construct the unique name
        unique_name = f'{name}_{self.repeat_elem_counter}'
        return tf.keras.layers.Lambda(lambda x, repnum: tf.keras.backend.repeat_elements(x, repnum, axis = 3), arguments = {'repnum': rep}, name=unique_name)(tensor)

    #
    def residual_conv_block(self, x, n_filters, padding = "same"):
        conv = tf.keras.layers.Conv2D(n_filters, kernel_size = (3,3), padding = padding, kernel_initializer = self.kernel_initializer)(x)
        if self.batch_norm == True:
            conv = tf.keras.layers.BatchNormalization(axis = 3)(conv)
        conv = tf.keras.layers.Activation(self.activation)(conv)
        #
        conv = tf.keras.layers.Conv2D(n_filters, kernel_size = (3,3), padding = padding, kernel_initializer = self.kernel_initializer)(conv)
        if self.batch_norm == True:
            conv = tf.keras.layers.BatchNormalization(axis = 3)(conv)
        #
        shortcut = tf.keras.layers.Conv2D(n_filters, kernel_size = (1,1), padding = padding)(x)
        if self.batch_norm == True:
            shortcut = tf.keras.layers.BatchNormalization(axis = 3)(shortcut)
        #
        res_path = tf.keras.layers.add([shortcut, conv])
        res_path = tf.keras.layers.Activation(self.activation)(res_path)
        #
        return(res_path)
    #
    def downsample_block(self, x, n_filters, pool_size = (2,2), strides = 2):
        f = self.residual_conv_block(x, n_filters)
        #
        if self.pooling_type == "Max":
            p = tf.keras.layers.MaxPool2D(pool_size = pool_size, strides = strides)(f)
        elif self.pooling_type == "Average":
            p = tf.keras.layers.AveragePooling2D(pool_size = pool_size, strides = strides)(f)
        #
        p = tf.keras.layers.Dropout(self.dropout)(p)
        return(f, p)
    def partial_downsample_block(self, x, n_filters, pool_size = (2,2), strides = 2):
        f = self.partial_residual_conv_block(x, n_filters)
        #
        if self.pooling_type == "Max":
            p = tf.keras.layers.MaxPool2D(pool_size = pool_size, strides = strides)(f)
            self.mask = tf.keras.layers.MaxPool2D(pool_size = pool_size, strides = strides)(self.mask)
        elif self.pooling_type == "Average":
            p = tf.keras.layers.AveragePooling2D(pool_size = pool_size, strides = strides)(f)
            self.mask = tf.keras.layers.AveragePooling2D(pool_size = pool_size, strides = strides)(self.mask)
        #
        p = tf.keras.layers.Dropout(self.dropout)(p)
        return(f, p)  
    #
    def upsample_block(self, x, conv_features, n_filters, kernel_size = (2,2), strides = 2, padding = "same"):
        up_att = tf.keras.layers.UpSampling2D(size = (2, 2), data_format = "channels_last")(x)
        up_att = tf.keras.layers.concatenate([up_att, conv_features], axis = 3)
        up_conv = self.residual_conv_block(up_att, n_filters)
        return(up_conv)
    #
    def upsample__transpose_block(self, x, conv_features, n_filters, kernel_size=(2, 2), strides=2, padding="same"):
        # Use Conv2DTranspose instead of UpSampling2D
        up_att = tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        # Concatenate with skip connection features
        up_att = tf.keras.layers.concatenate([up_att, conv_features], axis=3)
        
        # Pass through the residual convolution block
        up_conv = self.residual_conv_block(up_att, n_filters)
    
        return up_conv
    #
    def make_unet_model(self): 
        inputs = tf.keras.layers.Input(shape = (*self.patch_dim, self.n_predictors))
        
        # Encoder (downsample)
        f1, p1 = self.downsample_block(inputs, self.n_filters[0])
        f2, p2 = self.downsample_block(p1, self.n_filters[1])
        f3, p3 = self.downsample_block(p2, self.n_filters[2])
        f4, p4 = self.downsample_block(p3, self.n_filters[3])
        f5, p5 = self.downsample_block(p4, self.n_filters[4])
        # Bottleneck
        u5 = self.residual_conv_block(p5, self.n_filters[5])
        # Decoder (upsample)
        u4 = self.upsample_block(u5, f5, self.n_filters[4])
        u3 = self.upsample_block(u4, f4, self.n_filters[3])
        u2 = self.upsample_block(u3, f3, self.n_filters[2])
        u1 = self.upsample_block(u2, f2, self.n_filters[1])
        u0 = self.upsample_block(u1, f1, self.n_filters[0])
        # outputs
        residuals = tf.keras.layers.Conv2D(self.n_targets, (1, 1), padding = "same", activation = "linear", dtype = tf.float32, name = "residuals_output")(u0)
        if self.list_predictors[0] == 'dp_constrained':
            hr_output = tf.keras.layers.Add(name="HR_output")([inputs[:,:,:,0:50], residuals])
            positive_output = tf.keras.layers.ReLU()(hr_output)
            
            input_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=3, keepdims=True))(inputs[:,:,:,0:50])
            output_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=3, keepdims=True))(positive_output)
            
            scale_factor = tf.keras.layers.Lambda(lambda x: x[0] / x[1])([input_sum, output_sum])
            constrained_output = tf.keras.layers.Multiply()([positive_output, scale_factor])
        
            unet_model = tf.keras.Model(inputs, constrained_output, name = "Res-att-U-Net")
        else:
            #hr_output = tf.keras.layers.Add(name="HR_output")([inputs, residuals])
            #unet_model = tf.keras.Model(inputs, hr_output, name = "Res-att-U-Net")
            inputs_channels = inputs[:, :, :, 0:self.n_targets]
            hr_output = tf.keras.layers.Add(name="HR_output")([inputs_channels, residuals])
            unet_model = tf.keras.Model(inputs, hr_output, name = "Res-att-U-Net")
        #
        return(unet_model)