# Code from L6-196 comes directly from PVBM python package 
# Script webpage: https://github.com/aim-lab/LUNet/blob/main/src/lunet/model.py
# Author: Johnathan Fhima

import tensorflow as tf

#Define the cross attention block used by lunet between the input x and input g
#input x being the one from the skip connection while input_g being the one from the previous layer
def attention(input_x,input_g):
  x_original_shape = input_x.shape
  g_original_shape = input_g.shape
  x = tf.keras.layers.Conv2D(input_x.shape[-1],2,2)(input_x)
  g = tf.keras.layers.Conv2D(input_x.shape[-1],1,1)(input_g)
  s = tf.keras.layers.Add()([x,g])
  s = tf.keras.layers.Activation('relu')(s)
  s = tf.keras.layers.Conv2D(1,1,1)(s)
  attention_coef = tf.keras.layers.Activation('sigmoid')(s)
  attention_coef = tf.keras.layers.UpSampling2D(size=x_original_shape[-2] // attention_coef.shape[-2], 
                                              interpolation='bilinear')(attention_coef) #we should use trilinear instead
  x = tf.keras.layers.Multiply()([input_x,attention_coef])
  return x,attention_coef

#Define the structured dropout conv block used by lunet which includes Convolution, dropout, batch norm and RELU
def structured_dropout_conv_block(inputs,n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization=True,scale = True,dropblock = True):
    x = tf.keras.layers.Conv2D(n_filters,kernel_size,stride,padding='same')(inputs)
    if dropblock:
        x  = tf.keras.layers.SpatialDropout2D(1-keep_prob)(x)
    else:
        if keep_prob !=1:
            x = tf.keras.layers.Dropout(1-keep_prob)(x)
    if with_batch_normalization :
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x
#Same as before but the convolution is dilated with a (2x2) rate
def structured_dropout_dilated_conv_block(inputs,n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization=True,scale = True,dropblock = True):
    x = tf.keras.layers.Conv2D(n_filters,kernel_size,stride,padding='same',dilation_rate=(2, 2))(inputs)
    if dropblock:
        x = tf.keras.layers.SpatialDropout2D(1-keep_prob)(x)
    else:
        if keep_prob !=1:
            x = tf.keras.layers.Dropout(1-keep_prob)(x)
    if with_batch_normalization :
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

#Deine a conv block as 2 structured blocks
def conv_block(inputs,n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization=True,scale = True,dropblock = True):
    x = structured_dropout_conv_block(inputs,n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
    x = structured_dropout_conv_block(x,n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
    return x

#Deine a dilated conv block as 2 dilated structured blocks
def dilated_conv_block(inputs,n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization=True,scale = True,dropblock = True):
    x = structured_dropout_dilated_conv_block(inputs,n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
    x = structured_dropout_dilated_conv_block(x,n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
    return x

#Defining lunet encoder as 7 Double Dilated Conv Blocks followed by max poolings
def encoder(inputs,init_n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale = True,dropblock = True):
    skip1 = tf.keras.layers.concatenate([
        conv_block(inputs,init_n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock),
        dilated_conv_block(inputs,init_n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
         ])
    x = tf.keras.layers.MaxPool2D(2,2)(skip1)
    skip2 = tf.keras.layers.concatenate([
        conv_block(x,init_n_filters*2,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock),
        dilated_conv_block(x,init_n_filters*2,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
         ])
    
    x = tf.keras.layers.MaxPool2D(2,2)(skip2)
    skip3 = tf.keras.layers.concatenate([
        conv_block(x,init_n_filters*4,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock),
        dilated_conv_block(x,init_n_filters*4,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
         ])
    x = tf.keras.layers.MaxPool2D(2,2)(skip3)
    
    skip4 = tf.keras.layers.concatenate([
        conv_block(x,init_n_filters*4,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock),
        dilated_conv_block(x,init_n_filters*4,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
         ])
    x = tf.keras.layers.MaxPool2D(2,2)(skip4)
    
    skip5 = tf.keras.layers.concatenate([
        conv_block(x,init_n_filters*4,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock),
        dilated_conv_block(x,init_n_filters*4,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
         ])
    x = tf.keras.layers.MaxPool2D(2,2)(skip5)
    
    skip6 = tf.keras.layers.concatenate([
        conv_block(x,init_n_filters*4,kernel_size ,stride ,1,block_size,with_batch_normalization,scale,False),
        dilated_conv_block(x,init_n_filters*4,kernel_size ,stride ,1,block_size,with_batch_normalization,scale,False)
         ])
    x = tf.keras.layers.MaxPool2D(2,2)(skip6)
    
    
    
    bottleneck = tf.keras.layers.concatenate([
        structured_dropout_conv_block(x,init_n_filters*4,kernel_size ,stride ,1,block_size,with_batch_normalization,scale,False),
        structured_dropout_dilated_conv_block(x,init_n_filters*4,kernel_size ,stride ,1,block_size,with_batch_normalization,scale,False)       
    ])
    
    return bottleneck,skip1,skip2,skip3,skip4,skip5,skip6

#Defining lunet decoder using attention gated skip connection, double dilated conv block and transpose convolution
def attention_decoder(bottleneck,skip1,skip2,skip3,skip4,skip5,skip6,init_n_filters,kernel_size ,transpose_stride,stride ,keep_prob,block_size,with_batch_normalization=True,scale = True,dropblock = False, n_layer_output = 2):
    bottleneck = tf.keras.layers.concatenate([
        structured_dropout_conv_block(bottleneck,init_n_filters*4,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock),
        structured_dropout_dilated_conv_block(bottleneck,init_n_filters*4,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)       
    ])
    
    att,coef = attention(skip6,bottleneck)
    x = tf.keras.layers.Conv2DTranspose(init_n_filters*4,kernel_size,transpose_stride,activation = 'relu',padding='same')(bottleneck)
    x = tf.keras.layers.concatenate([att,x])
    x = tf.keras.layers.concatenate([
        conv_block(x,init_n_filters*4,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock),
        dilated_conv_block(x,init_n_filters*4,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
         ])
    
    att,coef = attention(skip5,x)
    x = tf.keras.layers.Conv2DTranspose(init_n_filters*4,kernel_size,transpose_stride,activation = 'relu',padding='same')(x)
    x = tf.keras.layers.concatenate([att,x])
    x = tf.keras.layers.concatenate([
        conv_block(x,init_n_filters*4,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock),
        dilated_conv_block(x,init_n_filters*4,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
         ])
    
    att,coef = attention(skip4,x)
    x = tf.keras.layers.Conv2DTranspose(init_n_filters*4,kernel_size,transpose_stride,activation = 'relu',padding='same')(x)
    x = tf.keras.layers.concatenate([att,x])
    x = tf.keras.layers.concatenate([
        conv_block(x,init_n_filters*4,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock),
        dilated_conv_block(x,init_n_filters*4,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
         ])
    
    att,coef = attention(skip3,x)
    x = tf.keras.layers.Conv2DTranspose(init_n_filters*4,kernel_size,transpose_stride,activation = 'relu',padding='same')(x)
    x = tf.keras.layers.concatenate([att,x])
    
    x = tf.keras.layers.concatenate([
        conv_block(x,init_n_filters*4,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock),
        dilated_conv_block(x,init_n_filters*4,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
         ])
    att,coef = attention(skip2,x)
    x = tf.keras.layers.Conv2DTranspose(init_n_filters*2,kernel_size,transpose_stride,activation = 'relu',padding='same')(x)
    x = tf.keras.layers.concatenate([att,x])
    x = tf.keras.layers.concatenate([
        conv_block(x,init_n_filters*2,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock),
        dilated_conv_block(x,init_n_filters*2,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
         ])
    att,coef = attention(skip1,x)
    x = tf.keras.layers.Conv2DTranspose(init_n_filters,kernel_size,transpose_stride,activation = 'relu',padding='same')(x)
    x = tf.keras.layers.concatenate([att,x])
    last0 =tf.keras.layers.concatenate([
        conv_block(x,init_n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock),
        dilated_conv_block(x,init_n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
         ])
    last0 =tf.keras.layers.concatenate([
        conv_block(last0,init_n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock),
        dilated_conv_block(last0,init_n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
         ]) + last0
         
    last0 =tf.keras.layers.concatenate([
        conv_block(last0,init_n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock),
        dilated_conv_block(last0,init_n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
         ]) + last0
         
    last =tf.keras.layers.concatenate([
        conv_block(last0,init_n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock),
        dilated_conv_block(last0,init_n_filters,kernel_size ,stride ,keep_prob,block_size,with_batch_normalization,scale,dropblock)
         ]) + last0
    
    x = tf.keras.layers.Conv2D( n_layer_output ,1,1,padding='same')(last)
    x = tf.keras.layers.concatenate([
        x,tf.maximum(x[:,:,:,:1],x[:,:,:,1:])
    ])
    return x,coef,last

## A small function that takes a tf.keras.layers.Input and some parameter and return the ouput layer of lunet
## as well as some attention, last layer and bottleneck
def lunet(inputs,init_n_filters, kernel_size,transpose_stride, stride, keep_prob,block_size,with_batch_normalization = True,scale = True,dropblock = False, n_layer_output = 2):
  bottleneck,skip1,skip2,skip3,skip4,skip5,skip6 =encoder(inputs,init_n_filters,kernel_size,stride,keep_prob,block_size,
                                                             with_batch_normalization,scale,dropblock)
  output,att,last = attention_decoder(bottleneck,skip1,skip2,skip3,skip4,skip5,skip6,init_n_filters,kernel_size ,transpose_stride,stride,keep_prob,block_size,with_batch_normalization,scale,dropblock, n_layer_output)
  return output,att,last,bottleneck

## A function that call the previous lunet function, mainly to instantiate lunet
def build_lunet(inputs,init_n_filters, kernel_size,transpose_stride, stride, keep_prob,block_size,with_batch_normalization = True,scale = True,dropblock = False, n_layer_output = 2):
  output,attention_coef,last,bottleneck = lunet(inputs = inputs,init_n_filters = init_n_filters,kernel_size = kernel_size,transpose_stride = transpose_stride,stride =  stride,keep_prob =  keep_prob, block_size = block_size,with_batch_normalization = with_batch_normalization,scale = scale,dropblock = dropblock,n_layer_output =  n_layer_output )
  model = tf.keras.Model(inputs = inputs,outputs = output)
  attention = tf.keras.Model(inputs = inputs,outputs = attention_coef)
  ds_model = tf.keras.Model(inputs=inputs, outputs=[output,last])
  return model,attention,bottleneck