from tensorflow import keras
import tensorflow as tf
# import tensorflow_addons as tfa
import tensorflow.keras.backend as K
# import tensorflow_addons as tfa

class InstanceNormalization(tf.keras.models.Model):
    def __init__(self,epsilon=1e-5,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):  
        assert len(input_shape)==4
        shape = (input_shape[-1],)
        self.built = True
        # self.pool = tf.keras.layers.AveragePooling2D(pool_size = (input_shape[1],input_shape[2]),padding='valid')



    def call(self,inputs):
        if inputs.shape[1] == None:
            return inputs
        else:
            mean = tf.keras.layers.AveragePooling2D(pool_size=(inputs.shape[1],inputs.shape[2]))(inputs)
            # print(type(mean),mean)
            mean = tf.keras.layers.Reshape(target_shape=(1,1,inputs.shape[-1]))(mean)
            variance = tf.keras.layers.AveragePooling2D(pool_size=(inputs.shape[1],inputs.shape[2]))((inputs-mean)*(inputs-mean))*inputs.shape[1]*inputs.shape[2]
            variance =tf.keras.layers.Reshape(target_shape=(1,1,inputs.shape[-1]))(variance)
            outputs = (inputs - mean) / (tf.exp(0.5*tf.math.log(variance + self.epsilon))) 

        return outputs

def res_block(input, filters, kernel_size=(3, 3), strides=(1, 1), normlization='InstanceNormalization'):
    initializer = tf.random_normal_initializer(0., 0.02)
    ## step size 1 , keep the feature size
    x = keras.layers.Conv2D(filters=filters,kernel_size=kernel_size, strides=strides, padding='same',kernel_initializer=initializer, use_bias=False)(input)    
    if normlization == 'batcnorm':
        x = keras.layers.BatchNormalization(momentum=0.0)(x,training=True)
    elif normlization == 'InstanceNormalization':
        x = InstanceNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,padding='same',kernel_initializer=initializer, use_bias=False)(x)    
    if normlization == 'batcnorm':
        x = keras.layers.BatchNormalization(momentum=0.0)(x,training=True)
    elif normlization == 'InstanceNormalization':
        x = InstanceNormalization()(x)
    out = keras.layers.Add()([input, x])#residual block
    out = keras.layers.Activation('relu')(out)
    return out

def Conv_block(input, filters, kernel_size=(3, 3), strides=(2, 2),iftanh=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    if iftanh:
        x = keras.layers.Conv2D(filters=filters,kernel_size=kernel_size, strides=strides, padding='same',kernel_initializer=initializer,activation='tanh', use_bias=False)(input) 
    else:
        x = keras.layers.Conv2D(filters=filters,kernel_size=kernel_size, strides=strides, padding='same',kernel_initializer=initializer, use_bias=False)(input)
        x = keras.layers.Activation('relu')(x)
    return x

def deconv_Relu(input,filters):
    x = keras.layers.Conv2DTranspose(filters,4,2,'same')(input)
    x = keras.layers.Activation('relu')(x)
    return x

def resnet(x,filters,num_block):
    for _ in tf.range(num_block):
        x = res_block(x,filters)
    return x

def model1(x,c,num_block):
    x1 = Conv_block(x,filters=c,strides=(1,1))
    x2 = Conv_block(x1,2*c)
    x3 = Conv_block(x2,4*c)
    x4 = Conv_block(x3,8*c)
    resout = resnet(x4,8*c,num_block)
    
    y1_0 = keras.layers.concatenate([resout,x4],3)
    y1_0 = deconv_Relu(y1_0,c*4)

    y1_1 = deconv_Relu(y1_0,c*2)
    y1_2 = deconv_Relu(y1_1,c)

    y_out1 = Conv_block(y1_2,3,strides=(1,1),iftanh=True)
    return y_out1

def model2(x,x2,c,num_block):

    x1_0 = keras.layers.concatenate([x2,x-x2],3)
    x1_0 = Conv_block(x1_0,c,strides=(1,1))
    x1_1 = Conv_block(x1_0,c*2)
    x1_2 = Conv_block(x1_1,c*4)
    x1_3 = Conv_block(x1_2,c*8)

    x_merge = x1_3
    x_merge = Conv_block(x_merge,c*8,strides=(1,1))
    x_resnet = resnet(x_merge,c*8,num_block=num_block)


    y1_0 = deconv_Relu(keras.layers.concatenate([x1_3,x_resnet],3),c*4)
    y1_1 = deconv_Relu(keras.layers.concatenate([x1_2,    y1_0],3),c*2)
    y1_2 = deconv_Relu(keras.layers.concatenate([x1_1,    y1_1],3),c*1) 

    gt = Conv_block(y1_2,3,strides=(1,1),iftanh=True)

    return(gt)

def xnet(c1=16,c2=24,res_num_block=9):
    inputs = keras.layers.Input(name='blur_image' ,shape=(None,None,3))
    y_out1=model1(inputs,c1,res_num_block)
    output = model2(inputs,y_out1,c2,res_num_block)
    model = keras.models.Model(inputs = inputs,outputs = [output,y_out1], name='Discriminator')
    return model








if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    # inputs = keras.layers.Input(shape=(1024,1408,3))
    model = xnet()
    model.summary()