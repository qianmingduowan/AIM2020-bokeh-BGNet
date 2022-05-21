from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import copy
from keras import regularizers
from keras.constraints import min_max_norm
from wnet_qn import InstanceNormalization
class NLayerDiscriminator(keras.models.Model):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        sequence = [
            keras.layers.Conv2D(ndf, kw, strides=2, padding="same", kernel_regularizer = regularizers.l2(), kernel_constraint=min_max_norm(-0.01, 0.01)),
            InstanceNormalization(),
            keras.layers.LeakyReLU(alpha=0.2, trainable=True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                keras.layers.Conv2D(ndf * nf_mult, kw, strides=2, padding="same", kernel_regularizer = regularizers.l2(), kernel_constraint=min_max_norm(-0.01, 0.01), bias_constraint=min_max_norm(-0.01, 0.01)),
                InstanceNormalization(),
                keras.layers.LeakyReLU(alpha=0.2, trainable=True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            keras.layers.Conv2D(ndf * nf_mult, kw, strides=1, padding="same", kernel_regularizer = regularizers.l2(), kernel_constraint=min_max_norm(-0.01, 0.01), bias_constraint=min_max_norm(-0.01, 0.01)),
            InstanceNormalization(),
            keras.layers.LeakyReLU(alpha=0.2, trainable=True)
        ]

        sequence += [keras.layers.Conv2D(ndf * nf_mult, 1, strides=1, padding="same", kernel_regularizer = regularizers.l2(), kernel_constraint=min_max_norm(-0.01, 0.01), bias_constraint=min_max_norm(-0.01, 0.01))]
        self.model = keras.models.Sequential(sequence)

    def __call__(self, inputs):
        return self.model(inputs)

def get_fullD():
    model_d = NLayerDiscriminator(n_layers=5)
    return model_d

def get_patchD():
    model_d = NLayerDiscriminator(n_layers=3)
    return model_d

def get_doublegan_D():
    patch_gan = get_patchD()
    full_gan = get_fullD()
    model_d = {'patch': patch_gan,
               'full': full_gan}
    return model_d

class DiscLossWGANGP(keras.losses.Loss):
    def __init__(self):
        super(DiscLossWGANGP, self).__init__()
        self.LAMBDA = 10

    def get_g_loss(self, net, fakeB, realB):
        # First, G(A) should fake the discriminator
        self.D_fake = net(fakeB)
        return -tf.reduce_mean(self.D_fake)

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        n = real_data.shape[0]

        alpha = tf.random.uniform((n, 1, 1, 1))
        alpha = tf.broadcast_to(alpha, real_data.shape)

        interplates = alpha * real_data + (1 - alpha) * fake_data

        # 在梯度环境中计算D 对插值样本的梯度
        with tf.GradientTape() as tape:
            tape.watch([interplates])  # 加入梯度观察列表
            d_interplote_logits = netD(interplates)
        grads = tape.gradient(d_interplote_logits, interplates)
  
        # 计算每个样本的梯度的范数:[b, h, w, c] => [b, -1]
        grads = tf.reshape(grads, [grads.shape[0], -1])
        gp = tf.norm(grads, axis=1)  # [b]
        # 计算梯度惩罚项
        gp = tf.reduce_mean((gp - 1.) ** 2)   
        return gp

    def get_loss(self, net, fakeB, realB):
        self.D_fake = net(fakeB)
        self.D_fake = tf.reduce_mean(self.D_fake)

        # Real
        self.D_real = net(realB)
        self.D_real = tf.reduce_mean(self.D_real)
        # Combined loss
        self.loss_D = self.D_fake - self.D_real
        gradient_penalty = self.calc_gradient_penalty(net, realB, fakeB)
        return self.loss_D + gradient_penalty
    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)

class DoubleGAN(keras.losses.Loss):
    def __init__(self, model_d, criterion):
        self.criterion = criterion
        self.patch_d = model_d['patch']
        self.full_d = model_d['full']
        self.full_criterion = copy.deepcopy(criterion)
    
    def loss_d(self, pred, gt):
        return (self.criterion(self.patch_d, pred, gt) + self.full_criterion(self.full_d, pred, gt)) / 2

    def loss_g(self, pred, gt):
        return (self.criterion.get_g_loss(self.patch_d, pred, gt) + self.full_criterion.get_g_loss(self.full_d, pred, gt)) / 2
    
    def get_trainable_variables(self):
        return self.patch_d.trainable_variables+self.full_d.trainable_variables


