import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import math

from utils.models_reg import SimpleA

def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    if freq_init == "random":
        # the frequence we use for each block, alpha in ICLR paper
        # freq_list shape: (frequency_num)
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":
        # freq_list = []
        # for cur_freq in range(frequency_num):
        #     base = 1.0/(np.power(max_radius, cur_freq*1.0/(frequency_num-1)))
        #     freq_list.append(base)

        # freq_list = np.asarray(freq_list)

        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) /
          (frequency_num*1.0 - 1))

        timescales = min_radius * np.exp(
            np.arange(frequency_num).astype(float) * log_timescale_increment)

        freq_list = 1.0/timescales

    return freq_list

def sample_gumbel(shape, eps=1e-20):
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)

def gumbel_softmax(logits, temperature, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(logits.shape)
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)

    if hard:
        k = logits.shape[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)),
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y
        

class SpatialModel(tf.keras.Model):
    '''
    Wraper for spatial embedding and a predictor model
    '''
    def __init__(self,args,image_model, fusion='concat'):
        assert fusion in ['concat','soft','hard']
        super(SpatialModel, self).__init__()
        self.args = args
        self.fusion = fusion
        self.lambda_reg = self.args.lambda_reg
        self.is_dropout = False
        self.image_model = image_model
        self.space_model = TheoryGridCellSpatialRelationEncoder(spa_embed_dim=64)

        if self.fusion != 'concat':
            self.attention_layer = layers.Conv2D(64+256,1, activation=None,padding='same', use_bias=True)


    def attention_fusion(self, features):

        if self.fusion == 'soft':
            at = self.attention_layer(features)
            at = tf.nn.sigmoid(at)
        elif self.fusion == 'hard':
            at = self.attention_layer(features)
            at = gumbel_softmax(at,temperature=10,hard=True)

        features_out = at * features    
        return features_out




    @tf.function
    def call(self,sample, is_training=True):

        img, coords = sample['feat_l'][...,0:11], sample['feat_l'][...,11:]

        img_feat = self.image_model.get_deep_features(img, is_training)

        space_feat = self.space_model(coords,is_training)

        last = tf.concat((img_feat,space_feat), axis=-1)
        
        if self.fusion != 'concat':    
            last = self.attention_fusion(last)


        if self.is_dropout:
            last = self.image_model.dropout_last(last, training=True)

        return_dict = {}
        if self.lambda_reg > 0.0:
            return_dict['reg'] = self.image_model.conv_reg(last)
        if self.lambda_reg < 1.0:
            return_dict['sem'] = self.image_model.conv_sem(last)

        return_dict['last'] = last
        
        return return_dict


class SpatialEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, freq_init, frequency_num,max_radius,min_radius):

        self.freq_init = freq_init
        self.frequency_num = frequency_num
        self.max_radius = max_radius
        self.min_radius = min_radius

        freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)
        freq_list = freq_list.reshape(1,1,1,1,self.frequency_num)
        self.freq_list = tf.convert_to_tensor(freq_list, dtype=tf.float32)


        self.unit_vectors = tf.convert_to_tensor(np.array([[1.0, 0.0],
                [-1.0/2.0, math.sqrt(3)/2.0],
                [-1.0/2.0, -math.sqrt(3)/2.0]
                ]).T.reshape(1,1,2,3), dtype=tf.float32)


    def call(self, coords):
        # b, w,h, c = tf.shape(coords)
        b, w,h, c = coords.get_shape().as_list()

        angle_mat = tf.nn.conv2d(coords, self.unit_vectors, strides=[1, 1, 1, 1], padding='VALID')
        angle_mat = tf.expand_dims(angle_mat,-1) * self.freq_list

        spr_embeds = tf.concat((tf.sin(angle_mat),tf.sin(angle_mat)),axis=-2)

        spr_embeds = tf.reshape(spr_embeds,shape=(b,w,h,-1))

        return spr_embeds

"""
The theory based Grid cell spatial relation encoder, 
See https://openreview.net/forum?id=Syx0Mh05YQ
Learning Grid Cells as Vector Representation of Self-Position Coupled with Matrix Representation of Self-Motion
"""
class TheoryGridCellSpatialRelationEncoder(tf.keras.layers.Layer):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """
    def __init__(self, spa_embed_dim, coord_dim = 2, frequency_num = 16, 
        max_radius = 10000,  min_radius = 1000, freq_init = "geometric", device = "cuda"):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(TheoryGridCellSpatialRelationEncoder, self).__init__()
        self.frequency_num = frequency_num
        self.coord_dim = coord_dim 
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.spa_embed_dim = spa_embed_dim
        self.freq_init = freq_init

        # the frequence we use for each block, alpha in ICLR paper
        # self.embedd_layer = SpatialModel(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)
        # freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)
        # freq_list = freq_list.reshape(1,1,1,1,self.frequency_num)
        # self.freq_list = tf.convert_to_tensor(freq_list, dtype=tf.float32)
        # self.cal_freq_list()
        # self.cal_freq_mat()

        # there unit vectors which is 120 degree apart from each other
        # self.unit_vec1 = np.asarray([1.0, 0.0])                        # 0
        # self.unit_vec2 = np.asarray([-1.0/2.0, math.sqrt(3)/2.0])      # 120 degree
        # self.unit_vec3 = np.asarray([-1.0/2.0, -math.sqrt(3)/2.0])     # 240 degree

        freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)
        freq_list = freq_list.reshape(1,1,1,1,self.frequency_num)
        self.freq_list = tf.convert_to_tensor(freq_list, dtype=tf.float32)


        self.unit_vectors = tf.convert_to_tensor(
                np.array([[1.0, 0.0],
                [-1.0/2.0, math.sqrt(3)/2.0],
                [-1.0/2.0, -math.sqrt(3)/2.0]
                ]).T.reshape(1,1,2,3), dtype=tf.float32)



        self.input_embed_dim = int(6 * self.frequency_num)

        self.ffn = tf.keras.Sequential()(tf.keras.layers.Conv2D(self.spa_embed_dim,1, activation='relu',padding='same', use_bias=False),
                                         tf.keras.layers.BatchNormalization())

        
    #@tf.function
    def call(self, coords, is_training):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        # spr_embeds = self.make_input_embeds(coords)
        # b, w,h, c = tf.shape(coords)
        b, w,h, c = coords.get_shape().as_list()
        angle_mat = tf.nn.conv2d(coords, self.unit_vectors, strides=[1, 1, 1, 1], padding='VALID')
        angle_mat = tf.expand_dims(angle_mat,-1) * self.freq_list

        spr_embeds = tf.concat((tf.sin(angle_mat),tf.sin(angle_mat)),axis=-2)

        spr_embeds = tf.reshape(spr_embeds,shape=(b,w,h,-1))



        # spr_embeds: (batch_size, num_context_pt, input_embed_dim)
        #spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)
        #spr_embeds = tf.convert_to_tensor(spr_embeds)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))

        # if self.use_post_mat:
        #     sprenc = self.post_linear_1(spr_embeds)
        #     sprenc = self.post_linear_2(self.dropout(sprenc))
        #     sprenc = self.f_act(self.dropout(sprenc))
        # else:
        #     sprenc = self.post_linear(spr_embeds)
        #     sprenc = self.f_act(self.dropout(sprenc))
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds

