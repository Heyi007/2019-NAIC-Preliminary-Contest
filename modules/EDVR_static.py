import tensorflow as tf
import functools
import numpy as np
from tensorflow import keras

from deformable_conv import deform_layer
from modules import module_util


class Predeblur_ResNet_Pyramid(object):
    def __init__(self, nf=128, HR_in=False):
        """
        :param nf: number of filters
        :param HR_in: True if the inputs are high spatial size
        """
        self.HR_in = True if HR_in else False
        if self.HR_in:
            self.conv_first_1 = keras.layers.Conv2D(nf, (3, 3), strides=(1, 1), padding="same", use_bias=True)
            self.conv_first_2 = keras.layers.Conv2D(nf, (3, 3), strides=(2, 2), padding="same", use_bias=True)
            self.conv_first_3 = keras.layers.Conv2D(nf, (3, 3), strides=(2, 2), padding="same", use_bias=True)
        else:
            self.conv_first = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same", use_bias=True)
        self.RB_L1_1 = module_util.ResidualBlock_noBN(nf=nf)
        self.RB_L1_2 = module_util.ResidualBlock_noBN(nf=nf)
        self.RB_L1_3 = module_util.ResidualBlock_noBN(nf=nf)
        self.RB_L1_4 = module_util.ResidualBlock_noBN(nf=nf)
        self.RB_L1_5 = module_util.ResidualBlock_noBN(nf=nf)
        self.RB_L2_1 = module_util.ResidualBlock_noBN(nf=nf)
        self.RB_L2_2 = module_util.ResidualBlock_noBN(nf=nf)
        self.RB_L3_1 = module_util.ResidualBlock_noBN(nf=nf)
        self.deblur_L2_conv = keras.layers.Conv2D(nf, (3, 3), strides=(2, 2), padding="same")
        self.deblur_L3_conv = keras.layers.Conv2D(nf, (3, 3), strides=(2, 2), padding="same")
        self.lrelu = keras.layers.LeakyReLU(0.1)

    def __call__(self, x):
        if self.HR_in:
            L1_fea = self.lrelu(self.conv_first_1(x))
            L1_fea = self.lrelu(self.conv_first_2(L1_fea))
            L1_fea = self.lrelu(self.conv_first_3(L1_fea))
        else:
            L1_fea = self.lrelu(self.conv_first(x))
        L2_fea = self.lrelu(self.deblur_L2_conv(L1_fea))
        L3_fea = self.lrelu(self.deblur_L3_conv(L2_fea))
        L3_fea = self.RB_L3_1(L3_fea)
        L3_fea_shape = tf.shape(L3_fea)
        L3_fea = tf.image.resize_images(L3_fea, (2 * L3_fea_shape[1], 2 * L3_fea_shape[2]))

        L2_fea = self.RB_L2_1(L2_fea) + L3_fea
        L2_fea = self.RB_L2_2(L2_fea)
        L2_fea_shape = tf.shape(L2_fea)
        L2_fea = tf.image.resize_images(L2_fea, (2 * L2_fea_shape[1], 2 * L2_fea_shape[2]))
        L1_fea = self.RB_L1_2(self.RB_L1_1(L1_fea)) + L2_fea
        out = self.RB_L1_5(self.RB_L1_4(self.RB_L1_3(L1_fea)))
        return out

class PCD_Align(object):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
        with 3 pyramid levels.
    '''
    def __init__(self, nf=64, groups=8):
        # L3: level3, 1/4 spatial size
        self.L3_offset_conv1 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")  # concat for diff
        self.L3_offset_conv2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        self.L3_dcnpack = deform_layer.DCN_seq(nf)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same")  # concat for diff
        self.L2_offset_conv2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same")  # concat for offset
        self.L2_offset_conv3 = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same")
        self.L2_dcnpack = deform_layer.DCN_seq(nf)
        self.L2_fea_conv = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same") # concat for diff
        self.L1_offset_conv2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same") # concat for offset
        self.L1_offset_conv3 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        self.L1_dcnpack = deform_layer.DCN_seq(nf)
        self.L1_fea_conv = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        # Cascading DCN
        self.cas_offset_conv1 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        self.cas_offset_conv2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        self.cas_dcnpack = deform_layer.DCN_seq(nf)
        self.lrelu = keras.layers.LeakyReLU(0.1)

    def __call__(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,H,W,C] features
        '''
        # L3
        L3_offset = tf.concat([nbr_fea_l[2], ref_fea_l[2]], axis=3)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset))
        # L2
        L2_offset = tf.concat([nbr_fea_l[1], ref_fea_l[1]], axis=3)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset_shape = tf.shape(L3_offset)
        L3_offset = tf.image.resize_images(L3_offset, (2*L3_offset_shape[1], 2*L3_offset_shape[2]))
        L2_offset = self.lrelu(self.L2_offset_conv2(tf.concat([L2_offset, L3_offset * 2], axis=3)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset)
        L3_fea_shape = tf.shape(L3_fea)
        L3_fea = tf.image.resize_images(L3_fea, (2*L3_fea_shape[1], 2*L3_fea_shape[2]))
        L2_fea = self.lrelu(self.L2_fea_conv(tf.concat([L2_fea, L3_fea], axis=3)))
        # L1
        L1_offset = tf.concat([nbr_fea_l[0], ref_fea_l[0]], axis=3)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset_shape = tf.shape(L2_offset)
        L2_offset = tf.image.resize_images(L2_offset, (2 * L2_offset_shape[1], 2 * L2_offset_shape[2]))
        L1_offset = self.lrelu(self.L1_offset_conv2(tf.concat([L1_offset, L2_offset*2], axis=3)))
        L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset)
        L2_fea_shape = tf.shape(L2_fea)
        L2_fea = tf.image.resize_images(L2_fea, (2 * L2_fea_shape[1], 2 * L2_fea_shape[2]))
        L1_fea = self.L1_fea_conv(tf.concat([L1_fea, L2_fea], axis=3))
        # Cascading
        offset = tf.concat([L1_fea, ref_fea_l[0]], axis=3)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack(L1_fea, offset))
        return L1_fea

class TSA_Fusion(object):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 3 pyramid levels.
    '''
    def __init__(self, nf=64, nframes=5, center=2):
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same")
        self.tAtt_2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same")
        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        # spatial attention (after fusion conv)
        self.sAtt_1 = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        self.maxpool = keras.layers.MaxPool2D((3, 3), (2, 2), padding="same")
        self.avgpool = keras.layers.AveragePooling2D((3, 3), (2, 2), padding="same")
        self.sAtt_2 = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        self.sAtt_3 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        self.sAtt_4 = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        self.sAtt_5 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        self.sAtt_L1 = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        self.sAtt_L2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        self.sAtt_L3 = keras.layers.Conv2D(nf, (3, 3), (1, 1), padding="same")
        self.sAtt_add_1 = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        self.sAtt_add_2 = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        self.lrelu = keras.layers.LeakyReLU(0.1)


    def __call__(self, aligned_fea):
        aligned_fea_shape = tf.shape(aligned_fea)
        B = aligned_fea_shape[0]
        N = aligned_fea_shape[1]
        H = aligned_fea_shape[2]
        W = aligned_fea_shape[3]
        C = aligned_fea_shape[4]
        
        #
        aligned_temp = aligned_fea[:, self.center, :, :, :]
        emb_ref = self.tAtt_2(aligned_temp)

        # ---------BUG------
        emb = tf.reshape(self.tAtt_1(tf.reshape(aligned_fea, [-1, H, W, C])), [B, N, H, W, -1])
        # temp_1 = tf.reshape(aligned_fea, [-1, H, W, C])
        # temp_2 = self.tAtt_1(temp_1)
        # emb = tf.reshape(temp_2, [B, N, H, W, -1])

        # ---------

        cor_l = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        def cond(i, N, input, arr):
            return tf.less(i, N)
        def body(i, N, input, arr):
            emb_nbr = input[:, i, :, :, :]
            cor_tmp = tf.reduce_sum(emb_nbr * emb_ref, axis=3) # B, H, W
            arr = arr.write(i, cor_tmp)
            i = tf.add(i, 1)
            return i, N, input, arr
        _, _, _, cor_l = tf.while_loop(cond, body, [0, N, emb, cor_l])
        cor_l = cor_l.stack()
        cor_l = tf.transpose(cor_l, [1, 0, 2, 3])
        cor_prob = tf.sigmoid(cor_l)  # B, N, H, W
        cor_prob = tf.expand_dims(cor_prob, axis=4)
        cor_prob = tf.tile(cor_prob, [1, 1, 1, 1, C])
        cor_prob = tf.reshape(cor_prob, [B, H, W, -1])
        aligned_fea = tf.reshape(aligned_fea, [B, H, W, -1]) * cor_prob
        ### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))
        ### spatial attention
        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(tf.concat([att_max, att_avg], axis=3)))
        ### pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(tf.concat([att_max, att_avg], axis=3)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L_shape = tf.shape(att_L)
        att_L = tf.image.resize_images(att_L, [2 * att_L_shape[1], 2 * att_L_shape[2]])
        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att_shape = tf.shape(att)
        att = tf.image.resize_images(att, [2 * att_shape[1], 2 * att_shape[2]])
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = tf.sigmoid(att)
        fea = fea * att * 2 + att_add
        return fea

class EDVR_Core(object):
    def __init__(self, nf=64, nframes=7, groups=8, front_RBs=5, back_RBs=10,
                 center=None, predeblur=False, HR_in=False, w_TSA=True):
        self.nframes = nframes
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(module_util.ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)
        if self.is_predeblur:
            self.pre_deblur = Predeblur_ResNet_Pyramid(nf=nf, HR_in=self.HR_in)
            self.conv_1x1 = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        else:
            if self.HR_in:
                self.conv_first_1 = keras.layers.Conv2D(nf, (3, 3), strides=(1, 1), padding="same", use_bias=True)
                self.conv_first_2 = keras.layers.Conv2D(nf, (3, 3), strides=(2, 2), padding="same", use_bias=True)
                self.conv_first_3 = keras.layers.Conv2D(nf, (3, 3), strides=(2, 2), padding="same", use_bias=True)
            else:
                self.conv_first = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same", use_bias=True)
        self.feature_extraction = module_util.Module(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = keras.layers.Conv2D(nf, (3, 3), (2, 2), "same")
        self.fea_L2_conv2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same")
        self.fea_L3_conv1 = keras.layers.Conv2D(nf, (3, 3), (2, 2), "same")
        self.fea_L3_conv2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same")
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        if self.w_TSA:
            self.tsa_fusion = TSA_Fusion(nf=nf, nframes=nf, center=self.center)
        else:
            self.tsa_fusion = keras.layers.Conv2D(nf, (1, 1), (1, 1))
        #### reconstruction
        self.recon_trunk = module_util.Module(ResidualBlock_noBN_f, back_RBs)
        #### upsampling
        self.upconv1 = keras.layers.Conv2D(nf * 4, (3, 3), (1, 1), "same")
        self.upconv2 = keras.layers.Conv2D(64 * 4, (3, 3), (1, 1), "same")
        self.pixel_shuffle = lambda x:tf.nn.depth_to_space(x, 2)
        self.HRconv = keras.layers.Conv2D(64, (3, 3), (1, 1), "same")
        self.conv_last = keras.layers.Conv2D(3, (3, 3), (1, 1), "same")
        #### activation function
        self.lrelu = keras.layers.LeakyReLU(0.1)

    def __call__(self, x):
        x_shape = tf.shape(x)
        B = x_shape[0]
        N = x_shape[1]
        H = x_shape[2]
        W = x_shape[3]
        C = x_shape[4]
        x_center = tf.to_float(x[:, self.center, :, :, :])

        #### extract LR features
        # L1
        if self.is_predeblur:
            L1_fea = self.pre_deblur(tf.reshape(x, [-1, H, W, C]))
            L1_fea = self.conv_1x1(L1_fea)
            if self.HR_in:
                H = tf.divide(H, 4)
                W = tf.divide(W, 4)
        else:
            if self.HR_in:
                L1_fea = self.lrelu(self.conv_first_1(tf.reshape(x, [-1, H, W, C])))
                L1_fea = self.lrelu(self.conv_first_2(L1_fea))
                L1_fea = self.lrelu(self.conv_first_3(L1_fea))
                H = tf.divide(H, 4)
                W = tf.divide(W, 4)
            else:
                L1_fea = self.lrelu(self.conv_first(tf.reshape(x, [-1, H, W, C])))
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = tf.reshape(L1_fea, [B, N, H, W, -1])
        L2_fea = tf.reshape(L2_fea, [B, N, H // 2, W // 2, -1])
        L3_fea = tf.reshape(L3_fea, [B, N, H // 4, W // 4, -1])

        #### pcd align
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :], L2_fea[:, self.center, :, :, :],
            L3_fea[:, self.center, :, :, :]
        ]
        aligned_fea = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        def cond(i, N, fea_col):
            return i < N
        def body(i, N, fea_col):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :], L2_fea[:, i, :, :, :],
                L3_fea[:, i, :, :, :]
            ]
            fea_col = fea_col.write(i, self.pcd_align(nbr_fea_l, ref_fea_l))
            i = tf.add(i, 1)
            return i, N, fea_col
        _, _, aligned_fea = tf.while_loop(cond, body, [0, N, aligned_fea])
        aligned_fea = aligned_fea.stack()

        aligned_fea = tf.transpose(aligned_fea, [1, 0, 2, 3, 4])  # [B, N, H, W, C]

        # =====================================================================================!!!!
        # For static graph 
        aligned_fea = tf.reshape(aligned_fea, [B, self.nframes, H, W, self.nf])
        # aligned_fea.set_shape([])
        # =====================================================================================!!!!
        
        if not self.w_TSA:
            aligned_fea = aligned_fea.view(B, -1, H, W)
        fea = self.tsa_fusion(aligned_fea)

        out = self.recon_trunk(fea)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        if self.HR_in:
            base = x_center
        else:
            x_center_shape = tf.shape(x_center)
            base = tf.image.resize_images(x_center, [4 * x_center_shape[1], 4 * x_center_shape[2]])
        # out = tf.add(out, base)
        out = out + base
        return out

def l1_loss(x, y, eps=1e-6):
    diff = x - y
    loss = tf.reduce_sum(tf.sqrt(diff * diff + eps))
    return loss


def static_test():
 
    x_p = tf.placeholder(tf.float32, shape=[16, 7, 64, 64, 3], name='L_train')
    x = np.ones((16, 7, 64, 64, 3))
    gt_p = tf.placeholder(tf.float32, shape=[16, 1, 64*4, 64*4, 3], name='gt')
    model = EDVR()
    out = model(x_p)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(out, feed_dict = {x_p: x})
    # loss = tf.reduce_mean(tf.sqrt((out-gt_p)**2+1e-6))
    # global_step=tf.Variable(initial_value=0, trainable=False)
    # vars_all=tf.trainable_variables()
    # training_op = tf.train.AdamOptimizer(0.5).minimize(loss, var_list=vars_all, global_step=global_step)
    # return training_op
    return out

def dynamic_test():
    tf.enable_eager_execution()
    x = tf.ones(shape=[16, 7, 64, 64, 3])
    gt = tf.ones(shape=[16, 64*4, 64*4, 3])
    model = EDVR()
    out = model(x)
    return out
    # global_step=tf.Variable(initial_value=0, trainable=False)
    # vars_all=tf.trainable_variables()
    # training_op = tf.train.AdamOptimizer(0.5).minimize(l1_loss(out, gt), var_list=vars_all, global_step=global_step)
    # return training_op

if __name__ == "__main__":

    out = static_test()
    out = out.flatten()
    print(out[:10])
    

    print('done')



