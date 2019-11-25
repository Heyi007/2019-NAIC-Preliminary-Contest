import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from os.path import join,exists
import glob
import random
import numpy as np
from PIL import Image
import scipy
import time
import os
from modules.EDVR_static import EDVR_Core
# from tensorflow.python.layers.convolutional import Conv2D,conv2d
# from utils import NonLocalBlock, DownSample, DownSample_4D, BLUR, get_num_params, cv2_imread, cv2_imsave, automkdir
from utils import get_num_params, cv2_imread, cv2_imsave, automkdir
from tqdm import tqdm,trange
from model.base_model import VSR
from skimage.measure import compare_ssim, compare_psnr
import cv2
'''This is the official code of PFNL (Progressive Fusion Video Super-Resolution Network via Exploiting Non-Local Spatio-Temporal Correlations).
The code is mainly based on https://github.com/psychopa4/MMCNN and https://github.com/jiangsutx/SPMC_VideoSR.
'''

class Parameters():
    def __init__(self):

        # network structure settings
        self.num_frames=7          #7
        self.main_channel_nums =64 #64

        # 
        self.scale=4
        self.in_size=32
        self.gt_size=self.in_size*self.scale
        self.batch_size=16

        self.learning_rate=1e-3
        self.end_lr=1e-5        
        self.max_step= int(1.5e10+1)
        self.decay_step=1.2e5
            
        self.reload=True
        self.save_iter_gap = 1000
        

        self.train_dir='./data/train.txt'
        self.eval_dir='./data/validation.txt'
        self.save_dir='./checkpoint/pfnl'
        self.log_dir='./pfnl.txt'
        self.exp_name = 'Temp'
        self.save_dir='./checkpoint/'+ self.exp_name
        self.log_dir='./txt_log/' + self.exp_name
        self.tensorboard_dir = './tb_log/' + self.exp_name
        self.start_epoch = 1


class EDVR(VSR):
    def __init__(self, parameters):
        self.num_frames = parameters.num_frames
        self.scale = parameters.scale
        self.in_size = parameters.in_size
        self.gt_size = self.in_size*self.scale
        self.batch_size = parameters.batch_size
        self.learning_rate = parameters.learning_rate
        self.end_lr = parameters.end_lr
        self.reload = parameters.reload
        self.max_step = parameters.max_step
        self.decay_step = parameters.decay_step
        self.train_dir = parameters.train_dir
        self.eval_dir = parameters.eval_dir
        self.save_dir = parameters.save_dir
        self.log_dir = parameters.log_dir
        self.tensorboard_dir = parameters.tensorboard_dir
        
        self.main_channel_nums =parameters.main_channel_nums
        self.save_iter_gap = parameters.save_iter_gap
        self.start_epoch = parameters.start_epoch

        # build the main network computational graph
        # # the main SR network: EDVR or PFNL
        self.model = EDVR_Core(nf = self.main_channel_nums, nframes = self.num_frames)

        self.GT = tf.placeholder(tf.float32, shape=[None, 1, None, None, 3], name='H_truth')
        self.L_train = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_frames, self.in_size, self.in_size, 3], name='L_train')
        self.SR = self.forward(self.L_train)

        self.L_test = tf.placeholder(tf.float32, shape=[1, self.num_frames, None, None, 3], name='L_test')
        self.SR_test=self.forward(self.L_test)
       
        

        self.loss = tf.reduce_mean(tf.sqrt((self.SR-self.GT)**2+1e-6))

        # data loader and training supports
        self.LR_one_batch, self.HR_one_batch= self.double_input_producer()
        global_step=tf.Variable(initial_value=0, trainable=False)
        self.global_step=global_step

        lr= tf.train.polynomial_decay(self.learning_rate, global_step, self.decay_step, end_learning_rate=self.end_lr, power=1.)
        vars_all=tf.trainable_variables()
        print('Params num of all:',get_num_params(vars_all))
        self.training_op = tf.train.AdamOptimizer(lr).minimize(self.loss, var_list=vars_all, global_step=global_step)


        # For tensorboard visualization

        # used in eval func
        self.loss_epoch = tf.placeholder(tf.float32, shape=[], name='epoch_loss_placeholder')
        self.epoch_loss_summary_op = tf.summary.scalar('loss/epoch loss', self.loss_epoch)
        self.psnr_eval = tf.placeholder(tf.float32, shape=[], name='eval_psnr_placeholder')
        self.eval_psnr_summary_op = tf.summary.scalar('metrics/eval psnr', self.psnr_eval)
        self.ssim_eval = tf.placeholder(tf.float32, shape=[], name='eval_ssim_placeholder')
        self.eval_ssim_summary_op = tf.summary.scalar('metrics/eval ssim', self.ssim_eval)
        self.merge_op_eval = tf.summary.merge([self.epoch_loss_summary_op, self.eval_psnr_summary_op, self.eval_ssim_summary_op])

        # used in iter training func
        iter_loss_summary_op = tf.summary.scalar("loss/iter loss", self.loss)
        lr_summary_op = tf.summary.scalar("lr", lr)
        self.merge_op_training = tf.summary.merge([iter_loss_summary_op, lr_summary_op])


        # writer, get session and hold it and some configs
        self.writer = tf.summary.FileWriter(self.tensorboard_dir, tf.get_default_graph())

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        print('[**] Initialzing global varibles ...')
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        if self.reload:
            print('[**] loading checkpoint in dir:'+ self.save_dir)
            self.load(self.sess, self.save_dir)


        # eval file prepare
        self.eval_frame_data_HR = []
        self.eval_frame_data_LR = []
        pathlists = open(self.eval_dir, 'rt').read().splitlines()
        for dataPath in pathlists:
            inList = sorted(glob.glob(os.path.join('H:/AI4K/data/frame_data/validation/LR', dataPath, '*.png')))
            gtList = sorted(glob.glob(os.path.join('H:/AI4K/data/frame_data/validation/HR', dataPath, '*.png')))
            assert(len(inList)==len(gtList))
            self.eval_frame_data_HR.append(gtList)
            self.eval_frame_data_LR.append(inList)
      
    def forward(self, x):
        # model = EDVR_Core(nf = self.main_channel_nums, nframes = self.num_frames)
        out = self.model(x)
        return out

    def eval(self, epoch, loss_epoch):
        print('Evaluating on the validation set...')
        psnr_all = 0
        ssim_all = 0
        count = 0
        for video_index in range(0,len(self.eval_frame_data_LR)):
            cur_video_LR = self.eval_frame_data_LR[video_index]
            path, _ = os.path.split(cur_video_LR[0])
            _, name = os.path.split(path)
            
            cur_video_HR = self.eval_frame_data_HR[video_index]
            max_frame = len(cur_video_LR)
            # temp_img = cv2_imread(cur_video_LR[0])
            # h,w,_ = temp_img.shape
            # L_eval = tf.placeholder(tf.float32, shape=[1, self.num_frames, h, w, 3], name='L_test')
            # SR_test = self.forward(L_eval)
            for i in range(max_frame):
                start_time = time.time()
                count+=1
                index=np.array([i for i in range(i-self.num_frames//2,i+self.num_frames//2+1)])
                index=np.clip(index,0,max_frame-1).tolist()
                lrs = np.array([cv2_imread(cur_video_LR[i]) for i in index])/255.0
                lrs = lrs.astype('float32')
                lrs = np.expand_dims(lrs, 0)

                sr = self.sess.run(self.SR_test, feed_dict={self.L_test: lrs})
                sr = sr*255
                sr = np.squeeze(sr, axis = (0))
                sr = np.clip(sr,0,255)
                sr = np.round(sr,0).astype(np.uint8)
                hr = cv2_imread(cur_video_HR[i])
                psnr = compare_psnr(sr, hr)
                ssim = compare_ssim(sr, hr, multichannel = True)
                psnr_all += psnr
                ssim_all += ssim
                cost_time = time.time() - start_time
                print('[*Epoch:{:05d}] val video:{} -> frame:{:05d}, psnr={:.4f}, ssim={:.4f}, cost time={:.2f} '.format(epoch, name, i, psnr, ssim, cost_time))


        a_psnr = psnr_all / count
        a_ssim = ssim_all / count

        eval_ss = self.sess.run(self.merge_op_eval, feed_dict={self.loss_epoch:loss_epoch, self.psnr_eval:a_psnr, self.ssim_eval:a_ssim})
        self.writer.add_summary(eval_ss, epoch)

        print('{'+'"Epoch": {:05d} , "Training Loss":{:.6f}, " Eval PSNR": {:.4f}, "Eval SSIM": {:.4f}'.format(epoch, loss_epoch, a_psnr, a_ssim)+'}')
        # write to log file
        with open(self.log_dir, 'a+') as f:
            f.write('{'+'"Epoch": {:05d} , "Training Loss":{:.6f}, " Eval PSNR": {:.4f}, "Eval SSIM": {:.4f}'.format(epoch, loss_epoch, a_psnr, a_ssim)+'}\n')


    def train(self):

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        loss_epoch = 0
        
        gs=self.sess.run(self.global_step)
        epoch = self.start_epoch
        for step in range(self.sess.run(self.global_step), self.max_step):

            lr1,hr=self.sess.run([self.LR_one_batch,self.HR_one_batch])
            SR,_,loss_v,ss=self.sess.run([self.SR, self.training_op, self.loss, self.merge_op_training],feed_dict={self.L_train:lr1, self.GT:hr})
            
            # SR = SR * 255.0
            # hr = hr * 255.0
            # for i in range(SR.shape[0]):
            #     a = SR[i]
            #     b = hr[i,0,:,:,:]
            #     a=np.clip(a, 0, 255)
            #     a=np.round(a,0).astype(np.uint8)
            #     b=np.clip(b, 0, 255)
            #     b=np.round(b,0).astype(np.uint8)
            #     cv2.imwrite(str(i)+".png", a)
            #     cv2.imwrite(str(i)+"_GT.png", b)
            # break

            loss_epoch += loss_v
            if step>gs and step % 1 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()),'Step:{}, loss:{}'.format(step,loss_v))
            self.writer.add_summary(ss, step)

            if step>500 and loss_v>10:
                print('Model collapsed with loss={}'.format(loss_v))
                break

            # eval and save model
            if step % self.save_iter_gap == 0 and step!=0:
                if step>gs:
                    print('saving model at global step: '+ str(step))
                    self.save(self.sess, self.save_dir, step)
                
                loss_epoch = loss_epoch / self.save_iter_gap
                if step % (self.save_iter_gap*2) == 0:
                    self.eval(epoch, loss_epoch)

                epoch += 1
                loss_epoch = 0

        self.writer.close()

  
    def test_video_lr(self, path, output_path, exp_name='PFNL_result'):
        num_once = 1
        _, video_name = os.path.split(path)
        save_path=join(output_path, exp_name, video_name)
        automkdir(save_path)
        imgs=sorted(glob.glob(join(path,'*.png')))
        max_frame=len(imgs)
        lrs=np.array([cv2_imread(i) for i in imgs])/255.
        lrs = lrs.astype('float32')
        h,w,_=lrs[0].shape
        lr_list=[]

        for i in range(max_frame):
            index=np.array([i for i in range(i-self.num_frames//2,i+self.num_frames//2+1)])
            index=np.clip(index,0,max_frame-1).tolist()
            lr_list.append(np.array([lrs[j] for j in index]))
        del lrs
        lr_list=np.array(lr_list)

        # L_test = tf.placeholder(tf.float32, shape=[1, self.num_frames, h, w, 3], name='L_test')
        # SR_test=self.forward(L_test)

        print('Save at {}'.format(save_path))
        print('{} Inputs With Shape {}'.format(max_frame,[h,w]))

        all_time=[]
        for i in trange(max_frame):
            st_time=time.time()
            sr=self.sess.run(self.SR_test,feed_dict={self.L_test : lr_list[i*num_once:(i+1)*num_once]})
            all_time.append(time.time()-st_time)
            for j in range(sr.shape[0]):
                img=sr[j]*255.
                img=np.clip(img,0,255)
                img=np.round(img,0).astype(np.uint8)
                cv2_imsave(join(save_path, '{:0>4}.jpg'.format(i*num_once+j)),img, 100)

        all_time=np.array(all_time)
        if max_frame>0:
            all_time=np.array(all_time)
            print('spent {} s in total and {} s in average'.format(np.sum(all_time),np.mean(all_time[1:])))

        del imgs
        # del L_test
        # del SR_test
        # del lrs
        del lr_list



if __name__=='__main__':
    pass
