import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from model.pfnl_hy import PFNL, Parameters

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
     
if __name__=='__main__':

    p = Parameters()
    p.num_frames=7 #7
    p.scale=4
    p.in_size=64
    p.gt_size=p.in_size*p.scale
    p.batch_size=8
    p.learning_rate=1e-4
    p.end_lr=1e-6
    p.reload=True
    p.max_step= int(1.5e10+1)
    p.decay_step=1.2e5
    p.num_blocks = 20 #20
    p.main_channel_nums =64 #64
    # we consider every save_iter_gap iterations to be one epoch
    p.save_iter_gap = 1000
    p.nonLocal_sub_sample_rate = 4

    p.exp_name = 'pfnl_hy_nonLocal_rx4_size_64_exp_1' #'pfnl_hy_nonLocal_subsmaplex4_no_eval_exp_3' #

    p.save_dir='./checkpoint/'+ p.exp_name
    p.log_dir='./txt_log/' + p.exp_name + '.txt'
    # with open(p.log_dir, 'a+') as f:
    #     f.writelines('Comment: concate the input and the output of Nonlocal module \n')
    p.tensorboard_dir = './tb_log/' + p.exp_name
    p.start_epoch = 58
    model=PFNL(p)
    
    # model.eval(21,-1.0)
    model.train()
    # model.test_video_lr(r'H:\AI4K\data\frame_data\training\LR\11044561', r'H:\AI4K\data\testing_540p_results', name='PFNL_exp_2_jpg100')
