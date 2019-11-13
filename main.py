import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from model.pfnl import PFNL

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
if __name__=='__main__':

    p = PFNL()
    p.num_frames=7 #7
    p.scale=4
    p.in_size=32
    p.gt_size=p.in_size*p.scale
    p.eval_in_size=[128,240]
    p.batch_size=16
    p.eval_basz=4
    p.learning_rate=1e-3
    p.end_lr=1e-5
    p.reload=True
    p.max_step= int(1.5e10+1)
    p.decay_step=1.2e5
    p.num_blocks = 20 #20
    p.main_channel_nums =64 #64
    p.save_iter_gap = 1000
    p.nonLocal_sub_sample_rate = 4

    p.exp_name = 'pfnl_hy_nonLocal_subsmaplex4_exp_2_no_eval_TEST'

    p.save_dir='./checkpoint/'+ p.exp_name
    p.log_dir='./txt_log/' + p.exp_name
    p.tensorboard_dir = './tb_log/' + p.exp_name

    p.train()
    # model.test_video_lr(r'H:\AI4K\data\frame_data\training\LR\11044561', r'H:\AI4K\data\testing_540p_results', name='PFNL_exp_2_jpg100')
