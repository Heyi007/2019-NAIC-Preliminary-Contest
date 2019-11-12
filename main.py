import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from model.pfnl import PFNL

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
if __name__=='__main__':
    model=PFNL()
    model.num_frames=7 #7
    model.scale=4
    model.in_size=32
    model.gt_size=model.in_size*model.scale
    model.eval_in_size=[128,240]
    model.batch_size=16
    model.eval_basz=4
    model.learning_rate=1e-3
    model.end_lr=1e-5
    model.reload=False
    model.max_step= int(1.5e10+1)
    model.decay_step=1.2e5
    model.num_blocks = 20 #20
    model.main_channel_nums =64 #64
    model.save_iter_gap = 1000
    model.nonLocal_sub_sample_rate = 4

    exp_name = 'pfnl_nonLocal_subsmaplex4_exp_2_no_eval'

    model.save_dir='./checkpoint/'+ exp_name
    model.log_dir='./txt_log/' + exp_name
    model.tensorboard_dir = './tb_log/' + exp_name


    model.train()
    # model.test_video_lr(r'H:\AI4K\data\frame_data\testing_540p_LR\16842928', r'H:\AI4K\data\testing_540p_results')
