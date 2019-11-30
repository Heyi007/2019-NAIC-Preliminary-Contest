import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from model.pfnl_hy import PFNL, Parameters

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
if __name__=='__main__':

    p = Parameters()
    p.num_frames=7 #7
    p.scale=4
    p.in_size=64
    p.gt_size=p.in_size*p.scale
    p.eval_in_size=[128,240]
    p.batch_size=4
    p.eval_basz=4
    p.learning_rate=1e-4
    p.end_lr=1e-6
    p.reload=True
    p.max_step= int(1.5e10+1)
    p.decay_step=1.2e5
    p.num_blocks = 20 #20
    p.main_channel_nums =128 #64
    p.save_iter_gap = 1000
    p.nonLocal_sub_sample_rate = 2

    p.exp_name = 'pfnl_hy_nonLocal_rx2_size_64_channel_128_blocks_20_exp_3_Finetune_exp_1'

    p.save_dir='./checkpoint/'+ p.exp_name
    p.reload_dir = './checkpoint/' + p.exp_name
    p.log_dir='./txt_log/' + p.exp_name
    p.tensorboard_dir = './tb_log/' + p.exp_name
    p.start_epoch = -1

    model=PFNL(p)

    test_dir = '../data/frame_data/testing_540p_LR'
    save_dir = '../testing_results/testing_540p_results_frame'
    
    names = os.listdir(test_dir)
    names = sorted(names)
    
    for name in names:
        
        test_path = os.path.join(test_dir, name) 
        model.test_video_lr(test_path, save_dir, p.exp_name+'_PNG_0')
