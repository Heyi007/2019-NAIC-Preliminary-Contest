import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from model.pfnl_hy import PFNL, Parameters

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
if __name__=='__main__':

    p = Parameters()
    p.num_frames=7 #7
    p.scale=4
    p.in_size=32
    p.gt_size=p.in_size*p.scale
    p.eval_in_size=[128,240]
    p.batch_size=16
    p.eval_basz=4
    p.learning_rate=1e-4
    p.end_lr=1e-6
    p.reload=True
    p.max_step= int(1.5e10+1)
    p.decay_step=1.2e5
    p.num_blocks = 20 #20
    p.main_channel_nums =64 #64
    p.save_iter_gap = 1000
    p.nonLocal_sub_sample_rate = 4

    p.exp_name = 'pfnl_hy_nonLocal_rx4_size_64_exp_1'

    p.save_dir='./checkpoint/'+ p.exp_name
    p.log_dir='./txt_log/' + p.exp_name
    p.tensorboard_dir = './tb_log/' + p.exp_name

    model=PFNL(p)

    test_dir = r'H:\AI4K\data\frame_data\testing_540p_LR'
    save_dir = r'D:\AI4K\testing_540p_results_frame'
 
    names = os.listdir(test_dir)
    names = sorted(names)
    names = names
    for name in names:
        
        test_path = os.path.join(test_dir, name) 
        # method1 = 'new_info'
        method2 = 'reflection'
        # model.makeup(test_path, save_dir, p.exp_name+'_makeup_'+method1, method=method1)
        model.makeup(test_path, save_dir, p.exp_name+'_makeup_'+method2, method=method2)
