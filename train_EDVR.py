import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# from model.pfnl_hy import PFNL, Parameters
from model.edvr_hy import EDVR, Parameters

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
     
if __name__=='__main__':

    p = Parameters()
    p.num_frames=7 #7
    p.scale=4
    p.in_size=64
    p.gt_size=p.in_size*p.scale
    p.batch_size= 8
    p.learning_rate=1e-4
    p.end_lr=1e-7
    p.reload=True
    p.max_step= int(1.5e10+1)
    p.decay_step=1.2e5
  
    p.main_channel_nums = 64 #64
    # we consider every save_iter_gap iterations to be one epoch
    p.save_iter_gap = 1000
  
    p.exp_name = 'edvr_tf_size_64_channel_40_exp_001' 

    p.save_dir='./checkpoint/'+ p.exp_name
    p.log_dir='./txt_log/' + p.exp_name + '.txt'
    # with open(p.log_dir, 'a+') as f:
    #     f.writelines('Comment: concate the input and the output of Nonlocal module \n')
    p.tensorboard_dir = './tb_log/' + p.exp_name
    p.start_epoch = 1

    model=EDVR(p)
    
    # model.eval(-1,-1.0)
    model.train()
    # model.test_video_lr(r'H:\AI4K\data\frame_data\training\LR\11044561', r'D:\AI4K\testing_540p_results_frame', exp_name=p.exp_name)
