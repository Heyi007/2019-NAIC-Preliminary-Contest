import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from model.edvr_hy import EDVR, Parameters

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
if __name__=='__main__':

    p = Parameters()
    p.num_frames=7 #7
    p.scale=4
    p.in_size=64
    p.gt_size=p.in_size*p.scale
    p.batch_size=4
    p.learning_rate=1e-4
    p.end_lr=1e-7
    p.reload=True
    p.max_step= int(1.5e10+1)
    p.decay_step=1.2e5
  
    p.main_channel_nums =32 #64
    # we consider every save_iter_gap iterations to be one epoch
    p.save_iter_gap = 1000
  
    p.exp_name = 'edvr_tf_size_64_channel_32_exp_1' 

    p.save_dir='./checkpoint/'+ p.exp_name
    p.log_dir='./txt_log/' + p.exp_name + '.txt'
    # with open(p.log_dir, 'a+') as f:
    #     f.writelines('Comment: concate the input and the output of Nonlocal module \n')
    p.tensorboard_dir = './tb_log/' + p.exp_name
    p.start_epoch = 1

    model=EDVR(p)

    test_dir = r'H:\AI4K\data\frame_data\testing_540p_LR'
    save_dir = r'D:\AI4K\testing_540p_results_frame'
 
    names = os.listdir(test_dir)
    names = sorted(names)
    names = names
    for name in names:
        
        test_path = os.path.join(test_dir, name) 
        model.test_video_lr(test_path, save_dir, p.exp_name)
