import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from model.pfnl_hy import PFNL, Parameters

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
if __name__=='__main__':

    p = Parameters()
    p.num_frames=7 #7
    p.scale=4
    p.in_size=64
    p.gt_size=p.in_size*p.scale
    p.batch_size=4
    p.learning_rate= 2e-5
    p.end_lr=1e-8
    p.reload=True
    p.max_step= int(1.5e10+1)
    p.decay_step=5e8
    p.num_blocks = 20        #20
    p.main_channel_nums =128 #64

    # we consider every save_iter_gap iterations to be one epoch
    p.save_iter_gap = 5000 #5000 #1000
    p.nonLocal_sub_sample_rate = 2
    p.train_dir = './data/train_698.txt'
    p.exp_name = 'pfnl_hy_nonLocal_rx2_size_64_channel_128_blocks_20_exp_3_Finetune_exp_1'  #'pfnl_hy_nonLocal_rx2_size_64_channel_128_blocks_26_exp_1' 
    p.reload_dir = './checkpoint/' + p.exp_name #'pfnl_hy_nonLocal_rx2_size_64_channel_128_blocks_20_exp_3'
    p.save_dir='./checkpoint/'+ p.exp_name
    p.log_dir='./txt_log/' + p.exp_name + '.txt'

    #with open(p.log_dir, 'a+') as f:
    #    f.writelines('Comment: Finetune model from pfnl_hy_rx2_size_64_channel_128_blocks_20_exp_3 \n')
    #    f.writelines('Comment: Add training video to 698 (original: 650) \n')
    #    f.writelines('Comment: Add the main channel number to 128 (original:64) \n')
    #    f.writelines('Comment: Keep the fusion block number to 20 (original:20) \n')
    #    f.writelines('Comment: Since the subsample rate is 2, will cost about 15GB Mem(float32) \n')
    #    f.writelines('Comment: So we have to put the NonLocal module in CPU \n')
    #    f.writelines('Comment: Params num of all: 11561236, batch_size = 4 \n')

    p.tensorboard_dir = './tb_log/' + p.exp_name

    remove_exp = False
    if remove_exp:
        cmd = 'rm -rf ' + p.save_dir + ' ' + p.log_dir + ' ' + p.tensorboard_dir
        print('[**System cmd**]: ' + cmd +'\n')
        os.system(cmd)

    p.start_epoch = 7  # epoch first uesed and then add 1

    model=PFNL(p)
    
    model.train()
    
    # model.eval(0,-1.0)
    # model.test_video_lr('../data/frame_data/training/LR/11044561', '../testing_results/testing_540p_results_frame', exp_name=p.exp_name)
