import os
import glob
def main():
    input_path = './pfnl_hy_nonLocal_rx2_size_64_channel_128_blocks_20_exp_3_Finetune_exp_1_PNG_0_add'
    output_path = '/home/mark/4k_game/result/pfnl_hy_nonLocal_rx2_size_64_channel_128_blocks_20_exp_3_PNG_0_H265_crf10_r29_7_finetune_add'
    #output_path = './temp_video'
    list_path = os.listdir(input_path)
    for path in list_path:
        in_path = os.path.join(input_path, path)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        #print(r'ffmpeg -r 24000/1001 -i {}/%04d.png -vcodec libx265 -pix_fmt yuv422p -crf 10 {}/{}.mp4'.format(in_path, out_path, path))
        os.system(r'ffmpeg -r 24000/1001 -i {}/%04d.png -vcodec libx265 -pix_fmt yuv422p -crf 10 {}/{}.mp4'.format(in_path, output_path, path))



if __name__ == '__main__':
    main()
