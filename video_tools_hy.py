# This is used to generate vedio from a image sequence
# @heyi 2019.11.9
import cv2 as cv2
import numpy as np
import os
from skimage.measure import compare_ssim, compare_psnr

# compute PSNR and SSIM of two video 
class VedioMetrics():
    def __init__(self, x, y):
        pass

    def psnr(self):
        pass

    def ssim(self):
        pass

# compute PSNR and SSIM of two image sequence
# the order of image frames and size must be correct
class ImageSeqMetrics():
    def __init__(self, x_seq_dir, y_seq_dir):
        names_x = os.listdir(x_seq_dir)
        names_y = os.listdir(y_seq_dir)
        assert(len(names_x) == len(names_y))
        self.img_num = len(names_x)
        self.x_imgs_path = []
        self.y_imgs_path = []
        for name in names_x:
            path = os.path.join(x_seq_dir, name)
            self.x_imgs_path.append(path)
        for name in names_y:
            path = os.path.join(y_seq_dir, name)
            self.y_imgs_path.append(path)
        self.x_imgs_path = sorted(self.x_imgs_path)
        self.y_imgs_path = sorted(self.y_imgs_path)
        
    def psnr(self):
        total_psnr = 0
    
        for i in range(self.img_num):
            # print('computing frame:'+str(i+1) + '...')
            x = cv2.imread(self.x_imgs_path[i])
            y = cv2.imread(self.y_imgs_path[i])
            total_psnr += compare_psnr(x, y)
           
        return total_psnr/self.img_num

    def ssim(self):
        total_ssim = 0
        
        for i in range(self.img_num):
            # print('computing frame:'+str(i+1) + '...')
            x = cv2.imread(self.x_imgs_path[i])
            y = cv2.imread(self.y_imgs_path[i])
            total_ssim += compare_ssim(x, y, multichannel=True)

        return total_ssim/self.img_num


# extract image frames from a video file
class Vextractor_single():
    def __init__(self, src_file = None):
        self.src_file = src_file
        self.cap = cv2.VideoCapture(self.src_file)

    def extract(self, output_dir = None):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        (filepath, filename) = os.path.split(self.src_file)
        (filename, ext) = os.path.splitext(filename)
        self.dst_file_dir = os.path.join(self.output_dir, filename)
        if not os.path.exists(self.dst_file_dir):
            os.mkdir(self.dst_file_dir)
        success, frame = self.cap.read()
        count = 1
        while success:
            print('Extracting file:' + self.src_file + '->No.%05d frame...'% count)
            frame_save_path = os.path.join(self.dst_file_dir, ('%05d.png' % count))
            cv2.imwrite(frame_save_path, frame)
            success, frame = self.cap.read()
            count += 1

class Vextractor_dir():
    def __init__(self, src_dir = None):
        self.src_dir = src_dir
        names = os.listdir(self.src_dir)
        self.vedio_paths = []
        for name in names:
            path = os.path.join(src_dir, name)
            self.vedio_paths.append(path)
    
    def extract(self, output_dir = None):
        self.output_dir = output_dir
        for video_path in self.vedio_paths:
            temp = Vextractor_single(video_path)
            temp.extract(output_dir)

            
# show a vedio file
class VReader():
    def __init__(self, src_file = None):
        self.src_file = src_file
        self.cap = cv2.VideoCapture(src_file)
    def show(self):
        success, frame = self.cap.read()
        while success:
            frame = cv2.resize(frame, (int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)))
            cv2.imshow(self.src_file, frame)
            ch = cv2.waitKey(int(1000/(25)))
            if ch == 27:
                break
            success, frame = self.cap.read()

# visualize and save the compare result
class LHCompare():
    def __init__(self, LR_file = None, HR_file = None):
        self.LR_file = LR_file
        self.HR_file = HR_file
        self.LR_cap = cv2.VideoCapture(self.LR_file)
        self.HR_cap = cv2.VideoCapture(self.HR_file)
        self.fps = self.HR_cap.get(5)
        self.size = (int(self.HR_cap.get(4)), int(self.HR_cap.get(3)))
        print('fps: '+ str(self.fps))
        print('size: ('+ str(self.size[0])+' , '+str(self.size[1]), ')' )

    def show_compare(self):
        success_L, frame_L = self.LR_cap.read()
        success_H, frame_H = self.HR_cap.read()
        H = frame_L.shape[0]
        W = frame_L.shape[1]
        while success_L and success_H:
            frame_H = cv2.resize(frame_H, (W, H))
            frame_L[:, int(W/2):, :] = frame_H[:, int(W/2):, :]
         
            frame_L[:, int(W/2):int(W/2)+1, :] = 255
            frame = np.concatenate((frame_L, frame_H), axis=0)

            cv2.imshow(self.LR_file, frame)
            ch = cv2.waitKey(int(1000/(25)))
            if ch == 27:
                break
            success_L, frame_L = self.LR_cap.read()
            success_H, frame_H = self.HR_cap.read()

    def save_compare(self, save_compare_path = None):
        success_L, frame_L = self.LR_cap.read()
        success_H, frame_H = self.HR_cap.read()
        H = frame_H.shape[0]
        W = frame_H.shape[1]
        size = (W, H)
        output_mp4_path = save_compare_path
        self.Vwriter = cv2.VideoWriter(output_mp4_path, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), self.fps, size)
        count = 1
        while success_L and success_H:
            print('writing frame '+ str(count) + '...')
            count += 1
            frame_L = cv2.resize(frame_L, (W, H))
            frame_L[:, int(W/2):, :] = frame_H[:, int(W/2):, :]
            frame_L[:, int(W/2):int(W/2)+1, :] = 255
            frame = np.concatenate((frame_L, frame_H), axis=0)
            self.Vwriter.write(frame_L)

            # cv2.imshow(self.LR_file, frame_L)
            # ch = cv2.waitKey(int(1000/(20)))
            # if ch == 27:
            #     break

            success_L, frame_L = self.LR_cap.read()
            success_H, frame_H = self.HR_cap.read()

# generate a video from a sequence of image frames
class VGenerator():
    def __init__(self, src_dir = None):
        self.src_dir = src_dir
        self.frame_name = os.listdir(self.src_dir)
        self.frame_path = []
        for name in self.frame_name:
            path = os.path.join(self.src_dir, name)
            self.frame_path.append(path)

        print('init done ...')

    def generate_video(self, output_file = None, fps = 24):
        print('Generating video: ' + output_file + ' with src dir: '+ self.src_dir + '...')
        temp_img = cv2.imread(self.frame_path[0])
        # temp_img = cv2.resize(temp_img,  (int(temp_img.shape[1]*0.5), int(temp_img.shape[0]*0.5)))
        size = (temp_img.shape[1], temp_img.shape[0])
        print('fps:' + str(fps) + ', size:(' + str(size[0]) + ',' + str(size[1]) + ')') 
        # self.Vwriter = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('H', 'E', 'V', 'C'), fps, size)
        self.Vwriter = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, size)

        count = 1
        for frame_path in self.frame_path:
            print('writing frame '+ str(count) + ' :' + frame_path)
            count = count + 1
            frame = cv2.imread(frame_path)
            # cv2.imshow('temp', frame)
            # cv2.waitKey(int(1000/(15)))
            self.Vwriter.write(frame)
            
        print('Generate done!')



def generate_test540p_video(exp_name):
    src_dir = os.path.join(r'I:\AI4K\testing_540p_results_frame', exp_name)
    video_names = os.listdir(src_dir)#[22:]
    dst_dir = os.path.join(r'I:\AI4K\testing_540p_results_video', exp_name)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for name in video_names:
        save_video_path = os.path.join(dst_dir, name+'.mp4')
        print('Generating Video:', save_video_path, '...')
        frame_dir = os.path.join(src_dir, name)
        g = VGenerator(frame_dir)
        g.generate_video(save_video_path)


def compare_test540p_video(exp_name):
    HR_dir = os.path.join(r'I:\AI4K\testing_540p_results_video', exp_name)
    video_names = os.listdir(HR_dir)
    saving_dir = os.path.join(r'I:\AI4K\compare_video', exp_name)
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    for name in video_names:
        
        HR_video = os.path.join(HR_dir, name)
        LR_video = os.path.join(r'I:\AI4K\testing_540p', name)
        c = LHCompare(LR_video, HR_video)
        saving_compare_video_path = os.path.join(saving_dir, 'C_'+name)
        c.save_compare(saving_compare_video_path)




if __name__ == '__main__':
    vid = '63171818'
    gt_dir = r'H:\AI4K\data\frame_data\validation\HR'+ '\\' + vid
    in_dir3 = r'D:\AI4K\TEMP\test_new_info'+ '\\' + vid
    in_dir4 = r'D:\AI4K\TEMP\test_new_info_crop_black_edge'+ '\\' + vid
    in_dir1 = r'D:\AI4K\TEMP\test_raw'+ '\\' + vid
    in_dir2 = r'D:\AI4K\TEMP\test_raw_crop_black_edge'+ '\\' + vid
    M = ImageSeqMetrics(in_dir1, gt_dir)
    psnr = M.psnr()
    ssim = M.ssim()
    print('raw:'+str(psnr)+ '|'+ str(ssim))

    M = ImageSeqMetrics(in_dir2, gt_dir)
    psnr = M.psnr()
    ssim = M.ssim()
    print('crop_edge:'+str(psnr)+ '|'+ str(ssim))

    M = ImageSeqMetrics(in_dir3, gt_dir)
    psnr = M.psnr()
    ssim = M.ssim()
    print('new_info:'+str(psnr)+ '|'+ str(ssim))

    M = ImageSeqMetrics(in_dir4, gt_dir)
    psnr = M.psnr()
    ssim = M.ssim()
    print('crop_edge_new_info:'+str(psnr)+ '|'+ str(ssim))




    # exp_name = 'pfnl_hy_nonLocal_rx4_size_64_exp_1_JPEG_Compress_100'
    # generate_test540p_video(exp_name)

    # compare_test540p_video(exp_name)
    # video_name = '16536366'

    # src_dir = os.path.join('I:/AI4K/testing_540p_results_frame', exp_name, video_name)
    # save_dir = os.path.join('I:/AI4K/testing_540p_results_video', exp_name)
    # save_path = os.path.join(save_dir, video_name+'.mp4')
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    

    # w = VGenerator(src_dir)
    # w.generate_video(save_path)

    # LR_video = os.path.join('H:/AI4K/data/testing_540p', video_name+'.mp4')
    # HR_video = save_path
    # compare_save_video_path = os.path.join('I:/AI4K/compare_video', video_name+'_'+exp_name+'.mp4')
    # c = LHCompare(LR_video, HR_video)
    # c.save_compare(compare_save_video_path)

    # c = LHCompare(r'H:\AI4K\data\testing_540p\16842928.mp4', r'H:\AI4K\data\testing_540p_results\16842928_PFNL_exp_2_x4.mp4')
    # c.save_compare(r'H:\AI4K\data\testing_540p_results\16842928_PFNL_exp_2_C_x4.mp4')


    # c = LHCompare(r'I:\training\SDR_540p\11044561.mp4', r'H:\AI4K\data\testing_540p_results\11044561_PFNL_exp_2_x4.mp4')
    # c.save_compare(r'H:\AI4K\data\testing_540p_results\11044561_exp_2_C_x4.mp4')
   

    # e = Vextractor_dir('H:/AI4K/data/training/SDR_540p')
    # e = Vextractor_dir('H:/AI4K/data/training/SDR_4K(Part1)')
    # e.extract('H:/AI4K/data/frame_data/training/HR')
    
    # TecoGAN -> calendar:psnr = 21.144, ssim = 0.774
    # FPNL -> calendar:psnr = 21.680, ssim = 0.815
    # c = ImageSeqMetrics(r'H:\AI4K\TecoGAN\HR\vid4_HR\calendar',r'H:\AI4K\TecoGAN\results\calendar')
    # a_psnr = c.ssim()
    # print(a_psnr)
 