import os
import threading
import cv2
import numpy as np

def single_processing(img_path, save_path, thresholding = 3840):
    img = cv2.imread(img_path)
    img_row_sum = np.sum(img, axis = (1,2))
    h, w, _ = img.shape
    # find up black edge
    up = 0
    bottom = 0
    for i in range(int(h*0.3)):
        if img_row_sum[i] > thresholding:
            break
        up += 1

    for i in range(h-1, int(h*0.7)-1, -1):
        if img_row_sum[i] > thresholding:
            break
        bottom += 1

    img[:up, :, :] = 0
    img[h - bottom : h, :, :] = 0

    cv2.imwrite(save_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def workers(video_path, frame_dir, save_dir):
    
    for video_name in video_path:
        print('Processing video:' + video_name)
        save_video_path = os.path.join(save_dir, video_name)
        if not os.path.exists(save_video_path):
            os.makedirs(save_video_path)
        
        video_path = os.path.join(frame_dir, video_name)
        for img in os.listdir(video_path):
            img_path = os.path.join(video_path, img)
            save_img_path = os.path.join(save_video_path, img)
            single_processing(img_path, save_img_path)



def processing(frame_dir, save_dir, workers_num = 2):

    video_names = os.listdir(frame_dir)
    threads = []
    l = len(video_names)
    step = int(l/workers_num)
    for i in range(workers_num):
        thread = threading.Thread(target=workers, args=(video_names[i*step:min((i+1)*step, l)], frame_dir, save_dir))
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()

    print('done!')






if __name__ == "__main__":

    processing(r'D:\AI4K\TEMP\test_raw',
        r'D:\AI4K\TEMP\test_raw_crop_black_edge')
    


