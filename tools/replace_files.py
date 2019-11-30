import os

dir = r'D:\AI4K\TEMP\test_new_info'

backup_dir = r'D:\AI4K\TEMP\new_info_backup'
input_dir = r'D:\AI4K\TEMP\new_info'

vid_names = os.listdir(input_dir)

for vid_name in vid_names:
    vid_path = os.path.join(input_dir, vid_name)
    frame_names = os.listdir(vid_path)
    backup_vid_dir = os.path.join(backup_dir, vid_name)
    if not os.path.exists(backup_vid_dir):
        os.makedirs(backup_vid_dir)
    for frame_name in frame_names:
        input_frame_path = os.path.join(input_dir, vid_name, frame_name)
        backup_frame_path = os.path.join(dir, vid_name, frame_name)
        backup_save_frame_path = os.path.join(backup_dir, vid_name, frame_name)

        cmd1 = 'move ' + backup_frame_path + ' ' + backup_save_frame_path
        cmd2 = 'move ' + input_frame_path + ' ' + backup_frame_path

        print(cmd1)
        print(cmd2)

        os.system(cmd1)
        os.system(cmd2)


