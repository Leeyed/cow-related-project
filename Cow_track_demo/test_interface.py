from cowTrack import trackModel
import os
import cv2
import yaml
import torch
import numpy as np
import time


if __name__ == '__main__':
    # configure
    with open('cfg.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    for k,v in config.items():
        print(k,v)

    track_model = trackModel(config)
    cap = cv2.VideoCapture(config['video_path'])

    num = 0
    begin = time.time()
    while True:
        num += 1
        ret, frame = cap.read()
        if not ret: break
        frame = frame.astype(np.uint8)
        start = time.time()
        img_show = track_model.forward(frame)
        print('interval:', time.time()-start)
        img_show = cv2.resize(img_show,(int(img_show.shape[1]/2), int(img_show.shape[0]/2)))
        cv2.putText(img_show, str(num), (5,30), 0, 1, (255, 255, 255), 4)
        cv2.imwrite(os.path.join('/home/liusheng/data/NFS120/cow/vedio/15组2021.5/DVR_Examiner_Export_2021-05-18 142552_Job_0002/2021-05-12/Native Video Files (MP4)/video_encoder/', str(num)+'.jpg'), img_show)
        print(num)
    print(f'total:{time.time()-begin}')
    cap.release()
    # cv2.destroyAllWindows()





