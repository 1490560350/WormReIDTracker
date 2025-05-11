import warnings
warnings.filterwarnings('ignore')
import cv2, os, shutil
import numpy as np
import csv
from pathlib import Path
from ultralytics import YOLO
from boxmot import BotSort

def get_video_cfg(path):
    video = cv2.VideoCapture(path)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return fourcc, size, fps

def counting(image_plot, result, class_names, colors):
    # 获取所有检测框的类别
    classes = result.boxes.cls.cpu().detach().numpy().astype(int)
    unique_classes, counts = np.unique(classes, return_counts=True)
    
    # 显示每个类别的计数
    y_offset = 50
    box_padding = 10  # 为框增加一些内边距，确保文字不被框遮挡
    box_height = 30   # 为每个标签设置固定高度
    
    for cls, count in zip(unique_classes, counts):
        label = f"{class_names[cls]}: {count}"
        color = tuple(int(c) for c in colors[cls])  # 转换为整数元组

        # 计算框的宽度，考虑到文字长度
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        box_width = text_width + 2 * box_padding
        
        # 绘制背景框
       # cv2.rectangle(image_plot, (10, y_offset - box_padding), (10 + box_width, y_offset + box_height), color, -1)

        # 在框内绘制文字
       # cv2.putText(image_plot, label, (10 + box_padding, y_offset + box_height - box_padding),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # 更新 y_offset，以便下一个框不会与上一个重叠
        y_offset += box_height + box_padding

    return image_plot

def count_worms(image_plot, result):
    box_count = result.boxes.shape[0]
    
    return image_plot

def transform_mot(result):
    mot_result = []
    for i in range(result.boxes.shape[0]):   
        mot_result.append(result.boxes.xyxy[i].cpu().detach().numpy().tolist() + [float(result.boxes.conf[i]), float(result.boxes.cls[i])])
    return np.array(mot_result)

def estimate_speed(tracks, fps, frame_id, speeds_dict, csv_writer):
    """
    估算目标的速度并实时保存每秒的平均速度到CSV文件，同时记录质心坐标。
    """
    for track in tracks:
        track_id = int(track[4])  # 提取轨迹ID
        x1, y1, x2, y2 = track[:4]
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])  # 计算中心点

        if isinstance(track, np.ndarray):
            track = track.tolist()

        if track_id not in speeds_dict:
            speeds_dict[track_id] = {"previous_center": center, "speeds": [], "frame_count": 0, "centers": []}

        if frame_id == 0:
            speed = 0
            track.append(center.tolist())
            speeds_dict[track_id]["previous_center"] = center
        else:
            previous_center = np.array(speeds_dict[track_id]["previous_center"])
            distance = np.linalg.norm(center - previous_center)
            speed = distance
            speeds_dict[track_id]["previous_center"] = center

        speeds_dict[track_id]["speeds"].append(speed)
        speeds_dict[track_id]["centers"].append(center)
        speeds_dict[track_id]["frame_count"] += 1

def save_average_speed(speeds_dict, frame_id, fps, csv_writer):
    second = frame_id // fps
    
    for track_id, data in speeds_dict.items():
        if data["frame_count"] >= fps:
            avg_speed = np.mean(data["speeds"])
            avg_center = np.mean(data["centers"], axis=0)
            
            avg_speed = f"{avg_speed:.3f}"
            avg_center_x = f"{avg_center[0]:.3f}"
            avg_center_y = f"{avg_center[1]:.3f}"
            
            csv_writer.writerow([second, track_id, avg_speed, avg_center_x, avg_center_y])
            
            speeds_dict[track_id]["speeds"] = []
            speeds_dict[track_id]["centers"] = []
            speeds_dict[track_id]["frame_count"] = 0

if __name__ == '__main__':
    output_base_dir = 'youtube-tracker'
    
    # 创建 result-boxmot 文件夹（如果不存在）
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    model = YOLO('/root/projects/WormReIDTracker-mogui/best/worm-egg.pt')
    
    video_base_path = 'video'
    class_names = model.model.names if hasattr(model.model, 'names') else model.names
    num_classes = len(class_names)
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype='uint8')
    
    # 设置是否启用速度估计
    estimate_speed_enabled = False  # 设置为 False 时不进行速度估算和 CSV 文件生成
    
    for video_path in os.listdir(video_base_path):
        if '.ipynb_checkpoints' in video_path:
            continue
        
        video_name = video_path.split('.')[0]  # 获取视频名（不带扩展名）
        video_output_dir = os.path.join(output_base_dir, video_name)
        
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        tracker = BotSort(
            reid_weights=Path('resnet50_model_best.pt'),
            device='cuda:0',
            half=False,
            per_class=True,
            track_high_thresh=0.5,
            track_low_thresh=0.01,
            new_track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.8,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            cmc_method="sof",
            frame_rate=2.5,
            fuse_first_associate=False,
            with_reid=True,
        )
        
        fourcc, size, fps = get_video_cfg(f'{video_base_path}/{video_path}')
        video_output = cv2.VideoWriter(f'{video_output_dir}/{video_name}.avi', fourcc, fps, size)

        imgsz = size[0]
        
        txt_output_path = os.path.join(video_output_dir, f'{video_name}.txt')
        with open(txt_output_path, 'w') as f:
            frame_id = 0
            
            # 只有启用速度估计时才会创建 CSV 文件
            if estimate_speed_enabled:
                csv_output_path = os.path.join(video_output_dir, f'{video_name}_speed_centroids.csv')
                with open(csv_output_path, 'w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["Second", "Track_ID", "Speed(m/s)", "Centroid_X", "Centroid_Y"])
                    
                    speeds_dict = {}
                
                    for result in model.predict(source=f'{video_base_path}/{video_path}',
                                                stream=True,
                                                imgsz=480,
                                                save=False,
                                                #conf=0.1
                                                ):
                        image_plot = result.orig_img
                        mot_input = transform_mot(result)

                        try:
                            tracks = tracker.update(mot_input, image_plot)
                        except:
                            continue        


                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy.cpu().detach().numpy().astype(int)[0]
                            cls = int(box.cls.cpu().detach().numpy()[0])
                            label = f"{class_names[cls]}"
                            if label == 'worm':  
                                color =(255, 0,0)  # 红色
                            else:
                                color = tuple(int(c) for c in colors[cls]) if cls < num_classes and cls >=0 else (255, 0,0)
                            cv2.rectangle(image_plot, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(image_plot, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        for track in tracks:
                            bbox = track[:4]
                            track_id = int(track[4])
                            f.write(f'{frame_id},{track_id},{bbox[0]},{bbox[1]},{bbox[2]-bbox[0]},{bbox[3]-bbox[1]},-1,-1,-1\n')

                        # 只有启用速度估计时才会进行速度计算
                        if estimate_speed_enabled:
                            estimate_speed(tracks, fps, frame_id, speeds_dict, csv_writer)
                            save_average_speed(speeds_dict, frame_id, fps, csv_writer)
                        
                        image_plot = counting(image_plot, result, class_names, colors)
                        video_output.write(image_plot)
                        
                        frame_id += 1
            else:
                # 当速度估算禁用时，跳过 CSV 文件相关操作
                for result in model.predict(source=f'{video_base_path}/{video_path}',
                                            stream=True,
                                            imgsz=640,
                                            save=False,
                                            #conf=0.1
                                            ):
                    image_plot = result.orig_img
                    mot_input = transform_mot(result)

                    try:
                        tracks = tracker.update(mot_input, image_plot)
                    except:
                        continue        

                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy.cpu().detach().numpy().astype(int)[0]
                        cls = int(box.cls.cpu().detach().numpy()[0])
                        label = f"{class_names[cls]}"                 
                        if label == 'worm':  
                            color = (0, 0, 255)  # 红色
                        else:
                            color = tuple(int(c) for c in colors[cls]) if cls < num_classes and cls >=0 else (255, 0,0)
                        cv2.rectangle(image_plot, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(image_plot, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    for track in tracks:
                        bbox = track[:4]
                        track_id = int(track[4])
                        f.write(f'{frame_id},{track_id},{bbox[0]},{bbox[1]},{bbox[2]-bbox[0]},{bbox[3]-bbox[1]},-1,-1,-1\n')
                    
                    image_plot = counting(image_plot, result, class_names, colors)
                    video_output.write(image_plot)
                    
                    frame_id += 1

        video_output.release()

    print(f"所有视频处理完成，输出结果保存在: {output_base_dir}")

