import cv2
import pytesseract
import os
from moviepy.editor import VideoFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from collections import deque

kill_queue = deque(maxlen=3)  # 存储最近三次识别的击杀数

os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/'

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    alpha = 1.5
    beta = 0
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    denoised = cv2.fastNlMeansDenoising(adjusted, h=30)
    _, binary = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def get_kill_words_frame(image, width_ratio=0.1, top_ratio=0.15, bottom_ratio=0.2):
    # h, w = image.shape[:2]  # 获取图像的高和宽
    # print("h : "+str(h)+",w ="+str(w))
    
    # # 计算裁剪区域的坐标
    # x1 = int((w * (1 - width_ratio)) / 2)  # 居中裁剪
    # x2 = int((w * (1 + width_ratio)) / 2)
    
    # y1 = int(h * top_ratio)    # top = 0.1 * height
    # y2 = int(h * bottom_ratio)  # bottom = 0.2 * height
    
    # # 输出裁剪的坐标，便于调试
    # print(f"裁剪区域：x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    x1, y1, x2, y2 = 1110, 234, 1285, 273
    # 裁剪图像
    return image[y1:y2, x1:x2]

def get_kda_image(image):
    x1, y1, x2, y2 = 1800, 127, 1839 , 158
    return image[y1:y2, x1:x2]

def detect_kill_events(video_path, log_file):
    print("Opening video:", video_path)
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print("Error: Could not open video.")
        return []

    fps = video.get(cv2.CAP_PROP_FPS)
    kill_times = []
    # kill_words = {"帅气的","干军","千军","团战","下无","无人能","二连", "三连", "四连", "五连"}
    i = 0
    
    with open(log_file, 'w', encoding='utf-8') as log:
        previous_kill = 0
        while True:
            ret, frame = video.read()
            if not ret:
                print("End of video or error reading frame.")
                break
            i += 1
            # if i <7700:
            #     continue
            # if i> 7711:
            #     break
            if i % 23 == 0:
                # kill_words_frame = get_kill_words_frame(frame)
                kda_frame = get_kda_image(frame)
                # kill_word_image_path = f"image/{i}killwords.png"
                full_image_path = f"image/{i}full.png"
                kda_image_path = f"image/{i}kda.png"
                cv2.imwrite(full_image_path, frame)     
                # cv2.imwrite(kill_word_image_path, kill_words_frame)  
                cv2.imwrite(kda_image_path, kda_frame)       
                # processed_frame = preprocess_image(kill_words_frame)
                # custom_config = r'--oem 3 --psm 6'
                # text = pytesseract.image_to_string(processed_frame, lang='chi_sim', config=custom_config)
                # 初始化变量
                # 初始化变量
                kill_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
                kill = pytesseract.image_to_string(kda_frame, lang='chi_sim', config=kill_config).strip()
                current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
                try:
                    # 尝试将识别到的 kill 转换为整数
                    kill_value = int(kill)
                    log.write(f"{current_time:.2f}, text = {kill}\n")
                    print(f"Time: {current_time:.2f}，当前击杀数：{kill}，{kill_value}")
                    kill_queue.append(kill_value)
                except ValueError:
                    kill_queue.clear
                # 检查条件：stable_kill_value 存在且递增且小于 30
                # 检查队列中是否有连续三次相同的值
                if len(kill_queue) == 3 and len(set(kill_queue)) == 1:
                    stable_kill_value = kill_queue[0]
                    if stable_kill_value is not None and stable_kill_value - previous_kill>0 and stable_kill_value - previous_kill<3 and kill_value < 30:
                        log_entry = f'Time: {current_time:.2f}s, Text: {kill}\n'
                        log.write(log_entry)
                        print(log_entry)
                        kill_times.append(current_time - 3)  # 记录时间戳
                        print(f"之前击杀数: {previous_kill}，更新为：{stable_kill_value}")
                        previous_kill = stable_kill_value  # 更新上一个 kill 值
                # if any(keyword in text for keyword in kill_words):
                #     kill_times.append(current_time)
                #     log_entry = f'Time: {current_time:.2f}s, Text: {text.strip()}\n'
                #     log.write(log_entry)
                #     print(log_entry)
    
    video.release()
    return kill_times

def clip_video_around_times(video_path, output_path, kill_times, duration=10):
    # Load the video
    video = VideoFileClip(video_path)
    clips = []
    
    # Sort the kill times in case they are not sorted
    kill_times.sort()
    
    # Initialize the start and end times of the first subclip
    current_start_time = max(kill_times[0] - duration / 2, 0)
    current_end_time = min(kill_times[0] + duration / 2, video.duration)
    
    for t in kill_times[1:]:
        start_time = max(t - duration / 2, 0)
        end_time = min(t + duration / 2, video.duration)
        
        # If the new clip overlaps or is adjacent to the previous one, merge them
        if start_time <= current_end_time:
            # Extend the current clip
            current_end_time = max(current_end_time, end_time)
        else:
            # Append the current clip and start a new one
            print(f"Creating subclip from {current_start_time} to {current_end_time}")
            clip = video.subclip(current_start_time, current_end_time)
            clips.append(clip)
            
            # Start a new clip
            current_start_time = start_time
            current_end_time = end_time
    
    # Append the last clip
    print(f"Creating subclip from {current_start_time} to {current_end_time}")
    clip = video.subclip(current_start_time, current_end_time)
    clips.append(clip)
    
    # Concatenate all the subclips and write them to the output video
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_path, codec="libx264")


# def clip_video_around_times(video_path, output_path, kill_times, duration=10):
#     # Load the video
#     video = VideoFileClip(video_path)
#     clips = []
    
#     # Create a subclip around each kill time (-10 to +10 seconds)
#     for t in kill_times:
#         start_time = max(t - duration / 2, 0)  # Ensure no negative start times
#         end_time = min(t + duration / 2, video.duration)
#         print(f"Creating subclip from {start_time} to {end_time}")
#         clip = video.subclip(start_time, end_time)
#         clips.append(clip)
    
#     # Concatenate all the subclips and write them to the output video
#     final_clip = concatenate_videoclips(clips)
#     final_clip.write_videofile(output_path, codec="libx264")

# 示例使用
video_path = 'a.mp4'
log_file = 'kill_events_log.txt'
kill_times = detect_kill_events(video_path, log_file)
# kill_times = [317.18100000000004, 416.63800000000003, 492.387, 720.653, 721.5980000000001, 729.173]
print("Detected kill times:", kill_times)
# 将每个击杀时间前后10秒剪辑到新视频中
output_path = 'kill_clips.mp4'
clip_video_around_times(video_path, output_path, kill_times)
