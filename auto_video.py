import cv2
import pytesseract
import os
from moviepy.editor import VideoFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from collections import deque
import shutil

kill_queue = deque(maxlen=3)  # 存储最近三次识别的击杀数
nagkill_queue = deque(maxlen=6)  # 存储最近6次识别的击杀数

# os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/'

before = 5
after = 4
w = 38
h = 30
# ipad:图像宽度: 1920, 高度: 1260
# 手机：图像宽度： 2400, 高度: 1080

coordinates = {
    "虎牙呆呆-安卓":   [1800, 127, 39, 30],  # x1, y1, width, height
    "虎牙呆呆-ipad":   [1561, 218, 38, 30], 
    "虎牙青帝-ipad":   [1516, 220, 38, 30], 
    "小七-安卓":   [1655, 141, 38, 30], 
    "虎牙小锦儿-ipad": [1516, 220, 38, 30]
}

def get_kda_image(image):
    # height, width = image.shape[:2]
    # print(f"输入图像宽度: {width}, 高度: {height}")
    x1, y1, w, h = coordinates["虎牙呆呆-ipad"]
    x2, y2 = x1 + w, y1 + h
    # x1, y1, x2, y2 = coordinates["虎牙呆呆-ipad"]
    return image[y1:y2, x1:x2]

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    alpha = 1.5
    beta = 0
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    denoised = cv2.fastNlMeansDenoising(adjusted, h=30)
    _, binary = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def get_kill_words_frame(image, width_ratio=0.1, top_ratio=0.15, bottom_ratio=0.2):
    x1, y1, x2, y2 = 1110, 234, 1285, 273
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
        previous_kill = None
        stable_kill_value = None
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
            if i % 13 == 0:
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
                kill_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
                kill = pytesseract.image_to_string(kda_frame, lang='chi_sim', config=kill_config).strip()
                current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
                print(f"{kill},,,current_time = "+str(current_time))
                try:
                    # 尝试将识别到的 kill 转换为整数
                    kill_value = int(kill)
                    log.write(f"{current_time:.2f}, text = {kill},{kda_image_path}\n")
                    print(f"Time: {current_time:.2f}，kill：{kill}，{i}kda.png")
                    kill_queue.append(kill_value)
                    nagkill_queue.append(kill_value)
                except ValueError:
                    print(f"error {current_time:.2f}, text = {kill}, {i}kda.png")
                    log.write(f"error {current_time:.2f}, text = {kill}, {kda_image_path}\n")
                    # kill_queue.clear
                    # nagkill_queue.clear
                # 检查条件：stable_kill_value 存在且递增且小于 30
                # 检查队列中是否有连续三次相同的值
                if len(kill_queue) == 3 and len(set(kill_queue)) == 1:
                    if previous_kill is None:
                        previous_kill = kill_queue[0]
                        continue
                    stable_kill_value = kill_queue[0]
                    if stable_kill_value - previous_kill>0 and stable_kill_value - previous_kill<5 and kill_value < 30:
                        log_entry = f'Time: {current_time:.2f}s, Text: {kill}\n'
                        log.write(log_entry)
                        print(log_entry)
                        kill_times.append(current_time - 1.5)  # 记录时间戳
                        print(str(kill_times))
                        print(f"kill:{previous_kill}->{stable_kill_value}, add time:{current_time - 1.5}")
                        previous_kill = stable_kill_value  # 更新上一个 kill 值
                if len(nagkill_queue) == 6 and len(set(nagkill_queue)) == 1:
                    if stable_kill_value!= nagkill_queue[0]:
                        stable_kill_value = nagkill_queue[0]
                        log_entry = f'fix!!! Time: {current_time:.2f}s, Text: {kill}\n'
                        log.write(log_entry)
                        print(log_entry)
                        kill_times.append(current_time - 3)  # 记录时间戳
                        print(str(kill_times))
                        print(f"fix kill:{previous_kill}->{stable_kill_value}, add time:{current_time - 3}")
                        previous_kill = stable_kill_value  # 更新上一个 kill 值戳
                # print(str(kill_times))
                # if any(keyword in text for keyword in kill_words):
                #     kill_times.append(current_time)
                #     log_entry = f'Time: {current_time:.2f}s, Text: {text.strip()}\n'
                #     log.write(log_entry)
                #     print(log_entry)
    
    video.release()
    return kill_times

def clip_video_around_times(video_path, output_path, kill_times):
    # Load the video
    video = VideoFileClip(video_path)
    clips = []

    # Sort the kill times in case they are not sorted
    kill_times.sort()

    # Initialize the start and end times for the first segment
    current_start_time = max(kill_times[0] - before, 0)  # 前10秒
    current_end_time = min(kill_times[0] + after, video.duration)  # 后5秒

    for t in kill_times[1:]:
        # 新的时间戳的起始和结束时间
        new_start_time = max(t - before, 0)
        new_end_time = min(t + after, video.duration)

        # 如果新时间戳在当前片段的范围内
        if new_start_time <= current_end_time + 10:  # 合并条件
            # Extend the current segment
            current_end_time = max(current_end_time, new_end_time)
        else:
            # Append the current segment and start a new one
            print(f"Creating subclip from {current_start_time} to {current_end_time}")
            clip = video.subclip(current_start_time, current_end_time)
            clips.append(clip)

            # Start a new segment
            current_start_time = new_start_time
            current_end_time = new_end_time

    # Don't forget to add the last segment
    print(f"Creating subclip from {current_start_time} to {current_end_time}")
    clip = video.subclip(current_start_time, current_end_time)
    clips.append(clip)

    # Concatenate all the subclips and write them to the output video
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_path, codec="libx264")

def process_video(video_path, output_path, log_file):
    kill_times = detect_kill_events(video_path, log_file)
    print(f"Detected kill times for {video_path}: {kill_times}")
    clip_video_around_times(video_path, output_path, kill_times)

def main(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        video_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f'clipped_{filename}')
        log_file = os.path.join(output_dir, f'{filename}_kill_events_log.txt')
        process_video(video_path, output_path, log_file)

if __name__ == "__main__":
    input_dir = 'input'
    output_dir = 'output'
    out_image_dir = "image"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    shutil.rmtree(out_image_dir)
    os.makedirs(out_image_dir, exist_ok=True)  # 重新创建空的 'image' 目录
    main(input_dir, output_dir)
