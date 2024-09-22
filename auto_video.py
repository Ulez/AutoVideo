import cv2
import pytesseract
import os

def preprocess_image(image):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 增强对比度
    alpha = 1.5  # 对比度控制（1.0-3.0）
    beta = 0    # 亮度控制（0-100）
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    # 去除噪声
    denoised = cv2.fastNlMeansDenoising(adjusted, h=30)
    
    # 二值化处理
    _, binary = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def crop_to_center(image, width_ratio=0.5, height_ratio=0.2):
    h, w = image.shape[:2]
    # print(f"原始宽高{w},{h}")
    x1 = 723
    y1 = 110
    
    x2= 877
    y2 = 142
    # x1 = int(w * (1 - width_ratio) / 2)
    # x2 = int(w * (1 + width_ratio) / 2)
    # y1 = int(h * (1 - height_ratio) / 2)-250
    # y2 = int(h * (1 + height_ratio) / 2)-250
    # print(f"裁剪到：{x1},{y1}, {x2}:{y2}")
    return image[y1:y2, x1:x2]

def detect_kill_events(video_path, log_file):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    kill_times = []
    kill_words = {"击败","二连","三连","三连","四连","五连"}
    i = 0
    with open(log_file, 'w', encoding='utf-8') as log:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            i+=1
            if i % 23 == 0:
                # 裁剪到中间区域
                cropped_frame = crop_to_center(frame)
                # 保存裁剪后的图片
                # output_image_path = f"{i}tt.png"
                # cv2.imwrite(output_image_path, cropped_frame)
                # 图像预处理
                processed_frame = preprocess_image(cropped_frame)
                # 使用OCR检测文本
                custom_config = r'--oem 3 --psm 6'  # 设置OCR引擎模式和页面分割模式
                text = pytesseract.image_to_string(processed_frame, lang='chi_sim', config=custom_config)  # 设置语言为简体中文
                if any(keyword in text for keyword in kill_words):
                    current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 当前帧的时间（秒）
                    print(text)
                    kill_times.append(current_time)
                    # 写入日志文件
                    log_entry = f'Time: {current_time:.2f}s, Text: {text.strip()}\n'
                    log.write(log_entry)
                    print(log_entry)
    
    video.release()
    return kill_times

# 示例使用
video_path = 'input_video.mp4'
log_file = 'kill_events_log.txt'
kill_times = detect_kill_events(video_path, log_file)
print("Detected kill times:", kill_times)
