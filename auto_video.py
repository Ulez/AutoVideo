import cv2
import pytesseract
import os

os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/'

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    alpha = 1.5
    beta = 0
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    denoised = cv2.fastNlMeansDenoising(adjusted, h=30)
    _, binary = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def crop_to_center(image, width_ratio=0.1, top_ratio=0.15, bottom_ratio=0.2):
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

def detect_kill_events(video_path, log_file):
    print("Opening video:", video_path)
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print("Error: Could not open video.")
        return []

    fps = video.get(cv2.CAP_PROP_FPS)
    kill_times = []
    kill_words = {"二连", "三连", "四连", "五连"}
    i = 0
    
    with open(log_file, 'w', encoding='utf-8') as log:
        while True:
            ret, frame = video.read()
            if not ret:
                print("End of video or error reading frame.")
                break
            i += 1
            if i % 23 == 0:
                cropped_frame = crop_to_center(frame)
                output_image_path = f"image/{i}tt.png"
                # output_image_path_full = f"image/{i}full.png"
                # cv2.imwrite(output_image_path_full, frame)     
                cv2.imwrite(output_image_path, cropped_frame)       
                processed_frame = preprocess_image(cropped_frame)
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(processed_frame, lang='chi_sim', config=custom_config)
                log.write(text)
                print(text)
                if any(keyword in text for keyword in kill_words):
                    current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    print(text)
                    kill_times.append(current_time)
                    log_entry = f'Time: {current_time:.2f}s, Text: {text.strip()}\n'
                    log.write(log_entry)
                    print(log_entry)
    
    video.release()
    return kill_times

# 示例使用
video_path = 'a.mp4'
log_file = 'kill_events_log.txt'
kill_times = detect_kill_events(video_path, log_file)
print("Detected kill times:", kill_times)
