import cv2
import numpy as np
import time
import serial
import struct
from collections import deque

# ==============================================================
# 配置参数（固定你调好的最终参数）
# ==============================================================
SERIAL_PORT = "/dev/ttyTHS1"
CAM_DEVICE = "/dev/video0"
W, H = 640, 480

A4_MIN_AREA = 5000
A4_MAX_AREA = 40000
CANNY_LOW = 30
CANNY_HIGH = 100
A4_EPSILON = 0.03
A4_RATIO_MIN = 0.5
A4_RATIO_MAX = 3.0

KF_PROCESS = 0.1
KF_MEASURE = 0.3
LOST_FRAMES_THRESHOLD = 3

ROI_BASE_EXPAND = 100
ROI_SPEED_FACTOR = 8.0
ROI_MIN_SIZE = 200
ROI_MAX_SIZE = W
MISS_RETRY_FRAMES = 2

SUBPIX_MARGIN = 5

# ==============================================================
# 串口状态机
# ==============================================================
STATE_WAIT_RESET = 0
STATE_WAIT_ACK = 1
STATE_DATA_TRANSFER = 2
current_state = STATE_WAIT_RESET
rx_buffer = bytearray()

# ==============================================================
# 工具函数
# ==============================================================
def float_to_bytes(x, y):
    return struct.pack('<ff', float(x), float(y))

def send_data(ser, data):
    try:
        ser.write(data)
    except Exception as e:
        pass

def order_points(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(1)
    d = np.diff(pts, 1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)]
    ], np.float32)

# ==============================================================
# 卡尔曼滤波
# ==============================================================
class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * KF_PROCESS
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KF_MEASURE
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.5
        self.last_pos = np.array([W / 2, H / 2], dtype=np.float32)
        self.last_speed = np.array([0.0, 0.0], dtype=np.float32)

    def correct(self, x, y):
        meas = np.array([[np.float32(x)], [np.float32(y)]], dtype=np.float32)
        self.kf.predict()
        res = self.kf.correct(meas)
        self.last_pos = np.array([res[0, 0], res[1, 0]], dtype=np.float32)
        self.last_speed = np.array([res[2, 0], res[3, 0]], dtype=np.float32)
        return self.last_pos

    def predict_only(self):
        pred = self.kf.predict()
        self.last_pos = np.array([pred[0, 0], pred[1, 0]], dtype=np.float32)
        self.last_speed = np.array([pred[2, 0], pred[3, 0]], dtype=np.float32)
        return self.last_pos

# ==============================================================
# 激光识别 —— 只在A4内部识别
# ==============================================================
def detect_laser(frame, a4_contour=None):
    if a4_contour is None:
        return None
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 紫色激光阈值
        lower_purple = np.array([0, 0, 200])
        upper_purple = np.array([150, 80, 255])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)

        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 3:
            return None

        M = cv2.moments(cnt)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # 核心：判断激光点是否在A4轮廓内部
        is_inside = cv2.pointPolygonTest(a4_contour, (cx, cy), False) >= 0
        if not is_inside:
            return None

        return (cx, cy)
    except:
        return None

# ==============================================================
#A4纸的检测识别
# ==============================================================
def detect_a4_predict(frame, roi_center=None, roi_expand=ROI_BASE_EXPAND):
    try:
        full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_h, img_w = full_gray.shape[:2]

        search_gray = full_gray
        offset_x, offset_y = 0, 0
        roi_used = False

        if roi_center is not None:
            cx, cy = roi_center
            half_size = int(roi_expand // 2)
            x1 = max(0, int(cx) - half_size)
            y1 = max(0, int(cy) - half_size)
            x2 = min(img_w, int(cx) + half_size)
            y2 = min(img_h, int(cy) + half_size)

            if (x2 - x1) >= ROI_MIN_SIZE and (y2 - y1) >= ROI_MIN_SIZE:
                search_gray = full_gray[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1
                roi_used = True

        gray = cv2.boxFilter(search_gray, -1, (3, 3))
        edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        try:
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None, roi_used

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        min_area = A4_MIN_AREA * 0.7 if roi_used else A4_MIN_AREA
        max_area = A4_MAX_AREA
        if area < min_area or area > max_area:
            return None, None, None, roi_used

        epsilon = A4_EPSILON * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) != 4 or not cv2.isContourConvex(approx):
            return None, None, None, roi_used

        approx[:, :, 0] += offset_x
        approx[:, :, 1] += offset_y

        ordered_pts = order_points(approx)
        tl, tr, br, bl = ordered_pts
        avg_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        avg_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2

        min_side = min(avg_w, avg_h)
        if min_side < 1.0:
            return None, None, None, roi_used

        ar = max(avg_w, avg_h) / min_side
        if not (A4_RATIO_MIN <= ar <= A4_RATIO_MAX):
            return None, None, None, roi_used

        final_pts = ordered_pts.copy()
        try:
            safe_pts = []
            for pt in ordered_pts.reshape(-1, 2):
                x, y = pt
                safe_x = np.clip(x, SUBPIX_MARGIN, img_w - SUBPIX_MARGIN)
                safe_y = np.clip(y, SUBPIX_MARGIN, img_h - SUBPIX_MARGIN)
                safe_pts.append([safe_x, safe_y])

            safe_pts = np.array(safe_pts, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)
            final_pts = cv2.cornerSubPix(full_gray, safe_pts, (5, 5), (-1, -1), criteria)
        except:
            final_pts = ordered_pts.reshape(-1, 2)

        c = np.mean(final_pts, axis=0)
        return (float(c[0]), float(c[1])), approx, None, roi_used

    except Exception as e:
        return None, None, None, False

def process_serial_state_machine(ser, delta_x, delta_y):
    global current_state, rx_buffer
    if ser.in_waiting > 0:
        new_data = ser.read(ser.in_waiting)
        rx_buffer.extend(new_data)

    reset_seq = b'\xA5\xFF\x5A'
    ack_seq = b'\xA5\x01\x5A'

    if current_state == STATE_WAIT_RESET:
        send_data(ser, b'\x01')
        if reset_seq in rx_buffer:
            rx_buffer.clear()
            current_state = STATE_WAIT_ACK
            print("[INFO] 收到重置指令，进入等待ACK")

    elif current_state == STATE_WAIT_ACK:
        send_data(ser, b'\x01')
        if ack_seq in rx_buffer:
            rx_buffer.clear()
            current_state = STATE_DATA_TRANSFER
            print("[INFO] 握手成功，进入数据发送")

    elif current_state == STATE_DATA_TRANSFER:
        # 发送：激光坐标 - A4中心坐标 的差值
        packet = b'\xB6' + float_to_bytes(delta_x, delta_y) + b'\xB6'
        send_data(ser, packet)
        if reset_seq in rx_buffer:
            rx_buffer.clear()
            current_state = STATE_WAIT_ACK
            print("[INFO] 收到重置，重新握手")

    if len(rx_buffer) > 100:
        rx_buffer = rx_buffer[-100:]

# ==============================================================
# 主程序
# ==============================================================
if __name__ == "__main__":
    # 初始化串口
    try:
        ser = serial.Serial(SERIAL_PORT, 115200, timeout=0)
        ser.reset_input_buffer()
        print("[OK] 串口就绪")
    except Exception as e:
        print(f"[ERROR] 串口失败: {e}")
        exit()

    # 初始化摄像头 默认参数
    cap = cv2.VideoCapture(CAM_DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("[ERROR] 摄像头打开失败")
        exit()
    print("[OK] 摄像头就绪")

    # 全局变量初始化
    kf = KalmanFilter()
    last_center = (W // 2, H // 2)
    laser_center = None
    lost_counter = 0
    target_locked = False
    last_approx = None

    fps_queue = deque(maxlen=30)
    last_tick = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # FPS
        now = time.time()
        fps = 1.0 / (now - last_tick) if (now - last_tick) > 0 else 0
        last_tick = now
        fps_queue.append(fps)
        avg_fps = int(np.mean(fps_queue))

        # 动态ROI
        predict_center = kf.predict_only() if target_locked else None
        dynamic_roi = ROI_BASE_EXPAND
        if target_locked:
            speed = np.linalg.norm(kf.last_speed)
            dynamic_roi = ROI_BASE_EXPAND + speed * ROI_SPEED_FACTOR
            dynamic_roi = np.clip(dynamic_roi, ROI_BASE_EXPAND, ROI_MAX_SIZE)

        # A4检测
        result, approx, _, roi_used = detect_a4_predict(frame, predict_center, dynamic_roi)

        # 激光识别：只传入A4轮廓，**仅纸面内生效**
        if approx is not None:
            laser_center = detect_laser(frame, approx)
        else:
            laser_center = None

        if result:
            x, y = result
            lost_counter = 0
            if not target_locked:
                target_locked = True
                kf = KalmanFilter()
            smooth = kf.correct(x, y)
            last_center = (int(smooth[0]), int(smooth[1]))
            last_approx = approx
        else:
            if target_locked:
                lost_counter += 1
                if lost_counter >= MISS_RETRY_FRAMES:
                    target_locked = False
                    lost_counter = 0

        # ===================== 核心修改：计算坐标差值 =====================
        if laser_center is not None:
            delta_x = laser_center[0] - last_center[0]
            delta_y = laser_center[1] - last_center[1]
        else:
            delta_x = 0.0
            delta_y = 0.0

        # 串口状态机（发送差值）
        process_serial_state_machine(ser, delta_x, delta_y)

        # 绘图
        if last_approx is not None:
            cv2.drawContours(frame, [last_approx], -1, (0, 255, 0), 2)
        cv2.circle(frame, last_center, 5, (0, 0, 255), -1)

        # 绘制激光点（只在A4里才会显示）
        if laser_center is not None:
            lx, ly = laser_center
            cv2.circle(frame, (int(lx), int(ly)), 6, (255, 0, 255), -1)

        # 状态文字
        state_text = {
            STATE_WAIT_RESET: "WAIT_RESET(SEND 01)",
            STATE_WAIT_ACK: "WAIT_ACK(SEND 01)",
            STATE_DATA_TRANSFER: "DATA_SEND"
        }[current_state]

        cv2.putText(frame, f"State: {state_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {avg_fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"A4 X:{last_center[0]:.1f} Y:{last_center[1]:.1f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # 显示差值
        cv2.putText(frame, f"Delta: X:{delta_x:.1f} Y:{delta_y:.1f}",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 显示激光坐标
        if laser_center:
            cv2.putText(frame, f"Laser: X{laser_center[0]:.1f} Y{laser_center[1]:.1f}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        cv2.imshow("Result", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    ser.close()
    cv2.destroyAllWindows()