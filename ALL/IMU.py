#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

# import numpy as np
import sys

import serial  # 导入模块
import serial.tools.list_ports
import threading
import struct
import time
import platform
import transforms3d as tfs  # pip install transforms3d -i https://pypi.tuna.tsinghua.edu.cn/simple
# from copy import deepcopy
# import sys
# import os
# import math

from serial import EIGHTBITS, PARITY_NONE, STOPBITS_ONE

# 宏定义参数
PI = 3.141592653589793
FRAME_HEAD = str('fc')
FRAME_END = str('fd')
TYPE_IMU = str('40')
TYPE_AHRS = str('41')
TYPE_INSGPS = str('42')
TYPE_GEODETIC_POS = str('5c')
TYPE_GROUND = str('f0')
TYPE_SYS_STATE = str('50')
TYPE_BODY_ACCELERATION = str('62')
TYPE_ACCELERATION = str('61')
TYPE_MSG_BODY_VEL = str('60')
IMU_LEN = str('38')  # //56
AHRS_LEN = str('30')  # //48
INSGPS_LEN = str('48')  # //72
GEODETIC_POS_LEN = str('20')  # //32
SYS_STATE_LEN = str('64')  # // 100
BODY_ACCELERATION_LEN = str('10')  # // 16
ACCELERATION_LEN = str('0c')  # 12
DEG_TO_RAD = 0.017453292519943295
RAD_TO_DEG = 180.0 / PI  # 弧度转角度
isrun = True


# 获取命令行输入参数
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--debugs', type=bool, default=False, help='if debug info output in terminal ')
    # 【唯一修改：默认端口改为 Ubuntu 标准串口 /dev/ttyUSB0】
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0', help='the models serial port receive data; example: '
                                                                 '    Windows: COM3'
                                                                 '    Linux: /dev/ttyUSB0')

    parser.add_argument('--bps', type=int, default=115200, help='the models baud rate set; default: 921600')
    parser.add_argument('--timeout', type=int, default=20, help='set the serial port timeout; default: 20')
    # parser.add_argument('--device_type', type=int, default=0, help='0: origin_data, 1: for single imu or ucar in ROS')

    receive_params = parser.parse_known_args()[0] if known else parser.parse_args()
    return receive_params


# 接收数据线程
def receive_data():
    open_port()
    # 尝试打开串口
    try:
        serial_ = serial.Serial(port=opt.port, baudrate=opt.bps, bytesize=EIGHTBITS, parity=PARITY_NONE,
                                stopbits=STOPBITS_ONE,
                                timeout=opt.timeout)
        print("baud rates = " + str(serial_.baudrate))
    except:
        print("error:  unable to open port .")
        exit(1)
    # 循环读取数据
    while serial_.isOpen() and tr.is_alive():
        if not threading.main_thread().is_alive():
            print('done')
            break
        check_head = serial_.read().hex()
        # 校验帧头
        if check_head != FRAME_HEAD:
            continue
        head_type = serial_.read().hex()
        # 校验数据类型
        if (head_type != TYPE_IMU and head_type != TYPE_AHRS and head_type != TYPE_INSGPS and
                head_type != TYPE_GEODETIC_POS and head_type != 0x50 and head_type != TYPE_GROUND and
                head_type != TYPE_SYS_STATE and head_type != TYPE_MSG_BODY_VEL and head_type != TYPE_BODY_ACCELERATION and head_type != TYPE_ACCELERATION):
            continue
        check_len = serial_.read().hex()
        # 校验数据类型的长度
        if head_type == TYPE_IMU and check_len != IMU_LEN:
            continue
        elif head_type == TYPE_AHRS and check_len != AHRS_LEN:
            continue
        elif head_type == TYPE_INSGPS and check_len != INSGPS_LEN:
            continue
        elif head_type == TYPE_GEODETIC_POS and check_len != GEODETIC_POS_LEN:
            continue
        elif head_type == TYPE_SYS_STATE and check_len != SYS_STATE_LEN:
            continue
        elif head_type == TYPE_GROUND or head_type == 0x50:
            continue
        elif head_type == TYPE_MSG_BODY_VEL and check_len != ACCELERATION_LEN:
            print("check head type " + str(TYPE_MSG_BODY_VEL) + " failed;" + " check_LEN:" + str(check_len))
            continue
        elif head_type == TYPE_BODY_ACCELERATION and check_len != BODY_ACCELERATION_LEN:
            print("check head type " + str(TYPE_BODY_ACCELERATION) + " failed;" + " check_LEN:" + str(check_len))
            continue
        elif head_type == TYPE_ACCELERATION and check_len != ACCELERATION_LEN:
            print("check head type " + str(TYPE_ACCELERATION) + " failed;" + " ckeck_LEN:" + str(check_len))
            continue
        check_sn = serial_.read().hex()
        head_crc8 = serial_.read().hex()
        crc16_H_s = serial_.read().hex()
        crc16_L_s = serial_.read().hex()

        # 读取并解析IMU数据
        if head_type == TYPE_IMU:
            data_s = serial_.read(int(IMU_LEN, 16))
            IMU_DATA = struct.unpack('12f ii', data_s[0:56])

        # 读取并解析AHRS数据 → 【仅修改这里，提取你需要的角度/角速度】
        elif head_type == TYPE_AHRS:
            data_s = serial_.read(int(AHRS_LEN, 16))
            AHRS_DATA = struct.unpack('10f ii', data_s[0:48])

            # ===================== 核心提取参数 =====================
            # 1. 俯仰角速度 (rad/s) → 转为 °/s
            pitch_speed = AHRS_DATA[1] * RAD_TO_DEG
            # 2. 偏航角速度 (rad/s) → 转为 °/s
            yaw_speed = AHRS_DATA[2] * RAD_TO_DEG

            # 3. 俯仰角 (Pitch) → 转为 °
            pitch_angle = AHRS_DATA[4] * RAD_TO_DEG
            # 4. 偏航角 (Yaw/Heading) → 转为 °
            yaw_angle = AHRS_DATA[5] * RAD_TO_DEG

            # 打印输出（实时刷新）
            print(
                f"[AHRS] 俯仰角: {pitch_angle:.2f}° | 偏航角: {yaw_angle:.2f}° | 俯仰角速度: {pitch_speed:.2f}°/s | 偏航角速度: {yaw_speed:.2f}°/s")

        # 读取并解析INSGPS数据
        elif head_type == TYPE_INSGPS:
            data_s = serial_.read(int(INSGPS_LEN, 16))
            INSGPS_DATA = struct.unpack('16f ii', data_s[0:72])

        # 读取并解析GPS数据
        elif head_type == TYPE_GEODETIC_POS:
            data_s = serial_.read(int(GEODETIC_POS_LEN, 16))

        elif head_type == TYPE_SYS_STATE:
            data_s = serial_.read(int(SYS_STATE_LEN, 16))

        elif head_type == TYPE_BODY_ACCELERATION:
            data_s = serial_.read(int(BODY_ACCELERATION_LEN, 16))

        elif head_type == TYPE_ACCELERATION:
            data_s = serial_.read(int(ACCELERATION_LEN, 16))

        elif head_type == TYPE_MSG_BODY_VEL:
            data_s = serial_.read(int(ACCELERATION_LEN, 16))
            Velocity_X = struct.unpack('f', data_s[0:4])[0]
            Velocity_Y = struct.unpack('f', data_s[4:8])[0]
            Velocity_Z = struct.unpack('f', data_s[8:12])[0]
            print(f"Velocity_X: {Velocity_X}, Velocity_Y: {Velocity_Y}, Velocity_Z: {Velocity_Z}")


# 寻找输入的port串口
def find_serial():
    port_list = list(serial.tools.list_ports.comports())
    for port in port_list:
        if port.device == opt.port:
            return True
    return False


def open_port():
    if find_serial():
        print("find this port : " + opt.port)
    else:
        print("error:  unable to find this port : " + opt.port)
        exit(1)


def UsePlatform():
    sys_str = platform.system()
    if sys_str == "Windows":
        print("Call Windows tasks")
    elif sys_str == "Linux":
        print("Call Linux tasks")
    else:
        print("Other System tasks: %s" % sys_str)
    return sys_str


if __name__ == "__main__":
    print(UsePlatform())
    opt = parse_opt()
    tr = threading.Thread(target=receive_data)
    tr.start()
    while True:
        try:
            if tr.is_alive():
                time.sleep(1)
            else:
                break
        except(KeyboardInterrupt, SystemExit):
            break