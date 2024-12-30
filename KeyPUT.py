import ctypes
from ctypes import wintypes

# WinAPI SendInput 함수 및 키 입력 구조체 정의
SendInput = ctypes.windll.user32.SendInput

# 스캔캔 코드 맵핑
SCAN_CODE = {
    'backspace': 0x0E, 'tab': 0x0F, 'enter': 0x1C, 'LeftShift': 0x2A, 'RightShift': 0x36,
    'ctrl': 0x1D, 'alt': 0x38, 'pause': 0x45, 'caps_lock': 0x3A,
    'esc': 0x01, 'space': 0x39, 'page_up': 0x49, 'page_down': 0x51,
    'end': 0x4F, 'home': 0x47, 'left_arrow': 0x4B, 'up_arrow': 0x48,
    'right_arrow': 0x4D, 'down_arrow': 0x50, 'print_screen': 0x37,
    'insert': 0x52, 'delete': 0x53, ',':0x33, '.':0x34, '?':0x35, 
    '0': 0x0B, '1': 0x02, '2': 0x03,'3': 0x04, '4': 0x05, '5': 0x06, 
    '6': 0x07, '7': 0x08, '8': 0x09,'9': 0x0A, 'a': 0x1E, 'b': 0x30, 
    'c': 0x2E, 'd': 0x20, 'e': 0x12,'f': 0x21, 'g': 0x22, 'h': 0x23, 
    'i': 0x17, 'j': 0x24, 'k': 0x25,'l': 0x26, 'm': 0x32, 'n': 0x31, 
    'o': 0x18, 'p': 0x19, 'q': 0x10,'r': 0x13, 's': 0x1F, 't': 0x14, 
    'u': 0x16, 'v': 0x2F, 'w': 0x11,'x': 0x2D, 'y': 0x15, 'z': 0x2C
}
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):		
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
    
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]
    
class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
    
class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]
    
class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]
    
def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

# 테스트 코드
if __name__ == "__main__":
    import time
    import os
    
    time.sleep(5)  # 메모장이 열릴 시간을 기다림

    for char in "hello world":
        if char == " ":
            PressKey(SCAN_CODE['space'])
            time.sleep(0.1)
            ReleaseKey(SCAN_CODE['space'])
        else:
            PressKey(SCAN_CODE[char])
            time.sleep(0.1)
            ReleaseKey(SCAN_CODE[char])