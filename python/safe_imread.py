import cv2
import numpy as np

def safe_imread(path):
    try:
        with open(path, 'rb') as f:
            img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"❌ 読み込み失敗: {path} - {e}")
        return None