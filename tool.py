import argparse
import cv2
import os
from khmt import camera, BreakException
from typing import Tuple


def generate_training_set(image: cv2.Mat, hand_detect: Tuple[bool, cv2.Mat]):
    """Hàm xử lý để tạo ra tập huấn luyện

    Args:
        image (cv2.Mat): frame từ camera
        hand_detect (Tuple[bool, cv2.Mat]): has_hand, hand_image = hand_detect

    Raises:
        BreakException: Dừng camera khi bấn ESC
    """
    global isCapture, i
    has_hand, hand_image = hand_detect
    cv2.putText(image, f'{i}/{args.iter}', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("test", image)

    if isCapture and has_hand and i < args.iter:
        hand_image = cv2.resize(hand_image, (32, 32))
        cv2.imwrite(f'{path}/{i}.jpg', hand_image)
        i += 1

    match cv2.waitKey(5) & 0xFF:
        case 115:
            isCapture = True
        case 27:
            raise BreakException("Shutdown...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training data generator')
    parser.add_argument('classname', type=str)
    parser.add_argument('-i', '--iter', type=int, default=100)

    args = parser.parse_args()
    i = 0
    isCapture = False
    path = os.path.join('./data', args.classname)

    if not os.path.exists(path):
        os.makedirs(path)

    camera(generate_training_set)
