import cv2

class Camera():
    def __init__(self) -> None:
        self.vid = cv2.VideoCapture(0)

    def __enter__(self):
        print('Camera is open.')
        return self.vid

    def __exit__(self, *args):
        self.vid.release()
        cv2.destroyAllWindows()
        print('Camera is closed.')

def cropped_frame(frame: cv2.Mat, x: int, y: int, w: int, h: int) -> cv2.Mat:
    return frame[y:y+h, x:x+w]
