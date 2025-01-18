'''import cv2
from threading import Thread
import time
import numpy as np


class WebcamVideoStream:
    def __init__(self, src = 0):
        print("init")
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        time.sleep(2.0)
    
    def start(self):
        print("start thread")
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    
    def update(self):
        print("read")
        while True:
            if self.stopped:
                return
            
            (self.grabbed, self.frame) = self.stream.read()
    
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True
        
        
'''



from threading import Thread
from picamera2 import Picamera2
import cv2
import face_rec1
import pandas as pd
import numpy as np

class WebcamVideoStream:
    def __init__(self):
        print("Initializing Picamera2")
        self.stream = Picamera2()
        self.stream.configure(self.stream.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
        self.stream.start()

        # Retrieve face recognition data
        name = 'ACADEMY:REGISTER'
        retrive_dict = face_rec1.r.hgetall(name)
        retrive_series = pd.Series(retrive_dict)
        retrive_series = retrive_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
        retrive_series.index = list(map(lambda x: x.decode(), retrive_series.index))
        self.retrive_df = retrive_series.to_frame().reset_index()
        self.retrive_df.columns = ['NAME_ROLE', 'FACIAL_FEATURES']
        self.retrive_df[['NAME', 'ROLE']] = self.retrive_df['NAME_ROLE'].apply(lambda x: x.split('@')).apply(pd.Series)

        self.stopped = False
        self.frame = None

    def start(self):
        print("Starting video stream thread")
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            frame = self.stream.capture_array()
            if frame.shape[2] == 4:  # If alpha channel exists, convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            # Perform face prediction
            self.frame = face_rec1.face_prediction(frame, self.retrive_df, 'FACIAL_FEATURES', ['NAME', 'ROLE'], thresh=0.5)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.stop()
