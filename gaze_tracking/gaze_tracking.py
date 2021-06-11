from __future__ import division
import os
import cv2
import dlib
from .eye import Eye
from .calibration import Calibration


class GazeTracking(object):
    
    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(
            cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):

        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):

        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)


        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):

        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):

        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):

        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """right 0.0 center 0.5 left 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / \
                (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / \
                (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):

        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / \
                (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / \
                (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):

        # if self.pupils_located:
        #     return self.horizontal_ratio() <= 0.35
        if self.pupils_located:
            if self.horizontal_ratio() <= 0.45:
                return True

    def is_left(self):

        # if self.pupils_located:
        #     return self.horizontal_ratio() >= 0.65 # and self.vertical_ratio >= 0.35
        if self.pupils_located:
            if self.horizontal_ratio() >= 0.7:
                return True

    def is_center(self):

        # if self.pupils_located:
        #     return self.is_right() is not True and self.is_left() is not True and self.is_up() is not True and self.is_down() is not True
        if self.pupils_located:
            if self.is_right() is not True and self.is_left() is not True and self.is_up() is not True and self.is_down() is not True:
                return True

    def is_up(self):
        """Returns true if the user is looking to the up"""
        # if self.pupils_located:
        #     return self.vertical_ratio() <= 0.65
        if self.pupils_located:
            if self.vertical_ratio() <= 0.5 and self.is_right() is not True and self.is_left() is not True:
                return True

    def is_down(self):

        # if self.pupils_located:
        #     return self.vertical_ratio() >= 0.65
        if self.pupils_located:
            if self.vertical_ratio() >= 0.65 and self.is_right() is not True and self.is_left() is not True:
                return True

    def head_direction(self):

        left_ear=landmarks.part(2)
        right_ear=landmarks.part(16)
        nose=landmarks.part(30)
        head_ratio= abs(left_ear-nose)/abs(right_ear-nose)

        if head_ratio > 1:
            return right
        if head_ration < 1:
            return left

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right),
                     (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5),
                     (x_right, y_right + 5), color)

        return frame
