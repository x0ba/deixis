"""Gaze sources. Each runs on its own thread and writes into a GazeHistory.

`CameraGaze` runs the webcam through MediaPipe and a calibrated GazeModel.
`MouseGaze` treats the pointer as the gaze, which also covers head mice and
hardware eye trackers that drive the pointer.
"""

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from deixis.gaze import features as feat
from deixis.gaze.filter import OneEuro
from deixis.gaze.history import GazeHistory
from deixis.gaze.model import GazeModel

log = logging.getLogger(__name__)

PointListener = Callable[[float, float], None]
FeatureListener = Callable[[float, np.ndarray], None]


class CameraGaze:
    def __init__(
        self,
        history: GazeHistory,
        model_path: Path,
        calibration_path: Path,
        camera_index: int = 0,
    ):
        self.history = history
        self.model_path = model_path
        self.calibration_path = calibration_path
        self.camera_index = camera_index
        self.model: GazeModel | None = GazeModel.load(calibration_path)
        self.face_visible = False
        self.on_point: PointListener | None = None
        self.on_features: FeatureListener | None = None
        self.error: str | None = None
        self._filter = OneEuro()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def calibrated(self) -> bool:
        return self.model is not None

    def set_model(self, model: GazeModel) -> None:
        model.save(self.calibration_path)
        self.model = model
        self._filter.reset()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="camera-gaze", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            self._loop()
        except Exception as err:
            self.error = str(err)
            log.exception("camera gaze stopped")

    def _loop(self) -> None:
        import cv2
        import mediapipe as mp

        if not self.model_path.exists():
            raise FileNotFoundError(f"missing face model at {self.model_path}")
        vision = mp.tasks.vision
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        capture = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not capture.isOpened():
            raise RuntimeError(
                "could not open the camera; allow Camera access for this app in System Settings"
            )
        last_ms = 0
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                now = time.monotonic()
                ms = max(int(now * 1000), last_ms + 1)
                last_ms = ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, ms)
                self._handle(now, result, frame.shape[1], frame.shape[0])
        capture.release()

    def _handle(self, now: float, result, width: int, height: int) -> None:
        if not result.face_landmarks:
            self.face_visible = False
            return
        self.face_visible = True
        blend = {c.category_name: c.score for c in result.face_blendshapes[0]}
        if feat.is_blinking(blend):
            return
        landmarks = np.array([(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=float)
        matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=float)
        try:
            vector = feat.extract(landmarks, matrix, blend, width, height)
        except ValueError:
            return
        if self.on_features:
            self.on_features(now, vector)
        if self.model is None:
            return
        x, y = self.model.predict(vector)
        x, y = self._filter(now, float(x), float(y))
        self.history.add(now, x, y)
        if self.on_point:
            self.on_point(x, y)


class MouseGaze:
    def __init__(self, history: GazeHistory, hz: float = 30.0):
        self.history = history
        self.on_point: PointListener | None = None
        self.face_visible = True
        self.calibrated = True
        self.error: str | None = None
        self._period = 1.0 / hz
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="mouse-gaze", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        from Quartz import CGEventCreate, CGEventGetLocation

        while not self._stop.is_set():
            point = CGEventGetLocation(CGEventCreate(None))
            now = time.monotonic()
            self.history.add(now, point.x, point.y)
            if self.on_point:
                self.on_point(point.x, point.y)
            time.sleep(self._period)
