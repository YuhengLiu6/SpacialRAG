import numpy as np
import pytest

from spatial_rag import detector as detector_module


class _FakeScalar:
    def __init__(self, value):
        self._value = value

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self._value)


class _FakeBox:
    def __init__(self, bbox, conf, cls_idx):
        self.xyxy = [_FakeScalar(bbox)]
        self.conf = [_FakeScalar(conf)]
        self.cls = [_FakeScalar(cls_idx)]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLOWorld:
    def __init__(self, model_path):
        self.model_path = model_path
        self.names = {0: "chair"}
        self.set_classes_calls = []

    def set_classes(self, classes):
        self.set_classes_calls.append(list(classes))

    def __call__(self, image, verbose=False, device=None):
        return [_FakeResult([_FakeBox([1, 2, 30, 40], 0.95, 0)])]


def test_detector_yolo_world_sets_classes_and_formats_detections(tmp_path, monkeypatch):
    model_path = tmp_path / "yolov8s-world.pt"
    model_path.write_bytes(b"weights")
    fake_model = _FakeYOLOWorld(str(model_path))
    monkeypatch.setattr(detector_module, "YOLO", lambda _path: fake_model)

    detector = detector_module.Detector(
        detector_type="YOLO_WORLD",
        model_path=str(model_path),
        class_names="chair, table",
    )
    detections = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))

    assert fake_model.set_classes_calls == [["chair", "table"]]
    assert len(detections) == 1
    assert detections[0]["label"] == "chair"
    assert detections[0]["bbox"] == pytest.approx(np.asarray([1, 2, 30, 40]))
    assert detections[0]["confidence"] == pytest.approx(0.95)


def test_detector_yolo_world_requires_local_model_file(tmp_path):
    missing_model = tmp_path / "missing-world.pt"

    with pytest.raises(FileNotFoundError, match="YOLO-World model not found"):
        detector_module.Detector(detector_type="YOLO_WORLD", model_path=str(missing_model), class_names="chair")
