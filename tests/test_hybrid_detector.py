import numpy as np

from dronetrackingrl.masking.background_subtraction import MaskResult
from dronetrackingrl.masking.hybrid_detector import HybridReloDetector


class ScriptedClassic:
    """Klasik dedektör yerine: sırayla verilen merkezleri döndürür (None = tespit yok)."""

    def __init__(self, centroids, crop_size=8):
        self.centroids = list(centroids)
        self.crop_size = crop_size
        self.index = -1
        self.last_response = None
        self.last_scale = 1.0

    def reset(self):
        self.index = -1

    def update(self, frame):
        self.index += 1
        self.last_response = frame.astype(np.float32)
        centroid = self.centroids[self.index]
        crop = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
        return MaskResult(mask_crop=crop, centroid=centroid, visible=centroid is not None)


class ScriptedTracker:
    """RELO yerine: init çağrılarını kaydeder, sırayla (merkez, güven) döndürür."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.inits = []
        self.index = -1

    def init(self, frame, box_xywh):
        self.inits.append(tuple(box_xywh))

    def track(self, frame):
        self.index += 1
        (cx, cy), confidence = self.outputs[self.index]
        return (cx - 5.0, cy - 5.0, 10.0, 10.0), confidence


def _detector(classic, tracker, **kwargs):
    defaults = dict(init_streak=3, init_box_px=20.0, lost_threshold=0.2, lost_patience=2, persist_radius=20.0)
    defaults.update(kwargs)
    return HybridReloDetector(crop_size=8, classic=classic, tracker=tracker, **defaults)


FRAME = np.zeros((100, 100), dtype=np.uint8)


def test_search_mode_passes_classic_output_through_until_streak_reached():
    classic = ScriptedClassic([(10.0, 10.0), (12.0, 10.0), None, (14.0, 10.0)])
    tracker = ScriptedTracker([])
    detector = _detector(classic, tracker)
    results = [detector.update(FRAME) for _ in range(4)]
    assert [r.centroid for r in results] == [(10.0, 10.0), (12.0, 10.0), None, (14.0, 10.0)]
    assert not any(r.relo_tracking for r in results)
    assert tracker.inits == []  # bir boşluk seriyi sıfırladı


def test_streak_of_consistent_detections_starts_tracker_centered_on_detection():
    classic = ScriptedClassic([(10.0, 10.0), (12.0, 10.0), (14.0, 10.0), (99.0, 99.0)])
    tracker = ScriptedTracker([((16.0, 11.0), 0.9)])
    detector = _detector(classic, tracker)
    results = [detector.update(FRAME) for _ in range(4)]
    assert tracker.inits == [(4.0, 0.0, 20.0, 20.0)]  # (14,10) merkezli 20 px kutu
    assert results[2].centroid == (14.0, 10.0) and results[2].relo_tracking  # başlatma karesi
    assert results[2].relo_score is None
    assert results[3].centroid == (16.0, 11.0)  # artık RELO; klasiğin (99,99)'u yok sayılır
    assert results[3].visible and results[3].relo_tracking
    assert results[3].relo_score == 0.9 and results[3].relo_box_wh == (10.0, 10.0)


def test_jump_larger_than_persist_radius_restarts_the_streak():
    classic = ScriptedClassic([(10.0, 10.0), (12.0, 10.0), (80.0, 80.0), (82.0, 80.0)])
    tracker = ScriptedTracker([])
    detector = _detector(classic, tracker)
    for _ in range(4):
        detector.update(FRAME)
    assert tracker.inits == []


def test_low_confidence_for_patience_frames_returns_to_search():
    classic = ScriptedClassic([(10.0, 10.0)] * 3 + [None, None, (50.0, 50.0)])
    tracker = ScriptedTracker([((11.0, 10.0), 0.1), ((12.0, 10.0), 0.05), ((13.0, 10.0), 0.9)])
    detector = _detector(classic, tracker, lost_patience=2)
    results = [detector.update(FRAME) for _ in range(6)]
    assert results[3].relo_tracking and results[3].centroid == (11.0, 10.0)  # 1. düşük kare: hâlâ takip
    assert not results[4].relo_tracking and results[4].centroid is None  # 2. düşük kare: ARAMA, klasiğin çıktısı
    assert results[5].centroid == (50.0, 50.0) and not results[5].relo_tracking
    assert tracker.index == 1  # ARAMA'dayken RELO çağrılmaz


def test_crop_is_taken_from_classic_response_around_output_center():
    frame = np.zeros((100, 100), dtype=np.uint8)
    frame[40, 60] = 50  # yanıt haritasında tek parlak nokta (ScriptedClassic yanıt = kare)
    classic = ScriptedClassic([(10.0, 10.0)] * 3 + [None])
    tracker = ScriptedTracker([((60.0, 40.0), 0.9)])
    detector = _detector(classic, tracker, response_scale=100.0)
    for _ in range(3):
        detector.update(frame)
    result = detector.update(frame)
    assert result.mask_crop.shape == (8, 8)
    assert np.isclose(result.mask_crop[4, 4], 0.5)  # merkez pikseli, 50/100


def test_reset_clears_state_and_resets_both_components():
    classic = ScriptedClassic([(10.0, 10.0)] * 3 + [(10.0, 10.0)] * 3)
    tracker = ScriptedTracker([])
    detector = _detector(classic, tracker)
    for _ in range(3):
        detector.update(FRAME)
    assert detector.tracking
    detector.reset()
    assert not detector.tracking and classic.index == -1


def test_persistent_classic_track_far_from_relo_reanchors_the_tracker():
    far = [(80.0, 80.0), (81.0, 80.0), (82.0, 80.0)]
    classic = ScriptedClassic([(10.0, 10.0)] * 3 + far + [None])
    tracker = ScriptedTracker([((11.0, 10.0), 0.99)] * 2 + [((83.0, 80.0), 0.99)])
    detector = _detector(classic, tracker, reanchor_px=50.0)
    results = [detector.update(FRAME) for _ in range(7)]
    assert results[4].centroid == (11.0, 10.0)  # uzak iz henüz 2 kare: RELO'ya güven
    assert len(tracker.inits) == 2 and tracker.inits[1] == (72.0, 70.0, 20.0, 20.0)  # 3. uzak karede yeniden çapa
    assert results[5].centroid == (82.0, 80.0) and results[5].relo_tracking
    assert results[6].centroid == (83.0, 80.0)  # yeni çapadan takip sürüyor


def test_nearby_classic_detections_do_not_reanchor():
    classic = ScriptedClassic([(10.0, 10.0)] * 6)
    tracker = ScriptedTracker([((14.0, 10.0), 0.99)] * 3)
    detector = _detector(classic, tracker, reanchor_px=50.0)
    for _ in range(6):
        detector.update(FRAME)
    assert len(tracker.inits) == 1


class BigBoxTracker(ScriptedTracker):
    def track(self, frame):
        self.index += 1
        return (0.0, 0.0, 300.0, 200.0), 0.99


def test_exploded_box_counts_as_lost_immediately():
    classic = ScriptedClassic([(10.0, 10.0)] * 3 + [None])
    tracker = BigBoxTracker([])
    detector = _detector(classic, tracker, max_box_px=60.0)
    results = [detector.update(FRAME) for _ in range(4)]
    assert not results[3].relo_tracking and results[3].centroid is None  # klasiğin çıktısı
    assert not detector.tracking
