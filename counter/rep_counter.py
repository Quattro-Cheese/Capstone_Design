from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Optional


@dataclass
class RepResult:
    timestamp_ms: int
    raw_signal: Optional[float]
    baseline: Optional[float]
    depth_now: Optional[float]
    peak_depth: Optional[float]
    count: int
    bpm: Optional[float]
    rate_feedback: str
    metronome_bpm: int
    beat_now: bool


class RepCounter:
    """
    CPR 반복 카운트 / BPM / 메트로놈 타이밍 계산용 독립 모듈

    입력:
        - timestamp_ms: 현재 시각(ms)
        - signal_value: 1차원 신호값
            예) 초음파 distance(cm), 혹은 다른 팀원이 만든 압박 관련 단일 값

    전제:
        - 압박할수록 값이 작아지는 신호라고 가정
          예: 초음파 distance는 누를수록 distance가 줄어듦
        - depth = baseline - current_value 로 계산
    """

    def __init__(
        self,
        calibration_samples: int = 15,
        enter_threshold_cm: float = 4.0,
        release_threshold_cm: float = 1.2,
        refractory_ms: int = 280,
        target_bpm: int = 110,
    ) -> None:
        self.calibration_samples = calibration_samples
        self.enter_threshold_cm = enter_threshold_cm
        self.release_threshold_cm = release_threshold_cm
        self.refractory_ms = refractory_ms
        self.target_bpm = target_bpm

        self._baseline_buffer: deque[float] = deque(maxlen=calibration_samples)
        self._baseline: Optional[float] = None

        self._count = 0
        self._compression_times: deque[int] = deque(maxlen=12)
        self._peak_depths: deque[float] = deque(maxlen=6)

        self._in_compression = False
        self._current_peak_depth = 0.0

        self._filtered_depth: Optional[float] = None
        self._last_peak_time: Optional[int] = None

        self._metronome_start_ms: Optional[int] = None

    def update(self, timestamp_ms: int, signal_value: Optional[float]) -> RepResult:
        if self._metronome_start_ms is None:
            self._metronome_start_ms = timestamp_ms

        if signal_value is None:
            bpm = self._calc_bpm()
            return RepResult(
                timestamp_ms=timestamp_ms,
                raw_signal=None,
                baseline=self._baseline,
                depth_now=None,
                peak_depth=self._latest_peak_depth(),
                count=self._count,
                bpm=bpm,
                rate_feedback=self._rate_feedback(bpm),
                metronome_bpm=self.target_bpm,
                beat_now=self._beat_now(timestamp_ms),
            )

        if self._baseline is None:
            self._baseline_buffer.append(signal_value)

            if len(self._baseline_buffer) >= self.calibration_samples:
                self._baseline = float(median(self._baseline_buffer))

            bpm = self._calc_bpm()
            return RepResult(
                timestamp_ms=timestamp_ms,
                raw_signal=signal_value,
                baseline=self._baseline,
                depth_now=None,
                peak_depth=self._latest_peak_depth(),
                count=self._count,
                bpm=bpm,
                rate_feedback="기준값 수집중",
                metronome_bpm=self.target_bpm,
                beat_now=self._beat_now(timestamp_ms),
            )

        # 압박이 아닐 때 baseline을 천천히 보정
        if not self._in_compression and signal_value >= self._baseline - 0.7:
            self._baseline = (self._baseline * 0.98) + (signal_value * 0.02)

        depth_now = max(0.0, self._baseline - signal_value)

        # smoothing
        if self._filtered_depth is None:
            self._filtered_depth = depth_now
        else:
            self._filtered_depth = (self._filtered_depth * 0.7) + (depth_now * 0.3)

        self._update_state(timestamp_ms, self._filtered_depth)

        bpm = self._calc_bpm()

        return RepResult(
            timestamp_ms=timestamp_ms,
            raw_signal=signal_value,
            baseline=self._baseline,
            depth_now=self._filtered_depth,
            peak_depth=self._latest_peak_depth(),
            count=self._count,
            bpm=bpm,
            rate_feedback=self._rate_feedback(bpm),
            metronome_bpm=self.target_bpm,
            beat_now=self._beat_now(timestamp_ms),
        )

    def reset(self) -> None:
        self._baseline_buffer.clear()
        self._baseline = None
        self._count = 0
        self._compression_times.clear()
        self._peak_depths.clear()
        self._in_compression = False
        self._current_peak_depth = 0.0
        self._filtered_depth = None
        self._last_peak_time = None
        self._metronome_start_ms = None

    def _update_state(self, timestamp_ms: int, depth_now: float) -> None:
        enough_gap = (
            self._last_peak_time is None
            or (timestamp_ms - self._last_peak_time) >= self.refractory_ms
        )

        if not self._in_compression:
            if depth_now >= self.enter_threshold_cm and enough_gap:
                self._in_compression = True
                self._current_peak_depth = depth_now
                self._count += 1
                self._compression_times.append(timestamp_ms)
                self._last_peak_time = timestamp_ms
        else:
            if depth_now > self._current_peak_depth:
                self._current_peak_depth = depth_now

            if depth_now <= self.release_threshold_cm:
                self._in_compression = False
                self._peak_depths.append(self._current_peak_depth)
                self._current_peak_depth = 0.0

    def _calc_bpm(self) -> Optional[float]:
        if len(self._compression_times) < 2:
            return None

        times = list(self._compression_times)
        intervals = [times[i] - times[i - 1] for i in range(1, len(times))]
        intervals = [x for x in intervals if x > 0]

        if not intervals:
            return None

        avg_interval = sum(intervals) / len(intervals)
        return 60000.0 / avg_interval

    def _rate_feedback(self, bpm: Optional[float]) -> str:
        if bpm is None:
            return "리듬 수집중"
        if bpm < 100:
            return "속도 느림"
        if bpm > 120:
            return "속도 빠름"
        return "속도 적절"

    def _latest_peak_depth(self) -> Optional[float]:
        if not self._peak_depths:
            return None
        return float(sum(self._peak_depths) / len(self._peak_depths))

    def _beat_now(self, timestamp_ms: int) -> bool:
        if self._metronome_start_ms is None:
            return False

        beat_interval = 60000 / self.target_bpm
        elapsed = timestamp_ms - self._metronome_start_ms
        phase = elapsed % beat_interval

        # 70ms 동안만 박자 ON
        return phase < 70


 #rep_counter.py 사용할 때 메인에 넣으면 될듯   
#    from counter.rep_counter import RepCounter

#counter = RepCounter()

#result = counter.update(timestamp_ms=123456, signal_value=10.0)

#print(result.count)
#print(result.bpm)
#print(result.rate_feedback)
#print(result.beat_now)