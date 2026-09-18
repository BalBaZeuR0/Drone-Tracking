from typing import Dict, Optional, Tuple


def parse_detections_file(path: str) -> Dict[int, Optional[Tuple[float, float]]]:
    signals: Dict[int, Optional[Tuple[float, float]]] = {}
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 3:
                continue  # header or blank line
            try:
                frame_id = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
            except ValueError:
                continue  # header
            signals[frame_id] = None if (x == 0.0 and y == 0.0) else (x, y)
    return signals
