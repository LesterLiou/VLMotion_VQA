"""Ground-plane homography, mask foot points, and polygon operations."""

import math

import numpy as np


def homography_matrix(values):
    """Return a validated 3x3 homography from nine row-major numbers."""
    matrix = np.asarray(values, dtype=float)
    if matrix.size != 9:
        raise ValueError("homography must contain nine values")
    matrix = matrix.reshape(3, 3)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("homography must contain finite values")
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("homography is singular")
    return matrix


def project_point(matrix, point):
    """Project a 2D point with a homography and normalize homogeneous scale."""
    vector = np.asarray([float(point[0]), float(point[1]), 1.0])
    result = np.asarray(matrix, dtype=float).reshape(3, 3) @ vector
    if abs(result[2]) < 1e-12:
        raise ValueError("homography projected point to infinity")
    return float(result[0] / result[2]), float(result[1] / result[2])


def image_to_ground(homography, footpoint_uv):
    return project_point(homography, footpoint_uv)


def ground_to_image(homography, point_xy):
    inverse = np.linalg.inv(np.asarray(homography, dtype=float).reshape(3, 3))
    return project_point(inverse, point_xy)


def footpoint_from_mask(mask, bottom_band_px=4):
    """Use the horizontal median near the lowest visible mask row as foot point."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        raise ValueError("person mask must be a non-empty 2D array")
    ys, _ = np.nonzero(binary)
    bottom_y = int(ys.max())
    band_start = max(0, bottom_y - max(1, int(bottom_band_px)) + 1)
    band_y, band_x = np.nonzero(binary[band_start : bottom_y + 1])
    if band_x.size == 0:
        raise ValueError("person mask has no pixels in its bottom band")
    actual_y = band_y + band_start
    return float(np.median(band_x)), float(actual_y.max())


def mask_foot_zone_overlap_ratio(person_mask, zone_mask, bottom_fraction=0.15):
    """Return how much of a person's lower mask band overlaps an image zone."""
    person = np.asarray(person_mask, dtype=bool)
    zone = np.asarray(zone_mask, dtype=bool)
    if person.ndim != 2 or zone.shape != person.shape or not person.any():
        return 0.0
    ys, _ = np.nonzero(person)
    top_y, bottom_y = int(ys.min()), int(ys.max())
    band_height = max(4, int(round((bottom_y - top_y + 1) * float(bottom_fraction))))
    band_start = max(top_y, bottom_y - band_height + 1)
    foot_band = person.copy()
    foot_band[:band_start] = False
    count = int(np.count_nonzero(foot_band))
    if count == 0:
        return 0.0
    return float(np.count_nonzero(foot_band & zone)) / count


def normalized_polygon_to_image(values, frame_width, frame_height):
    """Convert flat normalized x,y pairs to image pixels."""
    numbers = [float(value) for value in values]
    if len(numbers) < 6 or len(numbers) % 2:
        raise ValueError("normalized polygon needs at least three x,y pairs")
    width = max(1, int(frame_width))
    height = max(1, int(frame_height))
    return [
        (
            min(width - 1.0, max(0.0, numbers[index] * width)),
            min(height - 1.0, max(0.0, numbers[index + 1] * height)),
        )
        for index in range(0, len(numbers), 2)
    ]


def point_in_normalized_polygon(point_uv, values, frame_width, frame_height):
    return point_in_polygon(
        point_uv,
        normalized_polygon_to_image(values, frame_width, frame_height),
    )


def point_on_segment(point, start, end, tolerance=1e-9):
    px, py = point
    ax, ay = start
    bx, by = end
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > tolerance:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= tolerance


def point_in_polygon(point, polygon):
    """Ray-casting test that treats polygon boundaries as inside."""
    vertices = [(float(x), float(y)) for x, y in polygon]
    if len(vertices) < 3:
        raise ValueError("polygon requires at least three points")
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = vertices[-1]
    for current in vertices:
        if point_on_segment((x, y), previous, current):
            return True
        x1, y1 = previous
        x2, y2 = current
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside


def distance_to_polygon(point, polygon):
    """Return zero inside, otherwise the minimum Euclidean edge distance."""
    if point_in_polygon(point, polygon):
        return 0.0
    px, py = float(point[0]), float(point[1])
    vertices = [(float(x), float(y)) for x, y in polygon]
    minimum = math.inf
    previous = vertices[-1]
    for current in vertices:
        ax, ay = previous
        bx, by = current
        dx, dy = bx - ax, by - ay
        denominator = dx * dx + dy * dy
        if denominator <= 1e-12:
            projection = (ax, ay)
        else:
            t = ((px - ax) * dx + (py - ay) * dy) / denominator
            t = min(1.0, max(0.0, t))
            projection = (ax + t * dx, ay + t * dy)
        minimum = min(minimum, math.hypot(px - projection[0], py - projection[1]))
        previous = current
    return float(minimum)


def classify_zone(point, hazard_polygon, warning_polygon):
    if point_in_polygon(point, hazard_polygon):
        return "HAZARD_ZONE"
    if point_in_polygon(point, warning_polygon):
        return "WARNING_ZONE"
    return "OUTSIDE"
