"""Canonical marker-family names shared by detector and pose estimation."""

SUPPORTED_MARKER_FAMILIES = ("tag36h11", "aruco4x4_50")

_ALIASES = {
    "tag36h11": "tag36h11",
    "apriltag36h11": "tag36h11",
    "dict_apriltag_36h11": "tag36h11",
    "aruco4x4_50": "aruco4x4_50",
    "4x4_50": "aruco4x4_50",
    "4x4_dict_50": "aruco4x4_50",
    "dict_4x4_50": "aruco4x4_50",
}


def normalize_marker_family(value):
    key = str(value).strip().lower()
    try:
        return _ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"unsupported marker family {value!r}; expected one of "
            f"{', '.join(SUPPORTED_MARKER_FAMILIES)}"
        ) from exc


def marker_method(family, pnp=False):
    family = normalize_marker_family(family)
    suffix = "_pnp" if pnp else ""
    return ("apriltag36h11" if family == "tag36h11" else "aruco4x4_50") + suffix
