from __future__ import annotations
from matplotlib.colors import LinearSegmentedColormap

DEFAULT_HEX_COLORS = [
    "#ffffff", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"
]

def make_cmap(
        hex_colors: list[str] | None = None,
        *,
        n: int = 128
) -> LinearSegmentedColormap:
    
    """Creates the custom colormap used in plots."""

    if hex_colors is None:
        hex_colors = DEFAULT_HEX_COLORS

    colors = [[int(h[i:i+2], 16) / 255 for i in (1, 3, 5)] for h in hex_colors]
    return LinearSegmentedColormap.from_list("custom", colors, N=n)