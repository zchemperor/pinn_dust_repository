"""
Unified figure style for JEM submission.
All plotting code should import this module and call apply_style() once.

Based on Elsevier/JEM-compatible artwork settings:
- Maximum figure size: 7 in wide x 9 in high
- Preferred fonts: Arial/Helvetica, with DejaVu Sans fallback
- Most lettering near 8 pt; titles 9 pt; tick labels 7 pt
- PDF vector preferred; raster at 600 DPI
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
from pathlib import Path

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

CODE_DIR = Path(__file__).resolve().parent

# ============================================================
# rcParams: design at final print size
# ============================================================
PARAMS_JCP = {
    'figure.figsize': (7.0, 5.0),
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    # Figures are drawn at their intended manuscript width.  These values
    # retain at least roughly 8 pt lettering after LaTeX placement.
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 1.1,
    'lines.markersize': 4.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
}

# ============================================================
# Unified value ranges
# ============================================================
DUST_FLUX_VLIM = (-3.2, 3.2)
DIFFUSIVITY_VLIM = (0, 0.10)       # 99th pct near 0.071, rounded up
SCATTER_AXIS_LIM = (-4, 4)
ERROR_VLIM = (0, 1.5)

# ============================================================
# Unified colormaps
# ============================================================
CMAP_FLUX = 'viridis'
CMAP_DIFFUSIVITY = 'plasma'
CMAP_ERROR = 'Reds'
CMAP_DIVERGING = 'RdBu_r'
CMAP_DENSITY = 'YlOrRd'

# ============================================================
# Unified DPI for saving
# ============================================================
SAVE_DPI = 600

EPS_TEXT_REPLACEMENTS = {
    r'$\mu$': 'mu',
    r'$\sigma$': 'sigma',
    r'$\rho$': 'rho',
    r'$\Delta$': 'Delta',
    r'$\lambda$': 'lambda',
    r'$\theta$': 'theta',
    r'\mu': 'mu',
    r'\sigma': 'sigma',
    r'\rho': 'rho',
    r'\Delta': 'Delta',
    r'\lambda': 'lambda',
    r'\theta': 'theta',
    'μ': 'mu',
    'µ': 'mu',
    'σ': 'sigma',
    'ρ': 'rho',
    'Δ': 'Delta',
    'λ': 'lambda',
    'θ': 'theta',
}


def apply_style():
    """Apply JEM-compatible style globally."""
    mpl.rcParams.update(PARAMS_JCP)


def load_world():
    """Load Natural Earth basemap, compatible with old and new geopandas."""
    import geopandas as gpd
    try:
        return gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    except Exception:
        import geodatasets
        return gpd.read_file(geodatasets.get_path('naturalearth.land'))


def _replace_eps_text_for_core_fonts(figure):
    """Temporarily replace glyphs unsupported by PostScript core fonts."""
    original_text = []
    for text_obj in figure.findobj(Text):
        text = text_obj.get_text()
        new_text = text
        for old, new in EPS_TEXT_REPLACEMENTS.items():
            new_text = new_text.replace(old, new)
        if new_text != text:
            original_text.append((text_obj, text))
            text_obj.set_text(new_text)
    return original_text


def _restore_text(original_text):
    for text_obj, text in original_text:
        text_obj.set_text(text)


def save_figure_all_formats(base_path, fig=None, dpi=SAVE_DPI, bbox_inches='tight', save_png=True):
    """Save a figure to PDF/PNG/EPS using the same base filename."""
    target = Path(base_path)
    if not target.is_absolute():
        target = (CODE_DIR / target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = fig if fig is not None else plt.gcf()

    figure.savefig(target.with_suffix('.pdf'), dpi=dpi, bbox_inches=bbox_inches)
    if save_png:
        figure.savefig(target.with_suffix('.png'), dpi=dpi, bbox_inches=bbox_inches)
    # Some EPS viewers do not render embedded TrueType/Type42 subset glyphs
    # reliably. Use PostScript core Helvetica only for EPS output.
    original_text = _replace_eps_text_for_core_fonts(figure)
    try:
        with mpl.rc_context({
            'ps.useafm': True,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica'],
        }):
            figure.savefig(target.with_suffix('.eps'), dpi=dpi, bbox_inches=bbox_inches)
    finally:
        _restore_text(original_text)
