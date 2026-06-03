import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
from pathlib import Path

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

CODE_DIR = Path(__file__).resolve().parent

PARAMS_JCP = {
    'figure.figsize': (7.0, 5.0),
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
}

DUST_FLUX_VLIM = (-3.2, 3.2)
DIFFUSIVITY_VLIM = (0, 0.10)
SCATTER_AXIS_LIM = (-4, 4)
ERROR_VLIM = (0, 1.5)

CMAP_FLUX = 'viridis'
CMAP_DIFFUSIVITY = 'plasma'
CMAP_ERROR = 'Reds'
CMAP_DIVERGING = 'RdBu_r'
CMAP_DENSITY = 'YlOrRd'

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
    mpl.rcParams.update(PARAMS_JCP)


def load_world():
    import geopandas as gpd
    try:
        return gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    except Exception:
        import geodatasets
        return gpd.read_file(geodatasets.get_path('naturalearth.land'))


def _replace_eps_text_for_core_fonts(figure):
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
    target = Path(base_path)
    if not target.is_absolute():
        target = (CODE_DIR / target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure = fig if fig is not None else plt.gcf()

    figure.savefig(target.with_suffix('.pdf'), dpi=dpi, bbox_inches=bbox_inches)
    if save_png:
        figure.savefig(target.with_suffix('.png'), dpi=dpi, bbox_inches=bbox_inches)
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
