"""Backend and font initialization without machine-specific filesystem paths."""
import ctypes,ctypes.util,hashlib,platform
from functools import lru_cache
from pathlib import Path
from ._vendor.indigo import Indigo
from ._vendor.indigo.renderer import IndigoRenderer
ASSETS=Path(__file__).parent/'assets'

@lru_cache(maxsize=1)
def initialize_fonts():
    library=ctypes.util.find_library('fontconfig')
    if not library:raise RuntimeError('fontconfig is required. Install libfontconfig1/fontconfig on Linux.')
    fc=ctypes.CDLL(library);fc.FcConfigGetCurrent.restype=ctypes.c_void_p
    fc.FcConfigAppFontAddFile.argtypes=[ctypes.c_void_p,ctypes.c_char_p];fc.FcConfigAppFontAddFile.restype=ctypes.c_int
    fc.FcConfigBuildFonts.argtypes=[ctypes.c_void_p];fc.FcConfigBuildFonts.restype=ctypes.c_int
    config=fc.FcConfigGetCurrent()
    for path in sorted((ASSETS/'fonts').glob('*.ttf')):
        if not fc.FcConfigAppFontAddFile(config,str(path).encode()):raise RuntimeError(f'Cannot register bundled font: {path.name}')
    # Indigo exposes four logical font names. Pin their fontconfig matches to
    # bundled families instead of accepting host-dependent Arial/Times aliases.
    fc.FcConfigParseAndLoadFromMemory.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_int]
    fc.FcConfigParseAndLoadFromMemory.restype=ctypes.c_int
    xml='<fontconfig>'+''.join('<match target="pattern"><test name="family"><string>'+name+'</string></test><edit name="family" mode="prepend" binding="strong"><string>'+family+'</string></edit></match>' for name,family in [('Arial','DejaVu Sans'),('Helvetica','DejaVu Sans'),('Times','DejaVu Serif'),('Courier','DejaVu Sans Mono')])+'</fontconfig>'
    if not fc.FcConfigParseAndLoadFromMemory(config,xml.encode(),1):raise RuntimeError('Cannot register portable font aliases')
    if not fc.FcConfigBuildFonts(config):raise RuntimeError('Cannot build fontconfig application font set')
    return True


def create_backend():
    if platform.system()!='Linux' or platform.machine().lower() not in ('x86_64','amd64'):
        raise RuntimeError('Bundled backend supports Linux x86_64 only. Supply a matching MolScribe-patched Indigo backend for another platform.')
    initialize_fonts();ig=Indigo();renderer=IndigoRenderer(ig)
    ig.setOption('ignore-stereochemistry-errors',True)
    return ig,renderer


def set_font(ig,family):
    mapping={'DejaVu Sans':'Arial','DejaVu Serif':'Times','DejaVu Sans Mono':'Courier','sans':'Arial','serif':'Times','mono':'Courier'}
    engine=mapping.get(family,family)
    if engine not in ('Arial','Times','Courier','Helvetica'):raise ValueError('Supported body font families: DejaVu Sans/Serif/Sans Mono, Arial/Times/Courier/Helvetica')
    ig.setOption('render-font-family',engine)
    return engine


def doctor():
    from . import __version__
    import importlib.metadata,xml.etree.ElementTree as ET
    ig,r=create_backend();ig.setOption('render-output-format','svg');set_font(ig,'DejaVu Sans')
    mol=ig.loadMolecule('[NH3+]CC(=O)[O-]');svg=bytes(r.renderToBuffer(mol)).decode();coords=[a.coords() for a in mol.iterateAtoms()]
    width,height=map(float,ET.fromstring(svg).get('viewBox').split()[2:])
    if not all(0<=x<width+1 and 0<=y<height+1 for x,y in coords):raise RuntimeError('Patched Indigo pixel-coordinate probe failed')
    return {'status':'ok','version':__version__,'platform':platform.platform(),'backend':'bundled MolScribe-patched Indigo','pixel_coords_verified':True,'font_files':[p.name for p in sorted((ASSETS/'fonts').glob('*.ttf'))],'opencv_version':__import__('cv2').__version__,'dependencies':{p:importlib.metadata.version(p) for p in ['numpy','scipy','Pillow','CairoSVG','cairocffi','fonttools','networkx','PyYAML']}}
