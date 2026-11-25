# Third-party components

- `chem_prep/_vendor/indigo/`: copied from the local MolScribe distribution's modified Indigo binding and Linux x86_64 native libraries. Binding headers identify EPAM Systems' Apache-2.0 license. The MolScribe distribution is MIT-licensed. Both license texts are retained in `licenses/` and in the installed package's `_licenses/` directory. The required `atom.coords()` pixel-coordinate extension is preserved. Windows DLL string literals were made raw strings to remove Python escape warnings; Linux behavior is unchanged.
- Bundled DejaVu and Liberation fonts: verbatim font binaries; original font copyright/license notices are retained in `licenses/` and `_licenses/`.
- Rendering/annotation and augmentation modules were adapted from the data-preparation scripts developed in this workspace. The PCA/tapered band convention follows the referenced `legacy/prepare_band_pilot.py::region_quad`; actual bond SVG primitives are rasterized, and the short multistroke axis correction is retained.
- `assets/labels.json`: a static 86-alias snapshot from `det_engine.py::SUBSTITUTIONS`; records include the original file's SHA256 and example SMARTS/SMILES. This script is not imported or required at runtime.

The project intentionally has no neural-network model dependency. Its included native backend supports Linux x86_64; other operating systems/architectures require a compatible patched Indigo backend build.
