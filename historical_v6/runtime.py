import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;AUDIT=Path(os.environ.get('CHEM_V6_AUDIT',str(BASE/'pubchem_diversity_audit_20261121')));SOURCE=Path(os.environ.get('CHEM_V6_SOURCE',str(BASE/'indigo_subclass_balance_v5_20261120/dataset')));DATA=ROOT/'dataset';TOOL=ROOT/'chem_structure_prep' if (ROOT/'chem_structure_prep').is_dir() else BASE/'chem_structure_prep'
def workers(cap):return min(cap,int(os.environ.get('CHEM_PREP_WORKERS',max(1,(os.cpu_count() or 8)//2))))
