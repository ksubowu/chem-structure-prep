"""Bond center lines retain identity/type and track annotation coordinates."""
import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chem_prep.lines import attach,target

class LineTests(unittest.TestCase):
    def fixture(self):
        atoms=[{'id':i,'center':[10.+20*i,20.],'bbox':[5.+20*i,10.,25.+20*i,40.]} for i in range(7)]
        return {'atoms':atoms,'bonds':[{'id':i,'source':i,'target':i+1,'order':min(i+1,4),'stereo':0} for i in range(6)],'bands':[{'id':i,'bond_id':i,'category_id':i+1,'quad':[[0,0],[1,0],[1,1],[0,1]]} for i in range(6)],'width':200,'height':100,'image':'images/test.png'}

    def test_all_six_types_have_one_center_line_and_preserve_bands(self):
        a=self.fixture();before=copy.deepcopy(a['bands']);attach(a);t=target(a)
        self.assertEqual(a['bands'],before);self.assertEqual(t['labels'],list(range(6)))
        self.assertEqual(len(t['lines']),6)
        for b,line in zip(a['bonds'],a['lines']):
            self.assertEqual(line['endpoints'],[a['atoms'][b['source']]['center'],a['atoms'][b['target']]['center']])
            self.assertEqual(b['line'],line['endpoints'])
        # Wedge direction remains source -> target regardless of visible taper.
        self.assertEqual(a['lines'][5]['source'],5);self.assertEqual(a['lines'][5]['target'],6)

    def test_box_centers_are_explicit_and_lines_recompute_after_affine(self):
        a=self.fixture();attach(a,'box_center');self.assertEqual(a['lines'][0]['endpoints'],[[15.,25.],[35.,25.]])
        for atom in a['atoms']:
            atom['center']=[v*2+3 for v in atom['center']];atom['bbox']=[v*2+3 for v in atom['bbox']]
        attach(a,'atom_center');self.assertEqual(a['lines'][0]['endpoints'],[[23.,43.],[63.,43.]])
        self.assertEqual(target(a)['endpoint_mode'],'atom_center')

if __name__=='__main__':unittest.main()
