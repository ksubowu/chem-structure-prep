"""Create unique train manifests and actual long-tail/source-diversity statistics."""
from runtime import ROOT,DATA,TOOL,SOURCE,AUDIT
import sys,json,collections,hashlib,csv
from pathlib import Path
sys.path.insert(0,str(TOOL))
from chem_prep.io import write_json,write_manifest
from chem_prep.balance import verify_holdout
from plan import ATOMS,KEYS

def summarize(rows,details):
 counts={f:collections.Counter() for f in ['atoms','charges','bonds']};parents={f:collections.defaultdict(set) for f in counts};scafs={f:collections.defaultdict(set) for f in counts};top={f:collections.defaultdict(collections.Counter) for f in counts}
 for row,stats in zip(rows,details):
  source=row.get('original_parent_id',row['group']);sc=row.get('scaffold_cluster',row['group'])
  for f,k in KEYS:
   n=stats[f].get(k,0);counts[f][k]+=n
   if n:parents[f][k].add(source);scafs[f][k].add(sc);top[f][k][source]+=n
 return {'counts':{f:dict(c) for f,c in counts.items()},'fractions':{f:{k:n/sum(c.values()) for k,n in c.items()} for f,c in counts.items()},'parent_support':{f:{k:len(s) for k,s in d.items()} for f,d in parents.items()},'scaffold_support':{f:{k:len(s) for k,s in d.items()} for f,d in scafs.items()},'maximum_parent_share':{f:{k:max(c.values())/counts[f][k] for k,c in d.items()} for f,d in top.items()}}

def main():
 base=json.loads((ROOT/'base_train.json').read_text());results=json.loads((ROOT/'generation_results.json').read_text());new=[];stats=[];quarantined=[]
 for result in results:
  if result['status']!='ok':continue
  for row,counts in zip(result['rows'],result['subclass_counts']):
   if set(counts['charges'])-set(['-1','1']):quarantined.append({'sample_id':row['sample_id'],'reason':'Charge subclass outside this existing-class enrichment scope','charges':counts['charges']});continue
   new.append(row);stats.append(counts)
 write_json(ROOT/'scope_quarantine.json',quarantined)
 lookup={x['sample_id']:x for x in json.loads((ROOT/'source_asset_counts.json').read_text())};bs=[{f:lookup[x['sample_id']][f] for f in ['atoms','charges','bonds']} for x in base];rows=base+new;assert len({x['sample_id'] for x in rows})==len(rows);val=json.loads((DATA/'val.json').read_text());assert len(val)==1497
 # Remove the one no-longer-selected validation asset from this private output.
 active_paths={x[k] for x in rows+val for k in ['image','annotation','svg','masks','band_target','line_target','overlay','line_overlay','view_check','band_line_view_check']}
 for old in json.loads((SOURCE/'val.json').read_text()):
  if old['sample_id'] in {x['sample_id'] for x in val}:continue
  for key in ['image','annotation','svg','masks','band_target','line_target','overlay','line_overlay','view_check','band_line_view_check']:
   if old[key] not in active_paths:(DATA/old[key]).unlink(missing_ok=True)
 write_manifest(DATA,'train',rows);write_manifest(DATA,'unique_train',rows);write_manifest(DATA,'data_manifest',rows+val);write_json(DATA/'source_train.json',base)
 before=summarize(base,bs);after=summarize(rows,bs+stats);pubchem=summarize(new,stats);molecules={r['original_parent_id'] for r in new};hist=collections.Counter(collections.Counter(r['original_parent_id'] for r in new).values());aliases=sum(r['dummy_applied'] for r in new)
 charge=sum(r['has_charge'] for r in rows)/len(rows);color=sum(r['is_color'] for r in rows)/len(rows)
 report={'goal':'Long-tail relief with broad real chemical coverage; retain natural C/single dominance, no 2:1 target.','train_images':len(rows),'unique_train_images':len(rows),'validation_images':len(val),'new_pubchem_images':len(new),'real_pubchem_chemical_identities':len(molecules),'new_scaffold_clusters':len({r['scaffold_cluster'] for r in new}),'pubchem_images_per_chemical_histogram':dict(hist),'maximum_images_per_pubchem_chemical':max(hist),'training_sampling_repeats':1,'old_v5_template_expansion_used':False,'charged_image_fraction':charge,'color_image_fraction':color,'dummy_label_images':aliases,'before':before,'after':after,'pubchem_only':pubchem,'missing_charge_classes':[-4,-2,2],'generation_failed_styles':sum(r['status']!='ok' for r in results),'charge_typography_fallbacks':sum(bool(r.get('charge_typography_fallback')) for r in results),'geometric_fallbacks':sum(bool(r.get('geometric_fallback')) for r in results),'frozen_validation':verify_holdout(DATA)}
 if not .20<=charge<=.30:raise RuntimeError('Charged image fraction outside the retained 20–30% goal: '+str(charge))
 write_json(ROOT/'distribution_report.json',report);write_json(DATA/'distribution_report.json',report)
 with (DATA/'subclass_distribution.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['family','subclass','original_unique_instances','final_unique_instances','original_fraction','final_fraction','fraction_enrichment','new_real_pubchem_sources','new_scaffold_clusters','largest_pubchem_parent_share'])
  for fam,k in KEYS:
   b=before['fractions'][fam].get(k,0);a=after['fractions'][fam].get(k,0);w.writerow([fam,k,before['counts'][fam].get(k,0),after['counts'][fam].get(k,0),b,a,a/b if b else '',pubchem['parent_support'][fam].get(k,0),pubchem['scaffold_support'][fam].get(k,0),pubchem['maximum_parent_share'][fam].get(k,0)])
 write_json(DATA/'dataset_config.json',{'schema':'chem_structure_prep_dataset_v1','tool_version':'0.2.4','dataset_root':'.','train_manifest':'train.json','val_manifest':'val.json','unique_train_manifest':'unique_train.json','unique_assets_manifest':'data_manifest.json','bbox_format':'xyxy_pixels','band_target_schema':'independent_band_quad_v1','line_target_schema':'independent_bond_line_v1','line_endpoint_mode':'atom_center','sampling':'Uniform shuffle of unique train entries; each image once, no additional weighted sampler.','preserve_rgb':True,'unique_train_images':len(rows),'validation_images':len(val),'train_sampling_rows':len(rows),'source_validation_fraction':.1,'unique_validation_fraction':len(val)/(len(rows)+len(val)),'distribution_report':'distribution_report.json','charge_balance':{'enabled':True,'charged_fraction':charge,'note':'Achieved through different real charged molecules and bounded distinct drawing styles, not repeated epoch entries.'},'validation_reference':json.loads((ROOT/'validation_reference.json').read_text())})
 write_json(DATA/'missing_classes.json',{'charges':[-4,-2,2],'policy':'Existing charge subclasses +/-1 prioritized. Separate real PubChem supplies for missing classes remain in the audit report, not amplified here.'})
 print(json.dumps({k:v for k,v in report.items() if k not in ['before','after','pubchem_only']},indent=2),flush=True)
if __name__=='__main__':main()
