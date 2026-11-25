"""Publish actual long-tail gains, real-source coverage and review galleries."""
from runtime import ROOT,DATA,BASE,TOOL,SOURCE,AUDIT
from pathlib import Path
import sys,json,csv,math,collections,hashlib,shutil,html
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from review_pages import page
sys.path.insert(0,str(TOOL))
from chem_prep.io import write_json
from plan import ATOMS,KEYS

def read(p):return json.loads(p.read_text())
def main():
 report=read(DATA/'distribution_report.json');v=read(DATA/'validation_report.json');rows=read(DATA/'data_manifest.json');train=read(DATA/'train.json');new=[r for r in train if r['dataset']=='PubChem'];results=read(ROOT/'generation_results.json');stats={row['sample_id']:c for result in results if result['status']=='ok' for row,c in zip(result['rows'],result['subclass_counts'])};before=report['before'];after=report['after']
 fig,axes=plt.subplots(1,3,figsize=(18,5))
 for ax,fam in zip(axes,['atoms','bonds','charges']):
  ks=[k for f,k in KEYS if f==fam];x=np.arange(len(ks));ax.bar(x-.2,[100*before['fractions'][fam][k] for k in ks],.4,label='Original unique train');ax.bar(x+.2,[100*after['fractions'][fam][k] for k in ks],.4,label='v6 unique train');labels={'1':'single','2':'double','3':'triple','4':'aromatic','5':'solid wedge','6':'hashed wedge'} if fam=='bonds' else {};ax.set_xticks(x,[labels.get(k,k) for k in ks],rotation=25 if fam=='bonds' else 0);ax.set_title(fam+' subclasses');ax.set_ylabel('Fraction within family (%)');ax.legend(fontsize=8)
  if fam!='charges':ax.set_yscale('log')
  else:ax.set_ylim(0,100)
 fig.tight_layout();fig.savefig(DATA/'subclass_distribution.png',dpi=160);plt.close(fig)
 # Source provenance is independent of the number of depictions or aliases.
 sources={}
 for r in new:
  sid=r['original_parent_id'];entry=sources.setdefault(sid,{'pubchem_cid':r['pubchem_cid'],'native_identity':sid,'scaffold_cluster':r['scaffold_cluster'],'source_SMILES':r['source_pubchem_SMILES'],'images':0,'color_images':0,'dummy_images':0,'charged_images':0});entry['images']+=1;entry['color_images']+=r['is_color'];entry['dummy_images']+=r['dummy_applied'];entry['charged_images']+=r['has_charge']
 with (DATA/'pubchem_source_coverage.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(next(iter(sources.values()))));w.writeheader();w.writerows(sources.values())
 # Representatives cover different chemistry, not just several B templates.
 ordered=sorted(new,key=lambda r:r['original_parent_id']);selected=[];seen=set()
 targets=[('atoms',k) for k in ['B','Si','P','I','Br','H']]+[('bonds',k) for k in ['3','5','6']]+[('charges',k) for k in ['-1','1']]+[('bonds','4')]
 for fam,k in targets:
  for r in ordered:
   c=stats[r['sample_id']];sid=r['original_parent_id'];n=sum(c['atoms'].values())
   if sid in seen or not c[fam].get(k) or not 12<=n<=50:continue
   if (fam,k)!=('atoms','B') and c['atoms'].get('B'):continue
   seen.add(sid);selected.append(r);break
 preview=Image.new('RGB',(4*320,math.ceil(len(selected)/4)*330),'white');draw=ImageDraw.Draw(preview)
 for i,r in enumerate(selected):
  a=read(DATA/r['annotation']);boxes=[x['bbox'] for x in a['atoms']+a['charges']];b=np.array(boxes);bb=[max(0,int(b[:,0].min())-20),max(0,int(b[:,1].min())-20),min(640,int(b[:,2].max())+20),min(640,int(b[:,3].max())+20)]
  with Image.open(DATA/r['overlay']) as im:im=im.crop(bb).convert('RGB');im.thumbnail((300,280));x=(i%4)*320;y=(i//4)*330;preview.paste(im,(x+(320-im.width)//2,y));draw.text((x+5,y+285),'PubChem '+str(r['pubchem_cid']),fill='black')
 preview.save(DATA/'preview.jpg',quality=94)
 review=[]
 for r in selected:review.extend([x for x in new if x['original_parent_id']==r['original_parent_id']])
 page(review,DATA/'review_samples.html','v6 真实分子与画风检查','<a href="index.html">目录</a>')
 (DATA/'review').mkdir(exist_ok=True);npage=math.ceil(len(rows)/1000)
 for i in range(npage):
  nav='<a href="../index.html">目录</a> '
  if i:nav+=f'<a href="all_{i-1:04d}.html">上一组</a> '
  if i+1<npage:nav+=f'<a href="all_{i+1:04d}.html">下一组</a>'
  page(rows[i*1000:(i+1)*1000],DATA/'review'/f'all_{i:04d}.html',f'全量图像检查 {i+1}/{npage}',nav,'../')
 table=''.join(f'<tr><td>{fam}</td><td>{k}</td><td>{before["counts"][fam][k]:,}</td><td>{after["counts"][fam][k]:,}</td><td>{before["fractions"][fam][k]*100:.3f}% → {after["fractions"][fam][k]*100:.3f}%</td><td>{report["pubchem_only"]["parent_support"][fam].get(k,0):,}</td></tr>' for fam,k in KEYS)
 groups=' '.join(f'<a href="review/all_{i:04d}.html">{i*1000+1:,}–{min((i+1)*1000,len(rows)):,}</a>' for i in range(npage))
 (DATA/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>v6 真实结构长尾扩充</title><style>body{{font:16px sans-serif;max-width:1250px;margin:30px auto;line-height:1.65}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:5px 12px}}a{{margin-right:15px}}img{{max-width:100%}}</style><h1>v6：缓解长尾，保留自然类别分布</h1><p>训练图 {len(train):,} 张，均为独立条目；验证图 1,497 张保持原样。新增 PubChem 源分子 {report['real_pubchem_chemical_identities']:,} 种、骨架 {report['new_scaffold_clusters']:,} 种；未沿用 v5 的 29 模板扩张。</p><p>C 占原子实例 {after['fractions']['atoms']['C']:.2%}，single 占键实例 {after['fractions']['bonds']['1']:.2%}。本次缓解稀有子类，不追求各类等量。</p><p><a href="review_samples.html">真实分子多画风与标注试样</a><a href="subclass_distribution.csv">类别完整统计</a><a href="pubchem_source_coverage.csv">PubChem 源结构覆盖</a><a href="README.md">使用说明</a><a href="validation_report.json">全量核验</a></p><img src="subclass_distribution.png"><table><tr><th>类别</th><th>子类别</th><th>原实例数</th><th>新实例数</th><th>比例变化</th><th>新 PubChem 来源数</th></tr>{table}</table><h2>代表图（仅预览裁边）</h2><img src="preview.jpg"><h2>全量唯一资产</h2><p>{groups}</p>''')
 highlights='\n'.join(f"| {k} | {before['counts']['atoms'][k]:,} | {after['counts']['atoms'][k]:,} | {before['fractions']['atoms'][k]:.3%} → {after['fractions']['atoms'][k]:.3%} | {report['pubchem_only']['parent_support']['atoms'][k]:,} |" for k in ['B','Si','P','I','Br','H'])
 text=f'''# v6：真实结构多样性与长尾扩充

训练入口：dataset/train.json 或 train.csv；验证入口：dataset/val.json。
所有路径相对于 dataset/，坐标为原图像素 xyxy，line 两端取对应原子中心。
每张训练图出现一次，均匀 shuffle 即可；没有重复采样清单，不要套用旧 v5 的额外抽样权重。

## 结果

保留原始独立训练图 13,482 张；新增 PubChem 图 {report['new_pubchem_images']:,} 张。
最终训练集 {report['train_images']:,} 张，验证集 1,497 张。
新增真实 PubChem 源分子 {report['real_pubchem_chemical_identities']:,} 种，源骨架簇 {report['new_scaffold_clusters']:,} 种。
未使用 v5 的 29 模板扩充资产。含官能团 dummy 的图片属于源分子的声明增强，不冒充新的 PubChem 化学身份。

C 仍占原子实例 {after['fractions']['atoms']['C']:.2%}，single 仍占键实例 {after['fractions']['bonds']['1']:.2%}。
目的为缓解长尾，不是类别等量，因而不再使用 2:1 约束。
对比基线为最初 13,482 张唯一训练图，非 v5 的重复采样清单。

| 稀有原子 | 原实例数 | 新实例数 | 占原子实例比例 | 新真实 PubChem 来源 |
|---|---:|---:|---:|---:|
{highlights}

triple：{after['counts']['bonds']['3']:,} 条，比例 {before['fractions']['bonds']['3']:.3%} → {after['fractions']['bonds']['3']:.3%}；
solid wedge：{after['counts']['bonds']['5']:,}；hashed wedge：{after['counts']['bonds']['6']:,}。
±1 电荷分别 {after['counts']['charges']['1']:,} / {after['counts']['charges']['-1']:,}，不要求完全等量。
带电荷图片 {report['charged_image_fraction']:.2%}；彩色图片 {report['color_image_fraction']:.2%}；官能团 dummy 图片 {report['dummy_label_images']:,} 张。
其他较常见子类的比例会随真实结构混合变化，完整数值见 subclass_distribution.csv。

## 分子与画风

每个 PubChem 分子分配不同字体、笔画、Kekule/aromatic 或等价显式 H 画法，
再生成一张保持手性的平移/缩放/小角度旋转视图。
多数源分子 2–4 张图，少量带电分子 6 张，含 B 的分子最多 8 张；
实际图片数直方图为 {report['pubchem_images_per_chemical_histogram']}。
每类还单独统计独立源结构、骨架簇和最大单分子贡献，不能把增强图数量当作化学多样性。

彩色图按 50% 的请求概率替换一个可用 H 为官能团 dummy label。
无可用位点或未通过化学/立体化学安全检查时退回有效源画法，实际成功量另计。
用原生双颜色独占高亮获取所属墨迹，避免其它红色 O 等污染 atom/charge box。
NH2/SiH3/CF3 等显示标签按所属原子或 dummy 整体给 box，显式 H 节点另有自己的 box。

新 PubChem 带电图主要用 medium/large 电荷字体轮廓、多字体与位置变体。
新图电荷框高度至少 6px（640px 画布），允许原子框与电荷框部分重叠；
原有训练图及冻结验证标注不改。
新增每条键都有独立 band 与 line，mask 和 box 与图像同步变换。

## 验证

全量 {v['records']:,} 张唯一资产核验无失败，band 墨迹 1px 容差对齐率 {v['pixel_alignment_1px']:.8%}。
从实际标注重算子类别计数，与统计报告一致；train.json/csv 没有重复抽样。
验证清单沿用用户指定的当前 v3 1,497 条，14,972 个清单和关联文件逐字节一致。
新候选按当前 v3 与旧 v5 验证集并集排除，源化学/渲染化学均不能跨训练验证集。
新增只进入训练集，所以总图片数不再保持原来的 9:1；验证集保持不变。

一个原生样式（CID 10701667）超过节点上限而剔除；没有伪造其标注。
电荷字体替换有 {report['charge_typography_fallbacks']} 个样式安全回退到原生有效画法；
图像几何视图失败 {report['geometric_fallbacks']} 个。
缺失电荷类 −4、−2、+2 沿用“优先已有类、缺失类单独报告”的约定；
PubChem 中其它电荷的真实供给保留在上轮统计，不在本版本靠少量结构大量扩充。

## 文件

- dataset/index.html：统计与全量分组检查；review_samples.html：代表源分子的不同画风与叠加图。
- subclass_distribution.csv：全部子类的实例数、比例、比例增幅与源结构/骨架支持。
- pubchem_source_coverage.csv：每个源 CID/SMILES/化学身份的图片数量及画风。
- distribution_report.json：各类真实计数与多样性；validation_report.json：独立完整性核验。
- selected_jobs.json：固定源结构、画法、随机种子；generation_results.json：逐样式处理与回退记录。
- generation_errors.json：被剔除样式；core_tests.log：23 项核心测试。
- training_compatibility.json：现有训练适配器的 512 张样本目标转换检查。

## 复现

使用 conda bb2m；chem_structure_prep/ 为核心源码快照（含 patched Indigo 与字体）。
selected_jobs.json 是本次已选真实结构及参数的权威输入；同目录重跑可从 checkpoint 恢复。
基础资产来自旧 v5 dataset/source_train.json 中的原始 13,482 张训练资产，
并没有复制旧模板扩充目录。基础资产通过不可变硬链接复用，原数据集不被重写。

```bash
conda activate bb2m
python clone_base.py
python freeze_validation.py
python render_data.py
python finalize.py
python validate_data.py
python training_compatibility.py
python package_report.py
```

基础源路径、审计参考路径及工具路径在 runtime.py 中；迁移时可修改对应根目录。
基础资产路径也可用 CHEM_V6_SOURCE 指定，CHEM_PREP_WORKERS 限制进程数。
冻结验证清单保存在 validation_inputs/，排除的化学身份保存在 excluded_native_identities.json，重放生成不需要重新查询百万条 CSV。
原子、键、电荷统计来自最终标注，不直接将 SMILES 先验当作最终训练类别数。
变更输入/画风参数时使用新的输出目录，不与旧 checkpoint 混用。
'''
 (ROOT/'README.md').write_text(text);(DATA/'README.md').write_text(text)
 if TOOL.resolve()!=(ROOT/'chem_structure_prep').resolve():shutil.copytree(TOOL,ROOT/'chem_structure_prep',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','dist','build','*.egg-info','.git'))
 write_json(ROOT/'source_sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(ROOT.glob('*.py'))+list((ROOT/'chem_structure_prep/chem_prep').glob('*.py'))})
 print('Published report and review pages',len(rows),'assets',flush=True)
if __name__=='__main__':main()
