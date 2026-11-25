# chem-structure-prep

独立的数据准备仓库：CSV/SMILES → 化学结构图片、atom/charge boxes、bond bands 和 lines。
模型训练和推理见 [B2M](https://github.com/ksubowu/B2M)，本仓库不包含神经网络模型或权重。

```bash
git clone https://github.com/ksubowu/chem-structure-prep.git
cd chem-structure-prep
python -m pip install .
chem-prepare doctor
```

`historical_v6/` 保存历史 v6 生成脚本，含原机器路径；新任务请使用下面的独立 CLI。
`scripts/prepare_pubchem_bands.py` 和 `legacy/prepare_band_pilot.py` 保存早期 RDKit 流程，额外需要 RDKit。
原始训练数据、SMILES 数据表、人工标注及模型权重不在仓库中。

原创代码采用 [MIT License](LICENSE)，Indigo、MolScribe 衍生绑定与字体保留其原许可证，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

# 化学结构图片与标注准备工具

独立命令行工具 `chem-prepare`，把 CSV 中的 SMILES 转成化学结构图片、原子/电荷框、真实键墨迹 bands/masks，并完成彩色/dummy 扩增、分组划分及训练分布平衡。无需原工作目录、MolScribe 模型权重或 PyTorch。已有的黑白/彩色数据和冻结验证集未被这个工具的开发过程修改。

随包提供：修改版 Indigo Linux x86_64 后端、9 个字体文件、86 个官能团字符标签、示例 CSV、配置模板、测试和第三方许可。后端必须支持 `atom.coords()` 像素坐标扩展，`doctor` 会实际渲染分子验证它。

## 1. 安装与一键示例

**已安装依赖的环境（例如现有 Python 环境）**：

```bash
# 在仓库根目录执行
python -m chem_prep doctor
python -m chem_prep init /tmp/my_chemical_job
python -m chem_prep run --config /tmp/my_chemical_job/recipe.yaml
# 查看 /tmp/my_chemical_job/dataset/index.html
```

也可执行 `bash scripts/run_demo.sh /tmp/my_chemical_job`。目标任务目录应为空，已有同名配置不会被覆盖。

**迁移到新 Linux x86_64 环境**，推荐 Python 3.12：

```bash
cd chem-structure-prep
conda env create -f environment.yml
conda activate chem-prep
chem-prepare doctor
chem-prepare init ./my_job
chem-prepare run --config ./my_job/recipe.yaml
```

或使用现有 Python 环境：

```bash
# Debian/Ubuntu 系统库；已有这些库时无需重复安装。
# sudo apt-get install libcairo2 libfontconfig1 libfreetype6 libstdc++6
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps .
chem-prepare doctor
```

`requirements.lock.txt` 是当前 Linux / CPython 3.12 的测试版本；`pyproject.toml` 提供兼容范围。首次安装 Python/系统依赖可能需要网络，运行数据准备本身不访问外部服务。当前随包本地库仅支持 Linux x86_64；其他系统/架构需重新构建带像素坐标扩展的后端。

## 2. 使用自己的 CSV

最小 CSV：

```csv
image_id,SMILES
mol_001,CCO
mol_002,C[N+](C)(C)C.[Cl-]
mol_003,O=C(N1S[R'])CCC1=O
```

最小配置：

```yaml
inputs:
  - name: my_data
    csv: input.csv
    smiles_column: SMILES
    id_column: image_id
output: prepared
workers: 4
render:
  canvas_size: 640
```

```bash
chem-prepare run --config recipe.yaml
```

全部路径相对于 YAML 所在目录解析，支持 `${ENV_VAR}`。可配置任意数量、任意名称的数据集，`csv` 支持 glob；列名可改。ID 列缺省时用行号；同一数据集重复 ID、空 SMILES、缺失参考图会进入 `input_issues.json`，无效分子进入 `render_errors.json`。默认隔离错误记录并继续，设置 `quality.fail_on_render_error: true` 则使任务失败。没有有效记录会直接报错。

可选参考图片：

```yaml
inputs:
  - name: source_a
    csv: labels.csv
    image_column: file_path
    image_root: ./images_root
    font_families: [DejaVu Sans]
exclude_csv: [staker.csv]
render:
  canvas_size: [640, 480]
  match_reference: true
```

参考图用于近似匹配前景范围、留白和笔画宽度，不用于推断原图原子坐标。所有标注仍对应**新生成图**。没有参考图时，按固定画布和边距自动布局。字体逻辑名支持 DejaVu Sans/Serif/Sans Mono 及 Arial/Times/Courier/Helvetica，后几者在工具中固定映射到随包字体。任意 TTF 文件可用于后述电荷字形扩增。

## 3. 完整配置与历史流程预设

```yaml
inputs:
  - name: source_a
    csv: source_a.csv
  - name: source_b
    csv: source_b.csv
output: prepared
seed: 20261113
workers: 8
render:
  canvas_size: 640
  padding: 24
  font_families: [DejaVu Sans, DejaVu Serif]
color:
  count: 1000                 # 最多选 1000 个源记录；不超过有效输入数
  dummy_probability: 0.5
  labels_file: null           # null 使用随包 86 个标签
split:
  validation_fraction: 0.1
  seed: 20261113
  stratify: [dataset, variant, dummy_applied, has_charge]
charge_augmentation:
  count: 1200                # 仅来自训练集
  canvas_size: 640            # 支持其他正方形尺寸
  seed: 20261114
  fonts: null                # null 使用随包 8 种电荷字体/字重
balance:
  enabled: true
  charged_fraction: 0.25
  seed: 20261114
```

`examples/reproduce_previous.yaml` 提供 ACS/CLEF/UOB/USPTO 的流程级预设：参考图风格匹配、排除 staker、1,000 张彩色图、50% dummy 概率、9:1 分组划分、1,200 张训练电荷变体、25% 带电荷训练采样。

```bash
export DATA_ROOT=/path/to/chemcv_data/real
chem-prepare run --config examples/reproduce_previous.yaml
```

这是同类计算流程的可配置重放，**不宣称重新生成既往历史文件的全部字节或旧的 split ID**：通用工具现在使用固定随包字体、输出画布和包含电荷的分层规则。历史数据、旧脚本和既有冻结划分留在原位置；本预设写入新目录。

## 4. 计算流程

1. **读取/检查输入**：解析 CSV 和列映射，保存输入文件 SHA256，隔离坏记录，生成稳定且文件名安全的样本 ID。
2. **同一 SVG 渲染与标注**：读取 Indigo 绘图后像素坐标；由同一份 SVG 生成 PNG，避免原生 PNG/SVG 字体度量不一致。提取 atom boxes、独立 charge boxes、化学图、实际键墨迹 mask 和 band 几何。
3. **彩色与 dummy 扩增**：按数据集均衡选择源记录；每张用独立 seed 作 Bernoulli(p) 抽样。展开 H，把一个合适的非同位素 H 替换为带整体标签的 dummy，折叠其他 H，验证 H 减 1、非 H 节点加 1、总电荷保持；保留原始和实际绘制的 CXSMILES。无法可靠计算 H/价态或没有可替换 H 时，记录原因，仍保留彩色图。
4. **分组与划分**：芳香化规范 CXSMILES 识别重复分子；源结构、实际绘制结构、同源变体作为连通组共同划分。按数据集/变体/dummy/带电荷状态分层，使用整数优化满足计数。若分组使精确比例不可行，保持组不拆分并报告最小偏差；只有一组时不能产生非空验证集，会明确报告。
5. **冻结验证集**：保存 val.csv/json 以及所有引用素材的 SHA256。
6. **仅训练电荷字形扩增**：真实替换 ±1 电荷的字体轮廓，覆盖不同大小和相邻位置，检查不覆盖原子/键墨迹，并同步变换所有 boxes、bands、masks。化学图不变，继承训练组。
7. **训练采样平衡**：生成显式 epoch 清单，保留所有独立训练图片，再补采样达到目标比例；视原始分布可能重复带电荷或不带电荷图片。尽量同时保持原彩色/dummy 的整体比例；若类别组合不支持，会报告最接近的色彩混合。无电荷或无非电荷样本时，不可能靠采样构造目标比例，会明确报错。
8. **核查与人工检查**：验证几何、mask 对齐、charge 覆盖、组无泄漏及验证集不变，输出检查页和待复核 band 列表。

随机选择使用稳定样本级 seed，改变 worker 数不会改变样本身份、抽样和划分。输入/配方未变时 `run` 支持断点续跑；变更语义配置或输入快照时要求使用新输出目录，防止混用旧标签。固定验证集后修改训练采样使用独立 `balance` 命令。

## 5. 分步命令

```bash
chem-prepare render --config recipe.yaml    # 只渲染和扩增黑白/彩色图
chem-prepare split --config recipe.yaml     # 核查后分组，建立并冻结验证集
chem-prepare balance --dataset prepared --charged-fraction 0.25 --charge-variants 1200
chem-prepare validate --dataset prepared
chem-prepare review --dataset prepared
chem-prepare pack --dataset prepared --output prepared.tar.gz
```

`balance` 只读取 `source_train.json` 和训练源生成新变体，不重新划分验证集。首次可以从 `charge_augmentation.count: 0` 的数据开始增加变体；已有非空字形扩增后改其配方需使用新输出目录。仅修改 charged_fraction 可以在相同数据上重建训练采样。`balance` 不启动模型训练。

自定义标签：

```bash
chem-prepare extract-labels /path/to/det_engine.py --output labels.json
```

该命令只读取 AST 中的静态 `Substitution([...], ..., ...)`，不会导入模型。也可手写 JSON 数组 `["CF3", "OCH3"]`或 `{"labels":[{"label":"CF3"}]}`。标签应是 opaque pseudoatom 名称，不能直接使用真实元素符号代替。示例 SMARTS/SMILES 仅作来源说明，标签内部不会展开成真实原子。

## 6. 输出与数据格式

```text
prepared/
  assets/bw/                   # 各变体均有下列子目录
  assets/color/
  assets/charge/
    images/                    # PNG
    annotations/               # 原生逐图 JSON
    svg/                       # 同源 SVG
    masks/                     # 每键局部裁剪的 NPZ ink/region
    targets/                   # 独立 band 监督
    overlays/                  # PNG 标注叠加
    view_check/                # 参考/渲染/叠加拼图
  data_manifest.json,csv        # 独立素材：每张只出现一次
  render_manifest.json,csv      # 渲染任务状态，含失败记录
  source_train.json            # 冻结验证后、采样前的原训练子集
  unique_train.json            # 包含训练电荷变体的独立训练图
  train.json,csv               # 实际训练 epoch 清单，可有补采样
  val.json,csv                 # 固定验证集
  dataset_config.json
  holdout_snapshot.json
  run_metadata.json            # 配方指纹、输入 hash、依赖/后端版本
  split_report.json
  balance_report.json
  validation_report.json
  index.html, preview.jpg
```

所有素材路径相对于数据集根目录，数据集可以整体迁移。`run_metadata.json` 中原始输入路径仅作来源记录；验证和训练读取不依赖这些旧路径。

- 坐标：左上为原点，x 向右，y 向下，单位为当前渲染图像素。
- `bbox=[x_min,y_min,x_max,y_max]`，是 **xyxy**，不是 COCO xywh。
- `atoms`：id、symbol、formal_charge、center、bbox；无可见元素字样的碳顶点使用定位小框。
- `charges`：atom_id、带符号 value、label、独立 bbox。允许与 atom box 部分重叠；多价电荷的数值与正负号合并成一个 charge box。
- `bonds`：source/target、order/stereo、type、endpoints、对应 band。
- `bands`：真实墨迹 mask、凸包 `region_polygon`、有序四边形 `quad`。双/三键所有笔画、实楔填充、虚楔短划属于各自同一实例。不是原子连线加粗。
- `quad`：前两点为第一端面，后两点为第二端面；楔形窄端优先。短双键/三键的 PCA 轴歧义用实际平行 SVG 笔画方向纠正。
- `targets`：`independent_band_quad_v1`，仅 image、size、labels、quads 等，不含原子索引。类别 1..6 分别为单/双/三/芳香/实楔/虚楔，target labels 为类别减 1。
- `graph.edges`：与 MolScribe 风格一致；楔形反向编码 `11-code`。
- NPZ：`ink_N` 和 `region_N` 是局部 bool 数组，通过 band 的 `mask_bbox`（xyxy）放回全图。实例可重叠。
- 0.2.0 的新渲染使用逐个原子/键的原生双色独占探测确定归属；SVG 保存 owner ID，atom label ink mask 和 bbox 来自所拥有的实际字符。历史数据的 `audit-bands` 仍保留几何复核；原生 owner 标注不会被几何启发式重新分配。
- 基础渲染支持多价电荷；**电荷字形替换目前仅用于所有电荷均为 ±1 的训练图片**，其他价态保留其原图与框，仍可参与采样。没有合适父图时记录跳过，不会凭空改变分子的电荷。

## 7. 训练时读取

```python
import json
from pathlib import Path
from PIL import Image

root = Path('prepared')
record = json.loads((root / 'train.json').read_text())[0]
image = Image.open(root / record['image']).convert('RGB')
annotation = json.loads((root / record['annotation']).read_text())
band_target = json.loads((root / record['band_target']).read_text())
```

训练清单已实现目标采样比例，使用普通均匀读取并 shuffle 即可。`sample_id` 每个抽样条目唯一，`asset_sample_id` 标记重复引用的同一张图；保持清单中的补采样条目。若再次叠加权重采样，实际比例会改变。模型的尺寸变换要同步更新 boxes、quads、masks，并保持 RGB。

## 8. 核查、测试与复现边界

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
chem-prepare validate --dataset prepared
```

测试覆盖 H 替换精确计数、同位素 H、手性中心、长 dummy 标签、独立 charge box、多价电荷、短双键方向、原始参考图、非正方形画布、分组不可拆分、类/颜色比例不可行、跨 worker 重复运行、断点续跑、整体迁移后读取、固定验证集以及篡改检测。

固定输入、seed、后端/字体及锁定依赖时，测试要求不同 worker 数的 PNG 哈希和 train/val 清单一致。SVG 内部无语义的自动 ID、归档时间戳不属于跨机器字节一致保证；系统 Cairo/fontconfig/CPU 构建差异需通过 `doctor` / `run_metadata.json` 追踪。MOL 文件的程序/时间头统一为稳定值。迁移后的预览、训练清单与标注相对路径不依赖原机器目录。

第三方组件和许可见 `THIRD_PARTY_NOTICES.md`、`licenses/`。


## 12. 0.1.1：交叉桥环中的 band 笔画归属修复

增加直线笔画的一阶角度约束，修正 `UOB/maybridge-0442-496485002` 中 B0 只有一条双键笔画、另一条错误归入 B7 的问题。几何分配仍是启发式，变化候选需要人工查看，并不承诺所有桥环/遮挡情况都正确。多笔画双键保持一个 band；单键保持自身实际笔画。

可以复核已有数据的精确 SVG，无需重绘图片或重新计算分组。支持本工具的 JSON 清单及历史 CSV/JSON 清单：

```bash
chem-prepare audit-bands \
  --manifest /path/to/dataset/manifest.csv /path/to/color/manifest.csv \
  --output /path/to/new_band_audit --workers 8
```

输出 `report.json`、离线 `index.html`，以及发生归属变化样本的 annotation、target、mask 和修改前/后对照图。原数据和训练/验证清单不被覆盖，修正标注引用已有图片的绝对路径；此审计输出用于原环境中的人工检查，不是独立可迁移的数据集。历史归属重建会与保存的 mask 比较，不一致时列为失败待复核。新增 primitive ID 可追踪到 SVG 元素序号，重复审计已修正标注不会重复变化。

新版本记录 band 分配版本，禁止静默复用旧版绘图缓存。旧输出需要选择新目录生成；固定验证集需要继续使用原清单。版本更新不自动替换任何训练标注。


## 13. 0.2.0：独占高亮确定 atom/group/bond 的 mask 与 box

新生成流程默认使用 `exclusive_native_dual_color_v1`，不再用字形与原子距离分配 atom box，也不再用笔画与原子连线的距离确定 bond owner。

1. 分子只 layout 一次，保存正式 SVG、原子坐标和最终图片缩放/平移矩阵。
2. 每次清除所有 atom/bond 高亮，只选择当前一个原子、一个键或显式指定的 group。
3. 相同选择分别以红色和青色渲染。归属来自两次探测的变化部分，不按“所有红色像素”阈值提取，因此原本红色的 O 不会被带入别的标签框。正式图片的彩色配置最终恢复。
4. 每次有效探测必须保持 SVG 画布、全部字形边界、键/轮廓路径、线宽及原子坐标一致；不一致时报错并隔离该样本。不会静默降级为最近邻猜测。
5. N 的 NH₂ 是同一个原生 atom label，全部字符归入该 N。长 dummy 标签同理；电荷框单独保留。独立保留在分子图中的显式 H 有自己的原子框。
6. 当前后端会在高亮隐藏碳时新增 C/CHn，甚至移动其他 H 标签。程序先识别并丢弃这些改变几何的试探，最终使用未印字碳的锚点小框，不虚构字形 mask。拥挤桥环里已经印出的 CH₂ 则正常提取完整标签 mask。
7. 正式 SVG 中保存 `data-owner-atom-id` / `data-owner-bond-id`。电荷增强保留这些归属，并在最终仿射坐标重建 masks，不重新做几何猜测。

独立指定一个原子，例如 NH₂ 的 N 为 atom 0：

```bash
chem-prepare probe --smiles 'NC1=CC=CC=C1N1CCNCC1.O' \
  --atom-ids 0 --colored --output ./nh2_probe
```

逐个遍历分子中的所有原子、所有键（每次只高亮当前对象）：

```bash
chem-prepare probe --smiles 'NC1=CC=CC=C1N1CCNCC1' \
  --all-atoms --all-bonds --colored --output ./all_probes
```

按 SMARTS 指定一个 NH₂ 标签，或指定多个 atom ID 组成一个 group：

```bash
chem-prepare probe --smiles 'NCCO' --group-smarts '[NX3;H2]' --output ./amine_probe
chem-prepare probe --smiles 'NCCO' --atom-ids 0,1 --bond-ids 0 --output ./group_probe
```

`--group-smarts` 默认选择第一个匹配，可用 `--match-index` 指定；加 `--include-internal-bonds` 才同时高亮 group 内部的键。多原子 group 生成额外的区域检查结果，不合并分子图中的多个原子节点。

输出包含 `baseline.png/svg`、`index.html`，以及每个选择的 `probe_red.png`、`probe_cyan.png`、`mask.png/npz`、`overlay.png` 和 `selection.json`。`bbox_xyxy` 是最终图片像素坐标；未印字顶点的 `ink_bbox_xyxy` 为 null，`vertex_proxy_boxes` 明确记录代理框。参数 `--font`、`--size`、`--colored` 在每次探测之间固定。

普通 `run/render` 不必加新参数。annotations 的 `ownership` 保存逐个探测记录，`atoms[].mask_key/mask_bbox` 指向同一个 NPZ 中的实际字符墨迹 mask。`atoms[].bbox` 从最终像素 mask 的范围加少量边距生成。未印字顶点仍标记 `unlabeled_vertex_proxy`。完整流程有额外渲染开销，通常每个原子/键需要两次探测，普通碳还可能需要可见性试探。

原始训练集和冻结验证集不会因升级自动改变。重新生成请选新输出目录；新逻辑不会把不同字体/布局重绘的坐标直接套到旧图片。


## 14. 0.2.1：每个 bond 同时提供 band 和 line

新生成 `run/render` 默认同时输出两种几何标注。每条化学键只有一条 line，端点默认是 `source/target` 原子的 `atoms[].center`，按 source→target 排列；所有键型都使用同样的定义。双键不会生成两条 line，实楔/虚楔也不按尖端/宽端裁切 line；原来的键型标签保留。

- annotation 增加 `lines[]`、`line_endpoint_mode`、`line_target`；`bonds[]` 增加 `line_id` 和 `line`。
- `line_targets/*.json` 使用 `independent_bond_line_v1`：`image`、`size`、`endpoint_mode`、零起始 `labels`、`lines`。每个 `lines[i]` 的形状是 `[2,2]`，即 `[[x1,y1],[x2,y2]]`；相同 i 对应 band target 的相同键。
- `line_overlays/*.png` 标出连线和端点，`band_line_view_check/*.jpg` 并排显示原图、band、line；L 编号就是 bond ID。
- 电荷扩增在仿射变换后重新计算 line。验证同时检查 line、band、键型和原子中心是否一一对应，固定验证集的 SHA256 快照也包括新增文件。

给已有、尚未划分 train/val 的数据集另存并补充 line，无需重新绘图：

```bash
chem-prepare add-lines --dataset /path/to/unsplit_dataset \
  --output /path/to/new_band_line_dataset
```

默认 `--endpoint-mode atom_center`；若希望使用 atom box 几何中心，可指定 `--endpoint-mode box_center`。两者在 NH2 等长标签上可能不同，输出会明确记录模式。原子中心模式更贴近实际成键位置。原有图片、SVG、mask、band target、atom/band 数组均逐项核对不变；源数据集不修改。`add-lines` 不迁移冻结 train/val 清单，需要扩展已有冻结项目时应单独规划数据版本。


## 15. 0.2.2：全量生成与断点恢复

每个完成样本在 `_render_checkpoints/` 原子写入记录，中断后只重算未完成或失败样本；worker 数不参与语义指纹，可调整并发度继续运行。解析参考 SVG 几何在每次独占探测中复用，减少重复计算。有效探测仍保持严格几何检查；宽幅图里被丢弃的新增碳标签试探单独容许 float32 平移舍入误差。分层划分的降级优化也优先保留全局验证目标数。

本次全量 ACS/CLEF/UOB/USPTO 配方生成 12,780 黑白 +1,000 彩色基础图，再在训练中生成1,200电荷增强；最终唯一训练13,482、验证1,498，严格9:1。为了补偿训练专用的1,200张增强，基础阶段验证比例使用1498/13780，最终验证比例才是10%。训练抽样16,068条，其中带电荷25%。详见新数据目录的 recipe.yaml、README.md 和 campaign_summary.json。


## 16. 0.2.3：固定布局输入与子类别审计

`render_one(job)` 可接收 `job['source']['molfile']`，并在
`job['render']['preserve_source_coordinates']=True` 时保留输入布局与楔键方向。
调用方应先核对该 molfile 的化学身份与 `source.smiles` 一致；本次合成流程对此逐一检查。
这样可以绘制同一立体构型的双楔键变体，仍通过独占高亮提取真实墨迹归属。
未设置该选项的原有 CSV/SMILES 绘图流程继续自动 layout。
`check_one` 的检查结果新增 `subclass_counts`，分别记录 atom symbol、charge 值、bond category，
不计入 pseudoatom，供独立核对唯一资产和显式重复训练清单的子类别频次。
新增测试覆盖固定布局、化学身份保持、双楔键 native mask 归属及 line/band 一致性。


## 17. 0.2.4：元素同名官能团标签

Ac、Pr、Ar 等显示别名与真实元素符号重名时，不能通过 resetAtom(label) 指定为伪原子。
H 替换流程现在在 CXSMILES 的 atom-label 字段中显式写入别名，保留 dummy 类型、
立体化学和新原子的索引。禁止带 CXSMILES 保留分隔符或换行的标签。
新增测试覆盖与元素同名的别名、CXSMILES 重载，以及将 dummy 恢复为 H 后的立体化学一致性。
此前遇到元素同名标签会由安全检查拒绝，本次修复使这些标签可正确生成。
