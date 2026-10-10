# 分层抽取 20% 训练集

在项目根目录执行（需要 `pip install pyyaml`）：

```bash
python dataset/stratified_subset.py --data dataset/data.yaml --fraction 0.2 --seed 42 --output dataset/subset_20pct
```

默认读取 `dataset/data.yaml` 中的 `/root/autodl-tmp/HazyDet_YOLO`。
在实际存放数据的机器上运行。如果路径不同，用 `--root /实际路径/HazyDet_YOLO`
覆盖它；相对 `--root` 基于运行目录，相对 YAML `path` 基于 YAML 所在目录。
图片与标签必须采用 `images/...`、`labels/...` 的 YOLO 检测格式。

生成的文件：

- `train.txt`：选中图片的绝对路径，直接引用原图片及对应标签，不复制数据。
- `data.yaml`：新的训练配置，仅替换训练集，保留完整验证集和测试集。
- `sampling_report.json`：抽样前后的各类别图片数、目标框数及每层配额。

将 `train.py` 的 `model.train(...)` 中 `data` 参数改为：

```python
data='dataset/subset_20pct/data.yaml',
```

注意：现有 `train.py` 指向 VisDrone 配置。如果实际训练的是 VisDrone，
请先用 `--data /root/dataset/dataset_visdrone/data.yaml` 抽取该数据集，
再使用生成的配置。脚本从输入 YAML 自动读取类别，无需修改类别代码。

## 抽样方法

每张图片按出现的类别集合分层，例如 `{car}`、`{car, truck}`、
`{car, truck, bus}`；同图多个类别一起保留，空标签图片单独成层。
每层的理想配额为该层图片数乘以 0.2，先向下取整，再按小数余数
从大到小分配剩余名额；相同余数随机打破平局。各层内无放回随机抽样。
最终总数为总图片数乘以比例后四舍五入，至少保留一张。

固定数据、比例和种子可复现相同结果。此方法近似保持类别组合的图片比例，
不保证目标框比例恰好一致；很小的层可能抽到零张，类别缺失时会打印警告。
缺失标签默认报错，只有确认这些图片是背景时才加 `--allow-missing-labels`。
输出目录必须不存在，重新抽样时指定一个新的 `--output`。

绝对路径清单需在训练机器上生成；移动数据集后应重新生成。
减少训练图片可缩短每轮训练，但完整验证仍有开销，精度也可能下降。
