# RT-DETR-R18 的 GSD 对比注意力适配

来源是用户提供的 `code.zip/model.py`：`DenseContrastModule` 与 `SELayer`。
对应论文为 Lin 等人的 CVPR 2021
[Rich Context Aggregation with Reflection Prior for Glass Surface Detection](https://openaccess.thecvf.com/content/CVPR2021/html/Lin_Rich_Context_Aggregation_With_Reflection_Prior_for_Glass_Surface_Detection_CVPR_2021_paper.html)。
原始许可保存在 `ultralytics/nn/modules/GSD_LICENSE.txt`。

## 插入位置与理由

默认配置：`ultralytics/cfg/models/rt-detr/rtdetr-r18-gsd-p3p4.yaml`。

```text
Backbone P3 (stride 8)  -> 1x1 Conv 投影 -> GSD 残差对比增强 -> FPN 融合
Backbone P4 (stride 16) -> 1x1 Conv 投影 -> GSD 残差对比增强 -> FPN 融合
Backbone P5 (stride 32) -> 1x1 Conv 投影 -> 原 AIFI           -> FPN 融合
```

具体在原模型第 17 层（P3）和第 12 层（P4）投影之后、Concat 之前。
P3 保留相对精细的位置/边界信息，P4 提供更大范围的上下文；这是针对当前结构
优先验证的方案，不是已经测得的最优位置。模块可能同时增强背景边缘，不能保证提高 mAP。
P5 已经使用 AIFI，先保持它不变便于判断新增模块的独立效果。

## 与原代码的区别

- 保留 dilation=1/2/4/8 的四个卷积分支、六种两两差分及上下文/通道双重门控。
- 原 SELayer 的六组代表六种差分，不能直接套到不被 6 整除的 256 通道。
  现在对六组差分拼接后的 192 通道做注意力，再用 1x1 卷积恢复到 256 通道。
- 使用 `x + gain * project(attention(contrasts))`，`gain` 从零开始学习，
  初始输出等于原投影输出。首步分支权重梯度为零，gain 学习偏离零后分支获得梯度。
- 用兼容 Conv 的子类扩展原投影，保留层编号及原 conv/bn 参数名；预训练参数可按原键匹配，
  新增 contrast 参数需重新训练。融合 Conv/BN 的推理路径也保留增强模块。
- 不移植原分割预测、反射先验或 RefNet；这不是完整 GlassNet 的复现。

## 训练与对比

在原有可运行的项目 Python 环境中，从项目根目录执行。`train.py` 默认启用 P3+P4，
仍加载 `weights/rtdetr-r18.pt`，其余训练超参数保持原值。
默认 data 仍是原脚本的 VisDrone 路径；HazyDet 请显式指定，且先确认 YAML 内数据路径有效。

```bash
python train.py --data dataset/data.yaml --name r18_gsd_p3p4
python train.py --model ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml --data dataset/data.yaml --name r18_baseline
python train.py --model ultralytics/cfg/models/rt-detr/rtdetr-r18-gsd-p3.yaml --data dataset/data.yaml --name r18_gsd_p3
python train.py --model ultralytics/cfg/models/rt-detr/rtdetr-r18-gsd-p4.yaml --data dataset/data.yaml --name r18_gsd_p4
```

如使用 20% 子集，所有实验统一改为 `--data dataset/subset_20pct/data.yaml`。
保持相同的数据划分、预训练权重、seed、输入大小、训练轮数和 batch，比较验证集
mAP50-95、各类别 AP、召回率、显存与推理速度。先做相同预算的筛选，再对候选配置
完整训练并重复多个种子。用验证集选方案，最后用测试集报告最终结果。

## 验证

```bash
python -m unittest discover -s tests -p test_gsd_contrast.py
```

测试覆盖矩形特征、残差初始等价性、梯度、投影权重加载、Conv/BN 融合，以及三个配置
构建、完整模型前向和基线参数迁移。当前机器已检查的 Python 环境缺少完整 PyTorch，
因此运行测试尚未通过环境导入阶段；语法与 YAML 结构检查不代表完整训练验证。
尚未训练或测量精度提升。
