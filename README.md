# Fashion

___________________________________

Для задачи использовалась предобученная на COCO (большой датасет для сегментации и детекции, в частности людей) torchvision.models.detection.maskrcnn_resnet50_fp.

В задаче 46 классов одежды.

Картинки ресайзились до размера 512 для меньшей стороны.

Для maskrcnn_resnet50_fp во время обучения нужны: леблы, маски для каждого лейбла своя, bbox.

Cкрипты были сделаны для датасета COCO с похожей задаче, поэтом их можно адаптировать для задач instance segmentation.

----------------------------------------------------
|Date|Model|Runtime|IoU|
|-|-|-|-|
|18.06|Mask-RCNN with body|13h|0.1416|
|20.06|Mask-RCNN only heads|6h|0.1296|
|24.06|Mask-RCNN head after body|6h + 13h|0.146|
|24.06|Mask-RCNN body after head|13h + 6h|0.1693|
|25.06|Mask-RCNN body after head|13h + 6h * 2|0.172|
