# Fashion

___________________________________

Для задачи использовалась предобученная на COCO (большой датасет для сегментации и детекции, в частности людей) torchvision.models.detection.maskrcnn_resnet50_fp.

В задаче 46 классов одежды, некоторые из них с атрибутами, эксперементы показали, что лучшее качество получается если выбросить атрибуты и предсказывать только классы, доля объектов с атрибутами на трейне ~3%. По мимо этого, как визуально так и по IoU лучшее качество показывалось если отбросить первые 12 классов.

При фаинтюнинге заменялась голова на 34 + 1 класс.
Картинки ресайзились до размера 512 для маньшей стороны.

Для maskrcnn_resnet50_fp во время обучения нужны: леблы, маски для каждого лейбла своя, bbox. Все это писалось с нуля в скрипте mask_r_cnn.py.

Также изменялись скрипты engine.py, utils.py. Cкрипты были сделаны для датасета COCO с похожей задаче, поэтом их можно адаптировать для задач instance segmentation.

----------------------------------------------------
Эти скрипты позволяют обучить сетку, так чтобы она не упала. Было непросто этого добиться. 

Одна эпоха проходит за 13 часов.(Узнал что стоит обновить куду, должно работать быстрее)

|Date|Weights file name|Runtime|IoU|
|-|-|-|-|
|18.06|model_1_40000_iter.pkl|13h(CUDA V7.5.17)|0.145|
