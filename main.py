from mask_rcnn.dataset import UAVVasteDataset
from mask_rcnn.helpers.train import get_transform

d = UAVVasteDataset(get_transform(train=True))
imgs, targets = d[0]

print(len(d))
print()
print(imgs.shape)
print()
for k,v in targets.items():
    print(k, v.shape)

print("labels",  targets["labels"])
print(d.get_num_classes())
print(d.get_category_from_label(1))