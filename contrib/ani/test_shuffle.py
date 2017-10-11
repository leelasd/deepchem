import deepchem as dc

# def shard_generator():
#   for i in range(100):
#     X, y, w, ids = [], [], [], []
#     for j in range(2):
#       X.append([i] * 100)
#       y.append([i])
#       w.append(1)
#       ids.append(i * 2 + j)
#     yield X, y, w, ids
#
#
# ds1 = dc.data.DiskDataset.create_dataset(shard_generator(), './foo')
# sys.exit(1)
#
# ds1 = dc.data.DiskDataset('./foo')
ds_real = dc.data.DiskDataset('/home/leswing/ANI-1/datasets_2/all')
# ds2 = ds_real.shuffle(shard_size=500, merge_dir='./foo')
ds2 = dc.data.DiskDataset('./foo')

print(ds_real.ids)
print(ds2.ids)
