import sys
import numpy as np
from numpy.ma import vstack


def print_list(*args):
    print(f"list: {len(args)}")


def print_dict(**kwargs):
    for k, v in kwargs.items():
        print(f"{k}:{v}")

def iter_add_list():
    list = [1, 2, 3, 4]
    it = iter(list)  # 创建迭代器对象
    print("==== next ====")
    print(next(it))
    print("==== for ====")
    for i in it:
        print(i)

class MyNumbers:
    def __iter__(self):
        self.a = 1
        return self

    def __next__(self):
        x = self.a
        self.a += 1
        return x

def iter_add_class():
    myclass = MyNumbers()
    myiter = iter(myclass)
    print("==== next ====")
    print(next(myiter))
    print("==== for ====")
    for i in myiter:
        print(i)

def countdown(n):
    while n > 0:
        yield n
        n -= 1

def print_generator():
    generator = countdown(5)
    print("==== next ====")
    print(next(generator))
    print("==== for ====")
    for i in generator:
        print(i)


## 数据
array = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
print(f"## array: {array.shape}")

### 1. 增加新轴 np.newaxis
array_add0 = array[np.newaxis, :]
array_add1 = array[:, np.newaxis]
array_add2 = array[..., np.newaxis]
print(f"## array_add0: {array_add0.shape},array_add1: {array_add1.shape},array_add2: {array_add2.shape}")

### 2. 扩展维度 np.expand_dims
array_add0 = np.expand_dims(array, axis=0)
array_add1 = np.expand_dims(array, axis=1)
array_add2 = np.expand_dims(array, axis=2)
print(f"## array_add0: {array_add0.shape},array_add1: {array_add1.shape},array_add2: {array_add2.shape}")

### 3. 压缩维度 np.squeeze
squeezed_arr = np.squeeze(array_add2)
print(f"## array_add2: {array_add2.shape},squeezed_arr: {squeezed_arr.shape}")

array1 = np.array(np.arange(1, 13)).reshape(3, 4)
array2 = np.array(np.arange(11, 23)).reshape(3, 4)
array3 = np.array(np.arange(21, 33)).reshape(3, 4)
print(f"## array1: {array1.shape},array2: {array2.shape},array3: {array3.shape}")
concat_array = np.concatenate([array1, array2, array3])
concat_array1 = np.concatenate([array1, array2, array3], axis=0)
concat_array2 = np.concatenate([array1, array2, array3], axis=1)
print(f"## concat_array: {concat_array.shape},concat_array1: {concat_array1.shape},concat_array2: {concat_array2.shape}")
vstack_array = np.vstack([array1, array2, array3])
hstack_array = np.hstack([array1, array2, array3])
print(f"## vstack_array: {vstack_array.shape},hstack_array: {hstack_array.shape}")

arr = np.array([1, 2, 3, 4, 5, 6])
split_arr = np.split(arr, 3)  # 拆分为 3 份
print(f"## arr: {arr.shape},len_split_arr: {len(split_arr)},split_arr0: {split_arr[0].shape}")

arr2 = np.arange(0, 27).reshape(3, 3, 3)
vsplit_arr = np.vsplit(arr2, 3)  # 垂直拆分
hsplit_arr = np.hsplit(arr2, 3)  # 水平拆分
print(f"## arr2: {arr2.shape}")
print(f"## vsplit_arr: {len(vsplit_arr)},vsplit_arr0: {vsplit_arr[0].shape}")
print(f"## hsplit_arr: {len(hsplit_arr)},hsplit_arr0: {hsplit_arr[0].shape}")

### 坐标变换，(n,3)
tf_vins_ego = T = np.array([
    [1, 0, 0, 2],
    [0, 1, 0, 3],
    [0, 0, 1, 4],
    [0, 0, 0, 1]
])

points = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])