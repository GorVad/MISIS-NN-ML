import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd

def predicate(x, allowed_labels=tf.constant([0., 1., 2.])):
    label = x['label']
    isallowed = tf.equal(allowed_labels, tf.cast(label, tf.float32))
    reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
    return tf.greater(reduced, tf.constant(0.))

cifar100_builder = tfds.builder("cifar100")
cifar100_builder.download_and_prepare()
ds_train = cifar100_builder.as_dataset(split="train")
ds_test = cifar100_builder.as_dataset(split="test")

filtered_ds_train=ds_train.filter(predicate)
filtered_ds_test=ds_test.filter(predicate)


cifar100Train = pd.DataFrame(data=filtered_ds_train)
cifar100Test = pd.DataFrame(data=filtered_ds_test)

X_train_100 = cifar100Train["label"].astype('float32')
y_train_100 = cifar100Train["image"]
X_test_100 = cifar100Test["label"]
y_test_100 = cifar100Test["image"]

index = 0
for i in y_train_100:
   y_train_100[index] = np.asarray(i).astype(np.float32)
   index = index+1

print(np.array(y_train_100))

# for x in filtered_ds_train:
#   print(x['label'])