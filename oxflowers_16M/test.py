import oxflower17
import tensorflow as tf
import vgg16_slim
import numpy as np
import config
from tqdm import *

def test(ckpt_dir):

    x = tf.placeholder("float",[1,config.img_width,config.img_height,config.img_channel])
    logits = vgg16_slim.inference(x,1)

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    restorer = tf.train.Saver()
    restorer.restore(sess,ckpt_dir)
    dataset = oxflower17.Dataset()
    _, _, imgs, labels = dataset.read_data()

    total_accuracy = 0
    for iter in tqdm(range(imgs.shape[0])):
        img = imgs[iter].reshape([1,config.img_width,config.img_height,config.img_channel])
        label = labels[iter]
        logits_value = sess.run(logits,feed_dict={x:img})
        logits_value = logits_value[0]
        correct_prediction = np.equal(np.argmax(label,0),np.argmax(logits_value,0)).astype(np.float32)
        total_accuracy += correct_prediction
    print("accuracy: %f" %(total_accuracy/imgs.shape[0]))

if __name__ == '__main__':
    test('log/snapshot/epoch-251.ckpt')

