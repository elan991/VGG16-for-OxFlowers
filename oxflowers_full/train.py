import oxflower17
import tensorflow as tf
import vgg16_slim
import os
import sys
import config

def calc_recognition_loss(logits,y_):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=logits))
    return cross_entropy

def get_learning_rate(epoch):
    if epoch < 200:
        return 1e-4
    else:
        return 1e-5
def calc_weight_decay_loss():
    costs = []
    for var in tf.trainable_variables():
        if var.op.name.find(r'weights') > 0:
            costs.append(tf.nn.l2_loss(var))
            print(var.op.name)
    print('*'*10 + 'weight decay variables' + '*'*10)
    return tf.multiply(config.weight_decay_rate,tf.add_n(costs))

def train():
    work_path = sys.path[0]
    os.system('mkdir -p ' + work_path + '/log/snapshot')
    x = tf.placeholder("float",[config.batch_size,config.img_width,config.img_height,config.img_channel])
    y_ = tf.placeholder("float",[config.batch_size,config.nr_classes])
    learn_rate = tf.placeholder("float",[])

    logits = vgg16_slim.inference(x)
    recogition_loss = calc_recognition_loss(logits,y_)
    weight_decay_loss = calc_weight_decay_loss()
    total_loss = recogition_loss + weight_decay_loss
    tf.summary.scalar('recogition loss',recogition_loss)
    tf.summary.scalar('weight_decay_loss',weight_decay_loss)
    tf.summary.scalar('total_loss',total_loss)
    tf.summary.scalar('learn_rate',learn_rate)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(work_path+'/log')
    opt = tf.train.AdamOptimizer(learn_rate)
    train_op = opt.minimize(total_loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep= 100)
    dataset = oxflower17.Dataset()
    dataset.read_data()

    global_steps = 1
    for epoch in range(config.max_epoch):
        for iter in range(config.epoch_size // config.batch_size):
            batch_x,batch_y = dataset.next_batch(config.batch_size)
            #cv2.imshow('', batch_x[iter])
            #cv2.waitKey()
            #print(batch_x.shape,batch_y.shape)
            feed_dict = {x:batch_x, y_:batch_y, learn_rate: get_learning_rate(epoch)}
            [_, total_loss_value, recogition_loss_value,weight_decay_loss_value,summary] = \
                sess.run([train_op,total_loss,recogition_loss,weight_decay_loss,merged], feed_dict=feed_dict)

            print('epoch:%d %d/%d loss:%.4f recog_loss:%.4f wd_loss:%.4f lr:%.1e' %
                  (epoch+1, iter * config.batch_size, config.epoch_size, total_loss_value, recogition_loss_value,weight_decay_loss_value,
                   get_learning_rate(epoch)))
            train_writer.add_summary(summary, global_steps)
            global_steps += 1
        if epoch % 100 == 0:
            saver.save(sess,work_path + '/log/snapshot/epoch-%d.ckpt'%(epoch+1))
            print('epoch-%d saved.'%(epoch+1))



if __name__ == '__main__':
    train()
