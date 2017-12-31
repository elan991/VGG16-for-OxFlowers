import tensorflow as tf

slim = tf.contrib.slim
def inference(f):
    f = slim.conv2d(f, 64, [3, 3], stride=1)
    f = slim.conv2d(f, 64, [3, 3], stride=1)
    f = slim.max_pool2d(f, [2, 2], stride=2)

    f = slim.conv2d(f, 128, [3, 3], stride=1)
    f = slim.conv2d(f, 128, [3, 3], stride=1)
    f = slim.max_pool2d(f, [2, 2], stride=2)

    f = slim.conv2d(f, 256, [3, 3], stride=1)
    f = slim.conv2d(f, 256, [3, 3], stride=1)
    f = slim.conv2d(f, 256, [3, 3], stride=1)
    f = slim.max_pool2d(f, [2, 2], stride=2)

    f = slim.conv2d(f, 512, [3, 3], stride=1)
    f = slim.conv2d(f, 512, [3, 3], stride=1)
    f = slim.conv2d(f, 512, [3, 3], stride=1)
    f = slim.max_pool2d(f, [2, 2], stride=2)

    f = slim.conv2d(f, 512, [3, 3], stride=1)
    f = slim.conv2d(f, 512, [3, 3], stride=1)
    f = slim.conv2d(f, 512, [3, 3], stride=1)
    f = slim.max_pool2d(f, [2, 2], stride=2)

    f = slim.conv2d(f, 17, [1, 1], activation_fn=None,
                      normalizer_fn=None, scope='logits')
    f = tf.reduce_mean(f, [1, 2], name='global_average_pooling', keep_dims=True)
    f = tf.squeeze(f, [1, 2], name='SpatialSqueeze')


    #network = fully_connected(network,4096,activation='relu')   #batch*4096
    #network = dropout(network,keep_prob)
    #network = fully_connected(network,4096,activation='relu')   #batch*4096
    #network = dropout(network,keep_prob)
    #network = fully_connected(network,17,activation='softmax')  #batch*17

    logits = f
    #print(logits.shape)
    return logits       #batch*17
