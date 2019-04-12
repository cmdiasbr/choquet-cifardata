import tensorflow as tf

def MF(A,n):
  result=tf.pow(tf.truediv(tf.shape(A)[-1],n),0.5)
  return tf.cast(tf.expand_dims(result,0),tf.float32)

def fast_choquet_pooling(images, ksizes, strides, rates, padding, q=0.5):
  #funcao mais rapida, mas sem possibilidade de mudar a metrica fuzzy
  patch_depth=ksizes[0]*ksizes[1]*ksizes[2]*ksizes[3]
  patches=tf.extract_image_patches(images=images, ksizes=ksizes, strides=strides, rates=rates, padding=padding)  
  sh=tf.shape(patches)
  n_channels=images.shape[3]
  patches=tf.reshape(patches,[sh[0],sh[1],sh[2],patch_depth,n_channels])
  patches=tf.transpose(patches, [0,1,2,4,3])
  p_sorted,_=tf.nn.top_k(patches, patch_depth, sorted=True)
  #aumenta o peso do maior valor da janela em 20%
  #p_sorted=p_sorted*tf.pad(tf.constant([1.2]), [[0,patch_depth-1]], constant_values=1.0)
  p_sorted=p_sorted[:,:,:,:,::-1]
  p_diff=p_sorted-tf.pad(p_sorted, [[0,0],[0,0],[0,0],[0,0],[1,0]])[:,:,:,:,:-1]
  f=tf.range(start=(tf.shape(p_sorted)[4]), limit=0, delta=-1)
  f=tf.cast(tf.pow(tf.truediv(f,patch_depth),q),tf.float32)
  S=p_diff*f 
  return tf.reduce_sum(S, axis=4)

def fast_choquet_pooling_trainable(images, ksizes, strides, rates, padding):
  #funcao mais rapida, mas sem possibilidade de mudar a metrica fuzzy
  q=tf.Variable(initial_value=tf.random_uniform(shape=(), dtype=tf.float64), dtype=tf.float64, trainable=True)
  patch_depth=ksizes[0]*ksizes[1]*ksizes[2]*ksizes[3]
  patches=tf.extract_image_patches(images=images, ksizes=ksizes, strides=strides, rates=rates, padding=padding)  
  sh=tf.shape(patches)
  n_channels=images.shape[3]
  patches=tf.reshape(patches,[sh[0],sh[1],sh[2],patch_depth,n_channels])
  patches=tf.transpose(patches, [0,1,2,4,3])
  p_sorted,_=tf.nn.top_k(patches, patch_depth, sorted=True)
  #aumenta o peso do maior valor da janela em 20%
  #p_sorted=p_sorted*tf.pad(tf.constant([1.2]), [[0,patch_depth-1]], constant_values=1.0)
  p_sorted=p_sorted[:,:,:,:,::-1]
  p_diff=p_sorted-tf.pad(p_sorted, [[0,0],[0,0],[0,0],[0,0],[1,0]])[:,:,:,:,:-1]
  f=tf.range(start=(tf.shape(p_sorted)[4]), limit=0, delta=-1)
  f=tf.cast(tf.pow(tf.truediv(f,patch_depth),q),tf.float32)
  S=p_diff*f 
  return tf.reduce_sum(S, axis=4)

def choquet_pooling(images, ksizes, strides, rates, padding, fuzzy):
  patch_depth=ksizes[0]*ksizes[1]*ksizes[2]*ksizes[3]
  patches=tf.extract_image_patches(images=images, ksizes=ksizes, strides=strides, rates=rates, padding=padding)  
  sh=tf.shape(patches)
  n_channels=images.shape[3]
  patches=tf.reshape(patches,[sh[0],sh[1],sh[2],patch_depth,n_channels])
  patches=tf.transpose(patches, [0,1,2,4,3])
  p_sorted,p_index=tf.nn.top_k(patches, patch_depth, sorted=True)
  #aumenta o peso do maior valor da janela em 20%
  #p_sorted=p_sorted*tf.pad(tf.constant([1.2]), [[0,patch_depth-1]], constant_values=1.0)
  p_sorted=p_sorted[:,:,:,:,::-1]
  p_index=tf.expand_dims(p_index[:,:,:,:,::-1],-2)
  #T=tf.range(0,tf.shape(p_sorted)[4])
  #coloca T no mesmo shape de p_index, necessario para a funcao tf.sets.set_difference
  #T=tf.tile(T, [sh[0]*sh[1]*sh[2]*n_channels])
  #T=tf.reshape(T, [sh[0],sh[1],sh[2],n_channels,-1])
  p_diff=p_sorted-tf.pad(p_sorted, [[0,0],[0,0],[0,0],[0,0],[1,0]])[:,:,:,:,:-1]
  #S=tf.zeros(tf.shape(p_sorted)[:-1])
  #for i in range(patch_depth):
    #S=S+p_diff[:,:,:,:,i]*fuzzy(p_index[:,:,:,:,:,i:],patch_depth)
    #T=tf.sets.set_difference(T,p_index[:,:,:,:,:,i])
  #return S
  f=fuzzy(p_index,patch_depth)
  for i in range(1,patch_depth):
    f=tf.concat([f,fuzzy(p_index[:,:,:,:,:,i:],patch_depth)],-1)
  S=p_diff*f 
  return tf.reduce_sum(S, axis=4)

def _fast_choquet_pooling(x, ksizes, strides, rates, padding, F="prod", name='sort_pool2d', pool_weights = None):
    k = strides[1]*strides[2]
    #ksizes = [1, 2, 2, 1]
    #strides = [1, 2, 2, 1]
    #rates = [1, 1, 1, 1]

    batch_size, height, width, num_channels = x.get_shape().as_list()

    q = tf.Variable(initial_value=tf.random_uniform(shape=(), dtype=tf.float64), dtype=tf.float64, trainable=True)

    patch_depth = ksizes[0] * ksizes[1] * ksizes[2] * ksizes[3]
    patches = tf.extract_image_patches(images=x, ksizes=ksizes, strides=strides, rates=rates, padding=padding)
    #sh = patches.get_shape().as_list()
    patches = tf.reshape(patches, [-1, int(height / 2), int(width / 2), num_channels, patch_depth])
    #patches = tf.transpose(patches, [0, 1, 2, 4, 3])
    p_sorted, _ = tf.nn.top_k(patches, patch_depth, sorted=True)
    p_sorted = p_sorted[:, :, :, :, ::-1]
    #p_sorted = tf.reshape(p_sorted, [-1, int(height / 2), int(width / 2), num_channels, k])

    p_diff = p_sorted - tf.pad(p_sorted, [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0]])[:, :, :, :, :-1]

    mW = tf.range(start=(tf.shape(p_sorted)[4]), limit=0, delta=-1)
    mW = tf.cast(tf.pow(tf.truediv(mW, patch_depth), q), tf.float32)

    if pool_weights is None:
      with tf.variable_scope(name):
        pool_weights = tf.get_variable('pool_weights', [1,1,1,1,k], 
            tf.float32, initializer=tf.constant_initializer([1,1,1,1]))
      #pool_weights = tf.nn.softmax(pool_weights)

    #pool_weights, _ = tf.nn.top_k(pool_weights, patch_depth, sorted=True)
    pool_weights = tf.exp(pool_weights)
    pool_weights = pool_weights / tf.reduce_max(pool_weights, axis=4)



    # with tf.variable_scope(name):
    #   pool_weights = tf.get_variable('pool_weights', [1,1,1,1,k],
    #       tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
    # pool_weights = tf.nn.softmax(pool_weights)
    #
    # weighted_subsets = pool_weights * p_sorted
    #x = tf.reduce_sum(weighted_subsets, 4)

    S = p_diff * pool_weights
    #S = p_diff * mW
    x = tf.reduce_sum(S, 4)
    return x