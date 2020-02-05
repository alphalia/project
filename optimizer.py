import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
import tensorflow.keras.backend as K



class Eve(Optimizer):
  '''Eve optimizer.
  Default parameters follow those provided in the original paper.
  # Arguments
    lr: float >= 0. Learning rate.
    beta_1/beta_2/beta_3: floats, 0 < beta < 1. Generally close to 1.
    small_k/big_K: floats
    epsilon: float >= 0. Fuzz factor.
  # References
    - [Improving Stochastic Gradient Descent With FeedBack](http://arxiv.org/abs/1611.01505v1.pdf)
  '''

  def __init__(
      self, lr=0.001, beta_1=0.9, beta_2=0.999,
      beta_3=0.999, small_k=0.1, big_K=10,
      epsilon=1e-8, name='Eve', decay=0.0, **kwargs):
    super(Eve, self).__init__(name, **kwargs)
    self.__dict__.update(locals())
    self.iterations = K.variable(0)
    self.lr = K.variable(lr)
    self.beta_1 = K.variable(beta_1)
    self.beta_2 = K.variable(beta_2)
    self.beta_3 = K.variable(beta_3)
    self.small_k = K.variable(small_k)
    self.big_K = K.variable(big_K)
    self.decay = K.variable(decay)
    self.inital_decay = decay
  
  def _create_slots(self, var_list):
    for var in var_list:
      self.add_slot(var, 'ms')
    for var in var_list:
      self.add_slot(var, 'vs')

  def get_updates(self, params, loss):
    grads = self.get_gradients(loss, params)
    self.updates = [K.update_add(self.iterations, 1)]

    lr = self.lr
    if self.inital_decay > 0:
      lr *= (1. / (1. + self.decay * self.iterations))

    t = self.iterations + 1
    lr_t = lr * K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t))

    shapes = [K.get_variable_shape(p) for p in params]
    ms = [K.zeros(shape) for shape in shapes]
    vs = [K.zeros(shape) for shape in shapes]
    f = K.variable(0)
    d = K.variable(1)
    self.weights = [self.iterations] + ms + vs + [f, d]

    cond = K.greater(t, K.variable(1))
    small_delta_t = K.switch(K.greater(loss, f), self.small_k + 1, 1. / (self.big_K + 1))
    big_delta_t = K.switch(K.greater(loss, f), self.big_K + 1, 1. / (self.small_k + 1))

    c_t = K.minimum(K.maximum(small_delta_t, loss / (f + self.epsilon)), big_delta_t)
    f_t = c_t * f
    r_t = K.abs(f_t - f) / (K.minimum(f_t, f))
    d_t = self.beta_3 * d + (1 - self.beta_3) * r_t

    f_t = K.switch(cond, f_t, loss)
    d_t = K.switch(cond, d_t, K.variable(1.))

    self.updates.append(K.update(f, f_t))
    self.updates.append(K.update(d, d_t))

    for p, g, m, v in zip(params, grads, ms, vs):
      m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
      v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
      p_t = p - lr_t * m_t / (d_t * K.sqrt(v_t) + self.epsilon)

      self.updates.append(K.update(m, m_t))
      self.updates.append(K.update(v, v_t))

      new_p = p_t
      self.updates.append(K.update(p, new_p))
    return self.updates

  def get_config(self):
    config = {
      'lr': float(K.get_value(self.lr)),
      'beta_1': float(K.get_value(self.beta_1)),
      'beta_2': float(K.get_value(self.beta_2)),
      'beta_3': float(K.get_value(self.beta_3)),
      'small_k': float(K.get_value(self.small_k)),
      'big_K': float(K.get_value(self.big_K)),
      'epsilon': self.epsilon
      }
    base_config = super(Eve, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class RAdam(Optimizer):
  """RAdam optimizer.
  Default parameters follow those provided in the original Adam paper.
  # Arguments
    lr: float >= 0. Learning rate.\\
    beta_1: float, 0 < beta < 1. Generally close to 1.\\
    beta_2: float, 0 < beta < 1. Generally close to 1.\\
    epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.\\
    decay: float >= 0. Learning rate decay over each update.\\
    amsgrad: boolean. Whether to apply the AMSGrad variant of this
      algorithm from the paper "On the Convergence of Adam and
      Beyond".
  # References
  - [RAdam - A Method for Stochastic Optimization]
    (https://arxiv.org/abs/1908.03265)
  - [On The Variance Of The Adaptive Learning Rate And Beyond]
    (https://arxiv.org/abs/1908.03265)
  """

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=None,
               weight_decay=0.0,
               name='RAdam', **kwargs):
    super(RAdam, self).__init__(name, **kwargs)

    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self._set_hyper('decay', self._initial_decay)
    self.epsilon = epsilon or K.epsilon()
    self.weight_decay = weight_decay

  def _create_slots(self, var_list):
    for var in var_list:
      self.add_slot(var, 'm')
    for var in var_list:
      self.add_slot(var, 'v')

  def _resource_apply_dense(self, grad, var):
    var_dtype = var.dtype.base_dtype
    lr_t = self._decayed_lr(var_dtype)
    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    beta_1_t = self._get_hyper('beta_1', var_dtype)
    beta_2_t = self._get_hyper('beta_2', var_dtype)
    epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
    t = tf.cast(self.iterations + 1, var_dtype)

    m_t = (beta_1_t * m) + (1. - beta_1_t) * grad
    v_t = (beta_2_t * v) + (1. - beta_2_t) * tf.square(grad)

    beta2_t = beta_2_t ** t
    N_sma_max = 2 / (1 - beta_2_t) - 1
    N_sma = N_sma_max - 2 * t * beta2_t / (1 - beta2_t)

    # apply weight decay
    if self.weight_decay != 0.:
      p_wd = var - self.weight_decay * lr_t * var
    else:
      p_wd = None

    if p_wd is None:
      p_ = var
    else:
      p_ = p_wd

    def gt_path():
      step_size = lr_t * tf.sqrt(
          (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)
        ) / (1 - beta_1_t ** t)

      denom = tf.sqrt(v_t) + epsilon_t
      p_t = p_ - step_size * (m_t / denom)

      return p_t

    def lt_path():
      step_size = lr_t / (1 - beta_1_t ** t)
      p_t = p_ - step_size * m_t

      return p_t

    p_t = tf.cond(N_sma > 5, gt_path, lt_path)

    m_t = tf.compat.v1.assign(m, m_t)
    v_t = tf.compat.v1.assign(v, v_t)

    with tf.control_dependencies([m_t, v_t]):
      param_update = tf.compat.v1.assign(var, p_t)
      return tf.group(*[param_update, m_t, v_t])

  def _resource_apply_sparse(self, grad, handle, indices):
    raise NotImplementedError("Sparse data is not supported yet")

  def get_config(self):
    config = super(RAdam, self).get_config()
    config.update({
      'learning_rate': self._serialize_hyperparameter('learning_rate'),
      'decay': self._serialize_hyperparameter('decay'),
      'beta_1': self._serialize_hyperparameter('beta_1'),
      'beta_2': self._serialize_hyperparameter('beta_2'),
      'epsilon': self.epsilon,
      'weight_decay': self.weight_decay,
    })
    return config