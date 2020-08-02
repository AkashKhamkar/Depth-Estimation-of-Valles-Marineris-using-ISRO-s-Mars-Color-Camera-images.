LAMBDA = 100

def discriminator_loss(real, fake_img):
  real_loss = tf.reduce_mean(real)
  fake_loss = tf.reduce_mean(fake_img)
  loss1 = real_loss - fake_loss
  return loss1

def generator_loss(fake_img):
  gen_loss = tf.reduce_mean(fake_img)
 # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss = gen_loss + (LAMBDA * l1_loss)
  return total_gen_loss
  