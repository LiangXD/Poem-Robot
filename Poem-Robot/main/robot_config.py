import tensorflow as tf
import os

'''Train Data'''
batch_size = 64
# batch_size = 1


tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')
tf.app.flags.DEFINE_string('model_dir', "E:/pycharm_projects/Chinese_poem_generator-master/PoemRobot/model_dir/", 'model save path.')
tf.app.flags.DEFINE_string('file_path', "E:/pycharm_projects/Chinese_poem_generator-master/PoemRobot/static/poetry.txt", 'file name of poems.')
tf.app.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix.')
tf.app.flags.DEFINE_integer('epochs', 50, 'train how many epochs.')

FLAGS = tf.app.flags.FLAGS

'''Gen DATA'''
start_token = '['
end_token = ']'
model_dir = 'E:/pycharm_projects/Chinese_poem_generator-master/PoemRobot/model_dir/'
corpus_file = 'E:/pycharm_projects/Chinese_poem_generator-master/PoemRobot/static/poetry.txt'

lr = 0.0002

