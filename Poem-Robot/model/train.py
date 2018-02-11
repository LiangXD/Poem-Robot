import os
import numpy as np
import tensorflow as tf
from PoemRobot.model.model import *

def run_training():
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    poems_vector, word_to_int,vocabularies = poemProcess(FLAGS.file_path)
    batches_inputs, batches_output = generate_batch(FLAGS.batch_size, poems_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets,
                           vocab_size=len(vocabularies), rnn_size=128, batch_size=64,
              num_layers = 2,learning_rate=FLAGS.learning_rate)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:

        sess.run(init_op)

        start_epoch = 0
        check_point = tf.train.latest_checkpoint(FLAGS.model_dir)
        if check_point:
            saver.restore(sess, check_point)
            print("restore from the check point {0}".format(check_point))
            start_epoch += int(check_point.split('-')[-1])

        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                n_chunk = len(poems_vector) // FLAGS.batch_size
                for batch in range(n_chunk):
                    loss, _, _ = sess.run([end_points['total_loss'],
                                           end_points['last_state'],
                                           end_points['train_op']],
                                          feed_dict={input_data:batches_inputs[n], output_targets:
                                                     batches_output[n]})
                    n+=1
                    print("Epoch: %d, batch: %d, training loss: %.6f" % (epoch, batch, loss))
                if epoch % 6 == 0:
                    saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)

        except KeyboardInterrupt:
            print("Interrupt manually, try saving checkpoint for now")
            saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_prefix), global_step=epoch)
            print("Last epoch were saved, next time will start from epoch {}.".format(epoch))

def main():
    run_training()

if __name__ == '__main__':
    main()
