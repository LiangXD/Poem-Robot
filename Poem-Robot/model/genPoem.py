import tensorflow as tf
from PoemRobot.model.model import *
from PoemRobot.main.poemProcess import *

import numpy as np

def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]

def gen_poem(begin_word):
    batch_size = 1
    print("loading corpus from %s" % model_dir)
    poems_vector, word_int_map, vocabularies = poemProcess(corpus_file)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None,
                           vocab_size=len(vocabularies), rnn_size=128, batch_size=64,
              num_layers = 2,learning_rate=lr)

    saver = tf.train.Saver(tf.global_variables())
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)

        check_point = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, check_point)

        x = np.array([list(map(word_int_map.get, start_token))])

        [predict, last_state] = sess.run([end_points['prediction'],
                                          end_points['last_state']],
                                         feed_dict={input_data: x})

        if begin_word:
            word = begin_word
        else:
            word = to_word(predict, vocabularies)

        poem_ = ''

        i = 0

        while word != end_token:
            poem_ += word
            i += 1
            if i >= 24:
                break
            x = np.zeros((1,1))

            x[0, 0] = word_int_map[word]

            [predict, last_state] = sess.run([end_points['prediction'],
                                          end_points['last_state']],
                                         feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)

        return poem_

def pretty_print_poem(poem_):
    poem_sentences = poem_.split('。')
    for s in poem_sentences:
        if s!= '' and len(s) > 10:
            print(s + '。')

if __name__ == "__main__":
    begin_char = input("please input the first character:")
    poem = gen_poem(begin_char)
    pretty_print_poem(poem_= poem)