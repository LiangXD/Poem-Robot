import collections
import numpy as np
from math import ceil
from PoemRobot.main.robot_config import *

def poemProcess(filepath):
    # load dataset
    # poem_file = "E:/pycharm_projects/Chinese_poem_generator-master/PoemRobot/static/sample.txt"
    # poem_file = "E:/pycharm_projects/Chinese_poem_generator-master/PoemRobot/static/poetry.txt"
    poem_file = filepath
    # Create poem set
    poems = []
    titles = []
    titleTopoe = {}
    with open(poem_file, "r", encoding='utf-8') as f:
        for line in f:
            try:
                title, content = line.strip().split(":")
                content = content.replace(' ', '')
                # content = content.replace('，', ',')
                # content = content.replace('。', '.')
                if '_' in content or '(' in content or '（' in content or '《' in content \
                        or '[' in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = '[' + content + ']'
                poems.append(content)
                titles.append(title)
                # titleTopoe[title]=content;
            except Exception as e:
                pass

    # sort poems
    poems = sorted(poems, key=lambda line: len(line))
    # print("Total Poems: ", len(poems))

    # count words
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]

    # for poetry in titles: all_words += [word for word in poetry]

    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    words, _ = zip(*count_pairs)

    # counter words
    words = words[:len(words)] + (' ',)

    # transfer to num ID
    word_num_map = dict(zip(words, range(len(words))))
    # transfer to tensor
    to_num = lambda word: word_num_map.get(word, len(words))  # lambda 函数， 字典的get方法，获取诗词的出现次数并转化为向量。
    # 若字符不在字典中，将默认字典长度作为默认值。

    poetry_vector = [list(map(to_num, poetry)) for poetry in poems]

    titles_vector = [list(map(to_num, poetry)) for poetry in titles]

    return poetry_vector, word_num_map, words
    # max_length = max(map(len, titles_vector))
    #


def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size

    x_batches = []
    y_batches = []

    xtitle = []

    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        # tbatches = titles_vector[start_index:end_index]
        # tdata = np.full((batch_size, max_length), word_num_map[' '], np.int32)

        batches = poems_vec[start_index:end_index]
        length = max(map(len, batches))

        xdata = np.full((batch_size, length), word_to_int[' '], np.int32)

        for row in range(len(batches)):
            xdata[row, :len(batches[row])] = batches[row]

        ydata = np.copy(xdata)
        ydata[:, :-1] = xdata[:, 1:]

        x_batches.append(xdata)
        y_batches.append(ydata)
        # xtitle.append(tdata)
    return x_batches, y_batches