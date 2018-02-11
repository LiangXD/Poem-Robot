```python
import collections
import numpy as np
import tensorflow as tf
 
#-------------------------------数据预处理---------------------------#
 
poetry_file ='poetry.txt'
 
# 诗集
poetrys = []
titles=[] titletopoetry={}
with open(poetry_file, "r", encoding='utf-8',) as f:
    for line in f:
        try:
            title, content = line.strip().split(':')
            content = content.replace(' ','')
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                continue
            if len(content) < 5 or len(content) > 79:
                continue
            content = '[' + content + ']'
            poetrys.append(content)
            titles.append(title) #            titletopoetry[title]=content;
        except Exception as e: 
            pass
 
# 按诗的字数排序
poetrys = sorted(poetrys,key=lambda line: len(line))
print('唐诗总数: ', len(poetrys))
 
# 统计每个字出现次数
all_words = []
for poetry in poetrys:
    all_words += [word for word in poetry]
for poetry in titles: all_words += [word for word in poetry]    
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)
 
# 取前多少个常用字
words = words[:len(words)] + (' ',)
# 每个字映射为一个数字ID
word_num_map = dict(zip(words, range(len(words))))
# 把诗转换为向量形式，参考TensorFlow练习1
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys]
#[[314, 3199, 367, 1556, 26, 179, 680, 0, 3199, 41, 506, 40, 151, 4, 98, 1],
#[339, 3, 133, 31, 302, 653, 512, 0, 37, 148, 294, 25, 54, 833, 3, 1, 965, 1315, 377, 1700, 562, 21, 37, 0, 2, 1253, 21, 36, 264, 877, 809, 1]
#....]
titles_vector = [ list(map(to_num, poetry)) for poetry in titles]
maxlength = max(map(len,titles_vector)) 
# 每次取64首诗进行训练
batch_size = 64
n_chunk = len(poetrys_vector) // batch_size
x_batches = []
y_batches = []
xtitle=[]
for i in range(n_chunk):
    start_index = i * batch_size
    end_index = start_index + batch_size
    tdata = np.full((batch_size,maxlength), word_num_map[' '], np.int32) tbatches=titles_vector[start_index:end_index] batches = poetrys_vector[start_index:end_index] for row in range(batch_size): tdata[row,:len(tbatches[row])] = tbatches[row]
    length = max(map(len,batches))
    xdata = np.full((batch_size,length), word_num_map[' '], np.int32)
    for row in range(batch_size):
        xdata[row,:len(batches[row])] = batches[row]
    ydata = np.full((batch_size,length+1), word_num_map[' '], np.int32) ydata[:,0]=word_num_map['['] ydata[:,1:-1] = xdata[:,1:] """
    xdata             ydata
    [6,2,4,6,9]       [2,4,6,9,9]
    [1,4,2,8,5]       [4,2,8,5,5]
    """
    x_batches.append(xdata)
    y_batches.append(ydata)
    xtitle.append(tdata)
batch_size = 1 #生成诗句才用到，训练时注释掉
#---------------------------------------RNN--------------------------------------#
tf.reset_default_graph() 
input_data = tf.placeholder(tf.int32, [batch_size, None])
input_tdata = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])
# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2):
    if model == 'rnn':
        cell_fun = tf.nn.rnn_cell.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.nn.rnn_cell.GRUCell
    elif model == 'lstm':
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell
 
    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
 
    initial_state = cell.zero_state(batch_size, tf.float32)
 
    with tf.variable_scope('rnnlm'):
        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)+1])
        softmax_b = tf.get_variable("softmax_b", [len(words)+1])
        softmax_wt = tf.get_variable("softmax_wt", [maxlength*rnn_size, rnn_size]) softmax_bt = tf.get_variable("softmax_bt", [rnn_size])
        with tf.device("/gpu:0"):
            embedding = tf.get_variable("embedding", [len(words)+1, rnn_size])
            inputs = tf.nn.embedding_lookup(embedding, input_data)
            tdatas=tf.nn.embedding_lookup(embedding, input_tdata)
    midresult=tf.reshape(tdatas,[-1,maxlength*rnn_size]) mid=tf.matmul(midresult,softmax_wt)+softmax_bt mid1=tf.reshape(mid,[batch_size,1,rnn_size]) newinputs=tf.concat([mid1,inputs],1)
    outputs, last_state = tf.nn.dynamic_rnn(cell, newinputs, initial_state=initial_state, scope='rnnlm')
    output = tf.reshape(outputs,[-1, rnn_size])
 
    logits = tf.matmul(output, softmax_w) + softmax_b
    probs = tf.nn.softmax(logits)
    return logits, last_state, probs, cell, initial_state,newinputs,inputs
#训练
def train_neural_network():
    logits, last_state, _, _, _,_,_ = neural_network()
    targets = tf.reshape(output_targets, [-1])
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], len(words))
    cost = tf.reduce_mean(loss)
    learning_rate = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))
 
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
 
        saver = tf.train.Saver(tf.all_variables())
 
        for epoch in range(50):
            sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
            n = 0
            for batche in range(n_chunk):
                train_loss, _ , _ = sess.run([cost, last_state, train_op], feed_dict={input_data: x_batches[n], output_targets: y_batches[n],input_tdata:xtitle[n]})
                n += 1
                print(epoch, batche, train_loss)
            if epoch % 7 == 0:
                saver.save(sess, './my1/model.ckpt', global_step=epoch)
 
train_neural_network()#训练时用到，生成诗句时注释掉 def gen_poetry(begin):
    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sample = int(np.searchsorted(t, np.random.rand(1)*s))
        return words[sample]
 
    _, last_state, probs, cell, initial_state,newinputs,inputs = neural_network()
 
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
 
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess, './my1/model.ckpt-49')
 
        state_ = sess.run(cell.zero_state(1, tf.float32)) a=[a for a in begin] # print(map(word_num_map.get, a))
        x = np.array([list(map(word_num_map.get, '['))]) # [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
        x1 = np.array([list(map(word_num_map.get, a))]) print(x.shape) y=np.full([1,maxlength],word_num_map[' ']) y[0,0:x1.shape[1]]=x1[0,:]; [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x,input_tdata: y, initial_state: state_}) print(to_word(probs_[0])) word = to_word(probs_[1]) #word = words[np.argmax(probs_)]
        poem = ""
        while word != ']':
            poem += word
            x = np.zeros((1,1))
            x[0,0] = word_num_map[word]
           m=inputs.eval(session=sess,feed_dict={input_data: x}) [probs_, state_] = sess.run([probs, last_state], feed_dict={newinputs:m,initial_state: state_})
            word = to_word(probs_)
            #word = words[np.argmax(probs_)]
        return poem
def gen_poetry_with_head(head):
    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sample = int(np.searchsorted(t, np.random.rand(1)*s))
        return words[sample]
 
    _, last_state, probs, cell, initial_state = neural_network()
 
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
 
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess, './my/model.ckpt-49')
 
        state_ = sess.run(cell.zero_state(1, tf.float32))
        poem = ''
        i = 0
        for word in head:
            while word != '，' and word != '。':
                poem += word
                x = np.array([list(map(word_num_map.get, word))])
                [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
                word = to_word(probs_)
#                time.sleep(1)
            if i % 2 == 0:
                poem += '，'
            else:
                poem += '。'
            i += 1
        return poem
 
#print(gen_poetry_with_head('一二三四'))
str=input('输入诗的题目：') print(gen_poetry(str))#生成诗句用到，训练时注释掉
```