import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from IPython.display import clear_output, Image, display, HTML
import matplotlib.pyplot as plt


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.log_device_placement = False


def get_available_gpus():

    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

def print_logging_device():

    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.constant([1.0, 2.0, 3.0, 4.0, ], shape=[2, 2], name='c')

    mu = tf.matmul(a, b)

    sess = tf.Session(config=config)
    options = tf.RunOptions(output_partition_graphs=True)
    metadata = tf.RunMetadata()
    c_val = sess.run(mu, options=options, run_metadata=metadata)
    print(c_val)
    print(metadata.partition_graphs)

print_logging_device()

with tf.Session(config=config, graph=myGraph) as sess:
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.constant([1.0, 2.0, 3.0, 4.0, ], shape=[2, 2], name='c')

        mu = tf.matmul(a, b)

        with tf.device('/cpu:0'):
            ad = tf.add(mu, c)
        print(sess.run(ad))


def strip_consts(graph_def, max_const_size=32):
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n - strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes('<stripped %d bytes>'%size)
    return res_def

def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add()
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^' + rename_func(s[1:])
    return res_def

def show_graph(graph_def, max_const_size=32):
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """<script>
        function load() {{
            document.getElementById("{id}").pbtxt = {data};
            }}
    </script>
    <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
    <div style="height:600px">
        <tf-graph-basic id="{id}"></tf-graph-basic>
    </div>""".format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))
    iframe = """<iframe seamless style="width:800px; height:620px; border:0" srcdoc="{}"></iframe>""".format(code.replace('"', '&quot;'))
    display(HTML(iframe)

graph_def = myGraph.as_graph_def()
tmp_def = rename_nodes(graph_def, lambda s:'/'.join(s.split('_', 1)))
show_graph(tmp_def)

def matrix_mul(device_name, matrix_sizes):
        time_values = []
        for size in matrix_sizes:
            with tf.device(device_name):
                random_matrix = tf.random_uniform(shape(2,2), minval=0, maxval=1)
                dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
                sum_operation = tf.reduce_sum(dot_operation)

            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as session:
                startTime = datetime.now()
                result = session.run(sum_operation)

            td = datetime.now() - startTime
            time_values.append(td.microseconds/1000)
            print('Matrix shape:' + str(size) + ' --' + device_name + ' time: ' + str(td.microseconds/1000))

        return time_values

matrix_sizes = range(100, 1000, 100)
time_values_gpu = matrix_mul('/gpu:0', matrix_sizes)
time_values_cpu = matrix_mul('/cpu:0', matrix_sizes)

print('GPU time' + str(time_values_gpu))
print('CPU time' + str(time_values_cpu))

plt.plot(matrix_sizes[:len(time_values_gpu)], time_values_gpu, label='GPU')
plt.ylabel('Time (sec) ')
plt.xlabel('Size of matrix ')
plt.legend(loc='best')
plt.show()