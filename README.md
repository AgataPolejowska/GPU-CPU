## Graphics Processing Unit and Central Processing Unit


A GPU is a special purpose processor, designed for fast graphics processing.  
A CPU performs basic arithmetic, logic, controlling, and input/output (I/O) operations specified by the instructions.

Some of the differences between GPUs and regular CPUs:
- GPUs are attached processors, so any data they operate on has
to be transferred from the CPU. Since the memory bandwidth of this transfer is low, sufficient work has
to be done on the GPU to overcome this overhead.
- A CPU is optimized to handle a single stream of instructions, that can be very heterogeneous in character;
a GPU is made explicitly for data parallelism, and will perform badly on traditional codes.
- A CPU is made to handle one thread, or at best a small number of threads. A GPU needs a large number
of threads, far larger than the number of computational cores, to perform efficiently.


#### Why GPU overperforms CPU in Deep Learning?

Most of Deep Learning models, especially in their training phase, involve a lot of matrix and vector multiplications that can parallelized. In this case, GPUs can overperform CPUs, because GPUs were designed to handle these kind of matrix operations in parallel.
A single core CPU takes a matrix operation in serial, one element at a time. But, a single GPU could have hundreds or thousands of cores, while a CPU typically has no more than a few cores.


[*Introduction to High-Performance Scientific Computing*](https://s3.amazonaws.com/saylordotorg-resources/wwwresources/site/textbookuploads/5345_scicompbook.pdf)
