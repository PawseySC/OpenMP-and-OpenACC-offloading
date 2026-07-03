---
title: "Multi-GPU implementation"
teaching: 45
exercises: 30
questions:
- "Extra: implementing multi-GPU parallelisation"
objectives:
- "Use OpenMP to create multiple CPU threads"
- "Use OpenMP API functions to assign threads to devices"
- "Use OpenACC API functions to assign threads to devices"
- "Use update directives to synchronise halo-exchange boundaries"
keypoints:
- "We have learned how to parallelise OpenMP/OpenACC applications on multiple GPUs within the node"
- "We are now able to create codes that use all computational devices available in the node"
---

# Multi-GPU implementation

> ## Where to start?
> This episode starts in *5_single-gpu/* directory.   
{: .callout}

## Strategies

GPU-accelerated HPC systems are usually based on multi-GPU nodes. This means that in order to take advantage of the full computational power of the node we need to parallelise our application across multiple GPUs within the node. There are at least two strategies to achieve this:

* use MPI to create multiple processes and assign processes to GPUs,
* use a multithreading programming model, for example OpenMP, and assign threads to GPUs.

In this episode we will use OpenMP to generate multiple CPU threads and assign threads to GPUs. Each of the threads will be assigned to its unique GPU.

For GPU offloading we show both OpenMP target and OpenACC versions.

The computational nature of the Laplace equation solver will require synchronisation on the boundaries of domains assigned to various GPUs.

## Multithreading and thread-GPU affinity

We will start by creating an OpenMP parallel region around the *while* loop. We will set the default data type to be *shared*.  

```c
#pragma omp parallel default(shared) firstprivate(...)
{
  ...
  while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {
  ...
}
```

Now that we have created multiple CPU threads, we will create affinity between threads and GPUs. For this we will use OpenMP API query functions.

For this to work we must first include appropriate header files:

```c
#include <omp.h>
```

First, we will query for the number of available threads and set a thread ID.

```c
num_threads = omp_get_num_threads();
thread_id  = omp_get_thread_num();
```

Next, we will query for the number of devices and compute the GPU device ID for each thread by taking thread ID modulo number of available GPU devices.

```c
num_devices = omp_get_num_devices();
device_id  = thread_id % num_devices;
omp_set_default_device(device_id);
```

Finally, we can assign a thread to a GPU with the use of a language-specific function.

```c
omp_set_default_device(device_id);
```

Please note that we have used new integer variables: *num_threads, thread_id, num_devices, device_id*.

Those need to be defined earlier in the code:

```c
int num_threads = 1;
int thread_id  = 0;
int num_devices = 1;
int device_id  = 0;
```

and also declared as **firstprivate** variables.

## OpenACC thread-GPU affinity

The multi-GPU strategy remains the same for OpenACC.

We still use OpenMP to create multiple CPU threads.

Each CPU thread is assigned to one GPU.

The difference is that OpenACC directives and OpenACC runtime API calls are used for GPU offloading.

For OpenACC, we need to include the OpenACC header:

```c
#include <openacc.h>
```

The OpenMP version uses:

```c
num_devices = omp_get_num_devices();
device_id  = thread_id % num_devices;
omp_set_default_device(device_id);
```

The OpenACC equivalent is:

```c
num_devices = acc_get_num_devices(acc_device_default);
device_id  = thread_id % num_devices;
acc_set_device_num(device_id, acc_device_default);
```

Therefore, the thread-GPU affinity section becomes:

```c
num_threads = omp_get_num_threads();
thread_id   = omp_get_thread_num();

num_devices = acc_get_num_devices(acc_device_default);
device_id   = thread_id % num_devices;

acc_set_device_num(device_id, acc_device_default);
```

The OpenMP parallel region is still required because OpenACC by itself does not create multiple CPU threads.

```c
#pragma omp parallel default(shared) firstprivate(num_threads, thread_id, num_devices, device_id, i_start, i_end, chunk_size, dt, iteration, i, j)
{
    num_threads = omp_get_num_threads();
    thread_id   = omp_get_thread_num();

    num_devices = acc_get_num_devices(acc_device_default);
    device_id   = thread_id % num_devices;

    acc_set_device_num(device_id, acc_device_default);

    ...
}
```

## Domain decomposition

Parallelisation strategy and proper algorithm design is one of the most important steps for achieving high computational performance. In the case of our Laplace example we will simply divide the problem into *num_threads* chunks, or stripes, by dividing the grid in the X dimension. For this we need to:

* compute the size of the chunk for each thread:

```c
// calculate the chunk size based on number of threads
chunk_size=ceil((1.0*GRIDX)/num_threads);
```

* calculate X direction loop bounds for each thread:

```c
// calculate boundaries and process only inner region
i_start = thread_id * chunk_size + 1;
i_end   = i_start + chunk_size - 1;
```

Please note that all those variables need to be defined earlier as well as declared as **firstprivate**.

```c
int chunk_size = 0;
int i_start, i_end;
```

We can now change both i-loops to iterate within the precomputed boundaries:

```c
for(i = i_start; i <= i_end; i++)
```

## Firstprivate variables

We have set the CPU default OpenMP data type to *shared*. For this reason we need to be very careful with identifying variables that need to be treated as private or firstprivate within the OpenMP parallel region.

Clearly, arrays *T* and *T_new* should be treated as shared. Those arrays can be potentially very big in size and each thread will be working on a separate stripe/chunk.

We have also noted before that *num_threads, thread_id, num_devices, device_id, chunk_size, i_start* and *i_end* variables need to be treated as firstprivate.

In addition to this, variables *dt, iteration, i* and *j* need to be treated as firstprivate as well.

Therefore, the opening of the OpenMP parallel region should be of the following form:

```c
#pragma omp parallel default(shared) firstprivate(num_threads, thread_id, num_devices, device_id, i_start, i_end, chunk_size, dt, iteration, i, j)
```

## Halo-exchange

There is one very important component missing: halo-exchange. Left-hand side and right-hand side boundaries of each stripe, or chunk, need to be synchronised with other threads. Please continue reading to learn how that can be achieved with OpenMP and OpenACC. This is related to the computational nature of the Laplace equation solver. Value for each grid node is computed as an average of its 4 neighbours.

It is even more complicated since GPU threads are operating on a local copy of *T* and *T_new* arrays which were transferred to the GPU memory.

> ## Question
> Do we need to transfer the whole *T* and *T_new* arrays back to CPU memory to achieve the synchronisation? We have already seen that copying whole arrays back and forth is a significant overhead.
{: .callout}

Fortunately, both OpenMP and OpenACC provide *update* directives which can be used to transfer only single rows or columns of 2D arrays. Therefore, each thread will:

* first copy its stripe, or chunk, boundaries back to CPU memory,
* next copy neighbouring thread boundaries from CPU memory to GPU memory.

We will place those two events right after the main computational kernel in the code.

Please note that we had to introduce an additional OpenMP barrier between those two events. We need to be sure that boundaries of neighbouring threads are already available in the CPU memory.

This can be implemented in the following way for OpenMP:

```c
// main computational kernel, average over neighbours in the grid
#pragma omp target
#pragma omp teams distribute parallel for collapse(2)
for(i = i_start; i <= i_end; i++)
    for(j = 1; j <= GRIDY; j++)
        T_new[i][j] = 0.25 * (T[i+1][j] + T[i-1][j] +
                              T[i][j+1] + T[i][j-1]);

#pragma omp target update from(T[i_start:1][1:GRIDY])
#pragma omp target update from(T[i_end:1][1:GRIDY])

#pragma omp barrier

#pragma omp target update to(T[(i_start-1):1][1:GRIDY])
#pragma omp target update to(T[(i_end+1):1][1:GRIDY])
```

The OpenACC equivalent is:

```c
// main computational kernel, average over neighbours in the grid
#pragma acc parallel loop collapse(2) present(T, T_new)
for(i = i_start; i <= i_end; i++)
    for(j = 1; j <= GRIDY; j++)
        T_new[i][j] = 0.25 * (T[i+1][j] + T[i-1][j] +
                              T[i][j+1] + T[i][j-1]);

#pragma acc update self(T[i_start:1][1:GRIDY])
#pragma acc update self(T[i_end:1][1:GRIDY])

#pragma omp barrier

#pragma acc update device(T[(i_start-1):1][1:GRIDY])
#pragma acc update device(T[(i_end+1):1][1:GRIDY])
```

In OpenACC:

```c
#pragma acc update self(...)
```

copies data from GPU memory back to CPU memory.

```c
#pragma acc update device(...)
```

copies data from CPU memory to GPU memory.

The OpenMP barrier is still required because the CPU threads must wait until all neighbouring halo rows have been copied back to host memory.

> ## Note
> The example above follows the original OpenMP code and exchanges halo rows of *T*. If the halo exchange is intended to synchronise newly computed boundary values immediately after the stencil kernel, then *T_new* may need to be used in the update directives instead.
{: .callout}

## Synchronisation

Every thread is now able to compute correct result in each iteration, however we do not have proper synchronisation for the *dt* variable which is used to determine if the iterative algorithm converged. For that reason we will create a shared *dt_global* variable to compute global largest change of temperature.

This can be achieved in the following way for OpenMP:

```c
// reset dt
dt = 0.0;

#pragma omp single
dt_global = 0.0;

#pragma omp barrier

// compute the largest change and copy T_new to T
#pragma omp target map(dt)
#pragma omp teams distribute parallel for collapse(2) reduction(max:dt)
for(i = i_start; i <= i_end; i++){
    for(j = 1; j <= GRIDY; j++){
        dt = MAX(fabs(T_new[i][j] - T[i][j]), dt);
        T[i][j] = T_new[i][j];
    }
}

#pragma omp critical
dt_global = MAX(dt, dt_global);

#pragma omp barrier

dt = dt_global;
```

Local *dt* is first set to zero. One of the OpenMP threads is also setting *dt_global* to zero, this is followed by OpenMP barrier for proper synchronisation.

After computing local *dt*, each thread updates *dt_global*. This needs to be done in the OpenMP critical region to make sure that each thread's value is taken into account. Work is synchronised with the use of OpenMP barrier to make sure that contribution from all threads were calculated. Finally, each thread copies *dt_global* value to local *dt*.

The OpenACC equivalent is:

```c
// reset dt
dt = 0.0;

#pragma omp single
dt_global = 0.0;

#pragma omp barrier

// compute the largest change and copy T_new to T
#pragma acc parallel loop collapse(2) present(T, T_new) reduction(max:dt)
for(i = i_start; i <= i_end; i++){
    for(j = 1; j <= GRIDY; j++){
        dt = MAX(fabs(T_new[i][j] - T[i][j]), dt);
        T[i][j] = T_new[i][j];
    }
}

#pragma omp critical
dt_global = MAX(dt, dt_global);

#pragma omp barrier

dt = dt_global;
```

There is no direct OpenACC equivalent of:

```c
map(dt)
```

in this case.

The OpenACC reduction clause:

```c
reduction(max:dt)
```

creates private copies of `dt` on the GPU, performs the maximum reduction, and copies the final reduced value back to the host variable `dt`.

Therefore, for scalar reductions, this OpenMP pattern:

```c
#pragma omp target map(dt)
#pragma omp teams distribute parallel for collapse(2) reduction(max:dt)
```

becomes:

```c
#pragma acc parallel loop collapse(2) present(T, T_new) reduction(max:dt)
```

## OpenACC data region

For good performance, the arrays should remain on the GPU across iterations.

For example:

```c
#pragma acc data copy(T[0:GRIDX+2][0:GRIDY+2]) create(T_new[0:GRIDX+2][0:GRIDY+2])
{
    while (dt > MAX_TEMP_ERROR && iteration <= max_iterations) {
        ...
    }
}
```

Then the compute regions can use:

```c
present(T, T_new)
```

to tell the compiler that the arrays are already available on the current GPU.

This avoids copying the full arrays between CPU and GPU in every iteration.

Only halo rows are exchanged using OpenACC update directives.

## Diagnostic messages

We also make sure that diagnostic messages are printed by only a single thread, in that case by the master thread.

```c
// periodically print largest change
#pragma omp master
if((iteration % 100) == 0)
    printf("Iteration %4.0d, dt %f\n",iteration,dt);
```

> ## Note
> The size of the grid should be significantly increased to measure scalability across multiple GPUs. We suggest to use 8192x8192 grid size.
{: .callout}

