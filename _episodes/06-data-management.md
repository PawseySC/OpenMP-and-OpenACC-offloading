---
title: "Data management"
teaching: 15
exercises: 15
questions:
- "Usage of OpenMP/OpenACC data mapping/copy directives"
objectives:
- "Perform basic profiling of GPU events"
- "Apply data transfer OpenMP directives to improve the performance of the code"
keypoints:
- "We have successfully and significantly reduced the total number of memory transfers"
- "We have significantly increased the performance of GPU implementation"
---

# Data management

> ## Where to start?
> This episode starts in *4_data/* directory.  
{: .callout}

Non-optimal memory management (e.g. excessive memory transfers) can heavily impact the performance of any GPU accelerated code. Therefore it is very important to understand how memory is being mapped and copied between host and device.  

When using AMD GPUs and ROCm this can be achieved by using Rocprof profiler. 


The impact of memory transfers on the current performance of the GPU kernel on AMD GPUs can be measured with *rocprof* profiler by running:

```bash
bash-4.2$ srun -u -n 1 rocprof --hsa-trace ./laplace 4000
```

The above execution will generate multiple profiling files. We are especially interested in time spent for memory transfers which is reported in txt file with **.copy_stats.csv** extension. As can be seen from the report generated below, memory transfers represent about 19s (19096229752ns) and are executed more than 22,000 times. This is significant percentege of the whole runtime and indicates performance bottleneck.

```
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"async-copy",22470,19096229752,849854,100.0
```

## Analysing data transfers

As we've seen memory transfers can take significant amount of time if scheduled improperly. In the case of the Laplace example **T** and **T_new** arrays are being copied multiple times in every iteration of the algorithm. More precisely, in each iteration of the algorithm we have:
* *T* is being copied in to the device memory before the first loop nest and copied in and out of the device memory for the second loop nest,
* *T_new* is being copied out of the device memory after the first loop nest and copied in the device memory before the second loop nest.

This gives us 5 data transfers of a 33.5 MB buffer per iteration and **11,000** data transfers for the entire run. However if we analyse data accesses in the implementation, we can clearly see that there is no need for this, we don't need any results on the host until after the while loop exits. We will try to fix it by using OpenMP compiler directives to indicate when and which data transfers should occur.

We can achieve it in OpenMP with the use of *omp target data* directive placed right before the *while* loop:
```c
#pragma omp target data map(tofrom:T) map(alloc:T_new)
while ( dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {
```

There is actually a one to one mapping between OpenMP and OpenACC data transfer constructs.

| OpenMP construct | OpenACC construct | 
| ---------------- | ----------------- | 
| map(to:A)        | copyin(A)         | 
| map(from:A)      | copyout(A)        | 
| map(tofrom:A)    | copy(A)           | 
| map(alloc:A)     | create(A)         | 

Let's run the *rocprof* profiling again on the OpenMP version.
```bash
bash-4.2$ srun -u -n 1 rocprof --hsa-trace ./laplace 4000
```
```
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"async-copy",4496,28438985,6325,100.0
```

What we can notice is that the code runs much faster now and as can be seen from the profiler information the memory transfers are now taking only a fraction of second i.e. 0.028s (28438985ns).

**We have successfully and significantly reduced the total number of memory transfers of the large *T* and *T_new* arrays, as well as the total number of transfers to 4,496.**

## Key differences

Although we claim that we have significantly reduced the number of data transfers, the *rocprof* report is still indicating that there was around 4496 data transfers. Those transfers are related to the use of *dt* in the second loop nest. This scalar variable needs to be copied in and out in every iteration of the algorithm. 

### Default scalar mapping

In OpenMP a scalar variable that is not explicitly mapped is implicitly mapped as *firstprivate*, although this behaviour can be changed with the use of *defaultmap(tofrom:scalar)* clause.

This is why in the OpenMP implementation we need to explicitly map the *dt* variable which occurs in the *reduction* clause.

```c
// compute the largest change and copy T_new to T
#pragma omp target map(dt)
#pragma omp teams distribute parallel for collapse(2) reduction(max:dt)
for(i = 1; i <= GRIDX; i++){
    for(j = 1; j <= GRIDY; j++){
      dt = MAX( fabs(T_new[i][j]-T[i][j]), dt);
      T[i][j] = T_new[i][j];
    }
}
```
The OpenACC equivalent of this section will be:
```c
// compute the largest change and copy T_new to T
#pragma acc parallel loop collapse(2) copy(T[0:GRIDX+2][0:GRIDY+2]) copyin(T_new[0:GRIDX+2][0:GRIDY+2]) reduction(max:dt)
for(i = 1; i <= GRIDX; i++){
    for(j = 1; j <= GRIDY; j++){
        dt = MAX(fabs(T_new[i][j] - T[i][j]), dt);
        T[i][j] = T_new[i][j];
    }
}
In OpenACC, you usually do not write an explicit equivalent for dt when it is used in a reduction.
```
