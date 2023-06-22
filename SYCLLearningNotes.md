编译和运行 SYCL 程序的三个主要步骤是：

1. 初始化环境变量
2. 编译 SYCL 源代码
3. 运行应用程序

对于此培训，我们编写了一个脚本 （q） 来帮助开发人员在 DevCloud 上开发项目。此脚本将脚本提交到 DevCloud 上的 GPU 节点执行，等待作业完成并打印输出/错误。我们将使用此命令在 DevCloud 上运行：`run.sh``./q run.sh`

./q run.sh是一种运行脚本的方式，其中./q表示在当前目录下运行脚本。

希望这可以帮助你。

#### 在本地系统上编译和运行：

如果您已在本地系统上安装了英特尔® oneAPI 基本工具包，则可以使用以下命令编译和运行 SYCL 程序：

```
source /opt/intel/inteloneapi/setvars.sh

icpx -fsycl simple.cpp -o simple

./simple
```

*注意：run.sh 脚本是上述三个步骤的组合。*

constexpr：编译运行时常量

sycl代码需要包含：

```cpp
#include <sycl/sycl.hpp>
using namespace cl::sycl;
```

#### handler

`handler`是SYCL中的一个类，它表示一个命令组合器，可以用于将多个操作组合成单个命令

- `cgh.parallel_for()`：在GPU上并行执行操作。
- `cgh.single_task()`：在GPU上执行单个任务。
- `cgh.copy()`：将数据从一个访问器复制到另一个访问器。
- `cgh.fill()`：将缓冲区中的所有元素设置为指定值。
- `cgh.update_host()`：将缓冲区中的数据复制回主机

每个函数具体用法需要进一步搜索，如

```cpp
h.parallel_for(range<1>(1024), [=](id<1> i)
{
    A[i]= B[i]+C[i];
});
意为:for(int i=0; i < 1024; i++) {a[i]=b[i]+c[i];};
多个维度使用item类而非id
h.parallel_for(range<1>(1024), [=](item<1> item)
{
    auto i = item.get_id();
    auto R = item.get_range();
    // CODE THAT RUNS ON DEVICE

});
```

#### lambda表达式

`[&]` 是引用捕获，表示lambda函数内部可以通过引用访问当前作用域中的所有外部变量。在lambda函数内对捕获的变量进行修改时，会影响到外部变量本身。

 `[=]` 是值捕获，外部变量的副本在lambda函数创建时就被固定下来，在lambda函数内对捕获的变量进行修改时，不会影响到外部变量本身

#### 命令组  核函数

```cpp
class my_kernel;
queue gpuQueue;
gpuQueue.submit([&](handler &cgh)
{
     cgh.single_task<my_kernel>([=]() 
     {
     /* kernel code */
     }); 
}).wait();

gpuQueue.submit([&](sycl::handler &cgh) 
{
  auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
  cgh.parallel_for<class my_kernel>(sycl::range<1>{n_items}, [=](sycl::id<1> idx) 
  {
    acc[idx] *= 2;
  });
});
```

用lambda表达式定义kernel函数需要命名核

用cpp函数对象不用命名核，需要构造此类的一个实例对象传递过去

```cpp
struct my_kernel 
{ 
    void operator()()
    { 
    /* kernel function */
     }
};
queue gpuQueue;
gpuQueue.submit([&](handler &cgh)
{                        
  cgh.single_task(my_kernel{}); 
}).wait();
```

#### 流stream

```cpp
sycl::stream(size_t bufferSize, size_t workItemBufferSize, handler &cgh);
```

- The constructor takes a parameter specifying the total size of the buffer that will store the text.`size_t`
- It also takes a second parameter specifying the work-item buffer size.`size_t`
- The work-item buffer size represents the cache that each invocation of the kernel function (in the case of 1) has for composing a stream of text.`single_task`

```cpp
class my_kernel;

queue gpuQueue;
gpuQueue.submit([&](handler &cgh)
{

  auto os = sycl::stream(1024, 1024, cgh);
//在命令组中构造一个缓冲区大小为1024、工作项大小为1024的stream函数
//这意味着流可以接收的总文本为 1024 字节。
  cgh.single_task<my_kernel>([=]() 
  {
    os << "Hello world!\n";
//work-item size is the available size of the text after operator"<<"
  }); 
}).wait();
```

#### 内存分配

##### melloc

```cpp
//只在设备上
T *device_ptr = sycl::malloc_device<T>(n, myQueue);
myQueue.memcpy(device_ptr, cpu_ptr, n * sizeof(T));
// 
// Do some computation on device
// 
myQueue.memcpy(result_ptr, device_ptr, n * sizeof(T)).wait();
sycl::free(device_ptr, myQueue);
```

```cpp
//共享内存，设备和host都可访问
T *shared_ptr = sycl::malloc_shared<T>(n, myQueue);
for (auto i = 0; i < n; ++i) shared_ptr[i] = i;
// 
// Do some computation on device
// 
sycl::free(shared_ptr, myQueue);
```

#### BUFFERS & ACCESSORS

缓冲区和访问器负责内存迁移以及依赖关系分析。

如果两个内核使用同一个缓冲区，就会有依赖关系

`buffer`表示一个数据缓冲区，可以用于在主机和设备之间传递数据。它的构造函数有两个参数：第一个参数是指向数据的指针，第二个参数是表示缓冲区大小一个范围，最多三维。

`range`表示一个数据范围，<>中的数字是维数，{}之中的数字是每个维度的元素个数。如range<3>{3,4,5}、range<2>{4,4}

```cpp
T var = 42;
{
  auto buf = sycl::buffer{&var, sycl::range<1>{1}};
  // 
  // Do some computation on device. Use accessors to access buffer
  // 

} // var updated here

assert(var != 42);
```

`accessor`

accessor<元素type，维数（0-3），access::mode::(read/write/read_write/atomic...)，access::target::(host_buffer/local/global_buffer/constant_buffer)，access::placeholder>

可以缺省一些，由CTAD原则自动填充，默认为read_write

```cpp
auto acc = sycl::accessor{bufA, cgh};
auto readAcc = sycl::accessor{bufA, cgh, sycl::read_only};
auto writeAcc = sycl::accessor{bufB, cgh, sycl::write_only};
```

```cpp
T var = 42;
{

  auto bufA = sycl::buffer{&var, sycl::range<1>{1}};
  auto bufB = sycl::buffer{&var, sycl::range<1>{1}};

  q.submit([&](sycl::handler &cgh) 
 {
    auto accA = sycl::accessor{bufA, cgh, sycl::read_only};
    auto accB = sycl::accessor{bufA, cgh, sycl::no_init};//

    cgh.single_task<mykernel>(...); // Do some work
  });

} // var updated here

assert(var != 42);
```

#### 处理exceptions

`std::terminate`

```python
class add;

int main() {
  std::vector<float> dA{ 7, 5, 16, 8 }, dB{ 8, 16, 5, 7 }, dO{ 0, 0, 0, 0 };
  try{
    queue gpuQueue(gpu_selector{}, async_handler{});#

    buffer bufA{dA};
    buffer bufB{dB};
    buffer bufO{dO};

    gpuQueue.submit([&](handler &cgh) {
      auto inA = accessor{bufA, cgh, read_only};
      auto inB = accessor{bufB, cgh, read_only};
      auto out = accessor{bufO, cgh, write_only};

      cgh.single_task<add>(bufO.get_range(), [=](id<1> i) {
        out[i] = inA[i] + inB[i];
      });
    }).wait();

    gpuQueue.throw_asynchronous();
  } catch (...) { /* handle errors */
}
```

举例：一个简单向量加法

```python
void SYCL_code(int* a, int* b, int* c, int N) 
{
  //1: 创建队列
  //(developer can specify a device type via device selector or use default selector)
  auto R = range<1>(N);
  queue q;
  //2: 创建代表host和decive的buffers
  buffer buf_a(a, R);
  buffer buf_b(b, R);
  buffer buf_c(c, R);
  //3: 提交命令组
  q.submit([&](handler &h)
  {
  //4: 创建buffer accessors，获取device的buffer data
  accessor A(buf_a,h,read_only);
  accessor B(buf_b,h,read_only);
  accessor C(buf_c,h,write_only);

  //5: send a kernel (lambda) for execution
  h.parallel_for(range<1>(N), [=](auto i){
    //Step 6: write a kernel
    //Kernel invocations are executed in parallel
    //Kernel is invoked for each element of the range
    //Kernel invocation has access to the invocation id
    C[i] = A[i] + B[i];
    });
  });
}
```
