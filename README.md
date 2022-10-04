# MetaMalloc

---

**MetaMalloc** is a library that collects in information on the GPU-side memory allocation data, which includes the address allocated by the memory allocator , the size of the allocation, and the size of the heap.

It only supports CUDA as of now, but it supports any CUDA allocators as long as it consists of the standard `malloc` and `free` methods.

## Getting Started

### Installing

In order to get started using MetaMalloc, add the header files in the `include` directory to your `include` directory.

### Usage

MetaMalloc, as of now, requires the use of `LogDataArray` data structure to call a `malloc` function. In order to initialize a `LogDataArray` object, one must pass the name of the kernel, the dimensions of the grid, and the dimensions of the block. The `LogDataArray` object also has to be explicilty freed. To write the data stored in the `LogDataArray` object, use the `write_to_file` method.

Here is an example:

```cpp
#include "meta_malloc.cuh"
#include "meta_malloc_impl.cuh"

template <typename MemoryManager>
__global__ void foo_kernel (MemoryManager memory_manager, LogDataArray log_data_array) {
	size_t size = sizeof(int) * 10;
	int* ptr = memory_manager.malloc(size, log_data_array);
	memory_manager.free(ptr, log_data_array);
}

int main() {
	MemoryManager<MemoryAllocator> memory_manager(8192ULL * 1024ULL * 1024ULL);
	LogDataArray allocation_data("foo_kernel", 1, 128);
	foo_kernel (memory_manager, allocation_data);	
    allocation_data.write_to_file("tmp-file");
    allocation_data.free();
}
```

### Documentation

To generate documentation, run `doxygen Doxyfile` on the project root directory.

### Future Improvements

- Remove the need for manually adding and freeing `LogDataArray`