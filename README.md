# MetaMalloc

---

MetaMalloc is a single-header library that collects information on the GPU-side memory allocation data. Only CUDA is supported as of now, but it supports any CUDA allocators as long as it consists of the `malloc` and `free` methods.

## Getting Started

### Installing

Simply attach the header file to your project.

### Usage

MetaMalloc, as of now, requires the use of `LogDataArray` data structure to call a `malloc` function. In order to initialize a `LogDataArray` object, one must pass the name of the kernel, the dimensions of the grid, and the dimensions of the block.

Here is an example:

```cpp

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
}
```

### Future Improvements

- Remove the need for manually adding `LogDataArray`