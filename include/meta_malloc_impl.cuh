#ifndef MetaMallocImpl
#define MetaMallocImpl
#include <iostream>
#include <sstream>
// #include <fstream>
#include "meta_malloc.cuh"

// https://stackoverflow.com/a/28166605
#if defined(__GNUC__) || defined(__GNUG__)
	#define ALL_DEVICES __attribute__ ((device)) __attribute__ ((host))
	#define DEVICE __attribute__ ((device))
	#define HOST __attribute__ ((host))
	#define GLOBAL __attribute__ ((global))
#elif defined(_MSC_VER)
	#ifdef __CUDACC__
		#define ALL_DEVICES __device__ __host__
		#define DEVICE __device__
		#define HOST __host__
		#define GLOBAL __global__
	#else
		#define ALL_DEVICES
		#define DEVICE
		#define HOST
		#define GLOBAL
	#endif
#endif

/**
 * @brief      (HOST) Checks if the CUDA error is an actual error or not
 *
 * @param[in]  error  The error (or cudaSuccess)
 */
HOST void error_check(cudaError_t error) {
	if (error != cudaSuccess){
		std::cout << "Error: " << cudaGetErrorString(error) << '\n';
		exit(1);
	}
}

/**
 * @brief      (ALL_DEVICES) Gets the length of the log data array
 *
 * @return    Returns the number of threads for the given malloc call (not actual number of malloc's)
 */
ALL_DEVICES size_t LogDataArray::length() {
	return block_dim.x * block_dim.y * block_dim.z *
		grid_dim.x * grid_dim.y * grid_dim.z;
}


/**
 * @brief      Constructs a new instance of `LogDataArray`
 *
 * @param[in]  kernel_name_str  The name of the kernel
 * @param[in]  grid_dim         The grid dimension
 * @param[in]  block_dim        The block dimension
 */
HOST LogDataArray::LogDataArray(std::string kernel_name_str, const dim3& grid_dim, const dim3& block_dim) : block_dim(block_dim), grid_dim(grid_dim) {
	// to_CUDAResult(cudaMallocManaged(&kernel_name, sizeof(char) * kernel_name_str.size()));
	error_check(cudaMallocManaged(&kernel_name, sizeof(char) * kernel_name_str.size()));

	size_t i = 0;
	for(auto&& c : kernel_name_str) {
		kernel_name[i] = c;
		i++;
	}


	error_check(cudaMallocManaged(&clock_arr, sizeof(int64_t) * length()));
	error_check(cudaMallocManaged(&thread_id_arr, sizeof(dim3) * length()));
	error_check(cudaMallocManaged(&block_id_arr, sizeof(dim3) * length()));
	error_check(cudaMallocManaged(&address_arr, sizeof(void*) * length()));
	error_check(cudaMallocManaged(&memory_size_arr, sizeof(size_t) * length()));
	error_check(cudaMallocManaged(&type_arr, sizeof(size_t) * length()));
}


/**
 * @brief      Explicitly frees the `LogDataArray` object
 */
HOST void LogDataArray::free() {
	error_check(cudaFree(kernel_name));
	error_check(cudaFree(clock_arr));
	error_check(cudaFree(thread_id_arr));
	error_check(cudaFree(block_id_arr));
	error_check(cudaFree(address_arr));		
	error_check(cudaFree(memory_size_arr));
	error_check(cudaFree(type_arr));
}

/**
 * @brief      Prints the row of the LogDataArray in a CSV row format
 *
 * @param[in]  i     The index
 */
ALL_DEVICES void LogDataArray::print_at_index(size_t i) {
	// Kernel name, grid dim, block dim, type, clock, thread idx, block idx, address, memory size
	if (address_arr[i] != nullptr)
		printf(
			"%s,"	// kernel name
			"%s,"	// type
			"%li,"	// clock
			"%p,"	// address
			"%lu,"	// memory_size
			"%d,"	// gridDim.x
			"%d,"	// gridDim.y
			"%d,"	// gridDim.z
			"%d,"	// blockDim.x
			"%d,"	// blockDim.y
			"%d,"	// blockDim.z
			"%d,"	// blockIdx.x
			"%d,"	// blockIdx.y
			"%d,"	// blockIdx.z
			"%d,"	// threadIdx.x
			"%d,"	// threadIdx.y
			"%d"	// threadIdx.z
			"\n",
			kernel_name,
			type_arr[i] == 0 ? "malloc" : "free",
			clock_arr[i],
			address_arr[i],
			memory_size_arr[i],
			grid_dim.x,
			grid_dim.y,
			grid_dim.z,
			block_dim.x,
			block_dim.y,
			block_dim.z,
			block_id_arr[i].x,
			block_id_arr[i].y,
			block_id_arr[i].z,
			thread_id_arr[i].x,
			thread_id_arr[i].y,
			thread_id_arr[i].z
		);
}

HOST std::string LogDataArray::data_to_s(size_t i) {
	// CSV format
	// Kernel name, grid dim, block dim, type, clock, thread idx, block idx, address, memory size
	std::stringstream string_stream;

/*	string_stream << 
		kernel_name  << ',' <<
		grid_dim.x << ',' <<
		grid_dim.y << ',' <<
		grid_dim.z << ',' <<
		block_dim.x << ',' <<
		block_dim.y << ',' <<
		block_dim.z << ',' <<

		// type_arr[i] << ',' <<

		clock_arr[i] << ',' <<
		thread_id_arr[i] << ',' <<
		block_id_arr[i] << ',' <<
		address_arr[i] << ',' <<
		memory_size_arr[i]
		<< std::endl;
*/
	return string_stream.str();
}

HOST void LogDataArray::write_to_file(std::string filename) {
	for (size_t i = 0; i < length(); i++){
		print_at_index(i);
		// std::cout << data_to_s(i) << '\n'; // -> write to file
	}
}

template <typename MemoryAllocator>
HOST MemoryManager<MemoryAllocator>::MemoryManager(size_t size) : memory_allocator(MemoryAllocator(size)) {
	int device;
	size_t heap_size;
	int major_capability;
	int minor_capability;
	int runtime_version;

	cudaGetDevice(&device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
	cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor, device);
	cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor, device);
	cudaRuntimeGetVersion(&runtime_version);


	std::cout << "---\n";
	std::cout << 
		"device: " << prop.name << " " << prop.major << "." << prop.minor << 
		"(" << major_capability << "." << minor_capability << ")\n";
	std::cout << "device number: " << device << "\n"; 
	std::cout << "cuda version:" << runtime_version << "\n";
	std::cout << "heap size: " << heap_size << "\n";
	std::cout << "---\n";

	std::cout << 
		"kernel name,"
		"type,"
		"clock,"
		"address,"
		"memory_size,"
		"gridDim.x,"
		"gridDim.y,"
		"gridDim.z,"
		"blockDim.x,"
		"blockDim.y,"
		"blockDim.z,"
		"blockIdx.x,"
		"blockIdx.y,"
		"blockIdx.z,"
		"threadIdx.x,"
		"threadIdx.y,"
		"threadIdx.z," 
		<< std::endl;

}

template <typename MemoryAllocator>
DEVICE __forceinline__ void* MemoryManager<MemoryAllocator>::malloc(size_t size, LogDataArray log_data) {
	// 3 "heavy" calls: malloc, clock64 read, printf
	// not sure how to order

	auto pointer = memory_allocator.malloc(size);
	auto tid = threadIdx.x + blockIdx.x * blockDim.x;

	log_data.clock_arr[tid] = clock64();
	log_data.thread_id_arr[tid] = threadIdx;
	log_data.block_id_arr[tid] = blockIdx;
	log_data.address_arr[tid] = pointer;
	log_data.memory_size_arr[tid] = size;
	log_data.type_arr[tid] = 0; // MemoryOperation::Allocation;

	return pointer;
}

template <typename MemoryAllocator>
DEVICE __forceinline__ void MemoryManager<MemoryAllocator>::free(void* pointer, LogDataArray log_data) {
	memory_allocator.free(pointer);

	auto tid = threadIdx.x + blockIdx.x * blockDim.x;

	log_data.clock_arr[tid] = clock64();
	log_data.thread_id_arr[tid] = threadIdx;
	log_data.block_id_arr[tid] = blockIdx;
	log_data.address_arr[tid] = pointer;
	log_data.memory_size_arr[tid] = 0; // note: somehow get memory size
	log_data.type_arr[tid] = 1; // MemoryOperation::Free;
}

#endif