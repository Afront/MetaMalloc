#ifndef MetaMallocImpl
#define MetaMallocImpl
#include <iostream>
#include <sstream>
#include <fstream>
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
 * @brief Converts an element of a LogDataArray object into a string in a CSV-row format
 *
 * @param[in]  i     The index
 */
HOST std::string LogDataArray::data_to_s(size_t i) {
	if (address_arr[i] == nullptr) return "";
	std::stringstream string_stream;

	string_stream << 
		kernel_name  << ',' <<
		(type_arr[i] == 0 ? "malloc" : "free") << ',' <<
		clock_arr[i]   << ',' <<
		address_arr[i]   << ',' <<
		memory_size_arr[i]  << ',' <<
		grid_dim.x  << ',' <<
		grid_dim.y  << ',' <<
		grid_dim.z  << ',' <<
		block_dim.x  << ',' <<
		block_dim.y  << ',' <<
		block_dim.z  << ',' <<
		block_id_arr[i].x  << ',' <<
		block_id_arr[i].y  << ',' <<
		block_id_arr[i].z  << ',' <<
		thread_id_arr[i].x  << ',' <<
		thread_id_arr[i].y  << ',' <<
		thread_id_arr[i].z;

	return string_stream.str();
}

/**
 * @brief      Writes the input to a file
 *
 * @param[in]  filename  The filename
 */
HOST void LogDataArray::write_to_file(std::string filename) {
	std::fstream file(filename, std::ios::out | std::ios::app);

	for (size_t i = 0; i < length(); i++){
		auto data_str = data_to_s(i);
		if (data_str != "") file << data_str << '\n'; // -> write to file
	}
}

/**
 * @brief      Divides two numbers and then rounds it up to the nearest integer
 *
 * https://stackoverflow.com/a/30824434
 *
 * @param[in]  a     The dividend
 * @param[in]  b     The divisor
 *
 * @tparam     T     The type of the dividend
 * @tparam     U     The type of the divisor
 *
 * @return     The quotient that is rounded up to the nearest integer
 */
template <typename T, typename U>
ALL_DEVICES T ceil_div(T const a, U const b) {
	return a/b + (a%b > 0);
}

/**
 * @brief      Formats the size by adding a unit 
 *
 * @param[in]  size       The size
 * @param[in]  is_binary  Indicates if binary
 *
 * @tparam     T          The type of size (e.g. size_t)
 *
 * @return     The size formatted in a string
 */
template <typename T>
HOST std::string size_to_string(const T& size, const bool& is_binary = true) {
	std::stringstream string_stream;

	static char size_units[] = {'\0', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'};
	auto kb = is_binary ? 1028 : 1000;
	auto result = static_cast<double>(size);
	size_t i;

	for (i = 0; result > kb; i++){
		result /= kb;
	}

	auto number_of_chunks = ceil_div(size, 65536);

	string_stream 
		<< result << size_units[i] 
		<< (is_binary ? "iB (" : "B (") 
		<< number_of_chunks << " 64KB-chunk" 
		<< (number_of_chunks == 1 ? '\0' : 's') << ')';

	return string_stream.str();
}

/**
 * @brief      Initializes a `MemoryManager` object with the given heap size
 *
 * @param[in]  heap_size             The size of the heap
 *
 * @tparam     MemoryAllocator  { description }
 *
 * @return     { description_of_the_return_value }
 */
template <typename MemoryAllocator>
HOST MemoryManager<MemoryAllocator>::MemoryManager(size_t heap_size, std::string filename) : memory_allocator(MemoryAllocator(heap_size)) {
	int device;
	size_t cuda_heap_size;
	int runtime_version;
	uint major;
	uint minor;

	cudaGetDevice(&device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	cudaDeviceGetLimit(&cuda_heap_size, cudaLimitMallocHeapSize);
	cudaRuntimeGetVersion(&runtime_version);
	major = runtime_version/1000;
	minor = (runtime_version - (major * 1000))/10;
	std::fstream file(filename, std::ios::out);

	file << "---\n";
	file << "device: " << prop.name << " " << prop.major << "." << prop.minor << "\n";
	file << "device number: " << device << "\n"; 
	file << "cuda runtime version: " << major << "." << minor << "\n";
	file << "cuda heap size: " << size_to_string(cuda_heap_size) << "\n";
	file << "heap size: " << size_to_string(heap_size) << "\n";
	file << "---\n";

	file << 
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
		"threadIdx.z" 
		<< std::endl;
}

/**
 * @brief      Allocates a section of memory with the given `size` using the memory allocator and logs the allocation-related information
 *
 * @param[in]  size             The size of memory that would be allocated
 * @param[in]  log_data         The array where log information related to the dealloaction will be stored
 *
 * @tparam     MemoryAllocator  The memory allocator that will be used to free the section of memory
 *
 * @return     The pointer that refers to the allocated memory
 */
template <typename MemoryAllocator>
DEVICE __forceinline__ void* MemoryManager<MemoryAllocator>::malloc(size_t size, LogDataArray log_data) {
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

/**
 * @brief      Frees a section of memory pointed by the pointer using the memory allocator and logs the deallocation-related information
 *
 * @param      pointer          The pointer that points to the memory that would be freed
 * @param[in]  log_data         The array where log information related to the dealloaction will be stored
 *
 * @tparam     MemoryAllocator  The memory allocator that will be used to free the section of memory
 */
template <typename MemoryAllocator>
DEVICE __forceinline__ void MemoryManager<MemoryAllocator>::free(void* pointer, LogDataArray log_data) {
	memory_allocator.free(pointer);

	auto tid = threadIdx.x + blockIdx.x * blockDim.x;

	log_data.clock_arr[tid] = clock64();
	log_data.thread_id_arr[tid] = threadIdx;
	log_data.block_id_arr[tid] = blockIdx;
	log_data.address_arr[tid] = pointer;
	log_data.memory_size_arr[tid] = 0; // TODO: somehow get memory size
	log_data.type_arr[tid] = 1; 
}
#endif