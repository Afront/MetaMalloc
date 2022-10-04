#ifndef MetaMallocCUH
#define MetaMallocCUH 
#include <iostream>

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

enum memoryOperation {
	Free,
	Allocation
};


/**
 * @brief      Consists of allocation-related information to be stored in a CSV-row format
 */
struct LogDataArray {
	char* kernel_name;
	const dim3 block_dim;
	const dim3 grid_dim;

	int64_t* clock_arr;
	dim3* thread_id_arr;
	dim3* block_id_arr;
	void** address_arr;
	size_t* memory_size_arr;
	size_t* type_arr;

	ALL_DEVICES size_t length();
	HOST LogDataArray(std::string kernel_name_str, const dim3& grid_dim, const dim3& block_dim);
	HOST void free();
	HOST std::string data_to_s(size_t i);
	HOST void write_to_file(std::string filename);
};


/**
 * @brief      This class describes a memory manager.
 *
 * @tparam     MemoryAllocator  An allocator that can allocate and free a portion of GPU-side memory (e.g. CUDA-Alloc, Ouroboros).
 */
template <typename MemoryAllocator>
class MemoryManager {
	MemoryAllocator memory_allocator;

public:	
	HOST MemoryManager(size_t size, std::string filename);
	DEVICE __forceinline__ void* malloc(size_t size, LogDataArray log_data);
	DEVICE __forceinline__ void free(void* pointer, LogDataArray log_data);
};
#endif