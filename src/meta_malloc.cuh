#ifndef MetaMallocCUH
#define MetaMallocCUH 
#include <iostream>
#include <sstream>
#include <fstream>

// https://stackoverflow.com/a/28166605
#if defined(__GNUC__) || defined(__GNUG__)
	#define ALL_DEVICES __attribute__ ((device)) __attribute__ ((host))
	#define DEVICE __attribute__ ((device))
	#define HOST __attribute__ ((host))
	#define GLOBAL __attribute__ ((global))
#elif defined(_MSC_VER)
// https://learn.microsoft.com/en-us/previous-versions/visualstudio/visual-studio-2012/dabb5z75(v=vs.110)?redirectedfrom=MSDN
// https://stackoverflow.com/questions/28411283/dealing-with-attribute-in-msvc
/*	#define ALL_DEVICES __declspec(device) __declspec(host)
	#define DEVICE __declspec(device)
	#define HOST __declspec(host)
	#define GLOBAL __declspec(global)*/
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

/*class String {
	char* string;

public:
	ALL_DEVICES String(const char* string) {
		malloc();


	}

	~String(){ 


	}


	ALL_DEVICES const char* str() {
		return string;
	}

}
*/

using String = const char*;

enum class MemoryOperation {
	Free,
	Allocation
};

enum memoryOperation {
	Free,
	Allocation
};

const char* to_s(MemoryOperation memory_operation) {
	switch (memory_operation) {
		case MemoryOperation::Free:
			return "free";
		case MemoryOperation::Allocation:
			return "malloc";
	}
}

std::ostream& operator<< (std::ostream& os, MemoryOperation memory_operation) {
	return os << to_s(memory_operation);
}





/*std::stringstream& operator<< (std::stringstream& ss, MemoryOperation memory_operation) {
	switch (memory_operation) {
		case MemoryOperation::Free:
			return ss << "free";
		case MemoryOperation::Allocation:
			return ss << "malloc";
	}
}

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
	ALL_DEVICES void print_at_index(size_t i);
	HOST std::string data_to_s(size_t i);
	HOST void write_to_file(std::string filename);
};

template <typename MemoryAllocator>
class MemoryManager {
	MemoryAllocator memory_allocator;

public:	
	HOST MemoryManager(size_t size) : memory_allocator(MemoryAllocator(size)) {}
	DEVICE __forceinline__ void* malloc(size_t size, LogDataArray log_data);
	DEVICE __forceinline__ void free(void* pointer);
};

#endif