#include <iostream>
#include <sstream>
#include <fstream>
#include <tl/expected.hpp>

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

using CUDAResult = tl::expected<bool, cudaError_t>;
/*
CUDAResult to_CUDAResult(cudaError_t error) {

	return (error == cudaSuccess) ? true : tl::make_unexpected(error);
}
*/

HOST void error_check(cudaError_t error) {
	if (error != cudaSuccess){
		std::cout << "Error: " << cudaGetErrorString(error) << '\n';
		exit(1);
	}
}

struct LogDataArray {
	char* kernel_name;
	const dim3 block_dim;
	const dim3 grid_dim;

	int64_t* clock_arr;
	int32_t* thread_id_arr;
	int32_t* block_id_arr;
	void** address_arr;
	size_t* memory_size_arr;
	const char** type_arr;

	ALL_DEVICES size_t length() {
		return block_dim.x * block_dim.y * block_dim.z *
			grid_dim.x * grid_dim.y * grid_dim.z;
	}

	HOST LogDataArray(std::string kernel_name_str, const dim3& grid_dim, const dim3& block_dim) : block_dim(block_dim), grid_dim(grid_dim) {
		// to_CUDAResult(cudaMallocManaged(&kernel_name, sizeof(char) * kernel_name_str.size()));
		error_check(cudaMallocManaged(&kernel_name, sizeof(char) * kernel_name_str.size()));

		size_t i = 0;
		for(auto&& c : kernel_name_str) {
			kernel_name[i] = c;
			i++;
		}

		error_check(cudaMallocManaged(&clock_arr, sizeof(int64_t) * length()));
		error_check(cudaMallocManaged(&thread_id_arr, sizeof(int32_t) * length()));
		error_check(cudaMallocManaged(&block_id_arr, sizeof(int32_t) * length()));
		error_check(cudaMallocManaged(&address_arr, sizeof(void*) * length()));
		error_check(cudaMallocManaged(&memory_size_arr, sizeof(size_t) * length()));
		error_check(cudaMallocManaged(&type_arr, sizeof(char*) * length()));
	}

	HOST void free() {
		error_check(cudaFree(kernel_name));
		error_check(cudaFree(clock_arr));
		error_check(cudaFree(thread_id_arr));
		error_check(cudaFree(block_id_arr));
		error_check(cudaFree(address_arr));		
		error_check(cudaFree(memory_size_arr));
		error_check(cudaFree(type_arr));
	
	}

	ALL_DEVICES void print_at_index(size_t i) {
		printf(
			"kernel name: %s,"
			"clock: %li,"
			"threadIdx.x: %d,"
			"blockIdx.x: %d,"
			"blockDim.x: %d,"
			"blockDim.y: %d,"
			"blockDim.z: %d,"
			"address: %p"
			"memory_size: %lu"
//			""
			"\n",
			kernel_name,
			clock_arr[i],
			thread_id_arr[i],
			block_id_arr[i],
			block_dim.x,
			block_dim.y,
			block_dim.z,
			address_arr[i],
			memory_size_arr[i]
		);
	}

	HOST std::string data_to_s(size_t i) {
		// CSV format
		// Kernel name, grid dim, block dim, type, clock, thread idx, block idx, address, memory size

		std::stringstream string_stream;

		string_stream << 
			kernel_name  << ',' <<
			grid_dim.x << ',' <<
			grid_dim.y << ',' <<
			grid_dim.z << ',' <<
			block_dim.x << ',' <<
			block_dim.y << ',' <<
			block_dim.z << ',' <<

			type_arr[i] << ',' <<

			clock_arr[i] << ',' <<
			thread_id_arr[i] << ',' <<
			block_id_arr[i] << ',' <<
			address_arr[i] // << ',' <<
			// memory_size_arr[i]
			<< std::endl;

		return string_stream.str();
	}



	HOST void write_to_file(std::string filename) {
		for (size_t i = 0; i < length(); i++){
			std::cout << data_to_s(i) << '\n'; // -> write to file
		}


	}
};


template <typename MemoryAllocator>
class MemoryManager {
	MemoryAllocator memory_allocator;

public:	
	HOST MemoryManager(size_t size) : memory_allocator(MemoryAllocator(size)) {}

	DEVICE __forceinline__ void* malloc(
		size_t size,
		LogDataArray log_data
	) {
		// 3 "heavy" calls: malloc, clock64 read, printf
		// not sure how to order

		// technically should benchmark here instead
		auto pointer = memory_allocator.malloc(size);
		// should end benchmark here

		// printf("pointer %p\n", pointer);

		auto tid = threadIdx.x + blockIdx.x * blockDim.x;

		log_data.clock_arr[tid] = clock64();
		log_data.thread_id_arr[tid] = threadIdx.x;
		log_data.block_id_arr[tid] = blockIdx.x;
		log_data.address_arr[tid] = pointer;
		log_data.memory_size_arr[tid] = size;
		log_data.type_arr[tid] = "malloc";


		log_data.print_at_index(tid);


		return pointer;
	}

	DEVICE __forceinline__ void free(void* pointer) {
		return memory_allocator.free(pointer);
	}
};
