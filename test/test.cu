#include <iostream>
#include <sstream>
#include <fstream>

#include "UtilityFunctions.cuh"

#ifdef TEST_CUDA
#include "cuda/Instance.cuh"
using MemoryAllocator = MemoryManagerCUDA;
const std::string mem_name("CUDA");
#elif TEST_HALLOC
#include "halloc/Instance.cuh"
using MemoryAllocator = MemoryManagerHalloc;
const std::string mem_name("HALLOC");
#elif TEST_XMALLOC
#include "xmalloc/Instance.cuh"
using MemoryAllocator = MemoryManagerXMalloc;
const std::string mem_name("XMALLOC");
#elif TEST_SCATTERALLOC
#include "scatteralloc/Instance.cuh"
using MemoryAllocator = MemoryManagerScatterAlloc;
const std::string mem_name("ScatterAlloc");
#elif TEST_FDG
#include "fdg/Instance.cuh"
using MemoryAllocator = MemoryManagerFDG;
const std::string mem_name("FDGMalloc");
#elif TEST_OUROBOROS
#include "ouroboros/Instance.cuh"
	#ifdef TEST_PAGES
	#ifdef TEST_VIRTUALIZED_ARRAY
	using MemoryAllocator = MemoryManagerOuroboros<OuroVAPQ>;
	const std::string mem_name("Ouroboros-P-VA");
	#elif TEST_VIRTUALIZED_LIST
	using MemoryAllocator = MemoryManagerOuroboros<OuroVLPQ>;
	const std::string mem_name("Ouroboros-P-VL");
	#else
	using MemoryAllocator = MemoryManagerOuroboros<OuroPQ>;
	const std::string mem_name("Ouroboros-P-S");
	#endif
	#endif
	#ifdef TEST_CHUNKS
	#ifdef TEST_VIRTUALIZED_ARRAY
	using MemoryAllocator = MemoryManagerOuroboros<OuroVACQ>;
	const std::string mem_name("Ouroboros-C-VA");
	#elif TEST_VIRTUALIZED_LIST
	using MemoryAllocator = MemoryManagerOuroboros<OuroVLCQ>;
	const std::string mem_name("Ouroboros-C-VL");
	#else
	using MemoryAllocator = MemoryManagerOuroboros<OuroCQ>;
	const std::string mem_name("Ouroboros-C-S");
	#endif
	#endif
#elif TEST_REGEFF
#include "regeff/Instance.cuh"
	#ifdef TEST_ATOMIC
	using MemoryAllocator = MemoryManagerRegEff<RegEffVariants::AtomicMalloc>;
	const std::string mem_name("RegEff-A");
	#elif TEST_ATOMIC_WRAP
	using MemoryAllocator = MemoryManagerRegEff<RegEffVariants::AWMalloc>;
	const std::string mem_name("RegEff-AW");
	#elif TEST_CIRCULAR
	using MemoryAllocator = MemoryManagerRegEff<RegEffVariants::CMalloc>;
	const std::string mem_name("RegEff-C");
	#elif TEST_CIRCULAR_FUSED
	using MemoryAllocator = MemoryManagerRegEff<RegEffVariants::CFMalloc>;
	const std::string mem_name("RegEff-CF");
	#elif TEST_CIRCULAR_MULTI
	using MemoryAllocator = MemoryManagerRegEff<RegEffVariants::CMMalloc>;
	const std::string mem_name("RegEff-CM");
	#elif TEST_CIRCULAR_FUSED_MULTI
	using MemoryAllocator = MemoryManagerRegEff<RegEffVariants::CFMMalloc>;
	const std::string mem_name("RegEff-CFM");
	#endif
#endif

#include "../../MetaMalloc/src/meta_malloc.cuh"
#include "../../MetaMalloc/src/meta_malloc_impl.cuh"

template <typename MemoryManagerType>
__global__ void d_testAllocation(MemoryManagerType mm, int** verification_ptr, int num_allocations, int allocation_size,
	LogDataArray log_data_array
	) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;

	verification_ptr[tid] = reinterpret_cast<int*>(
		mm.malloc(
			allocation_size,
			log_data_array
		)
	);
}

__global__ void d_testWriteToMemory(
		int** verification_ptr,
		int num_allocations,
		int allocation_size

		// LogDataArray log_data_array
	) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations)
		return;
	
	auto ptr = verification_ptr[tid];

	for(auto i = 0; i < (allocation_size / sizeof(int)); ++i){
		ptr[i] = tid;
	}
}

__global__ void d_testReadFromMemory(int** verification_ptr, int num_allocations, int allocation_size){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations) return;
	
	auto ptr = verification_ptr[tid];

	for(auto i = 0; i < (allocation_size / sizeof(int)); ++i)
	{
		if(ptr[i] != tid)
		{
			printf("%d | We got a wrong value here! %d vs %d\n", tid, ptr[i], tid);
			__trap();
		}
	}
}

template <typename MemoryManagerType>
__global__ void d_testFree(MemoryManagerType mm, int** verification_ptr, int num_allocations){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= num_allocations) return;
	mm.free(verification_ptr[tid]);
}

int main(int argc, char* argv[]){
	int device{0};
	cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << "\n";

	int num_allocations{10000};
	int allocation_size_byte{16};
	int num_iterations {10};
	if(argc >= 2) {
		num_allocations = atoi(argv[1]);
		if(argc >= 3) {
			allocation_size_byte = atoi(argv[2]);
			if(argc >= 4) {
				num_iterations = atoi(argv[3]);
			}
		}
	}
	allocation_size_byte = Utils::alignment(allocation_size_byte, sizeof(int));
	std::cout << "Number of Allocations: " << num_allocations << " | Allocation Size: " << allocation_size_byte << " | Iterations: " << num_iterations << std::endl;
	std::cout << "--- " << mem_name << "---\n";

	MemoryManager<MemoryAllocator> memory_manager(8192ULL * 1024ULL * 1024ULL);

	int** d_memory{nullptr};
	CHECK_ERROR(cudaMalloc(&d_memory, sizeof(int*) * num_allocations));

	int blockSize {256};
	int gridSize {Utils::divup(num_allocations, blockSize)};
	float timing_allocation{0.0f};
	float timing_free{0.0f};
	cudaEvent_t start, end;
	for(auto i = 0; i < num_iterations; ++i){
		std::cout << "Iteration " << i + 1 << " / " << num_iterations << std::endl;

		LogDataArray allocation_data(
			"d_testAllocation",
			gridSize,
			blockSize
		);

		Utils::start_clock(start, end);

		d_testAllocation <decltype(memory_manager)> <<<gridSize, blockSize>>>(
			memory_manager, d_memory, num_allocations, allocation_size_byte, 
			allocation_data
		);

		allocation_data.write_to_file("tmp_file");

		timing_allocation += Utils::end_clock(start, end);
		CHECK_ERROR(cudaDeviceSynchronize());

		d_testWriteToMemory<<<gridSize, blockSize>>>(d_memory, num_allocations, allocation_size_byte);
		CHECK_ERROR(cudaDeviceSynchronize());

		d_testReadFromMemory<<<gridSize, blockSize>>>(d_memory, num_allocations, allocation_size_byte);
		CHECK_ERROR(cudaDeviceSynchronize());

		Utils::start_clock(start, end);
		d_testFree <decltype(memory_manager)> <<<gridSize, blockSize>>>(memory_manager, d_memory, num_allocations);
		timing_free += Utils::end_clock(start, end);
		CHECK_ERROR(cudaDeviceSynchronize());

		allocation_data.free();
		CHECK_ERROR(cudaDeviceSynchronize());
	}
	timing_allocation /= num_iterations;
	timing_free /= num_iterations;

	std::cout << "Timing Allocation: " << timing_allocation << "ms" << std::endl;
	std::cout << "Timing       Free: " << timing_free << "ms" << std::endl;

	printf("Testcase done!\n");

	return 0;
}