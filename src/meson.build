project('meta_malloc', 'cpp', 'cuda')

expected = dependency('expected')

lib = library('cuda_code', 'meta_malloc.cu', dependencies: expected)

# executable('main', 'main.cpp', link_with: lib, override_options: ['cpp_std=c++2a'], dependencies: expected)