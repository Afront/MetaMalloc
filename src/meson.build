project('meta_malloc', 'cpp', 'cuda')

expected = dependency('expected')

lib = library('meta_malloc', 'meta_malloc.cu', dependencies: expected)

# executable('main', 'main.cpp', link_with: lib, override_options: ['cpp_std=c++2a'], dependencies: expected)