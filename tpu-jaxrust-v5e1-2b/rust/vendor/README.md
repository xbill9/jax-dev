# Vendored

`pjrt_c_api.h` is `xla/pjrt/c/pjrt_c_api.h` from [openxla/xla](https://github.com/openxla/xla) at commit `261d4d4979f65a89b614a6a5baae1790ce51f048` (2026-10-21), PJRT C API 0.117, Apache-2.0. It depends only on the C standard headers. A plugin may be older than the header: `src/pjrt.c` checks each function-table entry against the plugin's `PJRT_Api.struct_size`, and argument structs sent at the header's size are accepted by plugins whose own size is smaller.
