// A small C shim over the PJRT C API (vendor/pjrt_c_api.h), the interface libtpu.so exports
// as GetPjrtApi and that JAX itself uses to reach the TPU. It plays the part aten.cpp plays in
// the torch rig: Rust calls these flat functions and never sees a PJRT struct.
//
// One process, one client. Every function returns 0 / a handle on success and -1 / NULL on
// failure, leaving the message in pj_last_error(). The header may be newer than the plugin:
// argument structs are sent at the header's size, which plugins accept when it is at least
// their own, and every function-table entry is checked against the plugin's PJRT_Api size
// before it is called.
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pjrt_c_api.h"

static const PJRT_Api* api = NULL;
static PJRT_Client* client = NULL;
static PJRT_Device* device = NULL;
static __thread char last_error[4096];

const char* pj_last_error(void) { return last_error; }

#define HAS(fn) (api && api->struct_size >= PJRT_STRUCT_SIZE(PJRT_Api, fn) && api->fn)
#define NEED(fn)                                                                    \
  do {                                                                              \
    if (!HAS(fn)) {                                                                 \
      snprintf(last_error, sizeof last_error, "plugin does not provide %s", #fn);   \
      return -1;                                                                    \
    }                                                                               \
  } while (0)

// Records a PJRT error's message and frees it. Returns -1 when err is set, else 0.
static int take(PJRT_Error* err, const char* what) {
  if (!err) return 0;
  PJRT_Error_Message_Args m = {.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE, .error = err};
  api->PJRT_Error_Message(&m);
  snprintf(last_error, sizeof last_error, "%s: %.*s", what, (int)m.message_size, m.message);
  PJRT_Error_Destroy_Args d = {.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE, .error = err};
  api->PJRT_Error_Destroy(&d);
  return -1;
}

static int await_and_free(PJRT_Event* ev, const char* what) {
  if (!ev) return 0;
  PJRT_Event_Await_Args a = {.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE, .event = ev};
  int rc = take(api->PJRT_Event_Await(&a), what);
  PJRT_Event_Destroy_Args d = {.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE, .event = ev};
  api->PJRT_Event_Destroy(&d);
  return rc;
}

int pj_header_version(int* major, int* minor) {
  *major = PJRT_API_MAJOR;
  *minor = PJRT_API_MINOR;
  return 0;
}

// dlopens the plugin, initializes it, creates the client and picks its first device.
int pj_init(const char* plugin_path, int* major, int* minor, int* num_devices) {
  if (client) {
    snprintf(last_error, sizeof last_error, "already initialized");
    return -1;
  }
  void* h = dlopen(plugin_path, RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    snprintf(last_error, sizeof last_error, "dlopen %s: %s", plugin_path, dlerror());
    return -1;
  }
  const PJRT_Api* (*get)(void) = (const PJRT_Api* (*)(void))dlsym(h, "GetPjrtApi");
  if (!get) {
    snprintf(last_error, sizeof last_error, "%s exports no GetPjrtApi", plugin_path);
    return -1;
  }
  api = get();
  *major = api->pjrt_api_version.major_version;
  *minor = api->pjrt_api_version.minor_version;
  if (HAS(PJRT_Plugin_Initialize)) {
    PJRT_Plugin_Initialize_Args a = {.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE};
    if (take(api->PJRT_Plugin_Initialize(&a), "PJRT_Plugin_Initialize")) return -1;
  }
  NEED(PJRT_Client_Create);
  PJRT_Client_Create_Args c = {.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE};
  if (take(api->PJRT_Client_Create(&c), "PJRT_Client_Create")) return -1;
  client = c.client;
  NEED(PJRT_Client_AddressableDevices);
  PJRT_Client_AddressableDevices_Args d = {
      .struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE, .client = client};
  if (take(api->PJRT_Client_AddressableDevices(&d), "PJRT_Client_AddressableDevices")) return -1;
  *num_devices = (int)d.num_addressable_devices;
  if (d.num_addressable_devices == 0) {
    snprintf(last_error, sizeof last_error, "client has no addressable devices");
    return -1;
  }
  device = d.addressable_devices[0];
  return 0;
}

// Compiles a StableHLO module given as MLIR text. compile_options is a serialized
// xla.CompileOptionsProto; empty means the plugin's defaults (one replica, one partition).
void* pj_compile(const char* mlir, size_t len, const char* opts, size_t opts_len) {
  if (!HAS(PJRT_Client_Compile)) {
    snprintf(last_error, sizeof last_error, "plugin does not provide PJRT_Client_Compile");
    return NULL;
  }
  PJRT_Program p = {.struct_size = PJRT_Program_STRUCT_SIZE,
                    .code = (char*)mlir,
                    .code_size = len,
                    .format = "mlir",
                    .format_size = 4};
  PJRT_Client_Compile_Args a = {.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
                                .client = client,
                                .program = &p,
                                .compile_options = opts,
                                .compile_options_size = opts_len};
  if (take(api->PJRT_Client_Compile(&a), "PJRT_Client_Compile")) return NULL;
  return a.executable;
}

long pj_num_outputs(void* exe) {
  PJRT_LoadedExecutable_GetExecutable_Args g = {
      .struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE,
      .loaded_executable = (PJRT_LoadedExecutable*)exe};
  if (take(api->PJRT_LoadedExecutable_GetExecutable(&g), "GetExecutable")) return -1;
  PJRT_Executable_NumOutputs_Args n = {.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE,
                                       .executable = g.executable};
  int rc = take(api->PJRT_Executable_NumOutputs(&n), "NumOutputs");
  PJRT_Executable_Destroy_Args d = {.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE,
                                    .executable = g.executable};
  api->PJRT_Executable_Destroy(&d);
  return rc ? -1 : (long)n.num_outputs;
}

void pj_free_exe(void* exe) {
  if (!exe) return;
  PJRT_LoadedExecutable_Destroy_Args d = {
      .struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE,
      .executable = (PJRT_LoadedExecutable*)exe};
  api->PJRT_LoadedExecutable_Destroy(&d);
}

// Element types the model uses. Rust passes these codes, never PJRT's enum.
static PJRT_Buffer_Type to_pjrt(int dtype) {
  switch (dtype) {
    case 0: return PJRT_Buffer_Type_F32;
    case 1: return PJRT_Buffer_Type_BF16;
    case 2: return PJRT_Buffer_Type_S32;
    default: return PJRT_Buffer_Type_INVALID;
  }
}

// Copies a dense row-major host array to the device. Returns once the host memory may be
// reused.
void* pj_upload(int dtype, const void* data, const int64_t* dims, size_t num_dims) {
  PJRT_Client_BufferFromHostBuffer_Args a = {
      .struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE,
      .client = client,
      .data = data,
      .type = to_pjrt(dtype),
      .dims = dims,
      .num_dims = num_dims,
      .host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes,
      .device = device};
  if (take(api->PJRT_Client_BufferFromHostBuffer(&a), "BufferFromHostBuffer")) return NULL;
  if (await_and_free(a.done_with_host_buffer, "BufferFromHostBuffer done")) return NULL;
  return a.buffer;
}

// Runs exe on the one device. outs must hold num_outputs slots. Inputs listed in
// non_donatable are never donated; the module's own aliasing decides the rest. Returns
// without waiting: output buffers become ready when the device finishes.
int pj_execute(void* exe, void* const* args, size_t num_args, void** outs,
               const int64_t* non_donatable, size_t num_non_donatable) {
  PJRT_ExecuteOptions o = {.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE,
                           .non_donatable_input_indices = non_donatable,
                           .num_non_donatable_input_indices = num_non_donatable};
  PJRT_Buffer* const* arg_list = (PJRT_Buffer* const*)args;
  PJRT_Buffer** out_list = (PJRT_Buffer**)outs;
  PJRT_LoadedExecutable_Execute_Args a = {
      .struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE,
      .executable = (PJRT_LoadedExecutable*)exe,
      .options = &o,
      .argument_lists = &arg_list,
      .num_devices = 1,
      .num_args = num_args,
      .output_lists = &out_list};
  return take(api->PJRT_LoadedExecutable_Execute(&a), "Execute");
}

// Copies a device buffer into dst (size bytes), waiting for the device.
int pj_download(void* buf, void* dst, size_t size) {
  PJRT_Buffer_ToHostBuffer_Args a = {.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE,
                                     .src = (PJRT_Buffer*)buf,
                                     .dst = dst,
                                     .dst_size = size};
  if (take(api->PJRT_Buffer_ToHostBuffer(&a), "ToHostBuffer")) return -1;
  return await_and_free(a.event, "ToHostBuffer done");
}

// Bytes the buffer occupies on the device, or -1.
long long pj_device_bytes(void* buf) {
  PJRT_Buffer_OnDeviceSizeInBytes_Args a = {
      .struct_size = PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE, .buffer = (PJRT_Buffer*)buf};
  if (take(api->PJRT_Buffer_OnDeviceSizeInBytes(&a), "OnDeviceSizeInBytes")) return -1;
  return (long long)a.on_device_size_in_bytes;
}

void pj_free(void* buf) {
  if (!buf) return;
  PJRT_Buffer_Destroy_Args d = {.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE,
                                .buffer = (PJRT_Buffer*)buf};
  api->PJRT_Buffer_Destroy(&d);
}
