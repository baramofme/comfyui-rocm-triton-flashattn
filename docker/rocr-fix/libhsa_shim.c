/* Stub HSA symbols missing from ROCr 1.18 that the torch 2.12+rocm7.14
 * bundled libs reference with ROCR_1 version. Needed because ROCr 1.21
 * busy-spins a core (ROCm/TheRock#7051); LD_PRELOADing 1.18 requires these
 * 10 versioned symbols to satisfy the linker. Never called during inference;
 * returning an error is safe. Build: gcc -shared -fPIC -Wl,--version-script=rocr1.map -o libhsa_shim.so libhsa_shim.c
 */
typedef int hsa_status_t;
#define HSA_STATUS_ERROR_INVALID_ARGUMENT ((hsa_status_t)0x1001)

#define STUB(name) hsa_status_t name(void) { return HSA_STATUS_ERROR_INVALID_ARGUMENT; }

STUB(hsa_amd_external_semaphore_handle_close)
STUB(hsa_amd_external_semaphore_handle_open)
STUB(hsa_amd_memory_async_batch_copy)
STUB(hsa_amd_svm_discard_batch_async)
STUB(hsa_amd_vmem_export_fabric_handle)
STUB(hsa_amd_vmem_import_fabric_handle)
STUB(hsa_ext_image_create_v2)
STUB(hsa_ext_image_data_get_info_v2)
STUB(hsa_ext_image_destroy_v2)
STUB(hsa_ext_image_mipmap_array_get_level)
