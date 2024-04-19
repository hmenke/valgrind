#define REGTEST 1
#define N_DEFAULT_ITERS 1

#define KNL

#ifdef KNL
#define AVX512F
#define AVX512CD
#define AVX512PF
#define AVX512ER
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "./../tests/malloc.h"

typedef unsigned char           UChar;
typedef unsigned long int       UWord;
typedef uint32_t UInt;
typedef uint64_t ULong;

#define IS_32_ALIGNED(_ptr) (0 == (0x1F & (UWord)(_ptr)))
#define IS_64_ALIGNED(_ptr) (0 == (0x3F & (UWord)(_ptr)))

typedef union {UChar u8[64]; ULong u64[8];} ZMM;
typedef struct {ZMM a1; ZMM a2; ZMM a3; ZMM a4; ULong u64; ULong m1; ULong m2;} Block;
UChar _randArray[2080] __attribute__((used));

void showZMM ( ZMM* vec )
{
   int i;
   assert(IS_64_ALIGNED(vec));
   for (i = 63; i >= 0; i--) {
      printf("%02x", (UInt)vec->u8[i]);
      if (i > 0 && 0 == ((i+0) & 7)) printf(".");
   }
}

void showBlock ( char* msg, Block* block )
{
   printf("  %s\n", msg);
   printf("    "); showZMM(&block->a1); printf("\n");
   printf("    "); showZMM(&block->a2); printf("\n");
   printf("    "); showZMM(&block->a3); printf("\n");
   printf("    "); showZMM(&block->a4); printf("\n");
   printf("    %016llx\n", block->u64);
   printf("    %016llx\n", block->m1);
   printf("    %016llx\n", block->m2);
}

UChar randUChar ( void )
{
   static UInt seed = 80021;
   seed = 1103515245 * seed + 12345;
   return (seed >> 17) & 0xFF;
}

void randBlock ( Block* b )
{
   int i;
   UChar* p = (UChar*)b;
   for (i = 0; i < sizeof(Block); i++)
      p[i] = randUChar();
}


/* Generate a function test_NAME, that tests the given insn, in both
   its mem and reg forms.  The reg form of the insn may mention, as
   operands only %zmm18, %zmm7, %zmm8, %zmm24 and %r14.  The mem form of
   the insn may mention as operands only (%rax), %zmm7, %zmm8, %zmm24
   and %r14.  It's OK for the insn to clobber zmm0, as this is needed
   for testing PCMPxSTRx, and zmm18, as this is needed for testing
   MOVMASK variants. */

#define GEN_test_empty(_name) __attribute__ ((noinline)) static void test_##_name ( void ) {}

/* use kmovw for masks */
#define GEN_test_RandM(_name, _reg_form, _mem_form)   \
   __attribute__ ((noinline)) static void test_##_name ( void )   \
{ \
   Block* b = memalign64(sizeof(Block)); \
   randBlock(b); \
   printf("%s(reg)\n", #_name); \
   showBlock("before", b); \
   __asm__ __volatile__( \
         "vmovdqa64   0(%0),%%zmm7"  "\n\t" \
         "vmovdqa64  64(%0),%%zmm8"  "\n\t" \
         "vmovdqa64 128(%0),%%zmm18"  "\n\t" \
         "vmovdqa64 192(%0),%%zmm24"  "\n\t" \
         "movq      256(%0),%%r14"   "\n\t" \
         "kmovw   264(%0),%%k4"    "\n\t" \
         "kmovw   272(%0),%%k5"    "\n\t" \
         _reg_form   "\n\t" \
         "vmovdqa64 %%zmm7,  0(%0)"  "\n\t" \
         "vmovdqa64 %%zmm8, 64(%0)"  "\n\t" \
         "vmovdqa64 %%zmm18,128(%0)"  "\n\t" \
         "vmovdqa64 %%zmm24,192(%0)"  "\n\t" \
         "movq      %%r14, 256(%0)"  "\n\t" \
         "kmovw   %%k4,  264(%0)"  "\n\t" \
         "kmovw   %%k5,  272(%0)"  "\n\t" \
         : /*OUT*/  \
         : /*IN*/"r"(b) \
         : /*TRASH*/"xmm0","xmm7","xmm8","r14","memory","cc" \
         ); \
   showBlock("after", b); \
   randBlock(b); \
   printf("%s(mem)\n", #_name); \
   showBlock("before", b); \
   __asm__ __volatile__( \
         "vmovdqa64   0(%0),%%zmm8"  "\n\t" \
         "leaq        0(%0),%%rax"   "\n\t" /* tests will use 64(%0) */ \
         "vmovdqa64 128(%0),%%zmm7"  "\n\t" \
         "vmovdqa64 192(%0),%%zmm24"  "\n\t" \
         "movq      256(%0),%%r14"   "\n\t" \
         "kmovw   264(%0),%%k4"    "\n\t" \
         "kmovw   272(%0),%%k5"    "\n\t" \
         _mem_form   "\n\t" \
         "vmovdqa64 %%zmm8, 64(%0)"  "\n\t" \
         "vmovdqa64 %%zmm7,128(%0)"  "\n\t" \
         "vmovdqa64 %%zmm24,192(%0)"  "\n\t" \
         "movq      %%r14, 256(%0)"  "\n\t" \
         "kmovw   %%k4,  264(%0)"  "\n\t" \
         "kmovw   %%k5,  272(%0)"  "\n\t" \
         : /*OUT*/  \
         : /*IN*/"r"(b) \
         : /*TRASH*/"xmm0","xmm7","xmm8","r14","rax","memory","cc" \
         ); \
   showBlock("after", b); \
   printf("\n"); \
   free(b); \
}

#define GEN_test_Mdisp(_name, _mem_disp_form)   \
   __attribute__ ((noinline)) static void test_##_name ( void )   \
{ \
   Block* b = memalign64(sizeof(Block)); \
   randBlock(b); \
   printf("%s(reg)\n", #_name); \
   showBlock("before", b); \
   __asm__ __volatile__( \
         "vmovdqa64   0(%0),%%zmm8"       "\n\t" \
         "leaq        0(%0),%%rax"        "\n\t" \
         "vmovdqa64 128(%0),%%zmm7"       "\n\t" \
         "vmovdqa64 192(%0),%%zmm24"      "\n\t" \
         "leaq _randArray(%%rip), %%r14"  "\n\t" \
         "mov $2, %%r12"  "\n\t" \
         "vpsllq $58, %%zmm18, %%zmm18"   "\n\t" \
         "vpsrlq $58, %%zmm18, %%zmm18"   "\n\t" \
         _mem_disp_form "\n\t" \
         "vmovdqa64 %%zmm8, 64(%0)"       "\n\t" \
         "vmovdqa64 %%zmm7,128(%0)"       "\n\t" \
         "vmovdqa64 %%zmm24,192(%0)"      "\n\t" \
         "xorl %%r14d, %%r14d"            "\n\t" \
         : /*OUT*/  \
         : /*IN*/"r"(b) \
         : /*TRASH*/"xmm0","xmm7","xmm8","r14","rax","memory","cc" \
         ); \
   showBlock("after", b); \
   printf("\n"); \
   free(b); \
}

#define GEN_test_Ronly(_name, _reg_form) \
   GEN_test_RandM(_name, _reg_form, "")
#define GEN_test_Monly(_name, _mem_form) \
   GEN_test_RandM(_name, "", _mem_form)

#define GEN_test_MergeMask(_name, _reg_form, _mem_form) \
   GEN_test_RandM(merge_##_name, _reg_form"%{%%k5%}", _mem_form"%{%%k5%}")
#define GEN_test_ZeroMask(_name, _reg_form, _mem_form) \
   GEN_test_RandM(zero_##_name, _reg_form"%{%%k5%}%{z%}", _mem_form"%{%%k5%}%{z%}")

// For instructions that fully support merge- and zero- masking
#define GEN_test_AllMaskModes(_name, _reg_form, _mem_form) \
   GEN_test_RandM(nomask_##_name, _reg_form,_mem_form)\
   GEN_test_MergeMask(_name, _reg_form, _mem_form)\
   GEN_test_ZeroMask(_name, _reg_form, _mem_form)

// For instructions that only use memory mode
#define GEN_test_M_AllMaskModes(_name, _mem_form) \
   GEN_test_Monly(nomask_##_name, _mem_form)\
   GEN_test_Monly(merge_##_name, _mem_form"%{%%k5%}")\
   GEN_test_Monly(zero_##_name, _mem_form"%{%%k5%}%{z%}")
// For instructions that only use register mode
#define GEN_test_R_AllMaskModes(_name, _reg_form) \
   GEN_test_Ronly(nomask_##_name, _reg_form)\
   GEN_test_Ronly(merge_##_name, _reg_form"%{%%k5%}")\
   GEN_test_Ronly(zero_##_name, _reg_form"%{%%k5%}%{z%}")

// For instructions with possible memory destination and zero-mask,
// do not generate zero-masking into memory
#define GEN_test_MemMaskModes(_name, _reg_form, _mem_form) \
   GEN_test_RandM(nomask_##_name, _reg_form,_mem_form)\
   GEN_test_MergeMask(_name, _reg_form, _mem_form)\
   GEN_test_Ronly(zero_##_name, _reg_form"%{%%k5%}%{z%}")

// For instructions that do not allow zero-masking or use zero-masking by default
#define GEN_test_OneMaskModeOnly(_name, _reg_form, _mem_form) \
   GEN_test_RandM(nomask_##_name, _reg_form,_mem_form)\
   GEN_test_MergeMask(_name, _reg_form, _mem_form)\
   GEN_test_empty(zero_##_name)


/* move */
GEN_test_AllMaskModes(VMOVDQU32_6F_512,
      "vmovdqu32 %%zmm7, %%zmm24",
      "vmovdqu32 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VMOVDQA32_6F_512,
      "vmovdqa32 %%zmm7, %%zmm24",
      "vmovdqa32 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VMOVDQA32_6F_256,
      "vmovdqa32 %%ymm7, %%ymm24",
      "vmovdqa32 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VMOVDQA32_6F_128,
      "vmovdqa32 %%xmm7, %%xmm24",
      "vmovdqa32 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VMOVDQU64_6F_512,
      "vmovdqu64 %%zmm7, %%zmm24",
      "vmovdqu64 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VMOVDQA64_6F_512,
      "vmovdqa64 %%zmm7, %%zmm24",
      "vmovdqa64 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VMOVDQA64_6F_256,
      "vmovdqa64 %%ymm7, %%ymm24",
      "vmovdqa64 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VMOVDQA64_6F_128,
      "vmovdqa64 %%xmm7, %%xmm24",
      "vmovdqa64 0x40(%%rax), %%xmm24")
GEN_test_RandM(VMOVQ_6E_128,
      "vmovq %%r14, %%xmm24",
      "vmovq 0x40(%%rax), %%xmm24")
GEN_test_RandM(VMOVD_6E_128,
      "vmovd %%r14d, %%xmm24",
      "vmovd 0x40(%%rax), %%xmm24")
GEN_test_Monly(VMOVNTDQ_E7_512,
      "vmovntdq %%zmm24, 0x40(%%rax)")
GEN_test_Monly(VMOVNTDQ_E7_256,
      "vmovntdq %%ymm24, 0x40(%%rax)")
GEN_test_Monly(VMOVNTDQ_E7_128,
      "vmovntdq %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVDQU32_7F_512,
      "vmovdqu32 %%zmm24, %%zmm7",
      "vmovdqu32 %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVDQA32_7F_512,
      "vmovdqa32 %%zmm24, %%zmm7",
      "vmovdqa32 %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVDQA32_7F_256,
      "vmovdqa32 %%ymm24, %%ymm7",
      "vmovdqa32 %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVDQA32_7F_128,
      "vmovdqa32 %%xmm24, %%xmm7",
      "vmovdqa32 %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVDQU64_7F_512,
      "vmovdqu64 %%zmm24, %%zmm7",
      "vmovdqu64 %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVDQA64_7F_512,
      "vmovdqa64 %%zmm24, %%zmm7",
      "vmovdqa64 %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVDQA64_7F_256,
      "vmovdqa64 %%ymm24, %%ymm7",
      "vmovdqa64 %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVDQA64_7F_128,
      "vmovdqa64 %%xmm24, %%xmm7",
      "vmovdqa64 %%xmm24, 0x40(%%rax)")
GEN_test_RandM(VMOVQ_7E_128,
      "vmovq %%xmm7, %%xmm24",
      "vmovq 0x40(%%rax), %%xmm24")
GEN_test_RandM(VMOVD_7E_128,
      "vmovd %%xmm24, %%r14d",
      "vmovd %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VMOVSHDUP_16_512,
      "vmovshdup %%zmm24, %%zmm8",
      "vmovshdup 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VMOVSLDUP_12_512,
      "vmovsldup %%zmm24, %%zmm8",
      "vmovsldup 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPMOVSXBQ_22_512,
      "vpmovsxbq %%xmm8, %%zmm24",
      "vpmovsxbq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPMOVZXBQ_32_512,
      "vpmovzxbq %%xmm8, %%zmm24",
      "vpmovzxbq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPMOVSXWD_23_512,
      "vpmovsxwd %%ymm8, %%zmm24",
      "vpmovsxwd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPMOVZXWD_33_512,
      "vpmovzxwd %%ymm8, %%zmm24",
      "vpmovzxwd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPMOVSXBD_21_512,
      "vpmovsxbd %%xmm8, %%zmm24",
      "vpmovsxbd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPMOVZXBD_31_512,
      "vpmovzxbd %%xmm8, %%zmm24",
      "vpmovzxbd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPMOVSXWQ_24_512,
      "vpmovsxwq %%xmm8, %%zmm24",
      "vpmovsxwq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPMOVZXWQ_34_512,
      "vpmovzxwq %%xmm8, %%zmm24",
      "vpmovzxwq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPMOVSXDQ_25_512,
      "vpmovsxdq %%ymm8, %%zmm24",
      "vpmovsxdq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPMOVZXDQ_35_512,
      "vpmovzxdq %%ymm8, %%zmm24",
      "vpmovzxdq 0x40(%%rax), %%zmm24")
GEN_test_RandM(VMOVQ_D6_128,
      "vmovq %%xmm24, %%xmm7",
      "vmovq %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVWB_30_512,
      "vpmovwb %%zmm24, %%ymm7",
      "vpmovwb %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVWB_30_256,
      "vpmovwb %%ymm24, %%xmm7",
      "vpmovwb %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVWB_30_128,
      "vpmovwb %%xmm24, %%xmm7",
      "vpmovwb %%xmm24, 0x40(%%rax)")
GEN_test_Ronly(VPMOVW2M_29_512,
      "vpmovw2m %%zmm7, %%k5")
GEN_test_Ronly(VPMOVW2M_29_256,
      "vpmovw2m %%ymm7, %%k5")
GEN_test_Ronly(VPMOVW2M_29_128,
      "vpmovw2m %%xmm7, %%k5")
GEN_test_MemMaskModes(VPMOVSWB_20_512,
      "vpmovswb %%zmm24, %%ymm7",
      "vpmovswb %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVSWB_20_256,
      "vpmovswb %%ymm24, %%xmm7",
      "vpmovswb %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVSWB_20_128,
      "vpmovswb %%xmm24, %%xmm7",
      "vpmovswb %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVUSWB_10_512,
      "vpmovuswb %%zmm24, %%ymm7",
      "vpmovuswb %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVUSWB_10_256,
      "vpmovuswb %%ymm24, %%xmm7",
      "vpmovuswb %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVUSWB_10_128,
      "vpmovuswb %%xmm24, %%xmm7",
      "vpmovuswb %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVQD_35_512,
      "vpmovqd %%zmm24, %%ymm7",
      "vpmovqd %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVQD_35_256,
      "vpmovqd %%ymm24, %%xmm7",
      "vpmovqd %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVQD_35_128,
      "vpmovqd %%xmm24, %%xmm7",
      "vpmovqd %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VPMOVZXDQ_35_256,
      "vpmovzxdq %%xmm7, %%ymm24",
      "vpmovzxdq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPMOVZXDQ_35_128,
      "vpmovzxdq %%xmm7, %%xmm24",
      "vpmovzxdq 0x40(%%rax), %%xmm24")
GEN_test_MemMaskModes(VPMOVQW_34_512,
      "vpmovqw %%zmm24, %%xmm7",
      "vpmovqw %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVQW_34_256,
      "vpmovqw %%ymm24, %%xmm7",
      "vpmovqw %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVQW_34_128,
      "vpmovqw %%xmm24, %%xmm7",
      "vpmovqw %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VPMOVZXWQ_34_256,
      "vpmovzxwq %%xmm7, %%ymm24",
      "vpmovzxwq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPMOVZXWQ_34_128,
      "vpmovzxwq %%xmm7, %%xmm24",
      "vpmovzxwq 0x40(%%rax), %%xmm24")
GEN_test_MemMaskModes(VPMOVDW_33_512,
      "vpmovdw %%zmm24, %%ymm7",
      "vpmovdw %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVDW_33_256,
      "vpmovdw %%ymm24, %%xmm7",
      "vpmovdw %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVDW_33_128,
      "vpmovdw %%xmm24, %%xmm7",
      "vpmovdw %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VPMOVZXWD_33_256,
      "vpmovzxwd %%xmm7, %%ymm24",
      "vpmovzxwd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPMOVZXWD_33_128,
      "vpmovzxwd %%xmm7, %%xmm24",
      "vpmovzxwd 0x40(%%rax), %%xmm24")
GEN_test_MemMaskModes(VPMOVQB_32_512,
      "vpmovqb %%zmm24, %%xmm7",
      "vpmovqb %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVQB_32_256,
      "vpmovqb %%ymm24, %%xmm7",
      "vpmovqb %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVQB_32_128,
      "vpmovqb %%xmm24, %%xmm7",
      "vpmovqb %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VPMOVZXBQ_32_256,
      "vpmovzxbq %%xmm7, %%ymm24",
      "vpmovzxbq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPMOVZXBQ_32_128,
      "vpmovzxbq %%xmm7, %%xmm24",
      "vpmovzxbq 0x40(%%rax), %%xmm24")
GEN_test_MemMaskModes(VPMOVDB_31_512,
      "vpmovdb %%zmm24, %%xmm7",
      "vpmovdb %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVDB_31_256,
      "vpmovdb %%ymm24, %%xmm7",
      "vpmovdb %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVDB_31_128,
      "vpmovdb %%xmm24, %%xmm7",
      "vpmovdb %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VPMOVZXBD_31_256,
      "vpmovzxbd %%xmm7, %%ymm24",
      "vpmovzxbd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPMOVZXBD_31_128,
      "vpmovzxbd %%xmm7, %%xmm24",
      "vpmovzxbd 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VMOVAPS_28_512,
      "vmovaps %%zmm7, %%zmm24",
      "vmovaps 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VMOVAPD_28_512,
      "vmovapd %%zmm7, %%zmm24",
      "vmovapd 0x40(%%rax), %%zmm24")
GEN_test_MemMaskModes(VPMOVSQD_25_512,
      "vpmovsqd %%zmm24, %%ymm7",
      "vpmovsqd %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVSQD_25_256,
      "vpmovsqd %%ymm24, %%xmm7",
      "vpmovsqd %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVSQD_25_128,
      "vpmovsqd %%xmm24, %%xmm7",
      "vpmovsqd %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VPMOVSXDQ_25_256,
      "vpmovsxdq %%xmm7, %%ymm24",
      "vpmovsxdq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPMOVSXDQ_25_128,
      "vpmovsxdq %%xmm7, %%xmm24",
      "vpmovsxdq 0x40(%%rax), %%xmm24")
GEN_test_MemMaskModes(VPMOVSQW_24_512,
      "vpmovsqw %%zmm24, %%xmm7",
      "vpmovsqw %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVSQW_24_256,
      "vpmovsqw %%ymm24, %%xmm7",
      "vpmovsqw %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVSQW_24_128,
      "vpmovsqw %%xmm24, %%xmm7",
      "vpmovsqw %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VPMOVSXWQ_24_256,
      "vpmovsxwq %%xmm7, %%ymm24",
      "vpmovsxwq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPMOVSXWQ_24_128,
      "vpmovsxwq %%xmm7, %%xmm24",
      "vpmovsxwq 0x40(%%rax), %%xmm24")
GEN_test_MemMaskModes(VPMOVSDW_23_512,
      "vpmovsdw %%zmm24, %%ymm7",
      "vpmovsdw %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVSDW_23_256,
      "vpmovsdw %%ymm24, %%xmm7",
      "vpmovsdw %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVSDW_23_128,
      "vpmovsdw %%xmm24, %%xmm7",
      "vpmovsdw %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VPMOVSXWD_23_256,
      "vpmovsxwd %%xmm7, %%ymm24",
      "vpmovsxwd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPMOVSXWD_23_128,
      "vpmovsxwd %%xmm7, %%xmm24",
      "vpmovsxwd 0x40(%%rax), %%xmm24")
GEN_test_MemMaskModes(VPMOVSQB_22_512,
      "vpmovsqb %%zmm24, %%xmm7",
      "vpmovsqb %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVSQB_22_256,
      "vpmovsqb %%ymm24, %%xmm7",
      "vpmovsqb %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVSQB_22_128,
      "vpmovsqb %%xmm24, %%xmm7",
      "vpmovsqb %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VPMOVSXBQ_22_256,
      "vpmovsxbq %%xmm7, %%ymm24",
      "vpmovsxbq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPMOVSXBQ_22_128,
      "vpmovsxbq %%xmm7, %%xmm24",
      "vpmovsxbq 0x40(%%rax), %%xmm24")
GEN_test_MemMaskModes(VPMOVSDB_21_512,
      "vpmovsdb %%zmm24, %%xmm7",
      "vpmovsdb %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVSDB_21_256,
      "vpmovsdb %%ymm24, %%xmm7",
      "vpmovsdb %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVSDB_21_128,
      "vpmovsdb %%xmm24, %%xmm7",
      "vpmovsdb %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VPMOVSXBD_21_256,
      "vpmovsxbd %%xmm7, %%ymm24",
      "vpmovsxbd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPMOVSXBD_21_128,
      "vpmovsxbd %%xmm7, %%xmm24",
      "vpmovsxbd 0x40(%%rax), %%xmm24")
GEN_test_Monly(VMOVNTPS_2B_512,
      "vmovntps %%zmm24, 0x40(%%rax)")
GEN_test_Monly(VMOVNTPS_2B_256,
      "vmovntps %%ymm24, 0x40(%%rax)")
GEN_test_Monly(VMOVNTPS_2B_128,
      "vmovntps %%xmm24, 0x40(%%rax)")
GEN_test_Monly(VMOVNTPD_2B_512,
      "vmovntpd %%zmm24, 0x40(%%rax)")
GEN_test_Monly(VMOVNTDQA_2A_512,
      "vmovntdqa 0x40(%%rax), %%zmm24")
GEN_test_Monly(VMOVNTDQA_2A_256,
      "vmovntdqa 0x40(%%rax), %%ymm24")
GEN_test_Monly(VMOVNTDQA_2A_128,
      "vmovntdqa 0x40(%%rax), %%xmm24")
GEN_test_MemMaskModes(VMOVAPS_29_512,
      "vmovaps %%zmm24, %%zmm7",
      "vmovaps %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVAPD_29_512,
      "vmovapd %%zmm24, %%zmm7",
      "vmovapd %%zmm24, 0x40(%%rax)")
GEN_test_Monly(VMOVHPS_17_128,
      "vmovhps %%xmm24, 0x40(%%rax)")
GEN_test_Monly(VMOVHPD_17_128,
      "vmovhpd %%xmm24, 0x40(%%rax)")
GEN_test_Monly(VMOVHPS_16_128,
      "vmovhps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_Monly(VMOVHPD_16_128,
      "vmovhpd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VMOVSHDUP_16_256,
      "vmovshdup %%ymm7, %%ymm24",
      "vmovshdup 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VMOVSHDUP_16_128,
      "vmovshdup %%xmm7, %%xmm24",
      "vmovshdup 0x40(%%rax), %%xmm24")
GEN_test_Ronly(VMOVLHPS_16_128,
      "vmovlhps %%xmm7, %%xmm8, %%xmm24")
GEN_test_MemMaskModes(VPMOVUSQD_15_512,
      "vpmovusqd %%zmm24, %%ymm7",
      "vpmovusqd %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVUSQD_15_256,
      "vpmovusqd %%ymm24, %%xmm7",
      "vpmovusqd %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVUSQD_15_128,
      "vpmovusqd %%xmm24, %%xmm7",
      "vpmovusqd %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVUSQW_14_512,
      "vpmovusqw %%zmm24, %%xmm7",
      "vpmovusqw %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVUSQW_14_256,
      "vpmovusqw %%ymm24, %%xmm7",
      "vpmovusqw %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVUSQW_14_128,
      "vpmovusqw %%xmm24, %%xmm7",
      "vpmovusqw %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVUSDW_13_512,
      "vpmovusdw %%zmm24, %%ymm7",
      "vpmovusdw %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVUSDW_13_256,
      "vpmovusdw %%ymm24, %%xmm7",
      "vpmovusdw %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVUSDW_13_128,
      "vpmovusdw %%xmm24, %%xmm7",
      "vpmovusdw %%xmm24, 0x40(%%rax)")
GEN_test_Monly(VMOVLPS_13_128,
      "vmovlps %%xmm24, 0x40(%%rax)")
GEN_test_Monly(VMOVLPD_13_128,
      "vmovlpd %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVUSQB_12_512,
      "vpmovusqb %%zmm24, %%xmm7",
      "vpmovusqb %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVUSQB_12_256,
      "vpmovusqb %%ymm24, %%xmm7",
      "vpmovusqb %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVUSQB_12_128,
      "vpmovusqb %%xmm24, %%xmm7",
      "vpmovusqb %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VMOVSLDUP_12_256,
      "vmovsldup %%ymm7, %%ymm24",
      "vmovsldup 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VMOVSLDUP_12_128,
      "vmovsldup %%xmm7, %%xmm24",
      "vmovsldup 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VMOVDDUP_12_512,
      "vmovddup %%zmm7, %%zmm24",
      "vmovddup 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VMOVDDUP_12_256,
      "vmovddup %%ymm7, %%ymm24",
      "vmovddup 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VMOVDDUP_12_128,
      "vmovddup %%xmm7, %%xmm24",
      "vmovddup 0x40(%%rax), %%xmm24")
GEN_test_Monly(VMOVLPS_12_128,
      "vmovlps %%xmm24, 0x40(%%rax)")
GEN_test_Ronly(VMOVHLPS_12_128,
      "vmovhlps %%xmm7, %%xmm8, %%xmm24")
GEN_test_Monly(VMOVLPD_12_128,
      "vmovlpd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_MemMaskModes(VPMOVUSDB_11_512,
      "vpmovusdb %%zmm24, %%xmm7",
      "vpmovusdb %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVUSDB_11_256,
      "vpmovusdb %%ymm24, %%xmm7",
      "vpmovusdb %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPMOVUSDB_11_128,
      "vpmovusdb %%xmm24, %%xmm7",
      "vpmovusdb %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVSS_11_128,
      "vmovss %%xmm24, %%xmm8, %%xmm7",
      "vmovss %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVSD_11_128,
      "vmovsd %%xmm24, %%xmm8, %%xmm7",
      "vmovsd %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVUPS_11_512,
      "vmovups %%zmm24, %%zmm7",
      "vmovups %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVUPS_11_256,
      "vmovups %%ymm24, %%ymm7",
      "vmovups %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVUPS_11_128,
      "vmovups %%xmm24, %%xmm7",
      "vmovups %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVUPD_11_512,
      "vmovupd %%zmm24, %%zmm7",
      "vmovupd %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVUPD_11_256,
      "vmovupd %%ymm24, %%ymm7",
      "vmovupd %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVUPD_11_128,
      "vmovupd %%xmm24, %%xmm7",
      "vmovupd %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VMOVSS_10_128,
      "vmovss %%xmm7, %%xmm8, %%xmm24",
      "vmovss 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VMOVSD_10_128,
      "vmovsd %%xmm7, %%xmm8, %%xmm24",
      "vmovsd 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VMOVUPS_10_512,
      "vmovups %%zmm7, %%zmm24",
      "vmovups 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VMOVUPS_10_256,
      "vmovups %%ymm7, %%ymm24",
      "vmovups 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VMOVUPS_10_128,
      "vmovups %%xmm7, %%xmm24",
      "vmovups 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VMOVUPD_10_512,
      "vmovupd %%zmm7, %%zmm24",
      "vmovupd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VMOVUPD_10_256,
      "vmovupd %%ymm7, %%ymm24",
      "vmovupd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VMOVUPD_10_128,
      "vmovupd %%xmm7, %%xmm24",
      "vmovupd 0x40(%%rax), %%xmm24")
GEN_test_Monly(VMOVNTPD_2B_256,
      "vmovntpd %%ymm24, 0x40(%%rax)")
GEN_test_Monly(VMOVNTPD_2B_128,
      "vmovntpd %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVAPS_29_256,
      "vmovaps %%ymm24, %%ymm7",
      "vmovaps %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVAPS_29_128,
      "vmovaps %%xmm24, %%xmm7",
      "vmovaps %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVAPD_29_256,
      "vmovapd %%ymm24, %%ymm7",
      "vmovapd %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVAPD_29_128,
      "vmovapd %%xmm24, %%xmm7",
      "vmovapd %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VMOVAPS_28_256,
      "vmovaps %%ymm7, %%ymm24",
      "vmovaps 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VMOVAPS_28_128,
      "vmovaps %%xmm7, %%xmm24",
      "vmovaps 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VMOVAPD_28_256,
      "vmovapd %%ymm7, %%ymm24",
      "vmovapd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VMOVAPD_28_128,
      "vmovapd %%xmm7, %%xmm24",
      "vmovapd 0x40(%%rax), %%xmm24")
GEN_test_MemMaskModes(VMOVDQU64_7F_256,
      "vmovdqu64 %%ymm24, %%ymm7",
      "vmovdqu64 %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVDQU64_7F_128,
      "vmovdqu64 %%xmm24, %%xmm7",
      "vmovdqu64 %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVDQU32_7F_256,
      "vmovdqu32 %%ymm24, %%ymm7",
      "vmovdqu32 %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVDQU32_7F_128,
      "vmovdqu32 %%xmm24, %%xmm7",
      "vmovdqu32 %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VMOVDQU64_6F_256,
      "vmovdqu64 %%ymm7, %%ymm24",
      "vmovdqu64 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VMOVDQU64_6F_128,
      "vmovdqu64 %%xmm7, %%xmm24",
      "vmovdqu64 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VMOVDQU32_6F_256,
      "vmovdqu32 %%ymm7, %%ymm24",
      "vmovdqu32 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VMOVDQU32_6F_128,
      "vmovdqu32 %%xmm7, %%xmm24",
      "vmovdqu32 0x40(%%rax), %%xmm24")
GEN_test_Ronly(VPMOVQ2M_39_512,
      "vpmovq2m %%zmm7, %%k5")
GEN_test_Ronly(VPMOVQ2M_39_256,
      "vpmovq2m %%ymm7, %%k5")
GEN_test_Ronly(VPMOVQ2M_39_128,
      "vpmovq2m %%xmm7, %%k5")
GEN_test_Ronly(VPMOVD2M_39_512,
      "vpmovd2m %%zmm7, %%k5")
GEN_test_Ronly(VPMOVD2M_39_256,
      "vpmovd2m %%ymm7, %%k5")
GEN_test_Ronly(VPMOVD2M_39_128,
      "vpmovd2m %%xmm7, %%k5")
GEN_test_Ronly(VPMOVB2M_29_512,
      "vpmovb2m %%zmm7, %%k5")
GEN_test_Ronly(VPMOVB2M_29_256,
      "vpmovb2m %%ymm7, %%k5")
GEN_test_Ronly(VPMOVB2M_29_128,
      "vpmovb2m %%xmm7, %%k5")
GEN_test_AllMaskModes(VMOVDQU16_6F_512,
      "vmovdqu16 %%zmm7, %%zmm24",
      "vmovdqu16 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VMOVDQU16_6F_256,
      "vmovdqu16 %%ymm7, %%ymm24",
      "vmovdqu16 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VMOVDQU16_6F_128,
      "vmovdqu16 %%xmm7, %%xmm24",
      "vmovdqu16 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VMOVDQU8_6F_512,
      "vmovdqu8 %%zmm7, %%zmm24",
      "vmovdqu8 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VMOVDQU8_6F_256,
      "vmovdqu8 %%ymm7, %%ymm24",
      "vmovdqu8 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VMOVDQU8_6F_128,
      "vmovdqu8 %%xmm7, %%xmm24",
      "vmovdqu8 0x40(%%rax), %%xmm24")
GEN_test_Ronly(VPMOVM2Q_38_512,
      "vpmovm2q %%k4, %%zmm24")
GEN_test_Ronly(VPMOVM2Q_38_256,
      "vpmovm2q %%k4, %%ymm24")
GEN_test_Ronly(VPMOVM2Q_38_128,
      "vpmovm2q %%k4, %%xmm24")
GEN_test_Ronly(VPMOVM2D_38_512,
      "vpmovm2d %%k4, %%zmm24")
GEN_test_Ronly(VPMOVM2D_38_256,
      "vpmovm2d %%k4, %%ymm24")
GEN_test_Ronly(VPMOVM2D_38_128,
      "vpmovm2d %%k4, %%xmm24")
GEN_test_AllMaskModes(VPMOVZXBW_30_512,
      "vpmovzxbw %%ymm7, %%zmm24",
      "vpmovzxbw 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPMOVZXBW_30_256,
      "vpmovzxbw %%xmm7, %%ymm24",
      "vpmovzxbw 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPMOVZXBW_30_128,
      "vpmovzxbw %%xmm7, %%xmm24",
      "vpmovzxbw 0x40(%%rax), %%xmm24")
GEN_test_Ronly(VPMOVM2W_28_512,
      "vpmovm2w %%k4, %%zmm24")
GEN_test_Ronly(VPMOVM2W_28_256,
      "vpmovm2w %%k4, %%ymm24")
GEN_test_Ronly(VPMOVM2W_28_128,
      "vpmovm2w %%k4, %%xmm24")
GEN_test_Ronly(VPMOVM2B_28_512,
      "vpmovm2b %%k4, %%zmm24")
GEN_test_Ronly(VPMOVM2B_28_256,
      "vpmovm2b %%k4, %%ymm24")
GEN_test_Ronly(VPMOVM2B_28_128,
      "vpmovm2b %%k4, %%xmm24")
GEN_test_AllMaskModes(VPMOVSXBW_20_512,
      "vpmovsxbw %%ymm7, %%zmm24",
      "vpmovsxbw 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPMOVSXBW_20_256,
      "vpmovsxbw %%xmm7, %%ymm24",
      "vpmovsxbw 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPMOVSXBW_20_128,
      "vpmovsxbw %%xmm7, %%xmm24",
      "vpmovsxbw 0x40(%%rax), %%xmm24")

/* Arithmetic instructions */

GEN_test_AllMaskModes(VPADDD_FE_512,
      "vpaddd %%zmm7, %%zmm8, %%zmm24",
      "vpaddd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPADDD_FE_256,
      "vpaddd %%ymm7, %%ymm8, %%ymm24",
      "vpaddd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPADDD_FE_128,
      "vpaddd %%xmm7, %%xmm8, %%xmm24",
      "vpaddd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VADDSS_58_128,
      "vaddss %%xmm7, %%xmm8, %%xmm24",
      "vaddss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VADDSD_58_128,
      "vaddsd %%xmm7, %%xmm8, %%xmm24",
      "vaddsd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VADDPS_58_512,
      "vaddps %%zmm7, %%zmm8, %%zmm24",
      "vaddps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VADDPD_58_512,
      "vaddpd %%zmm7, %%zmm8, %%zmm24",
      "vaddpd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPADDQ_D4_512,
      "vpaddq %%zmm7, %%zmm8, %%zmm24",
      "vpaddq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSUBQ_FB_512,
      "vpsubq %%zmm7, %%zmm8, %%zmm24",
      "vpsubq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSUBD_FA_512,
      "vpsubd %%zmm7, %%zmm8, %%zmm24",
      "vpsubd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSUBD_FA_256,
      "vpsubd %%ymm7, %%ymm8, %%ymm24",
      "vpsubd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSUBD_FA_128,
      "vpsubd %%xmm7, %%xmm8, %%xmm24",
      "vpsubd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMULUDQ_F4_512,
      "vpmuludq %%zmm7, %%zmm8, %%zmm24",
      "vpmuludq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMULUDQ_F4_256,
      "vpmuludq %%ymm7, %%ymm8, %%ymm24",
      "vpmuludq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMULUDQ_F4_128,
      "vpmuludq %%xmm7, %%xmm8, %%xmm24",
      "vpmuludq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VMAXSS_5F_128,
      "vmaxss %%xmm7, %%xmm8, %%xmm24",
      "vmaxss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VMAXSD_5F_128,
      "vmaxsd %%xmm7, %%xmm8, %%xmm24",
      "vmaxsd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VMAXPS_5F_512,
      "vmaxps %%zmm7, %%zmm8, %%zmm24",
      "vmaxps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VMAXPD_5F_512,
      "vmaxpd %%zmm7, %%zmm8, %%zmm24",
      "vmaxpd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VDIVSS_5E_128,
      "vdivss %%xmm7, %%xmm8, %%xmm24",
      "vdivss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VDIVSD_5E_128,
      "vdivsd %%xmm7, %%xmm8, %%xmm24",
      "vdivsd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VDIVPS_5E_512,
      "vdivps %%zmm7, %%zmm8, %%zmm24",
      "vdivps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VDIVPD_5E_512,
      "vdivpd %%zmm7, %%zmm8, %%zmm24",
      "vdivpd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VMINSS_5D_128,
      "vminss %%xmm7, %%xmm8, %%xmm24",
      "vminss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VMINSD_5D_128,
      "vminsd %%xmm7, %%xmm8, %%xmm24",
      "vminsd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VMINPS_5D_512,
      "vminps %%zmm7, %%zmm8, %%zmm24",
      "vminps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VMINPD_5D_512,
      "vminpd %%zmm7, %%zmm8, %%zmm24",
      "vminpd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VSUBSS_5C_128,
      "vsubss %%xmm7, %%xmm8, %%xmm24",
      "vsubss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VSUBSD_5C_128,
      "vsubsd %%xmm7, %%xmm8, %%xmm24",
      "vsubsd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VSUBPS_5C_512,
      "vsubps %%zmm7, %%zmm8, %%zmm24",
      "vsubps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VSUBPD_5C_512,
      "vsubpd %%zmm7, %%zmm8, %%zmm24",
      "vsubpd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VMULSS_59_128,
      "vmulss %%xmm7, %%xmm8, %%xmm24",
      "vmulss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VMULSD_59_128,
      "vmulsd %%xmm7, %%xmm8, %%xmm24",
      "vmulsd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VMULPS_59_512,
      "vmulps %%zmm7, %%zmm8, %%zmm24",
      "vmulps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VMULPD_59_512,
      "vmulpd %%zmm7, %%zmm8, %%zmm24",
      "vmulpd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMULDQ_28_512,
      "vpmuldq %%zmm7, %%zmm8, %%zmm24",
      "vpmuldq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMULDQ_28_256,
      "vpmuldq %%ymm7, %%ymm8, %%ymm24",
      "vpmuldq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMULDQ_28_128,
      "vpmuldq %%xmm7, %%xmm8, %%xmm24",
      "vpmuldq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VSQRTSS_51_128,
      "vsqrtss %%xmm7, %%xmm8, %%xmm24",
      "vsqrtss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VSQRTSD_51_128,
      "vsqrtsd %%xmm7, %%xmm8, %%xmm24",
      "vsqrtsd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VSQRTPS_51_512,
      "vsqrtps %%zmm7, %%zmm24",
      "vsqrtps 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VSQRTPD_51_512,
      "vsqrtpd %%zmm7, %%zmm24",
      "vsqrtpd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPMULLD_40_512,
      "vpmulld %%zmm7, %%zmm8, %%zmm24",
      "vpmulld 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMULLD_40_256,
      "vpmulld %%ymm7, %%ymm8, %%ymm24",
      "vpmulld 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMULLD_40_128,
      "vpmulld %%xmm7, %%xmm8, %%xmm24",
      "vpmulld 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMAXUQ_3F_512,
      "vpmaxuq %%zmm7, %%zmm8, %%zmm24",
      "vpmaxuq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMAXUQ_3F_256,
      "vpmaxuq %%ymm7, %%ymm8, %%ymm24",
      "vpmaxuq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMAXUQ_3F_128,
      "vpmaxuq %%xmm7, %%xmm8, %%xmm24",
      "vpmaxuq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMAXUD_3F_512,
      "vpmaxud %%zmm7, %%zmm8, %%zmm24",
      "vpmaxud 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMAXUD_3F_256,
      "vpmaxud %%ymm7, %%ymm8, %%ymm24",
      "vpmaxud 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMAXUD_3F_128,
      "vpmaxud %%xmm7, %%xmm8, %%xmm24",
      "vpmaxud 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMAXSQ_3D_512,
      "vpmaxsq %%zmm7, %%zmm8, %%zmm24",
      "vpmaxsq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMAXSQ_3D_256,
      "vpmaxsq %%ymm7, %%ymm8, %%ymm24",
      "vpmaxsq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMAXSQ_3D_128,
      "vpmaxsq %%xmm7, %%xmm8, %%xmm24",
      "vpmaxsq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMAXSD_3D_512,
      "vpmaxsd %%zmm7, %%zmm8, %%zmm24",
      "vpmaxsd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMAXSD_3D_256,
      "vpmaxsd %%ymm7, %%ymm8, %%ymm24",
      "vpmaxsd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMAXSD_3D_128,
      "vpmaxsd %%xmm7, %%xmm8, %%xmm24",
      "vpmaxsd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMINUQ_3B_512,
      "vpminuq %%zmm7, %%zmm8, %%zmm24",
      "vpminuq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMINUQ_3B_256,
      "vpminuq %%ymm7, %%ymm8, %%ymm24",
      "vpminuq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMINUQ_3B_128,
      "vpminuq %%xmm7, %%xmm8, %%xmm24",
      "vpminuq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMINUD_3B_512,
      "vpminud %%zmm7, %%zmm8, %%zmm24",
      "vpminud 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMINUD_3B_256,
      "vpminud %%ymm7, %%ymm8, %%ymm24",
      "vpminud 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMINUD_3B_128,
      "vpminud %%xmm7, %%xmm8, %%xmm24",
      "vpminud 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPADDQ_D4_256,
      "vpaddq %%ymm7, %%ymm8, %%ymm24",
      "vpaddq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPADDQ_D4_128,
      "vpaddq %%xmm7, %%xmm8, %%xmm24",
      "vpaddq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VMULPD_59_256,
      "vmulpd %%ymm7, %%ymm8, %%ymm24",
      "vmulpd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VMULPD_59_128,
      "vmulpd %%xmm7, %%xmm8, %%xmm24",
      "vmulpd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VMULPS_59_256,
      "vmulps %%ymm7, %%ymm8, %%ymm24",
      "vmulps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VMULPS_59_128,
      "vmulps %%xmm7, %%xmm8, %%xmm24",
      "vmulps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VADDPS_58_256,
      "vaddps %%ymm7, %%ymm8, %%ymm24",
      "vaddps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VADDPS_58_128,
      "vaddps %%xmm7, %%xmm8, %%xmm24",
      "vaddps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VADDPD_58_256,
      "vaddpd %%ymm7, %%ymm8, %%ymm24",
      "vaddpd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VADDPD_58_128,
      "vaddpd %%xmm7, %%xmm8, %%xmm24",
      "vaddpd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPADDW_FD_512,
      "vpaddw %%zmm7, %%zmm8, %%zmm24",
      "vpaddw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPADDW_FD_256,
      "vpaddw %%ymm7, %%ymm8, %%ymm24",
      "vpaddw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPADDW_FD_128,
      "vpaddw %%xmm7, %%xmm8, %%xmm24",
      "vpaddw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPADDB_FC_512,
      "vpaddb %%zmm7, %%zmm8, %%zmm24",
      "vpaddb 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPADDB_FC_256,
      "vpaddb %%ymm7, %%ymm8, %%ymm24",
      "vpaddb 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPADDB_FC_128,
      "vpaddb %%xmm7, %%xmm8, %%xmm24",
      "vpaddb 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMAXSW_EE_512,
      "vpmaxsw %%zmm7, %%zmm8, %%zmm24",
      "vpmaxsw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMAXSW_EE_256,
      "vpmaxsw %%ymm7, %%ymm8, %%ymm24",
      "vpmaxsw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMAXSW_EE_128,
      "vpmaxsw %%xmm7, %%xmm8, %%xmm24",
      "vpmaxsw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPADDSW_ED_512,
      "vpaddsw %%zmm7, %%zmm8, %%zmm24",
      "vpaddsw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPADDSW_ED_256,
      "vpaddsw %%ymm7, %%ymm8, %%ymm24",
      "vpaddsw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPADDSW_ED_128,
      "vpaddsw %%xmm7, %%xmm8, %%xmm24",
      "vpaddsw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPADDSB_EC_512,
      "vpaddsb %%zmm7, %%zmm8, %%zmm24",
      "vpaddsb 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPADDSB_EC_256,
      "vpaddsb %%ymm7, %%ymm8, %%ymm24",
      "vpaddsb 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPADDSB_EC_128,
      "vpaddsb %%xmm7, %%xmm8, %%xmm24",
      "vpaddsb 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMINSW_EA_512,
      "vpminsw %%zmm7, %%zmm8, %%zmm24",
      "vpminsw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMINSW_EA_256,
      "vpminsw %%ymm7, %%ymm8, %%ymm24",
      "vpminsw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMINSW_EA_128,
      "vpminsw %%xmm7, %%xmm8, %%xmm24",
      "vpminsw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMULHW_E5_512,
      "vpmulhw %%zmm7, %%zmm8, %%zmm24",
      "vpmulhw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMULHW_E5_256,
      "vpmulhw %%ymm7, %%ymm8, %%ymm24",
      "vpmulhw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMULHW_E5_128,
      "vpmulhw %%xmm7, %%xmm8, %%xmm24",
      "vpmulhw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMULHUW_E4_512,
      "vpmulhuw %%zmm7, %%zmm8, %%zmm24",
      "vpmulhuw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMULHUW_E4_256,
      "vpmulhuw %%ymm7, %%ymm8, %%ymm24",
      "vpmulhuw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMULHUW_E4_128,
      "vpmulhuw %%xmm7, %%xmm8, %%xmm24",
      "vpmulhuw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPABSQ_1F_512,
      "vpabsq %%zmm7, %%zmm24",
      "vpabsq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPABSQ_1F_256,
      "vpabsq %%ymm7, %%ymm24",
      "vpabsq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPABSQ_1F_128,
      "vpabsq %%xmm7, %%xmm24",
      "vpabsq 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VPABSD_1E_512,
      "vpabsd %%zmm7, %%zmm24",
      "vpabsd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPABSD_1E_256,
      "vpabsd %%ymm7, %%ymm24",
      "vpabsd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPABSD_1E_128,
      "vpabsd %%xmm7, %%xmm24",
      "vpabsd 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VPABSW_1D_512,
      "vpabsw %%zmm7, %%zmm24",
      "vpabsw 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPABSW_1D_256,
      "vpabsw %%ymm7, %%ymm24",
      "vpabsw 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPABSW_1D_128,
      "vpabsw %%xmm7, %%xmm24",
      "vpabsw 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VPABSB_1C_512,
      "vpabsb %%zmm7, %%zmm24",
      "vpabsb 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPABSB_1C_256,
      "vpabsb %%ymm7, %%ymm24",
      "vpabsb 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPABSB_1C_128,
      "vpabsb %%xmm7, %%xmm24",
      "vpabsb 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VPMAXUB_DE_512,
      "vpmaxub %%zmm7, %%zmm8, %%zmm24",
      "vpmaxub 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMAXUB_DE_256,
      "vpmaxub %%ymm7, %%ymm8, %%ymm24",
      "vpmaxub 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMAXUB_DE_128,
      "vpmaxub %%xmm7, %%xmm8, %%xmm24",
      "vpmaxub 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPADDUSB_DC_512,
      "vpaddusb %%zmm7, %%zmm8, %%zmm24",
      "vpaddusb 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPADDUSB_DC_256,
      "vpaddusb %%ymm7, %%ymm8, %%ymm24",
      "vpaddusb 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPADDUSB_DC_128,
      "vpaddusb %%xmm7, %%xmm8, %%xmm24",
      "vpaddusb 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMINUB_DA_512,
      "vpminub %%zmm7, %%zmm8, %%zmm24",
      "vpminub 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMINUB_DA_256,
      "vpminub %%ymm7, %%ymm8, %%ymm24",
      "vpminub 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMINUB_DA_128,
      "vpminub %%xmm7, %%xmm8, %%xmm24",
      "vpminub 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMULLW_D5_512,
      "vpmullw %%zmm7, %%zmm8, %%zmm24",
      "vpmullw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMULLW_D5_256,
      "vpmullw %%ymm7, %%ymm8, %%ymm24",
      "vpmullw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMULLW_D5_128,
      "vpmullw %%xmm7, %%xmm8, %%xmm24",
      "vpmullw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMAXUW_3E_512,
      "vpmaxuw %%zmm7, %%zmm8, %%zmm24",
      "vpmaxuw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMAXUW_3E_256,
      "vpmaxuw %%ymm7, %%ymm8, %%ymm24",
      "vpmaxuw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMAXUW_3E_128,
      "vpmaxuw %%xmm7, %%xmm8, %%xmm24",
      "vpmaxuw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMAXSB_3C_512,
      "vpmaxsb %%zmm7, %%zmm8, %%zmm24",
      "vpmaxsb 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMAXSB_3C_256,
      "vpmaxsb %%ymm7, %%ymm8, %%ymm24",
      "vpmaxsb 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMAXSB_3C_128,
      "vpmaxsb %%xmm7, %%xmm8, %%xmm24",
      "vpmaxsb 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMINUW_3A_512,
      "vpminuw %%zmm7, %%zmm8, %%zmm24",
      "vpminuw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMINUW_3A_256,
      "vpminuw %%ymm7, %%ymm8, %%ymm24",
      "vpminuw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMINUW_3A_128,
      "vpminuw %%xmm7, %%xmm8, %%xmm24",
      "vpminuw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMINSB_38_512,
      "vpminsb %%zmm7, %%zmm8, %%zmm24",
      "vpminsb 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMINSB_38_256,
      "vpminsb %%ymm7, %%ymm8, %%ymm24",
      "vpminsb 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMINSB_38_128,
      "vpminsb %%xmm7, %%xmm8, %%xmm24",
      "vpminsb 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMINSQ_39_512,
      "vpminsq %%zmm7, %%zmm8, %%zmm24",
      "vpminsq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMINSQ_39_256,
      "vpminsq %%ymm7, %%ymm8, %%ymm24",
      "vpminsq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMINSQ_39_128,
      "vpminsq %%xmm7, %%xmm8, %%xmm24",
      "vpminsq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMINSD_39_512,
      "vpminsd %%zmm7, %%zmm8, %%zmm24",
      "vpminsd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMINSD_39_256,
      "vpminsd %%ymm7, %%ymm8, %%ymm24",
      "vpminsd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMINSD_39_128,
      "vpminsd %%xmm7, %%xmm8, %%xmm24",
      "vpminsd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMADDUBSW_04_512,
      "vpmaddubsw %%zmm7, %%zmm8, %%zmm24",
      "vpmaddubsw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMADDUBSW_04_256,
      "vpmaddubsw %%ymm7, %%ymm8, %%ymm24",
      "vpmaddubsw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMADDUBSW_04_128,
      "vpmaddubsw %%xmm7, %%xmm8, %%xmm24",
      "vpmaddubsw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSUBW_F9_512,
      "vpsubw %%zmm7, %%zmm8, %%zmm24",
      "vpsubw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSUBW_F9_256,
      "vpsubw %%ymm7, %%ymm8, %%ymm24",
      "vpsubw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSUBW_F9_128,
      "vpsubw %%xmm7, %%xmm8, %%xmm24",
      "vpsubw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSUBB_F8_512,
      "vpsubb %%zmm7, %%zmm8, %%zmm24",
      "vpsubb 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSUBB_F8_256,
      "vpsubb %%ymm7, %%ymm8, %%ymm24",
      "vpsubb 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSUBB_F8_128,
      "vpsubb %%xmm7, %%xmm8, %%xmm24",
      "vpsubb 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_RandM(VPSADBW_F6_512,
      "vpsadbw %%zmm7, %%zmm8, %%zmm24",
      "vpsadbw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_RandM(VPSADBW_F6_256,
      "vpsadbw %%ymm7, %%ymm8, %%ymm24",
      "vpsadbw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_RandM(VPSADBW_F6_128,
      "vpsadbw %%xmm7, %%xmm8, %%xmm24",
      "vpsadbw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMADDWD_F5_512,
      "vpmaddwd %%zmm7, %%zmm8, %%zmm24",
      "vpmaddwd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMADDWD_F5_256,
      "vpmaddwd %%ymm7, %%ymm8, %%ymm24",
      "vpmaddwd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMADDWD_F5_128,
      "vpmaddwd %%xmm7, %%xmm8, %%xmm24",
      "vpmaddwd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSUBSW_E9_512,
      "vpsubsw %%zmm7, %%zmm8, %%zmm24",
      "vpsubsw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSUBSW_E9_256,
      "vpsubsw %%ymm7, %%ymm8, %%ymm24",
      "vpsubsw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSUBSW_E9_128,
      "vpsubsw %%xmm7, %%xmm8, %%xmm24",
      "vpsubsw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSUBSB_E8_512,
      "vpsubsb %%zmm7, %%zmm8, %%zmm24",
      "vpsubsb 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSUBSB_E8_256,
      "vpsubsb %%ymm7, %%ymm8, %%ymm24",
      "vpsubsb 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSUBSB_E8_128,
      "vpsubsb %%xmm7, %%xmm8, %%xmm24",
      "vpsubsb 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSUBUSW_D9_512,
      "vpsubusw %%zmm7, %%zmm8, %%zmm24",
      "vpsubusw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSUBUSW_D9_256,
      "vpsubusw %%ymm7, %%ymm8, %%ymm24",
      "vpsubusw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSUBUSW_D9_128,
      "vpsubusw %%xmm7, %%xmm8, %%xmm24",
      "vpsubusw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSUBUSB_D8_512,
      "vpsubusb %%zmm7, %%zmm8, %%zmm24",
      "vpsubusb 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSUBUSB_D8_256,
      "vpsubusb %%ymm7, %%ymm8, %%ymm24",
      "vpsubusb 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSUBUSB_D8_128,
      "vpsubusb %%xmm7, %%xmm8, %%xmm24",
      "vpsubusb 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPMULLQ_40_512,
      "vpmullq %%zmm7, %%zmm8, %%zmm24",
      "vpmullq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMULLQ_40_256,
      "vpmullq %%ymm7, %%ymm8, %%ymm24",
      "vpmullq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMULLQ_40_128,
      "vpmullq %%xmm7, %%xmm8, %%xmm24",
      "vpmullq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSUBQ_FB_256,
      "vpsubq %%ymm7, %%ymm8, %%ymm24",
      "vpsubq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSUBQ_FB_128,
      "vpsubq %%xmm7, %%xmm8, %%xmm24",
      "vpsubq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VDIVPD_5E_256,
      "vdivpd %%ymm7, %%ymm8, %%ymm24",
      "vdivpd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VDIVPD_5E_128,
      "vdivpd %%xmm7, %%xmm8, %%xmm24",
      "vdivpd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VDIVPS_5E_256,
      "vdivps %%ymm7, %%ymm8, %%ymm24",
      "vdivps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VDIVPS_5E_128,
      "vdivps %%xmm7, %%xmm8, %%xmm24",
      "vdivps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VSUBPD_5C_256,
      "vsubpd %%ymm7, %%ymm8, %%ymm24",
      "vsubpd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VSUBPD_5C_128,
      "vsubpd %%xmm7, %%xmm8, %%xmm24",
      "vsubpd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VSUBPS_5C_256,
      "vsubps %%ymm7, %%ymm8, %%ymm24",
      "vsubps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VSUBPS_5C_128,
      "vsubps %%xmm7, %%xmm8, %%xmm24",
      "vsubps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VMAXPD_5F_256,
      "vmaxpd %%ymm7, %%ymm8, %%ymm24",
      "vmaxpd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VMAXPD_5F_128,
      "vmaxpd %%xmm7, %%xmm8, %%xmm24",
      "vmaxpd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VMAXPS_5F_256,
      "vmaxps %%ymm7, %%ymm8, %%ymm24",
      "vmaxps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VMAXPS_5F_128,
      "vmaxps %%xmm7, %%xmm8, %%xmm24",
      "vmaxps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VRSQRT28SD_CD_128,
      "vrsqrt28sd %%xmm7, %%xmm8, %%xmm24",
      "vrsqrt28sd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VRSQRT28SS_CD_128,
      "vrsqrt28ss %%xmm7, %%xmm8, %%xmm24",
      "vrsqrt28ss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VRSQRT28PS_CC_512,
      "vrsqrt28ps %%zmm7, %%zmm24",
      "vrsqrt28ps 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VRCP28SD_CB_128,
      "vrcp28sd %%xmm7, %%xmm8, %%xmm24",
      "vrcp28sd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VRCP28SS_CB_128,
      "vrcp28ss %%xmm7, %%xmm8, %%xmm24",
      "vrcp28ss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VRCP28PD_CA_512,
      "vrcp28pd %%zmm7, %%zmm24",
      "vrcp28pd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VRCP28PS_CA_512,
      "vrcp28ps %%zmm7, %%zmm24",
      "vrcp28ps 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VEXP2PD_C8_512,
      "vexp2pd %%zmm7, %%zmm24",
      "vexp2pd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VEXP2PS_C8_512,
      "vexp2ps %%zmm7, %%zmm24",
      "vexp2ps 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPMULHRSW_0B_512,
      "vpmulhrsw %%zmm7, %%zmm8, %%zmm24",
      "vpmulhrsw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPMULHRSW_0B_256,
      "vpmulhrsw %%ymm7, %%ymm8, %%ymm24",
      "vpmulhrsw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPMULHRSW_0B_128,
      "vpmulhrsw %%xmm7, %%xmm8, %%xmm24",
      "vpmulhrsw 0x40(%%rax), %%xmm8, %%xmm24")

/* Logic instructions */
GEN_test_AllMaskModes(VPANDNQ_DF_512,
      "vpandnq %%zmm7, %%zmm8, %%zmm24",
      "vpandnq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPANDNQ_DF_256,
      "vpandnq %%ymm7, %%ymm8, %%ymm24",
      "vpandnq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPANDNQ_DF_128,
      "vpandnq %%xmm7, %%xmm8, %%xmm24",
      "vpandnq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPANDND_DF_512,
      "vpandnd %%zmm7, %%zmm8, %%zmm24",
      "vpandnd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPANDND_DF_256,
      "vpandnd %%ymm7, %%ymm8, %%ymm24",
      "vpandnd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPANDND_DF_128,
      "vpandnd %%xmm7, %%xmm8, %%xmm24",
      "vpandnd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPANDQ_DB_512,
      "vpandq %%zmm7, %%zmm8, %%zmm24",
      "vpandq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPANDD_DB_512,
      "vpandd %%zmm7, %%zmm8, %%zmm24",
      "vpandd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPXORQ_EF_512,
      "vpxorq %%zmm7, %%zmm8, %%zmm24",
      "vpxorq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPXORD_EF_512,
      "vpxord %%zmm7, %%zmm8, %%zmm24",
      "vpxord 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPXORD_EF_256,
      "vpxord %%ymm7, %%ymm8, %%ymm24",
      "vpxord 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPXORD_EF_128,
      "vpxord %%xmm7, %%xmm8, %%xmm24",
      "vpxord 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPORQ_EB_512,
      "vporq %%zmm7, %%zmm8, %%zmm24",
      "vporq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPORD_EB_512,
      "vpord %%zmm7, %%zmm8, %%zmm24",
      "vpord 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VANDPD_54_512,
      "vandpd %%zmm7, %%zmm8, %%zmm24",
      "vandpd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VANDPD_54_256,
      "vandpd %%ymm7, %%ymm8, %%ymm24",
      "vandpd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VANDPD_54_128,
      "vandpd %%xmm7, %%xmm8, %%xmm24",
      "vandpd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPTERNLOGQ_25_512,
      "vpternlogq $0xAB, %%zmm7, %%zmm8, %%zmm24",
      "vpternlogq $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPTERNLOGD_25_512,
      "vpternlogd $0xAB, %%zmm7, %%zmm8, %%zmm24",
      "vpternlogd $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPTERNLOGD_25_256,
      "vpternlogd $0xAB, %%ymm7, %%ymm8, %%ymm24",
      "vpternlogd $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPTERNLOGD_25_128,
      "vpternlogd $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vpternlogd $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VXORPS_57_512,
      "vxorps %%zmm7, %%zmm8, %%zmm24",
      "vxorps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VXORPS_57_256,
      "vxorps %%ymm7, %%ymm8, %%ymm24",
      "vxorps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VXORPS_57_128,
      "vxorps %%xmm7, %%xmm8, %%xmm24",
      "vxorps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPTERNLOGQ_25_256,
      "vpternlogq $0xAB, %%ymm7, %%ymm8, %%ymm24",
      "vpternlogq $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPTERNLOGQ_25_128,
      "vpternlogq $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vpternlogq $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPXORQ_EF_256,
      "vpxorq %%ymm7, %%ymm8, %%ymm24",
      "vpxorq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPXORQ_EF_128,
      "vpxorq %%xmm7, %%xmm8, %%xmm24",
      "vpxorq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPORQ_EB_256,
      "vporq %%ymm7, %%ymm8, %%ymm24",
      "vporq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPORQ_EB_128,
      "vporq %%xmm7, %%xmm8, %%xmm24",
      "vporq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPORD_EB_256,
      "vpord %%ymm7, %%ymm8, %%ymm24",
      "vpord 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPORD_EB_128,
      "vpord %%xmm7, %%xmm8, %%xmm24",
      "vpord 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPANDQ_DB_256,
      "vpandq %%ymm7, %%ymm8, %%ymm24",
      "vpandq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPANDQ_DB_128,
      "vpandq %%xmm7, %%xmm8, %%xmm24",
      "vpandq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPANDD_DB_256,
      "vpandd %%ymm7, %%ymm8, %%ymm24",
      "vpandd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPANDD_DB_128,
      "vpandd %%xmm7, %%xmm8, %%xmm24",
      "vpandd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_OneMaskModeOnly(VPTESTMQ_27_512,
      "vptestmq %%zmm7, %%zmm8, %%k5",
      "vptestmq 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTMQ_27_256,
      "vptestmq %%ymm7, %%ymm8, %%k5",
      "vptestmq 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTMQ_27_128,
      "vptestmq %%xmm7, %%xmm8, %%k5",
      "vptestmq 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTMD_27_512,
      "vptestmd %%zmm7, %%zmm8, %%k5",
      "vptestmd 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTMD_27_256,
      "vptestmd %%ymm7, %%ymm8, %%k5",
      "vptestmd 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTMD_27_128,
      "vptestmd %%xmm7, %%xmm8, %%k5",
      "vptestmd 0x40(%%rax), %%xmm8, %%k5")
GEN_test_AllMaskModes(VXORPD_57_512,
      "vxorpd %%zmm7, %%zmm8, %%zmm24",
      "vxorpd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VXORPD_57_256,
      "vxorpd %%ymm7, %%ymm8, %%ymm24",
      "vxorpd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VXORPD_57_128,
      "vxorpd %%xmm7, %%xmm8, %%xmm24",
      "vxorpd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VORPD_56_512,
      "vorpd %%zmm7, %%zmm8, %%zmm24",
      "vorpd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VORPD_56_256,
      "vorpd %%ymm7, %%ymm8, %%ymm24",
      "vorpd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VORPD_56_128,
      "vorpd %%xmm7, %%xmm8, %%xmm24",
      "vorpd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VANDNPD_55_512,
      "vandnpd %%zmm7, %%zmm8, %%zmm24",
      "vandnpd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VANDNPD_55_256,
      "vandnpd %%ymm7, %%ymm8, %%ymm24",
      "vandnpd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VANDNPD_55_128,
      "vandnpd %%xmm7, %%xmm8, %%xmm24",
      "vandnpd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VORPS_56_512,
      "vorps %%zmm7, %%zmm8, %%zmm24",
      "vorps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VORPS_56_256,
      "vorps %%ymm7, %%ymm8, %%ymm24",
      "vorps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VORPS_56_128,
      "vorps %%xmm7, %%xmm8, %%xmm24",
      "vorps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VANDNPS_55_512,
      "vandnps %%zmm7, %%zmm8, %%zmm24",
      "vandnps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VANDNPS_55_256,
      "vandnps %%ymm7, %%ymm8, %%ymm24",
      "vandnps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VANDNPS_55_128,
      "vandnps %%xmm7, %%xmm8, %%xmm24",
      "vandnps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VANDPS_54_512,
      "vandps %%zmm7, %%zmm8, %%zmm24",
      "vandps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VANDPS_54_256,
      "vandps %%ymm7, %%ymm8, %%ymm24",
      "vandps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VANDPS_54_128,
      "vandps %%xmm7, %%xmm8, %%xmm24",
      "vandps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_OneMaskModeOnly(VPTESTNMQ_27_512,
      "vptestnmq %%zmm7, %%zmm8, %%k5",
      "vptestnmq 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTNMQ_27_256,
      "vptestnmq %%ymm7, %%ymm8, %%k5",
      "vptestnmq 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTNMQ_27_128,
      "vptestnmq %%xmm7, %%xmm8, %%k5",
      "vptestnmq 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTNMD_27_512,
      "vptestnmd %%zmm7, %%zmm8, %%k5",
      "vptestnmd 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTNMD_27_256,
      "vptestnmd %%ymm7, %%ymm8, %%k5",
      "vptestnmd 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTNMD_27_128,
      "vptestnmd %%xmm7, %%xmm8, %%k5",
      "vptestnmd 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTNMW_26_512,
      "vptestnmw %%zmm7, %%zmm8, %%k5",
      "vptestnmw 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTNMW_26_256,
      "vptestnmw %%ymm7, %%ymm8, %%k5",
      "vptestnmw 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTNMW_26_128,
      "vptestnmw %%xmm7, %%xmm8, %%k5",
      "vptestnmw 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTNMB_26_512,
      "vptestnmb %%zmm7, %%zmm8, %%k5",
      "vptestnmb 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTNMB_26_256,
      "vptestnmb %%ymm7, %%ymm8, %%k5",
      "vptestnmb 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTNMB_26_128,
      "vptestnmb %%xmm7, %%xmm8, %%k5",
      "vptestnmb 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTMW_26_512,
      "vptestmw %%zmm7, %%zmm8, %%k5",
      "vptestmw 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTMW_26_256,
      "vptestmw %%ymm7, %%ymm8, %%k5",
      "vptestmw 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTMW_26_128,
      "vptestmw %%xmm7, %%xmm8, %%k5",
      "vptestmw 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTMB_26_512,
      "vptestmb %%zmm7, %%zmm8, %%k5",
      "vptestmb 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTMB_26_256,
      "vptestmb %%ymm7, %%ymm8, %%k5",
      "vptestmb 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPTESTMB_26_128,
      "vptestmb %%xmm7, %%xmm8, %%k5",
      "vptestmb 0x40(%%rax), %%xmm8, %%k5")

/* Shift and rotate */

GEN_test_AllMaskModes(VPSRLVD_512,
      "vpslld $27, %%zmm18, %%zmm18;"
      "vpsrld $27, %%zmm18, %%zmm18;"
      "vpsrlvd %%zmm18, %%zmm8, %%zmm7",
      "andl $31, 0x40(%%rax);"
      "andl $31, 0x44(%%rax);"
      "andl $31, 0x48(%%rax);"
      "andl $31, 0x52(%%rax);"
      "andl $31, 0x56(%%rax);"
      "andl $31, 0x60(%%rax);"
      "andl $31, 0x64(%%rax);"
      "andl $31, 0x68(%%rax);"
      "andl $31, 0x72(%%rax);"
      "andl $31, 0x76(%%rax);"
      "andl $31, 0x80(%%rax);"
      "andl $31, 0x84(%%rax);"
      "andl $31, 0x88(%%rax);"
      "andl $31, 0x92(%%rax);"
      "andl $31, 0x96(%%rax);"
      "andl $31, 0x100(%%rax);"
      "vpsrlvd 0x40(%%rax), %%zmm8, %%zmm7")
GEN_test_AllMaskModes(VPSRAVD_512,
      "vpslld $27, %%zmm18, %%zmm18;"
      "vpsrld $27, %%zmm18, %%zmm18;"
      "vpsravd %%zmm18, %%zmm8, %%zmm7",
      "andl $31, 0x40(%%rax);"
      "andl $31, 0x44(%%rax);"
      "andl $31, 0x48(%%rax);"
      "andl $31, 0x52(%%rax);"
      "andl $31, 0x56(%%rax);"
      "andl $31, 0x60(%%rax);"
      "andl $31, 0x64(%%rax);"
      "andl $31, 0x68(%%rax);"
      "andl $31, 0x72(%%rax);"
      "andl $31, 0x76(%%rax);"
      "andl $31, 0x80(%%rax);"
      "andl $31, 0x84(%%rax);"
      "andl $31, 0x88(%%rax);"
      "andl $31, 0x92(%%rax);"
      "andl $31, 0x96(%%rax);"
      "andl $31, 0x100(%%rax);"
      "vpsravd 0x40(%%rax), %%zmm8, %%zmm7")
GEN_test_AllMaskModes(VPSLLVD_512,
      "vpslld $27, %%zmm18, %%zmm18;"
      "vpsrld $27, %%zmm18, %%zmm18;"
      "vpsllvd %%zmm18, %%zmm8, %%zmm7",
      "andl $31, 0x40(%%rax);"
      "andl $31, 0x44(%%rax);"
      "andl $31, 0x48(%%rax);"
      "andl $31, 0x52(%%rax);"
      "andl $31, 0x56(%%rax);"
      "andl $31, 0x60(%%rax);"
      "andl $31, 0x64(%%rax);"
      "andl $31, 0x68(%%rax);"
      "andl $31, 0x72(%%rax);"
      "andl $31, 0x76(%%rax);"
      "andl $31, 0x80(%%rax);"
      "andl $31, 0x84(%%rax);"
      "andl $31, 0x88(%%rax);"
      "andl $31, 0x92(%%rax);"
      "andl $31, 0x96(%%rax);"
      "andl $31, 0x100(%%rax);"
      "vpsllvd 0x40(%%rax), %%zmm8, %%zmm7")

// Note - "andl $63, 8(%%rax);" from avx2-1.c does not zero [63:32] bits
GEN_test_AllMaskModes(VPSRLVQ_512,
      "vpsllq $58, %%zmm18, %%zmm18;"
      "vpsrlq $58, %%zmm18, %%zmm18;"
      "vpsrlvq %%zmm18, %%zmm8, %%zmm7",
      "andl $31,   0x40(%%rax);"
      "movl $0x0,  0x44(%%rax);"
      "andl $31,   0x48(%%rax);"
      "movl $0x0,  0x52(%%rax);"
      "andl $31,   0x56(%%rax);"
      "movl $0x0,  0x60(%%rax);"
      "andl $31,   0x64(%%rax);"
      "movl $0x0,  0x68(%%rax);"
      "andl $31,   0x72(%%rax);"
      "movl $0x0,  0x76(%%rax);"
      "andl $31,   0x80(%%rax);"
      "movl $0x0,  0x84(%%rax);"
      "andl $31,   0x88(%%rax);"
      "movl $0x0,  0x92(%%rax);"
      "andl $31,   0x96(%%rax);"
      "movl $0x0,  0x100(%%rax);"
      "vpsrlvq 0x40(%%rax), %%zmm8, %%zmm7")
GEN_test_AllMaskModes(VPSRAVQ_512,
      "vpsllq $58, %%zmm18, %%zmm18;"
      "vpsrlq $58, %%zmm18, %%zmm18;"
      "vpsravq %%zmm18, %%zmm8, %%zmm7",
      "andl $31,   0x40(%%rax);"
      "movl $0x0,  0x44(%%rax);"
      "andl $31,   0x48(%%rax);"
      "movl $0x0,  0x52(%%rax);"
      "andl $31,   0x56(%%rax);"
      "movl $0x0,  0x60(%%rax);"
      "andl $31,   0x64(%%rax);"
      "movl $0x0,  0x68(%%rax);"
      "andl $31,   0x72(%%rax);"
      "movl $0x0,  0x76(%%rax);"
      "andl $31,   0x80(%%rax);"
      "movl $0x0,  0x84(%%rax);"
      "andl $31,   0x88(%%rax);"
      "movl $0x0,  0x92(%%rax);"
      "andl $31,   0x96(%%rax);"
      "movl $0x0,  0x100(%%rax);"
      "vpsravq 0x40(%%rax), %%zmm8, %%zmm7")
GEN_test_AllMaskModes(VPSLLVQ_512,
      "vpsllq $58, %%zmm18, %%zmm18;"
      "vpsrlq $58, %%zmm18, %%zmm18;"
      "vpsllvq %%zmm18, %%zmm8, %%zmm7",
      "andl $31,   0x40(%%rax);"
      "movl $0x0,  0x44(%%rax);"
      "andl $31,   0x48(%%rax);"
      "movl $0x0,  0x52(%%rax);"
      "andl $31,   0x56(%%rax);"
      "movl $0x0,  0x60(%%rax);"
      "andl $31,   0x64(%%rax);"
      "movl $0x0,  0x68(%%rax);"
      "andl $31,   0x72(%%rax);"
      "movl $0x0,  0x76(%%rax);"
      "andl $31,   0x80(%%rax);"
      "movl $0x0,  0x84(%%rax);"
      "andl $31,   0x88(%%rax);"
      "movl $0x0,  0x92(%%rax);"
      "andl $31,   0x96(%%rax);"
      "movl $0x0,  0x100(%%rax);"
      "vpsllvq 0x40(%%rax), %%zmm8, %%zmm7")

GEN_test_AllMaskModes(VPSRAD_E2_512,
      "vpsrad %%xmm24, %%zmm8, %%zmm7",
      "vpsrad 0x40(%%rax), %%zmm24, %%zmm8")
GEN_test_AllMaskModes(VPSLLQ_F3_512,
      "vpsllq %%xmm7, %%zmm8, %%zmm24",
      "vpsllq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSRAQ_E2_512,
      "vpsraq %%xmm7, %%zmm8, %%zmm24",
      "vpsraq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSRLQ_D3_512,
      "vpsrlq %%xmm7, %%zmm8, %%zmm24",
      "vpsrlq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSLLQ_73_512,
      "vpsllq $0xAB, %%zmm7, %%zmm24",
      "vpsllq $0xAB, 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPSRLQ_73_512,
      "vpsrlq $0xAB, %%zmm7, %%zmm24",
      "vpsrlq $0xAB, 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPSRAQ_72_512,
      "vpsraq $0xAB, %%zmm7, %%zmm8",
      "vpsraq $0xAB, 0x40(%%rax), %%zmm8")
GEN_test_AllMaskModes(VPSRAQ_72_256,
      "vpsraq $0xAB, %%ymm7, %%ymm18",
      "vpsraq $0xAB, 0x40(%%rax), %%ymm18")
GEN_test_AllMaskModes(VPSRAQ_72_128,
      "vpsraq $0xAB, %%xmm7, %%xmm18",
      "vpsraq $0xAB, 0x40(%%rax), %%xmm18")
GEN_test_AllMaskModes(VPROLQ_72_512,
      "vprolq $0xAB, %%zmm7, %%zmm8",
      "vprolq $0xAB, 0x40(%%rax), %%zmm8")
GEN_test_AllMaskModes(VPROLQ_72_256,
      "vprolq $0xAB, %%ymm7, %%ymm18",
      "vprolq $0xAB, 0x40(%%rax), %%ymm18")
GEN_test_AllMaskModes(VPROLQ_72_128,
      "vprolq $0xAB, %%xmm7, %%xmm18",
      "vprolq $0xAB, 0x40(%%rax), %%xmm18")
GEN_test_AllMaskModes(VPRORQ_72_512,
      "vprorq $0xAB, %%zmm7, %%zmm18",
      "vprorq $0xAB, 0x40(%%rax), %%zmm18")
GEN_test_AllMaskModes(VPRORQ_72_256,
      "vprorq $0xAB, %%ymm7, %%ymm18",
      "vprorq $0xAB, 0x40(%%rax), %%ymm18")
GEN_test_AllMaskModes(VPRORQ_72_128,
      "vprorq $0xAB, %%xmm7, %%xmm18",
      "vprorq $0xAB, 0x40(%%rax), %%xmm18")
GEN_test_AllMaskModes(VPSLLD_72_512,
      "vpslld $0xAB, %%zmm7, %%zmm18",
      "vpslld $0xAB, 0x40(%%rax), %%zmm18")
GEN_test_AllMaskModes(VPSLLD_72_256,
      "vpslld $0xAB, %%ymm7, %%ymm18",
      "vpslld $0xAB, 0x40(%%rax), %%ymm18")
GEN_test_AllMaskModes(VPSLLD_72_128,
      "vpslld $0xAB, %%xmm7, %%xmm18",
      "vpslld $0xAB, 0x40(%%rax), %%xmm18")
GEN_test_AllMaskModes(VPSRAD_72_512,
      "vpsrad $0xAB, %%zmm7, %%zmm18",
      "vpsrad $0xAB, 0x40(%%rax), %%zmm18")
GEN_test_AllMaskModes(VPSRAD_72_256,
      "vpsrad $0xAB, %%ymm7, %%ymm18",
      "vpsrad $0xAB, 0x40(%%rax), %%ymm18")
GEN_test_AllMaskModes(VPSRAD_72_128,
      "vpsrad $0xAB, %%xmm7, %%xmm18",
      "vpsrad $0xAB, 0x40(%%rax), %%xmm18")
GEN_test_AllMaskModes(VPSRLD_72_512,
      "vpsrld $0xAB, %%zmm7, %%zmm18",
      "vpsrld $0xAB, 0x40(%%rax), %%zmm18")
GEN_test_AllMaskModes(VPSRLD_72_256,
      "vpsrld $0xAB, %%ymm7, %%ymm18",
      "vpsrld $0xAB, 0x40(%%rax), %%ymm18")
GEN_test_AllMaskModes(VPSRLD_72_128,
      "vpsrld $0xAB, %%xmm7, %%xmm18",
      "vpsrld $0xAB, 0x40(%%rax), %%xmm18")
GEN_test_AllMaskModes(VPROLD_72_512,
      "vprold $0xAB, %%zmm7, %%zmm18",
      "vprold $0xAB, 0x40(%%rax), %%zmm18")
GEN_test_AllMaskModes(VPROLD_72_256,
      "vprold $0xAB, %%ymm7, %%ymm18",
      "vprold $0xAB, 0x40(%%rax), %%ymm18")
GEN_test_AllMaskModes(VPROLD_72_128,
      "vprold $0xAB, %%xmm7, %%xmm18",
      "vprold $0xAB, 0x40(%%rax), %%xmm18")
GEN_test_AllMaskModes(VPRORD_72_512,
      "vprord $0xAB, %%zmm7, %%zmm18",
      "vprord $0xAB, 0x40(%%rax), %%zmm18")
GEN_test_AllMaskModes(VPRORD_72_256,
      "vprord $0xAB, %%ymm7, %%ymm18",
      "vprord $0xAB, 0x40(%%rax), %%ymm18")
GEN_test_AllMaskModes(VPRORD_72_128,
      "vprord $0xAB, %%xmm7, %%xmm18",
      "vprord $0xAB, 0x40(%%rax), %%xmm18")
GEN_test_AllMaskModes(VPROLVQ_15_512,
      "vprolvq %%zmm7, %%zmm8, %%zmm24",
      "vprolvq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPROLVQ_15_256,
      "vprolvq %%ymm7, %%ymm8, %%ymm24",
      "vprolvq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPROLVQ_15_128,
      "vprolvq %%xmm7, %%xmm8, %%xmm24",
      "vprolvq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPROLVD_15_512,
      "vprolvd %%zmm7, %%zmm8, %%zmm24",
      "vprolvd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPROLVD_15_256,
      "vprolvd %%ymm7, %%ymm8, %%ymm24",
      "vprolvd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPROLVD_15_128,
      "vprolvd %%xmm7, %%xmm8, %%xmm24",
      "vprolvd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VUNPCKHPD_15_256,
      "vunpckhpd %%ymm7, %%ymm8, %%ymm24",
      "vunpckhpd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VUNPCKHPD_15_128,
      "vunpckhpd %%xmm7, %%xmm8, %%xmm24",
      "vunpckhpd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VUNPCKHPS_15_256,
      "vunpckhps %%ymm7, %%ymm8, %%ymm24",
      "vunpckhps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VUNPCKHPS_15_128,
      "vunpckhps %%xmm7, %%xmm8, %%xmm24",
      "vunpckhps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPRORVQ_14_512,
      "vprorvq %%zmm7, %%zmm8, %%zmm24",
      "vprorvq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPRORVQ_14_256,
      "vprorvq %%ymm7, %%ymm8, %%ymm24",
      "vprorvq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPRORVQ_14_128,
      "vprorvq %%xmm7, %%xmm8, %%xmm24",
      "vprorvq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPRORVD_14_512,
      "vprorvd %%zmm7, %%zmm8, %%zmm24",
      "vprorvd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPRORVD_14_256,
      "vprorvd %%ymm7, %%ymm8, %%ymm24",
      "vprorvd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPRORVD_14_128,
      "vprorvd %%xmm7, %%xmm8, %%xmm24",
      "vprorvd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSRAVQ_47_512,
      "vpsravq %%zmm7, %%zmm8, %%zmm24",
      "vpsravq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSRAVQ_47_256,
      "vpsravq %%ymm7, %%ymm8, %%ymm24",
      "vpsravq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSRAVQ_47_128,
      "vpsravq %%xmm7, %%xmm8, %%xmm24",
      "vpsravq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSRAVD_47_512,
      "vpsravd %%zmm7, %%zmm8, %%zmm24",
      "vpsravd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSRAVD_47_256,
      "vpsravd %%ymm7, %%ymm8, %%ymm24",
      "vpsravd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSRAVD_47_128,
      "vpsravd %%xmm7, %%xmm8, %%xmm24",
      "vpsravd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSRAVQ_46_512,
      "vpsravq %%zmm7, %%zmm8, %%zmm24",
      "vpsravq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSRAVQ_46_256,
      "vpsravq %%ymm7, %%ymm8, %%ymm24",
      "vpsravq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSRAVQ_46_128,
      "vpsravq %%xmm7, %%xmm8, %%xmm24",
      "vpsravq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSRAVD_46_512,
      "vpsravd %%zmm7, %%zmm8, %%zmm24",
      "vpsravd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSRAVD_46_256,
      "vpsravd %%ymm7, %%ymm8, %%ymm24",
      "vpsravd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSRAVD_46_128,
      "vpsravd %%xmm7, %%xmm8, %%xmm24",
      "vpsravd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSRLVQ_45_512,
      "vpsrlvq %%zmm7, %%zmm8, %%zmm24",
      "vpsrlvq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSRLVQ_45_256,
      "vpsrlvq %%ymm7, %%ymm8, %%ymm24",
      "vpsrlvq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSRLVQ_45_128,
      "vpsrlvq %%xmm7, %%xmm8, %%xmm24",
      "vpsrlvq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSRLVD_45_512,
      "vpsrlvd %%zmm7, %%zmm8, %%zmm24",
      "vpsrlvd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSRLVD_45_256,
      "vpsrlvd %%ymm7, %%ymm8, %%ymm24",
      "vpsrlvd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSRLVD_45_128,
      "vpsrlvd %%xmm7, %%xmm8, %%xmm24",
      "vpsrlvd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSLLQ_F3_256,
      "vpsllq %%xmm7, %%ymm8, %%ymm24",
      "vpsllq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSLLQ_F3_128,
      "vpsllq %%xmm7, %%xmm8, %%xmm24",
      "vpsllq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSRAQ_E2_256,
      "vpsraq %%xmm7, %%ymm8, %%ymm24",
      "vpsraq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSRAQ_E2_128,
      "vpsraq %%xmm7, %%xmm8, %%xmm24",
      "vpsraq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSRAD_E2_256,
      "vpsrad %%xmm7, %%ymm8, %%ymm24",
      "vpsrad 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSRAD_E2_128,
      "vpsrad %%xmm7, %%xmm8, %%xmm24",
      "vpsrad 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSRLQ_D3_256,
      "vpsrlq %%xmm7, %%ymm8, %%ymm24",
      "vpsrlq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSRLQ_D3_128,
      "vpsrlq %%xmm7, %%xmm8, %%xmm24",
      "vpsrlq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSLLQ_73_256,
      "vpsllq $0xAB, %%ymm7, %%ymm18",
      "vpsllq $0xAB, 0x40(%%rax), %%ymm18")
GEN_test_AllMaskModes(VPSLLQ_73_128,
      "vpsllq $0xAB, %%xmm7, %%xmm18",
      "vpsllq $0xAB, 0x40(%%rax), %%xmm18")
GEN_test_AllMaskModes(VPSRLQ_73_256,
      "vpsrlq $0xAB, %%ymm7, %%ymm18",
      "vpsrlq $0xAB, 0x40(%%rax), %%ymm18")
GEN_test_AllMaskModes(VPSRLQ_73_128,
      "vpsrlq $0xAB, %%xmm7, %%xmm18",
      "vpsrlq $0xAB, 0x40(%%rax), %%xmm18")
GEN_test_AllMaskModes(VPSLLW_F1_512,
      "vpsllw %%xmm7, %%zmm8, %%zmm24",
      "vpsllw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSLLW_F1_256,
      "vpsllw %%xmm7, %%ymm8, %%ymm24",
      "vpsllw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSLLW_F1_128,
      "vpsllw %%xmm7, %%xmm8, %%xmm24",
      "vpsllw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSRAW_E1_512,
      "vpsraw %%xmm7, %%zmm8, %%zmm24",
      "vpsraw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSRAW_E1_256,
      "vpsraw %%xmm7, %%ymm8, %%ymm24",
      "vpsraw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSRAW_E1_128,
      "vpsraw %%xmm7, %%xmm8, %%xmm24",
      "vpsraw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSRLW_D1_512,
      "vpsrlw %%xmm7, %%zmm8, %%zmm24",
      "vpsrlw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSRLW_D1_256,
      "vpsrlw %%xmm7, %%ymm8, %%ymm24",
      "vpsrlw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSRLW_D1_128,
      "vpsrlw %%xmm7, %%xmm8, %%xmm24",
      "vpsrlw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_RandM(VPSRLDQ_73_512,
      "vpsrldq $0xAB, %%zmm7, %%zmm8",
      "vpsrldq $0xAB, 0x40(%%rax), %%zmm8")
GEN_test_RandM(VPSRLDQ_73_256,
      "vpsrldq $0xAB, %%ymm7, %%ymm8",
      "vpsrldq $0xAB, 0x40(%%rax), %%ymm8")
GEN_test_RandM(VPSRLDQ_73_128,
      "vpsrldq $0xAB, %%xmm7, %%xmm8",
      "vpsrldq $0xAB, 0x40(%%rax), %%xmm8")
GEN_test_AllMaskModes(VPSLLW_71_512,
      "vpsllw $0xAB, %%zmm7, %%zmm8",
      "vpsllw $0xAB, 0x40(%%rax), %%zmm8")
GEN_test_AllMaskModes(VPSLLW_71_256,
      "vpsllw $0xAB, %%ymm7, %%ymm8",
      "vpsllw $0xAB, 0x40(%%rax), %%ymm8")
GEN_test_AllMaskModes(VPSLLW_71_128,
      "vpsllw $0xAB, %%xmm7, %%xmm8",
      "vpsllw $0xAB, 0x40(%%rax), %%xmm8")
GEN_test_AllMaskModes(VPSRAW_71_512,
      "vpsraw $0xAB, %%zmm7, %%zmm8",
      "vpsraw $0xAB, 0x40(%%rax), %%zmm8")
GEN_test_AllMaskModes(VPSRAW_71_256,
      "vpsraw $0xAB, %%ymm7, %%ymm8",
      "vpsraw $0xAB, 0x40(%%rax), %%ymm8")
GEN_test_AllMaskModes(VPSRAW_71_128,
      "vpsraw $0xAB, %%xmm7, %%xmm8",
      "vpsraw $0xAB, 0x40(%%rax), %%xmm8")
GEN_test_AllMaskModes(VPSRLW_71_512,
      "vpsrlw $0xAB, %%zmm7, %%zmm8",
      "vpsrlw $0xAB, 0x40(%%rax), %%zmm8")
GEN_test_AllMaskModes(VPSRLW_71_256,
      "vpsrlw $0xAB, %%ymm7, %%ymm8",
      "vpsrlw $0xAB, 0x40(%%rax), %%ymm8")
GEN_test_AllMaskModes(VPSRLW_71_128,
      "vpsrlw $0xAB, %%xmm7, %%xmm8",
      "vpsrlw $0xAB, 0x40(%%rax), %%xmm8")
GEN_test_AllMaskModes(VPSLLVW_12_512,
      "vpsllvw %%zmm7, %%zmm8, %%zmm24",
      "vpsllvw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSLLVW_12_256,
      "vpsllvw %%ymm7, %%ymm8, %%ymm24",
      "vpsllvw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSLLVW_12_128,
      "vpsllvw %%xmm7, %%xmm8, %%xmm24",
      "vpsllvw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSRAVW_11_512,
      "vpsravw %%zmm7, %%zmm8, %%zmm24",
      "vpsravw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSRAVW_11_256,
      "vpsravw %%ymm7, %%ymm8, %%ymm24",
      "vpsravw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSRAVW_11_128,
      "vpsravw %%xmm7, %%xmm8, %%xmm24",
      "vpsravw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSRLVW_10_512,
      "vpsrlvw %%zmm7, %%zmm8, %%zmm24",
      "vpsrlvw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSRLVW_10_256,
      "vpsrlvw %%ymm7, %%ymm8, %%ymm24",
      "vpsrlvw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSRLVW_10_128,
      "vpsrlvw %%xmm7, %%xmm8, %%xmm24",
      "vpsrlvw 0x40(%%rax), %%xmm8, %%xmm24")

/* Vector compare */

GEN_test_OneMaskModeOnly(VPCMPD_512,
      "vpcmpd $0x1, %%zmm18, %%zmm8, %%k4",
      "vpcmpd $0x1, 0x40(%%rax),%%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VPCMPUD_512,
      "vpcmpud $0x5, %%zmm18, %%zmm8, %%k4",
      "vpcmpud $0x5, 0x40(%%rax),%%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VPCMPQ_512,
      "vpcmpq $0x6, %%zmm18, %%zmm8, %%k4",
      "vpcmpd $0x6, 0x40(%%rax),%%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VPCMPUQ_512,
      "vpcmpuq $0x2, %%zmm18, %%zmm8, %%k4",
      "vpcmpuq $0x2, 0x40(%%rax),%%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VPCMPEQUD_512,
      "vpcmpequd %%zmm18, %%zmm8, %%k4",
      "vpcmpequd 0x40(%%rax),%%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VPCMPLTUD_512,
      "vpcmpltud %%zmm18, %%zmm8, %%k4",
      "vpcmpltud 0x40(%%rax),%%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VPCMPGTD_512,
      "vpcmpgtd %%zmm18, %%zmm8, %%k4",
      "vpcmpgtd 0x40(%%rax),%%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VPCMPLEUD_512,
      "vpcmpleud %%zmm18, %%zmm8, %%k4",
      "vpcmpleud 0x40(%%rax),%%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VPTESTMD_512,
      "vptestmd %%zmm18, %%zmm8, %%k4",
      "vptestmd 0x40(%%rax),%%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VPTESTMQ_512,
      "vptestmq %%zmm18, %%zmm8, %%k4",
      "vptestmq 0x40(%%rax),%%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VPTESTNMD_512,
      "vptestnmd %%zmm18, %%zmm8, %%k4",
      "vptestnmd 0x40(%%rax),%%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VPTESTNMQ_512,
      "vptestnmq %%zmm18, %%zmm8, %%k4",
      "vptestnmq 0x40(%%rax),%%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPEQ_UQPD_512,
      "vcmpeq_uqpd %%zmm18, %%zmm8, %%k4",
      "vcmpeq_uqpd 0x40(%%rax),%%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPNEQPD_512,
      "vcmpneqpd %%zmm18, %%zmm8, %%k4",
      "vcmpneqpd 0x40(%%rax),%%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x0,
      "vcmppd $0, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $0, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x1,
      "vcmppd $1, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $1, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x2,
      "vcmppd $2, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $2, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x3,
      "vcmppd $3, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $3, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x4,
      "vcmppd $4, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $4, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x5,
      "vcmppd $5, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $5, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x6,
      "vcmppd $6, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $6, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x7,
      "vcmppd $7, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $7, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x8,
      "vcmppd $8, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $8, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x9,
      "vcmppd $9, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $9, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0xA,
      "vcmppd $10, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $10, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0xB,
      "vcmppd $11, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $11, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0xC,
      "vcmppd $12, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $12, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0xD,
      "vcmppd $13, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $13, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0xE,
      "vcmppd $14, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $14, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0xF,
      "vcmppd $15, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $15, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x10,
      "vcmppd $16, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $16, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x11,
      "vcmppd $17, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $17, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x12,
      "vcmppd $18, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $18, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x13,
      "vcmppd $19, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $19, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x14,
      "vcmppd $20, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $20, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x15,
      "vcmppd $21, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $21, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x16,
      "vcmppd $22, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $22, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x17,
      "vcmppd $23, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $23, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x18,
      "vcmppd $24, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $24, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x19,
      "vcmppd $25, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $25, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x1A,
      "vcmppd $26, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $26, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x1B,
      "vcmppd $27, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $27, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x1C,
      "vcmppd $28, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $28, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x1D,
      "vcmppd $29, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $29, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x1E,
      "vcmppd $30, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $30, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_512_0x1F,
      "vcmppd $31, %%zmm8,  %%zmm24, %%k4",
      "vcmppd $31, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x0,
      "vcmpps $0, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $0, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x1,
      "vcmpps $1, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $1, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x2,
      "vcmpps $2, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $2, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x3,
      "vcmpps $3, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $3, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x4,
      "vcmpps $4, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $4, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x5,
      "vcmpps $5, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $5, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x6,
      "vcmpps $6, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $6, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x7,
      "vcmpps $7, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $7, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x8,
      "vcmpps $8, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $8, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x9,
      "vcmpps $9, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $9, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0xA,
      "vcmpps $10, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $10, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0xB,
      "vcmpps $11, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $11, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0xC,
      "vcmpps $12, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $12, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0xD,
      "vcmpps $13, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $13, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0xE,
      "vcmpps $14, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $14, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0xF,
      "vcmpps $15, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $15, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x10,
      "vcmpps $16, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $16, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x11,
      "vcmpps $17, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $17, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x12,
      "vcmpps $18, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $18, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x13,
      "vcmpps $19, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $19, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x14,
      "vcmpps $20, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $20, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x15,
      "vcmpps $21, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $21, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x16,
      "vcmpps $22, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $22, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x17,
      "vcmpps $23, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $23, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x18,
      "vcmpps $24, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $24, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x19,
      "vcmpps $25, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $25, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x1A,
      "vcmpps $26, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $26, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x1B,
      "vcmpps $27, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $27, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x1C,
      "vcmpps $28, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $28, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x1D,
      "vcmpps $29, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $29, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x1E,
      "vcmpps $30, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $30, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPPS_512_0x1F,
      "vcmpps $31, %%zmm8,  %%zmm24, %%k4",
      "vcmpps $31, 0x40(%%rax), %%zmm24, %%k4")
GEN_test_RandM(VCMPPD_C4_512,
      "vcmppd $0xAB, %%zmm7, %%zmm8, %%k4",
      "vcmppd $0xAB, 0x40(%%rax), %%zmm8, %%k4")
GEN_test_RandM(VCMPPD_C4_256,
      "vcmppd $0xAB, %%ymm7, %%ymm8, %%k4",
      "vcmppd $0xAB, 0x40(%%rax), %%ymm8, %%k4")
GEN_test_RandM(VCMPPD_C4_128,
      "vcmppd $0xAB, %%xmm7, %%xmm8, %%k4",
      "vcmppd $0xAB, 0x40(%%rax), %%xmm8, %%k4")
GEN_test_RandM(VCMPPS_C2_512,
      "vcmpps $0xAB, %%zmm7, %%zmm8, %%k4",
      "vcmpps $0xAB, 0x40(%%rax), %%zmm8, %%k4")
GEN_test_RandM(VCMPPS_C2_256,
      "vcmpps $0xAB, %%ymm7, %%ymm8, %%k4",
      "vcmpps $0xAB, 0x40(%%rax), %%ymm8, %%k4")
GEN_test_RandM(VCMPPS_C2_128,
      "vcmpps $0xAB, %%xmm7, %%xmm8, %%k4",
      "vcmpps $0xAB, 0x40(%%rax), %%xmm8, %%k4")
GEN_test_OneMaskModeOnly(VPCMPEQD_76_512,
      "vpcmpeqd %%zmm24, %%zmm8, %%k4",
      "vpcmpeqd 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VPCMPEQQ_29_512,
      "vpcmpeqq %%zmm24, %%zmm8, %%k4",
      "vpcmpeqq 0x40(%%rax), %%zmm24, %%k4")
GEN_test_OneMaskModeOnly(VPCMPGTQ_37_512,
      "vpcmpgtq %%zmm24, %%zmm8, %%k4",
      "vpcmpgtq 0x40(%%rax), %%zmm24, %%k4")
GEN_test_RandM(VPCMPGTD_66_512,
      "vpcmpgtd %%zmm7, %%zmm8, %%k4",
      "vpcmpgtd 0x40(%%rax), %%zmm8, %%k4")
GEN_test_RandM(VPCMPGTD_66_256,
      "vpcmpgtd %%ymm7, %%ymm8, %%k4",
      "vpcmpgtd 0x40(%%rax), %%ymm8, %%k4")
GEN_test_RandM(VPCMPGTD_66_128,
      "vpcmpgtd %%xmm7, %%xmm8, %%k4",
      "vpcmpgtd 0x40(%%rax), %%xmm8, %%k4")
GEN_test_OneMaskModeOnly(VCMPPD_C2_512,
      "vcmppd $0xAB, %%zmm7, %%zmm8, %%k5",
      "vcmppd $0xAB, 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VCMPPD_C2_256,
      "vcmppd $0xAB, %%ymm7, %%ymm8, %%k5",
      "vcmppd $0xAB, 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VCMPPD_C2_128,
      "vcmppd $0xAB, %%xmm7, %%xmm8, %%k5",
      "vcmppd $0xAB, 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VCMPSS_C2_128,
      "vcmpss $0xAB, %%xmm7, %%xmm8, %%k5",
      "vcmpss $0xAB, 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPEQD_76_256,
      "vpcmpeqd %%ymm7, %%ymm8, %%k5",
      "vpcmpeqd 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPEQD_76_128,
      "vpcmpeqd %%xmm7, %%xmm8, %%k5",
      "vpcmpeqd 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPGTQ_37_256,
      "vpcmpgtq %%ymm7, %%ymm8, %%k5",
      "vpcmpgtq 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPGTQ_37_128,
      "vpcmpgtq %%xmm7, %%xmm8, %%k5",
      "vpcmpgtq 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPEQQ_29_256,
      "vpcmpeqq %%ymm7, %%ymm8, %%k5",
      "vpcmpeqq 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPEQQ_29_128,
      "vpcmpeqq %%xmm7, %%xmm8, %%k5",
      "vpcmpeqq 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPQ_1F_512,
      "vpcmpq $0xAB, %%zmm7, %%zmm8, %%k5",
      "vpcmpq $0xAB, 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPQ_1F_256,
      "vpcmpq $0xAB, %%ymm7, %%ymm8, %%k5",
      "vpcmpq $0xAB, 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPQ_1F_128,
      "vpcmpq $0xAB, %%xmm7, %%xmm8, %%k5",
      "vpcmpq $0xAB, 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPD_1F_512,
      "vpcmpd $0xAB, %%zmm7, %%zmm8, %%k5",
      "vpcmpd $0xAB, 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPD_1F_256,
      "vpcmpd $0xAB, %%ymm7, %%ymm8, %%k5",
      "vpcmpd $0xAB, 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPD_1F_128,
      "vpcmpd $0xAB, %%xmm7, %%xmm8, %%k5",
      "vpcmpd $0xAB, 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPUQ_1E_512,
      "vpcmpuq $0xAB, %%zmm7, %%zmm8, %%k5",
      "vpcmpuq $0xAB, 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPUQ_1E_256,
      "vpcmpuq $0xAB, %%ymm7, %%ymm8, %%k5",
      "vpcmpuq $0xAB, 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPUQ_1E_128,
      "vpcmpuq $0xAB, %%xmm7, %%xmm8, %%k5",
      "vpcmpuq $0xAB, 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPUD_1E_512,
      "vpcmpud $0xAB, %%zmm7, %%zmm8, %%k5",
      "vpcmpud $0xAB, 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPUD_1E_256,
      "vpcmpud $0xAB, %%ymm7, %%ymm8, %%k5",
      "vpcmpud $0xAB, 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPUD_1E_128,
      "vpcmpud $0xAB, %%xmm7, %%xmm8, %%k5",
      "vpcmpud $0xAB, 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPEQW_75_512,
      "vpcmpeqw %%zmm7, %%zmm8, %%k5",
      "vpcmpeqw 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPEQW_75_256,
      "vpcmpeqw %%ymm7, %%ymm8, %%k5",
      "vpcmpeqw 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPEQW_75_128,
      "vpcmpeqw %%xmm7, %%xmm8, %%k5",
      "vpcmpeqw 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPEQB_74_512,
      "vpcmpeqb %%zmm7, %%zmm8, %%k5",
      "vpcmpeqb 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPEQB_74_256,
      "vpcmpeqb %%ymm7, %%ymm8, %%k5",
      "vpcmpeqb 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPEQB_74_128,
      "vpcmpeqb %%xmm7, %%xmm8, %%k5",
      "vpcmpeqb 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPGTW_65_512,
      "vpcmpgtw %%zmm18, %%zmm7, %%k5",
      "vpcmpgtw 0x40(%%rax), %%zmm18, %%k5")
GEN_test_OneMaskModeOnly(VPCMPGTW_65_256,
      "vpcmpgtw %%ymm18, %%ymm7, %%k5",
      "vpcmpgtw 0x40(%%rax), %%ymm18, %%k5")
GEN_test_OneMaskModeOnly(VPCMPGTW_65_128,
      "vpcmpgtw %%xmm18, %%xmm7, %%k5",
      "vpcmpgtw 0x40(%%rax), %%xmm18, %%k5")
GEN_test_OneMaskModeOnly(VPCMPGTB_64_512,
      "vpcmpgtb %%zmm18, %%zmm7, %%k5",
      "vpcmpgtb 0x40(%%rax), %%zmm18, %%k5")
GEN_test_OneMaskModeOnly(VPCMPGTB_64_256,
      "vpcmpgtb %%ymm18, %%ymm7, %%k5",
      "vpcmpgtb 0x40(%%rax), %%ymm18, %%k5")
GEN_test_OneMaskModeOnly(VPCMPGTB_64_128,
      "vpcmpgtb %%xmm18, %%xmm7, %%k5",
      "vpcmpgtb 0x40(%%rax), %%xmm18, %%k5")
GEN_test_OneMaskModeOnly(VPCMPW_3F_512,
      "vpcmpw $0xAB, %%zmm7, %%zmm8, %%k5",
      "vpcmpw $0xAB, 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPW_3F_256,
      "vpcmpw $0xAB, %%ymm7, %%ymm8, %%k5",
      "vpcmpw $0xAB, 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPW_3F_128,
      "vpcmpw $0xAB, %%xmm7, %%xmm8, %%k5",
      "vpcmpw $0xAB, 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPB_3F_512,
      "vpcmpb $0xAB, %%zmm7, %%zmm8, %%k5",
      "vpcmpb $0xAB, 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPB_3F_256,
      "vpcmpb $0xAB, %%ymm7, %%ymm8, %%k5",
      "vpcmpb $0xAB, 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPB_3F_128,
      "vpcmpb $0xAB, %%xmm7, %%xmm8, %%k5",
      "vpcmpb $0xAB, 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPUW_3E_512,
      "vpcmpuw $0xAB, %%zmm7, %%zmm8, %%k5",
      "vpcmpuw $0xAB, 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPUW_3E_256,
      "vpcmpuw $0xAB, %%ymm7, %%ymm8, %%k5",
      "vpcmpuw $0xAB, 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPUW_3E_128,
      "vpcmpuw $0xAB, %%xmm7, %%xmm8, %%k5",
      "vpcmpuw $0xAB, 0x40(%%rax), %%xmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPUB_3E_512,
      "vpcmpub $0xAB, %%zmm7, %%zmm8, %%k5",
      "vpcmpub $0xAB, 0x40(%%rax), %%zmm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPUB_3E_256,
      "vpcmpub $0xAB, %%ymm7, %%ymm8, %%k5",
      "vpcmpub $0xAB, 0x40(%%rax), %%ymm8, %%k5")
GEN_test_OneMaskModeOnly(VPCMPUB_3E_128,
      "vpcmpub $0xAB, %%xmm7, %%xmm8, %%k5",
      "vpcmpub $0xAB, 0x40(%%rax), %%xmm8, %%k5")

/* Extract, broadcast, expand */
GEN_test_MemMaskModes(VEXTRACTI64x4_3B_512,
      "vextracti64x4 $0xAB, %%zmm24, %%ymm7",
      "vextracti64x4 $0xAB, %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VEXTRACTI32x4_39_512,
      "vextracti32x4 $0xAB, %%zmm24, %%xmm7",
      "vextracti32x4 $0xAB, %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VEXTRACTI32x4_39_256,
      "vextracti32x4 $0xAB, %%ymm24, %%xmm7",
      "vextracti32x4 $0xAB, %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VEXTRACTF64x4_1B_512,
      "vextractf64x4 $0xAB, %%zmm24, %%ymm7",
      "vextractf64x4 $0xAB, %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VEXTRACTF32x4_19_512,
      "vextractf32x4 $0xAB, %%zmm24, %%xmm7",
      "vextractf32x4 $0xAB, %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VEXTRACTF32x4_19_256,
      "vextractf32x4 $0xAB, %%ymm24, %%xmm7",
      "vextractf32x4 $0xAB, %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VEXTRACTI64x2_39_512,
      "vextracti64x2 $0xAB, %%zmm24, %%xmm7",
      "vextracti64x2 $0xAB, %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VEXTRACTI64x2_39_256,
      "vextracti64x2 $0xAB, %%ymm24, %%xmm7",
      "vextracti64x2 $0xAB, %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VEXTRACTF64x2_19_512,
      "vextractf64x2 $0xAB, %%zmm24, %%xmm7",
      "vextractf64x2 $0xAB, %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VEXTRACTF64x2_19_256,
      "vextractf64x2 $0xAB, %%ymm24, %%xmm7",
      "vextractf64x2 $0xAB, %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VEXTRACTI32x8_3B_512,
      "vextracti32x8 $0xAB, %%zmm24, %%ymm7",
      "vextracti32x8 $0xAB, %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VEXTRACTF32x8_1B_512,
      "vextractf32x8 $0xAB, %%zmm24, %%ymm7",
      "vextractf32x8 $0xAB, %%zmm24, 0x40(%%rax)")
GEN_test_Ronly(VPBROADCASTQ_7C_512,
      "vpbroadcastq %%r14, %%zmm24")
GEN_test_Ronly(VPBROADCASTQ_7C_256,
      "vpbroadcastq %%r14, %%ymm24")
GEN_test_Ronly(VPBROADCASTQ_7C_128,
      "vpbroadcastq %%r14, %%xmm24")
GEN_test_Ronly(VPBROADCASTD_7C_512,
      "vpbroadcastd %%r14d, %%zmm24")
GEN_test_Ronly(VPBROADCASTD_7C_256,
      "vpbroadcastd %%r14d, %%ymm24")
GEN_test_Ronly(VPBROADCASTD_7C_128,
      "vpbroadcastd %%r14d, %%xmm24")
GEN_test_M_AllMaskModes(VBROADCASTI64x4_5B_512,
      "vbroadcasti64x4 0x40(%%rax), %%zmm24")
GEN_test_M_AllMaskModes(VBROADCASTI32x4_5A_512,
      "vbroadcasti32x4 0x40(%%rax), %%zmm24")
GEN_test_M_AllMaskModes(VBROADCASTI32x4_5A_256,
      "vbroadcasti32x4 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPBROADCASTQ_59_512,
      "vpbroadcastq %%xmm7, %%zmm24",
      "vpbroadcastq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPBROADCASTQ_59_256,
      "vpbroadcastq %%xmm7, %%ymm24",
      "vpbroadcastq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPBROADCASTQ_59_128,
      "vpbroadcastq %%xmm7, %%xmm24",
      "vpbroadcastq 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VPBROADCASTD_58_512,
      "vpbroadcastd %%xmm7, %%zmm24",
      "vpbroadcastd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPBROADCASTD_58_256,
      "vpbroadcastd %%xmm7, %%ymm24",
      "vpbroadcastd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPBROADCASTD_58_128,
      "vpbroadcastd %%xmm7, %%xmm24",
      "vpbroadcastd 0x40(%%rax), %%xmm24")
GEN_test_M_AllMaskModes(VBROADCASTF64x4_1B_512,
      "vbroadcastf64x4 0x40(%%rax), %%zmm24")
GEN_test_M_AllMaskModes(VBROADCASTF32x4_1A_512,
      "vbroadcastf32x4 0x40(%%rax), %%zmm24")
GEN_test_M_AllMaskModes(VBROADCASTF32x4_1A_256,
      "vbroadcastf32x4 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VBROADCASTSD_19_512,
      "vbroadcastsd %%xmm7, %%zmm24",
      "vbroadcastsd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VBROADCASTSD_19_256,
      "vbroadcastsd %%xmm7, %%ymm24",
      "vbroadcastsd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VBROADCASTSS_18_512,
      "vbroadcastss %%xmm7, %%zmm24",
      "vbroadcastss 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VBROADCASTSS_18_256,
      "vbroadcastss %%xmm7, %%ymm24",
      "vbroadcastss 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VBROADCASTSS_18_128,
      "vbroadcastss %%xmm7, %%xmm24",
      "vbroadcastss 0x40(%%rax), %%xmm24")
GEN_test_Ronly(VPBROADCASTMW2D_3A_512,
      "vpbroadcastmw2d %%k5, %%zmm24")
GEN_test_Ronly(VPBROADCASTMW2D_3A_256,
      "vpbroadcastmw2d %%k5, %%ymm24")
GEN_test_Ronly(VPBROADCASTMW2D_3A_128,
      "vpbroadcastmw2d %%k5, %%xmm24")
GEN_test_Ronly(VPBROADCASTMB2Q_2A_512,
      "vpbroadcastmb2q %%k5, %%zmm24")
GEN_test_Ronly(VPBROADCASTMB2Q_2A_256,
      "vpbroadcastmb2q %%k5, %%ymm24")
GEN_test_Ronly(VPBROADCASTMB2Q_2A_128,
      "vpbroadcastmb2q %%k5, %%xmm24")
GEN_test_R_AllMaskModes(VPBROADCASTW_7B_512,
      "vpbroadcastw %%r14d, %%zmm24")
GEN_test_R_AllMaskModes(VPBROADCASTW_7B_256,
      "vpbroadcastw %%r14d, %%ymm24")
GEN_test_R_AllMaskModes(VPBROADCASTW_7B_128,
      "vpbroadcastw %%r14d, %%xmm24")
GEN_test_R_AllMaskModes(VPBROADCASTB_7A_512,
      "vpbroadcastb %%r14d, %%zmm24")
GEN_test_R_AllMaskModes(VPBROADCASTB_7A_256,
      "vpbroadcastb %%r14d, %%ymm24")
GEN_test_R_AllMaskModes(VPBROADCASTB_7A_128,
      "vpbroadcastb %%r14d, %%xmm24")
GEN_test_M_AllMaskModes(VBROADCASTF64x2_1A_512,
      "vbroadcastf64x2 0x40(%%rax), %%zmm24")
GEN_test_M_AllMaskModes(VBROADCASTF64x2_1A_256,
      "vbroadcastf64x2 0x40(%%rax), %%ymm24")
GEN_test_M_AllMaskModes(VBROADCASTF32x2_19_512,
      "vbroadcastf32x2 0x40(%%rax), %%zmm24")
GEN_test_M_AllMaskModes(VBROADCASTF32x2_19_256,
      "vbroadcastf32x2 0x40(%%rax), %%ymm24")
GEN_test_M_AllMaskModes(VBROADCASTI32x8_5B_512,
      "vbroadcasti32x8 0x40(%%rax), %%zmm24")
GEN_test_M_AllMaskModes(VBROADCASTI64x2_5A_512,
      "vbroadcasti64x2 0x40(%%rax), %%zmm24")
GEN_test_M_AllMaskModes(VBROADCASTI64x2_5A_256,
      "vbroadcasti64x2 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VBROADCASTI32x2_59_512,
      "vbroadcasti32x2 %%xmm7, %%zmm24",
      "vbroadcasti32x2 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VBROADCASTI32x2_59_256,
      "vbroadcasti32x2 %%xmm7, %%ymm24",
      "vbroadcasti32x2 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VBROADCASTI32x2_59_128,
      "vbroadcasti32x2 %%xmm7, %%xmm24",
      "vbroadcasti32x2 0x40(%%rax), %%xmm24")
GEN_test_M_AllMaskModes(VBROADCASTF32x8_1B_512,
      "vbroadcastf32x8 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPEXPANDQ_89_512,
      "vpexpandq %%zmm7, %%zmm24",
      "vpexpandq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPEXPANDQ_89_256,
      "vpexpandq %%ymm7, %%ymm24",
      "vpexpandq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPEXPANDQ_89_128,
      "vpexpandq %%xmm7, %%xmm24",
      "vpexpandq 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VPEXPANDD_89_512,
      "vpexpandd %%zmm7, %%zmm24",
      "vpexpandd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPEXPANDD_89_256,
      "vpexpandd %%ymm7, %%ymm24",
      "vpexpandd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPEXPANDD_89_128,
      "vpexpandd %%xmm7, %%xmm24",
      "vpexpandd 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VEXPANDPD_88_512,
      "vexpandpd %%zmm7, %%zmm24",
      "vexpandpd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VEXPANDPD_88_256,
      "vexpandpd %%ymm7, %%ymm24",
      "vexpandpd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VEXPANDPD_88_128,
      "vexpandpd %%xmm7, %%xmm24",
      "vexpandpd 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VEXPANDPS_88_512,
      "vexpandps %%zmm7, %%zmm24",
      "vexpandps 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VEXPANDPS_88_256,
      "vexpandps %%ymm7, %%ymm24",
      "vexpandps 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VEXPANDPS_88_128,
      "vexpandps %%xmm7, %%xmm24",
      "vexpandps 0x40(%%rax), %%xmm24")

/* Scatter and gather*/

GEN_test_Ronly(VPGATHERDD_512,
      "leaq _randArray(%%rip), %%r14;"
      /* cut down masked elements to 8 bits, leave unmasked random */
      "vpslld $25,%%zmm18, %%zmm8;"
      "vpsrld $25,%%zmm8, %%zmm8;"
      "vmovdqa32 %%zmm8, %%zmm18%{%%k4%};"
      /* should only try to gather according to the mask */
      "vpgatherdd 0x40(%%r14,%%zmm18,4), %%zmm24%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VPGATHERQD_512,
      "leaq _randArray(%%rip), %%r14;"
      "vpsllq $58, %%zmm18, %%zmm18%{%%k4%};"
      "vpsrlq $58, %%zmm18, %%zmm18%{%%k4%};"
      "vpgatherqd 0x40(%%r14,%%zmm18,4), %%ymm24%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VGATHERQPS_512,
      "leaq _randArray(%%rip), %%r14;"
      "vpsllq $58, %%zmm18, %%zmm18%{%%k4%};"
      "vpsrlq $58, %%zmm18, %%zmm18%{%%k4%};"
      "vgatherqps 0x40(%%r14,%%zmm18,4), %%ymm24%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VGATHERDPS_512,
      "leaq _randArray(%%rip), %%r14;"
      "vpslld $25, %%zmm18, %%zmm18%{%%k4%};"
      "vpsrld $25, %%zmm18, %%zmm18%{%%k4%};"
      "vgatherdps 0x40(%%r14,%%zmm18,4), %%zmm24%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VGATHERDPD_512,
      "leaq _randArray(%%rip), %%r14;"
      "vpslld $25, %%zmm18, %%zmm18%{%%k4%};"
      "vpsrld $25, %%zmm18, %%zmm18%{%%k4%};"
      "vgatherdpd 0x40(%%r14,%%ymm18,4), %%zmm24%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VPGATHERDQ_512,
      "leaq _randArray(%%rip), %%r14;"
      "vpslld $25, %%zmm18, %%zmm18%{%%k4%};"
      "vpsrld $25, %%zmm18, %%zmm18%{%%k4%};"
      "vpgatherdq 0x40(%%r14,%%ymm18,4), %%zmm24%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VPGATHERQQ_512,
      "leaq _randArray(%%rip), %%r14;"
      "vpsllq $58, %%zmm18, %%zmm18%{%%k4%};"
      "vpsrlq $58, %%zmm18, %%zmm18%{%%k4%};"
      "vpgatherqq 0x40(%%r14,%%zmm18,4), %%zmm24%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VGATHERQPD_512,
      "leaq _randArray(%%rip), %%r14;"
      "vpsllq $58, %%zmm18, %%zmm18%{%%k4%};"
      "vpsrlq $58, %%zmm18, %%zmm18%{%%k4%};"
      "vgatherqpd 0x40(%%r14,%%zmm18,4), %%zmm24%{%%k4%};"
      "xorl %%r14d, %%r14d")

GEN_test_Ronly(VPSCATTERDD_512,
      "leaq _randArray(%%rip), %%r14;"
      "vpslld $25, %%zmm18, %%zmm18%{%%k4%};"
      "vpsrld $25, %%zmm18, %%zmm18%{%%k4%};"
      "vpscatterdd %%zmm8, 0x40(%%r14,%%zmm18,4)%{%%k4%};"
      "vpgatherdd 0x40(%%r14,%%zmm18,4), %%zmm24%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VPSCATTERDQ_512,
      "leaq _randArray(%%rip), %%r14;"
      "vpslld $25, %%zmm18, %%zmm18%{%%k4%};"
      "vpsrld $25, %%zmm18, %%zmm18%{%%k4%};"
      "vpscatterdq %%zmm8, 0x40(%%r14,%%ymm18,4)%{%%k4%};"
      "vpgatherdq 0x40(%%r14,%%ymm18,4), %%zmm24%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VPSCATTERQD_512,
      "leaq _randArray(%%rip), %%r14;"
      "vpsllq $58, %%zmm18, %%zmm18%{%%k4%};"
      "vpsrlq $58, %%zmm18, %%zmm18%{%%k4%};"
      "vpscatterqd %%ymm8, 0x40(%%r14,%%zmm18,4)%{%%k4%};"
      "vpgatherqd 0x40(%%r14,%%zmm18,4), %%ymm24%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VPSCATTERQQ_512,
      "leaq _randArray(%%rip), %%r14;"
      "vpsllq $58, %%zmm18, %%zmm18%{%%k4%};"
      "vpsrlq $58, %%zmm18, %%zmm18%{%%k4%};"
      "vpscatterqq %%zmm8, 0x40(%%r14,%%zmm18,4)%{%%k4%};"
      "vpgatherqq 0x40(%%r14,%%zmm18,4), %%zmm24%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VSCATTERDPS_512,
      "leaq _randArray(%%rip), %%r14;"
      "vpslld $25, %%zmm18, %%zmm18%{%%k4%};"
      "vpsrld $25, %%zmm18, %%zmm18%{%%k4%};"
      "vscatterdps %%zmm8, 0x40(%%r14,%%zmm18,4)%{%%k4%};"
      "vgatherdps 0x40(%%r14,%%zmm18,4), %%zmm24%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VSCATTERQPS_512,
      "leaq _randArray(%%rip), %%r14;"
      "vpsllq $58, %%zmm18, %%zmm18%{%%k4%};"
      "vpsrlq $58, %%zmm18, %%zmm18%{%%k4%};"
      "vscatterqps %%ymm8, 0x40(%%r14,%%zmm18,4)%{%%k4%};"
      "vgatherqps 0x40(%%r14,%%zmm18,4), %%ymm24%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VSCATTERDPD_512,
      "leaq _randArray(%%rip), %%r14;"
      "vpslld $25, %%zmm18, %%zmm18%{%%k4%};"
      "vpsrld $25, %%zmm18, %%zmm18%{%%k4%};"
      "vscatterdpd %%zmm8, 0x40(%%r14,%%ymm18,4)%{%%k4%};"
      "vgatherdpd 0x40(%%r14,%%ymm18,4), %%zmm24%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VSCATTERQPD_512,
      "leaq _randArray(%%rip), %%r14;"
      "vpsllq $58, %%zmm18, %%zmm18%{%%k4%};"
      "vpsrlq $58, %%zmm18, %%zmm18%{%%k4%};"
      "vscatterqpd %%zmm8, 0x40(%%r14,%%zmm18,4)%{%%k4%};"
      "vgatherqpd 0x40(%%r14,%%zmm18,4), %%zmm24%{%%k4%};"
      "xorl %%r14d, %%r14d")

/* prefetch */
GEN_test_Ronly(VGATHERPF0DPS_512,
      "leaq _randArray(%%rip), %%r14;"
      "vgatherpf0dps (%%r14,%%zmm18,4)%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VGATHERPF0DPD_512,
      "leaq _randArray(%%rip), %%r14;"
      "vgatherpf0dpd (%%r14,%%ymm18,4)%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VGATHERPF1DPS_512,
      "leaq _randArray(%%rip), %%r14;"
      "vgatherpf1dps (%%r14,%%zmm18,4)%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VGATHERPF1DPD_512,
      "leaq _randArray(%%rip), %%r14;"
      "vgatherpf1dpd (%%r14,%%ymm18,4)%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VGATHERPF0QPS_512,
      "leaq _randArray(%%rip), %%r14;"
      "vgatherpf0qps (%%r14,%%zmm18,4)%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VGATHERPF0QPD_512,
      "leaq _randArray(%%rip), %%r14;"
      "vgatherpf0qpd (%%r14,%%zmm18,4)%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VGATHERPF1QPS_512,
      "leaq _randArray(%%rip), %%r14;"
      "vgatherpf1qps (%%r14,%%zmm18,4)%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VGATHERPF1QPD_512,
      "leaq _randArray(%%rip), %%r14;"
      "vgatherpf1qpd (%%r14,%%zmm18,4)%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VSCATTERPF0DPS_512,
      "leaq _randArray(%%rip), %%r14;"
      "vscatterpf0dps (%%r14,%%zmm18,4)%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VSCATTERPF0DPD_512,
      "leaq _randArray(%%rip), %%r14;"
      "vscatterpf0dpd (%%r14,%%ymm18,4)%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VSCATTERPF1DPS_512,
      "leaq _randArray(%%rip), %%r14;"
      "vscatterpf1dps (%%r14,%%zmm18,4)%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VSCATTERPF1DPD_512,
      "leaq _randArray(%%rip), %%r14;"
      "vscatterpf1dpd (%%r14,%%ymm18,4)%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VSCATTERPF0QPS_512,
      "leaq _randArray(%%rip), %%r14;"
      "vscatterpf0qps (%%r14,%%zmm18,4)%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VSCATTERPF0QPD_512,
      "leaq _randArray(%%rip), %%r14;"
      "vscatterpf0qpd (%%r14,%%zmm18,4)%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VSCATTERPF1QPS_512,
      "leaq _randArray(%%rip), %%r14;"
      "vscatterpf1qps (%%r14,%%zmm18,4)%{%%k4%};"
      "xorl %%r14d, %%r14d")
GEN_test_Ronly(VSCATTERPF1QPD_512,
      "leaq _randArray(%%rip), %%r14;"
      "vscatterpf1qpd (%%r14,%%zmm18,4)%{%%k4%};"
      "xorl %%r14d, %%r14d")

/* Convert */
GEN_test_AllMaskModes(VCVTDQ2PD_E6_512,
      "vcvtdq2pd %%ymm7, %%zmm24",
      "vcvtdq2pd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTDQ2PD_E6_256,
      "vcvtdq2pd %%xmm7, %%ymm24",
      "vcvtdq2pd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTDQ2PD_E6_128,
      "vcvtdq2pd %%xmm7, %%xmm24",
      "vcvtdq2pd 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VCVTPD2UDQ_79_512,
      "vcvtpd2udq %%zmm7, %%ymm24",
      "vcvtpd2udq 0x40(%%rax), %%ymm24")
GEN_test_Ronly(VCVTPD2UDQ_79_256,
      "vcvtpd2udq %%ymm7, %%xmm24")
GEN_test_Ronly(VCVTPD2UDQ_79_128,
      "vcvtpd2udq %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VCVTPS2UDQ_79_512,
      "vcvtps2udq %%zmm7, %%zmm24",
      "vcvtps2udq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTPS2UDQ_79_256,
      "vcvtps2udq %%ymm7, %%ymm24",
      "vcvtps2udq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTPS2UDQ_79_128,
      "vcvtps2udq %%xmm7, %%xmm24",
      "vcvtps2udq 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VCVTTPS2UDQ_78_512,
      "vcvttps2udq %%zmm7, %%zmm24",
      "vcvttps2udq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTTPS2UDQ_78_256,
      "vcvttps2udq %%ymm7, %%ymm24",
      "vcvttps2udq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTTPS2UDQ_78_128,
      "vcvttps2udq %%xmm7, %%xmm24",
      "vcvttps2udq 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VCVTTPD2UDQ_78_512,
      "vcvttpd2udq %%zmm7, %%ymm24",
      "vcvttpd2udq 0x40(%%rax), %%ymm24")
GEN_test_Ronly(VCVTTPD2UDQ_78_256,
      "vcvttpd2udq %%ymm7, %%xmm24")
GEN_test_Ronly(VCVTTPD2UDQ_78_128,
      "vcvttpd2udq %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VCVTUDQ2PD_7A_512,
      "vcvtudq2pd %%ymm7, %%zmm24",
      "vcvtudq2pd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTUDQ2PD_7A_256,
      "vcvtudq2pd %%xmm7, %%ymm24",
      "vcvtudq2pd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTUDQ2PD_7A_128,
      "vcvtudq2pd %%xmm7, %%xmm24",
      "vcvtudq2pd 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VCVTUDQ2PS_7A_512,
      "vcvtudq2ps %%zmm7, %%zmm24",
      "vcvtudq2ps 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTUDQ2PS_7A_256,
      "vcvtudq2ps %%ymm7, %%ymm24",
      "vcvtudq2ps 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTUDQ2PS_7A_128,
      "vcvtudq2ps %%xmm7, %%xmm24",
      "vcvtudq2ps 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VCVTTPS2DQ_5B_512,
      "vcvttps2dq %%zmm7, %%zmm24",
      "vcvttps2dq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTTPS2DQ_5B_256,
      "vcvttps2dq %%ymm7, %%ymm24",
      "vcvttps2dq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTTPS2DQ_5B_128,
      "vcvttps2dq %%xmm7, %%xmm24",
      "vcvttps2dq 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VCVTPS2DQ_5B_512,
      "vcvtps2dq %%zmm7, %%zmm24",
      "vcvtps2dq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTPS2DQ_5B_256,
      "vcvtps2dq %%ymm7, %%ymm24",
      "vcvtps2dq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTPS2DQ_5B_128,
      "vcvtps2dq %%xmm7, %%xmm24",
      "vcvtps2dq 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VCVTDQ2PS_5B_512,
      "vcvtdq2ps %%zmm7, %%zmm24",
      "vcvtdq2ps 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTDQ2PS_5B_256,
      "vcvtdq2ps %%ymm7, %%ymm24",
      "vcvtdq2ps 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTDQ2PS_5B_128,
      "vcvtdq2ps %%xmm7, %%xmm24",
      "vcvtdq2ps 0x40(%%rax), %%xmm24")
GEN_test_RandM(VCVTSS2SI_2D_128,
      "vcvtss2si %%xmm7, %%r14",
      "vcvtss2si 0x40(%%rax), %%r14")
GEN_test_RandM(VCVTSD2SI_2D_128,
      "vcvtsd2si %%xmm7, %%r14",
      "vcvtsd2si 0x40(%%rax), %%r14")
GEN_test_RandM(VCVTTSS2SI_2C_128,
      "vcvttss2si %%xmm7, %%r14",
      "vcvttss2si 0x40(%%rax), %%r14")
GEN_test_RandM(VCVTTSD2SI_2C_128,
      "vcvttsd2si %%xmm7, %%r14",
      "vcvttsd2si 0x40(%%rax), %%r14")
GEN_test_RandM(VCVTSI2SD_2A_128,
      "vcvtsi2sd %%r14, %%xmm8, %%xmm24",
      "vcvtsi2sd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VCVTSD2SS_5A_128,
      "vcvtsd2ss %%xmm7, %%xmm8, %%xmm24",
      "vcvtsd2ss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_RandM(VCVTSI2SS_2A_128,
      "vcvtsi2ss %%r14, %%xmm8, %%xmm24",
      "vcvtsi2ss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_RandM(VCVTSI2SS32_2A_128,
      "vcvtsi2ss %%r14d, %%xmm24, %%xmm8",
      "vcvtsi2ss 0x40(%%rax), %%xmm24, %%xmm8")
GEN_test_AllMaskModes(VCVTUQQ2PD_7A_512,
      "vcvtuqq2pd %%zmm7, %%zmm24",
      "vcvtuqq2pd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTUQQ2PD_7A_256,
      "vcvtuqq2pd %%ymm7, %%ymm24",
      "vcvtuqq2pd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTUQQ2PD_7A_128,
      "vcvtuqq2pd %%xmm7, %%xmm24",
      "vcvtuqq2pd 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VCVTPD2UQQ_79_512,
      "vcvtpd2uqq %%zmm7, %%zmm24",
      "vcvtpd2uqq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTPD2UQQ_79_256,
      "vcvtpd2uqq %%ymm7, %%ymm24",
      "vcvtpd2uqq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTPD2UQQ_79_128,
      "vcvtpd2uqq %%xmm7, %%xmm24",
      "vcvtpd2uqq 0x40(%%rax), %%xmm24")
GEN_test_RandM(VCVTUSI2SS_7B_128,
      "vcvtusi2ss %%r14, %%xmm8, %%xmm24",
      "vcvtusi2ss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_RandM(VCVTUSI2SD_7B_128,
      "vcvtusi2sd %%r14, %%xmm8, %%xmm24",
      "vcvtusi2sd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_RandM(VCVTSS2USI_79_128,
      "vcvtss2usi %%xmm7, %%r14",
      "vcvtss2usi 0x40(%%rax), %%r14")
GEN_test_RandM(VCVTSD2USI_79_128,
      "vcvtsd2usi %%xmm7, %%r14",
      "vcvtsd2usi 0x40(%%rax), %%r14")
GEN_test_AllMaskModes(VCVTSS2SD_5A_128,
      "vcvtss2sd %%xmm7, %%xmm8, %%xmm24",
      "vcvtss2sd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VCVTPD2DQ_E6_512,
      "vcvtpd2dq %%zmm24, %%ymm8",
      "vcvtpd2dq 0x40(%%rax), %%ymm24")
GEN_test_R_AllMaskModes(VCVTPD2DQ_E6_256,
      "vcvtpd2dq %%ymm7, %%xmm24")
GEN_test_R_AllMaskModes(VCVTPD2DQ_E6_128,
      "vcvtpd2dq %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VCVTPD2PS_5A_512,
      "vcvtpd2ps %%zmm24, %%ymm8",
      "vcvtpd2ps 0x40(%%rax), %%ymm24")
GEN_test_R_AllMaskModes(VCVTPD2PS_5A_256,
      "vcvtpd2ps %%ymm7, %%xmm24")
GEN_test_R_AllMaskModes(VCVTPD2PS_5A_128,
      "vcvtpd2ps %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VCVTPS2PD_5A_512,
      "vcvtps2pd %%ymm24, %%zmm8",
      "vcvtps2pd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTPS2PD_5A_256,
      "vcvtps2pd %%xmm7, %%ymm24",
      "vcvtps2pd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTPS2PD_5A_128,
      "vcvtps2pd %%xmm7, %%xmm24",
      "vcvtps2pd 0x40(%%rax), %%xmm24")
GEN_test_MemMaskModes(VCVTPS2PH_1D_512,
      "vcvtps2ph $0x0, %%zmm7, %%ymm24",
      "vcvtps2ph $0x1, %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VCVTPS2PH_1D_256,
      "vcvtps2ph $0x2, %%ymm7, %%xmm24",
      "vcvtps2ph $0x0, %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VCVTPS2PH_1D_128,
      "vcvtps2ph $0x1, %%xmm7, %%xmm24",
      "vcvtps2ph $0x2, %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VCVTPH2PS_13_512,
      "vcvtph2ps %%ymm7, %%zmm24",
      "vcvtph2ps 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTPH2PS_13_256,
      "vcvtph2ps %%xmm7, %%ymm24",
      "vcvtph2ps 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTPH2PS_13_128,
      "vcvtph2ps %%xmm7, %%xmm24",
      "vcvtph2ps 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VCVTQQ2PD_E6_512,
      "vcvtqq2pd %%zmm7, %%zmm24",
      "vcvtqq2pd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTQQ2PD_E6_256,
      "vcvtqq2pd %%ymm7, %%ymm24",
      "vcvtqq2pd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTQQ2PD_E6_128,
      "vcvtqq2pd %%xmm7, %%xmm24",
      "vcvtqq2pd 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VCVTPD2QQ_7B_512,
      "vcvtpd2qq %%zmm7, %%zmm24",
      "vcvtpd2qq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTPD2QQ_7B_256,
      "vcvtpd2qq %%ymm7, %%ymm24",
      "vcvtpd2qq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTPD2QQ_7B_128,
      "vcvtpd2qq %%xmm7, %%xmm24",
      "vcvtpd2qq 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VCVTPS2QQ_7B_512,
      "vcvtps2qq %%ymm7, %%zmm24",
      "vcvtps2qq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTPS2QQ_7B_256,
      "vcvtps2qq %%xmm7, %%ymm24",
      "vcvtps2qq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTPS2QQ_7B_128,
      "vcvtps2qq %%xmm7, %%xmm24",
      "vcvtps2qq 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VCVTUQQ2PS_7A_512,
      "vcvtuqq2ps %%zmm7, %%ymm24",
      "vcvtuqq2ps 0x40(%%rax), %%ymm24")
GEN_test_R_AllMaskModes(VCVTUQQ2PS_7A_256,
      "vcvtuqq2ps %%ymm7, %%xmm24")
GEN_test_R_AllMaskModes(VCVTUQQ2PS_7A_128,
      "vcvtuqq2ps %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VCVTTPD2QQ_7A_512,
      "vcvttpd2qq %%zmm7, %%zmm24",
      "vcvttpd2qq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTTPD2QQ_7A_256,
      "vcvttpd2qq %%ymm7, %%ymm24",
      "vcvttpd2qq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTTPD2QQ_7A_128,
      "vcvttpd2qq %%xmm7, %%xmm24",
      "vcvttpd2qq 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VCVTTPS2QQ_7A_512,
      "vcvttps2qq %%ymm7, %%zmm24",
      "vcvttps2qq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTTPS2QQ_7A_256,
      "vcvttps2qq %%xmm7, %%ymm24",
      "vcvttps2qq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTTPS2QQ_7A_128,
      "vcvttps2qq %%xmm7, %%xmm24",
      "vcvttps2qq 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VCVTPS2UQQ_79_512,
      "vcvtps2uqq %%ymm7, %%zmm24",
      "vcvtps2uqq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTPS2UQQ_79_256,
      "vcvtps2uqq %%xmm7, %%ymm24",
      "vcvtps2uqq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTPS2UQQ_79_128,
      "vcvtps2uqq %%xmm7, %%xmm24",
      "vcvtps2uqq 0x40(%%rax), %%xmm24")
GEN_test_RandM(VCVTTSS2USI_78_128,
      "vcvttss2usi %%xmm7, %%r14",
      "vcvttss2usi 0x40(%%rax), %%r14")
GEN_test_RandM(VCVTTSD2USI_78_128,
      "vcvttsd2usi %%xmm7, %%r14",
      "vcvttsd2usi 0x40(%%rax), %%r14")
GEN_test_AllMaskModes(VCVTTPD2UQQ_78_512,
      "vcvttpd2uqq %%zmm7, %%zmm24",
      "vcvttpd2uqq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTTPD2UQQ_78_256,
      "vcvttpd2uqq %%ymm7, %%ymm24",
      "vcvttpd2uqq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTTPD2UQQ_78_128,
      "vcvttpd2uqq %%xmm7, %%xmm24",
      "vcvttpd2uqq 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VCVTTPS2UQQ_78_512,
      "vcvttps2uqq %%ymm7, %%zmm24",
      "vcvttps2uqq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VCVTTPS2UQQ_78_256,
      "vcvttps2uqq %%xmm7, %%ymm24",
      "vcvttps2uqq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VCVTTPS2UQQ_78_128,
      "vcvttps2uqq %%xmm7, %%xmm24",
      "vcvttps2uqq 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VCVTQQ2PS_5B_512,
      "vcvtqq2ps %%zmm7, %%ymm24",
      "vcvtqq2ps 0x40(%%rax), %%ymm24")
GEN_test_R_AllMaskModes(VCVTQQ2PS_5B_256,
      "vcvtqq2ps %%ymm7, %%xmm24")
GEN_test_R_AllMaskModes(VCVTQQ2PS_5B_128,
      "vcvtqq2ps %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VCVTTPD2DQ_E6_512,
      "vcvttpd2dq %%zmm24, %%ymm8",
      "vcvttpd2dq 0x40(%%rax), %%ymm24")
GEN_test_R_AllMaskModes(VCVTTPD2DQ_E6_256,
      "vcvttpd2dq %%ymm7, %%xmm24")
GEN_test_R_AllMaskModes(VCVTTPD2DQ_E6_128,
      "vcvttpd2dq %%xmm7, %%xmm24")


/* Conflict detection */
GEN_test_AllMaskModes(VPCONFLICTQ_C4_512,
      "vpconflictq %%zmm7, %%zmm24",
      "vpconflictq 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPCONFLICTQ_C4_256,
      "vpconflictq %%ymm7, %%ymm24",
      "vpconflictq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPCONFLICTQ_C4_128,
      "vpconflictq %%xmm7, %%xmm24",
      "vpconflictq 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VPCONFLICTD_C4_512,
      "vpconflictd %%zmm7, %%zmm24",
      "vpconflictd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPCONFLICTD_C4_256,
      "vpconflictd %%ymm7, %%ymm24",
      "vpconflictd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPCONFLICTD_C4_128,
      "vpconflictd %%xmm7, %%xmm24",
      "vpconflictd 0x40(%%rax), %%xmm24")
GEN_test_Ronly(VPLZCNTQ_44_512,
      "vpsrlq $60, %%zmm24, %%zmm24;"
      "vpsrlq %%xmm24, %%zmm7, %%zmm7;"
      "vplzcntq %%zmm7, %%zmm24")
GEN_test_Ronly(VPLZCNTD_44_512,
      "vpsrlq $60, %%zmm24, %%zmm24;"
      "vpsrld %%xmm24, %%zmm7, %%zmm7;"
      "vplzcntd %%zmm7, %%zmm24")
GEN_test_AllMaskModes(VPLZCNTQ_44_256,
      "vplzcntq %%ymm7, %%ymm24",
      "vplzcntq 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPLZCNTQ_44_128,
      "vplzcntq %%xmm7, %%xmm24",
      "vplzcntq 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VPLZCNTD_44_256,
      "vplzcntd %%ymm7, %%ymm24",
      "vplzcntd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPLZCNTD_44_128,
      "vplzcntd %%xmm7, %%xmm24",
      "vplzcntd 0x40(%%rax), %%xmm24")

/* Permute */
GEN_test_AllMaskModes(VPERMI2D_76_512,
      "vpermi2d %%zmm7, %%zmm8, %%zmm24",
      "vpermi2d 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMI2D_76_256,
      "vpermi2d %%ymm7, %%ymm8, %%ymm24",
      "vpermi2d 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMI2D_76_128,
      "vpermi2d %%xmm7, %%xmm8, %%xmm24",
      "vpermi2d 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPERMQ_00_512,
      "vpermq $0x7C, %%zmm24, %%zmm8",
      "vpermq $0xBA, 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPERMQ_36_512,
      "vpermq %%zmm24, %%zmm8, %%zmm7",
      "vpermq 0x40(%%rax), %%zmm24, %%zmm8")
GEN_test_AllMaskModes(VPERMT2PD_7F_512,
      "vpermt2pd %%zmm7, %%zmm8, %%zmm24",
      "vpermt2pd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMT2PD_7F_256,
      "vpermt2pd %%ymm7, %%ymm8, %%ymm24",
      "vpermt2pd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMT2PD_7F_128,
      "vpermt2pd %%xmm7, %%xmm8, %%xmm24",
      "vpermt2pd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPERMI2PD_77_512,
      "vpermi2pd %%zmm7, %%zmm8, %%zmm24",
      "vpermi2pd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMI2PD_77_256,
      "vpermi2pd %%ymm7, %%ymm8, %%ymm24",
      "vpermi2pd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMI2PD_77_128,
      "vpermi2pd %%xmm7, %%xmm8, %%xmm24",
      "vpermi2pd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPERMI2PS_77_512,
      "vpermi2ps %%zmm7, %%zmm8, %%zmm24",
      "vpermi2ps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMI2PS_77_256,
      "vpermi2ps %%ymm7, %%ymm8, %%ymm24",
      "vpermi2ps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMI2PS_77_128,
      "vpermi2ps %%xmm7, %%xmm8, %%xmm24",
      "vpermi2ps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPERMD_36_512,
      "vpermd %%zmm7, %%zmm8, %%zmm24",
      "vpermd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMPD_16_512,
      "vpermpd %%zmm7, %%zmm8, %%zmm24",
      "vpermpd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMT2PS_7F_512,
      "vpermt2ps %%zmm7, %%zmm8, %%zmm24",
      "vpermt2ps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMT2PS_7F_256,
      "vpermt2ps %%ymm7, %%ymm8, %%ymm24",
      "vpermt2ps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMT2PS_7F_128,
      "vpermt2ps %%xmm7, %%xmm8, %%xmm24",
      "vpermt2ps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPERMT2Q_7E_512,
      "vpermt2q %%zmm7, %%zmm8, %%zmm24",
      "vpermt2q 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMT2Q_7E_256,
      "vpermt2q %%ymm7, %%ymm8, %%ymm24",
      "vpermt2q 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMT2Q_7E_128,
      "vpermt2q %%xmm7, %%xmm8, %%xmm24",
      "vpermt2q 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPERMT2D_7E_512,
      "vpermt2d %%zmm7, %%zmm8, %%zmm24",
      "vpermt2d 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMT2D_7E_256,
      "vpermt2d %%ymm7, %%ymm8, %%ymm24",
      "vpermt2d 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMT2D_7E_128,
      "vpermt2d %%xmm7, %%xmm8, %%xmm24",
      "vpermt2d 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPERMI2Q_76_512,
      "vpermi2q %%zmm7, %%zmm8, %%zmm24",
      "vpermi2q 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMI2Q_76_256,
      "vpermi2q %%ymm7, %%ymm8, %%ymm24",
      "vpermi2q 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMI2Q_76_128,
      "vpermi2q %%xmm7, %%xmm8, %%xmm24",
      "vpermi2q 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPERMQ_36_256,
      "vpermq %%ymm7, %%ymm8, %%ymm24",
      "vpermq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMD_36_256,
      "vpermd %%ymm7, %%ymm8, %%ymm24",
      "vpermd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMPD_16_256,
      "vpermpd %%ymm7, %%ymm8, %%ymm24",
      "vpermpd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMPS_16_512,
      "vpermps %%zmm7, %%zmm8, %%zmm24",
      "vpermps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMPS_16_256,
      "vpermps %%ymm7, %%ymm8, %%ymm24",
      "vpermps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMILPD_0D_512,
      "vpermilpd %%zmm7, %%zmm8, %%zmm24",
      "vpermilpd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMILPD_0D_256,
      "vpermilpd %%ymm7, %%ymm8, %%ymm24",
      "vpermilpd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMILPD_0D_128,
      "vpermilpd %%xmm7, %%xmm8, %%xmm24",
      "vpermilpd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPERMILPS_0C_512,
      "vpermilps %%zmm7, %%zmm8, %%zmm24",
      "vpermilps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMILPS_0C_256,
      "vpermilps %%ymm7, %%ymm8, %%ymm24",
      "vpermilps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMILPS_0C_128,
      "vpermilps %%xmm7, %%xmm8, %%xmm24",
      "vpermilps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPERMILPD_05_512,
      "vpermilpd $0xAB, %%zmm7, %%zmm24",
      "vpermilpd $0xAB, 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPERMILPD_05_256,
      "vpermilpd $0xAB, %%ymm7, %%ymm24",
      "vpermilpd $0xAB, 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPERMILPD_05_128,
      "vpermilpd $0xAB, %%xmm7, %%xmm24",
      "vpermilpd $0xAB, 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VPERMILPS_04_512,
      "vpermilps $0xAB, %%zmm7, %%zmm24",
      "vpermilps $0xAB, 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPERMILPS_04_256,
      "vpermilps $0xAB, %%ymm7, %%ymm24",
      "vpermilps $0xAB, 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPERMILPS_04_128,
      "vpermilps $0xAB, %%xmm7, %%xmm24",
      "vpermilps $0xAB, 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VPERMPD_01_512,
      "vpermpd $0xAB, %%zmm7, %%zmm24",
      "vpermpd $0xAB, 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPERMPD_01_256,
      "vpermpd $0xAB, %%ymm7, %%ymm24",
      "vpermpd $0xAB, 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPERMQ_00_256,
      "vpermq $0xAB, %%ymm7, %%ymm24",
      "vpermq $0xAB, 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPERMW_8D_512,
      "vpermw %%zmm8, %%zmm7, %%zmm24",
      "vpermw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMW_8D_256,
      "vpermw %%ymm8, %%ymm7, %%ymm24",
      "vpermw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMW_8D_128,
      "vpermw %%xmm8, %%xmm7, %%xmm24",
      "vpermw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPERMB_8D_512,
      "vpermb %%zmm8, %%zmm7, %%zmm24",
      "vpermb 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMB_8D_256,
      "vpermb %%ymm8, %%ymm7, %%ymm24",
      "vpermb 0x40(%%rax),  %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMB_8D_128,
      "vpermb %%xmm8, %%xmm7, %%xmm24",
      "vpermb 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPERMT2W_7D_512,
      "vpermt2w %%zmm7, %%zmm8, %%zmm24",
      "vpermt2w 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMT2W_7D_256,
      "vpermt2w %%ymm7, %%ymm8, %%ymm24",
      "vpermt2w 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMT2W_7D_128,
      "vpermt2w %%xmm7, %%xmm8, %%xmm24",
      "vpermt2w 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPERMT2B_7D_512,
      "vpermt2b %%zmm7, %%zmm8, %%zmm24",
      "vpermt2b 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMT2B_7D_256,
      "vpermt2b %%ymm7, %%ymm8, %%ymm24",
      "vpermt2b 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMT2B_7D_128,
      "vpermt2b %%xmm7, %%xmm8, %%xmm24",
      "vpermt2b 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPERMI2W_75_512,
      "vpermi2w %%zmm7, %%zmm8, %%zmm24",
      "vpermi2w 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMI2W_75_256,
      "vpermi2w %%ymm7, %%ymm8, %%ymm24",
      "vpermi2w 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMI2W_75_128,
      "vpermi2w %%xmm7, %%xmm8, %%xmm24",
      "vpermi2w 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPERMI2B_75_512,
      "vpermi2b %%zmm7, %%zmm8, %%zmm24",
      "vpermi2b 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPERMI2B_75_256,
      "vpermi2b %%ymm7, %%ymm8, %%ymm24",
      "vpermi2b 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPERMI2B_75_128,
      "vpermi2b %%xmm7, %%xmm8, %%xmm24",
      "vpermi2b 0x40(%%rax), %%xmm8, %%xmm24")

/* FMA */
GEN_test_AllMaskModes(VFNMSUB231SD_BF_128,
      "vfnmsub231sd %%xmm7, %%xmm8, %%xmm24",
      "vfnmsub231sd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMSUB231SS_BF_128,
      "vfnmsub231ss %%xmm7, %%xmm8, %%xmm24",
      "vfnmsub231ss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMSUB231PD_BE_512,
      "vfnmsub231pd %%zmm7, %%zmm8, %%zmm24",
      "vfnmsub231pd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFNMSUB231PD_BE_256,
      "vfnmsub231pd %%ymm7, %%ymm8, %%ymm24",
      "vfnmsub231pd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFNMSUB231PD_BE_128,
      "vfnmsub231pd %%xmm7, %%xmm8, %%xmm24",
      "vfnmsub231pd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMSUB231PS_BE_512,
      "vfnmsub231ps %%zmm7, %%zmm8, %%zmm24",
      "vfnmsub231ps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFNMSUB231PS_BE_256,
      "vfnmsub231ps %%ymm7, %%ymm8, %%ymm24",
      "vfnmsub231ps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFNMSUB231PS_BE_128,
      "vfnmsub231ps %%xmm7, %%xmm8, %%xmm24",
      "vfnmsub231ps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMADD231SD_BD_128,
      "vfnmadd231sd %%xmm7, %%xmm8, %%xmm24",
      "vfnmadd231sd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMADD231SS_BD_128,
      "vfnmadd231ss %%xmm7, %%xmm8, %%xmm24",
      "vfnmadd231ss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMADD231PD_BC_512,
      "vfnmadd231pd %%zmm7, %%zmm8, %%zmm24",
      "vfnmadd231pd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFNMADD231PD_BC_256,
      "vfnmadd231pd %%ymm7, %%ymm8, %%ymm24",
      "vfnmadd231pd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFNMADD231PD_BC_128,
      "vfnmadd231pd %%xmm7, %%xmm8, %%xmm24",
      "vfnmadd231pd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMADD231PS_BC_512,
      "vfnmadd231ps %%zmm7, %%zmm8, %%zmm24",
      "vfnmadd231ps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFNMADD231PS_BC_256,
      "vfnmadd231ps %%ymm7, %%ymm8, %%ymm24",
      "vfnmadd231ps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFNMADD231PS_BC_128,
      "vfnmadd231ps %%xmm7, %%xmm8, %%xmm24",
      "vfnmadd231ps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMSUB231SD_BB_128,
      "vfmsub231sd %%xmm7, %%xmm8, %%xmm24",
      "vfmsub231sd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMSUB231SS_BB_128,
      "vfmsub231ss %%xmm7, %%xmm8, %%xmm24",
      "vfmsub231ss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMSUB231PD_BA_512,
      "vfmsub231pd %%zmm7, %%zmm8, %%zmm24",
      "vfmsub231pd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFMSUB231PD_BA_256,
      "vfmsub231pd %%ymm7, %%ymm8, %%ymm24",
      "vfmsub231pd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFMSUB231PD_BA_128,
      "vfmsub231pd %%xmm7, %%xmm8, %%xmm24",
      "vfmsub231pd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMSUB231PS_BA_512,
      "vfmsub231ps %%zmm7, %%zmm8, %%zmm24",
      "vfmsub231ps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFMSUB231PS_BA_256,
      "vfmsub231ps %%ymm7, %%ymm8, %%ymm24",
      "vfmsub231ps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFMSUB231PS_BA_128,
      "vfmsub231ps %%xmm7, %%xmm8, %%xmm24",
      "vfmsub231ps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMADD231SD_B9_128,
      "vfmadd231sd %%xmm7, %%xmm8, %%xmm24",
      "vfmadd231sd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMADD231SS_B9_128,
      "vfmadd231ss %%xmm7, %%xmm8, %%xmm24",
      "vfmadd231ss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMADD231PD_B8_512,
      "vfmadd231pd %%zmm7, %%zmm8, %%zmm24",
      "vfmadd231pd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFMADD231PD_B8_256,
      "vfmadd231pd %%ymm7, %%ymm8, %%ymm24",
      "vfmadd231pd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFMADD231PD_B8_128,
      "vfmadd231pd %%xmm7, %%xmm8, %%xmm24",
      "vfmadd231pd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMADD231PS_B8_512,
      "vfmadd231ps %%zmm7, %%zmm8, %%zmm24",
      "vfmadd231ps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFMADD231PS_B8_256,
      "vfmadd231ps %%ymm7, %%ymm8, %%ymm24",
      "vfmadd231ps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFMADD231PS_B8_128,
      "vfmadd231ps %%xmm7, %%xmm8, %%xmm24",
      "vfmadd231ps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMSUBADD231PD_B7_512,
      "vfmsubadd231pd %%zmm7, %%zmm8, %%zmm24",
      "vfmsubadd231pd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFMSUBADD231PD_B7_256,
      "vfmsubadd231pd %%ymm7, %%ymm8, %%ymm24",
      "vfmsubadd231pd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFMSUBADD231PD_B7_128,
      "vfmsubadd231pd %%xmm7, %%xmm8, %%xmm24",
      "vfmsubadd231pd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMSUBADD231PS_B7_512,
      "vfmsubadd231ps %%zmm7, %%zmm8, %%zmm24",
      "vfmsubadd231ps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFMSUBADD231PS_B7_256,
      "vfmsubadd231ps %%ymm7, %%ymm8, %%ymm24",
      "vfmsubadd231ps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFMSUBADD231PS_B7_128,
      "vfmsubadd231ps %%xmm7, %%xmm8, %%xmm24",
      "vfmsubadd231ps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMADDSUB231PD_B6_512,
      "vfmaddsub231pd %%zmm7, %%zmm8, %%zmm24",
      "vfmaddsub231pd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFMADDSUB231PD_B6_256,
      "vfmaddsub231pd %%ymm7, %%ymm8, %%ymm24",
      "vfmaddsub231pd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFMADDSUB231PD_B6_128,
      "vfmaddsub231pd %%xmm7, %%xmm8, %%xmm24",
      "vfmaddsub231pd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMADDSUB231PS_B6_512,
      "vfmaddsub231ps %%zmm7, %%zmm8, %%zmm24",
      "vfmaddsub231ps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFMADDSUB231PS_B6_256,
      "vfmaddsub231ps %%ymm7, %%ymm8, %%ymm24",
      "vfmaddsub231ps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFMADDSUB231PS_B6_128,
      "vfmaddsub231ps %%xmm7, %%xmm8, %%xmm24",
      "vfmaddsub231ps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMSUB213SD_AF_128,
      "vfnmsub213sd %%xmm7, %%xmm8, %%xmm24",
      "vfnmsub213sd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMSUB213SS_AF_128,
      "vfnmsub213ss %%xmm7, %%xmm8, %%xmm24",
      "vfnmsub213ss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMSUB213PD_AE_512,
      "vfnmsub213pd %%zmm7, %%zmm8, %%zmm24",
      "vfnmsub213pd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFNMSUB213PD_AE_256,
      "vfnmsub213pd %%ymm7, %%ymm8, %%ymm24",
      "vfnmsub213pd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFNMSUB213PD_AE_128,
      "vfnmsub213pd %%xmm7, %%xmm8, %%xmm24",
      "vfnmsub213pd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMSUB213PS_AE_512,
      "vfnmsub213ps %%zmm7, %%zmm8, %%zmm24",
      "vfnmsub213ps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFNMSUB213PS_AE_256,
      "vfnmsub213ps %%ymm7, %%ymm8, %%ymm24",
      "vfnmsub213ps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFNMSUB213PS_AE_128,
      "vfnmsub213ps %%xmm7, %%xmm8, %%xmm24",
      "vfnmsub213ps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMADD213SD_AD_128,
      "vfnmadd213sd %%xmm7, %%xmm8, %%xmm24",
      "vfnmadd213sd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMADD213SS_AD_128,
      "vfnmadd213ss %%xmm7, %%xmm8, %%xmm24",
      "vfnmadd213ss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMADD213PD_AC_512,
      "vfnmadd213pd %%zmm7, %%zmm8, %%zmm24",
      "vfnmadd213pd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFNMADD213PD_AC_256,
      "vfnmadd213pd %%ymm7, %%ymm8, %%ymm24",
      "vfnmadd213pd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFNMADD213PD_AC_128,
      "vfnmadd213pd %%xmm7, %%xmm8, %%xmm24",
      "vfnmadd213pd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMADD213PS_AC_512,
      "vfnmadd213ps %%zmm7, %%zmm8, %%zmm24",
      "vfnmadd213ps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFNMADD213PS_AC_256,
      "vfnmadd213ps %%ymm7, %%ymm8, %%ymm24",
      "vfnmadd213ps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFNMADD213PS_AC_128,
      "vfnmadd213ps %%xmm7, %%xmm8, %%xmm24",
      "vfnmadd213ps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMSUB213SD_AB_128,
      "vfmsub213sd %%xmm7, %%xmm8, %%xmm24",
      "vfmsub213sd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMSUB213SS_AB_128,
      "vfmsub213ss %%xmm7, %%xmm8, %%xmm24",
      "vfmsub213ss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMSUB213PD_AA_512,
      "vfmsub213pd %%zmm7, %%zmm8, %%zmm24",
      "vfmsub213pd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFMSUB213PD_AA_256,
      "vfmsub213pd %%ymm7, %%ymm8, %%ymm24",
      "vfmsub213pd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFMSUB213PD_AA_128,
      "vfmsub213pd %%xmm7, %%xmm8, %%xmm24",
      "vfmsub213pd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMSUB213PS_AA_512,
      "vfmsub213ps %%zmm7, %%zmm8, %%zmm24",
      "vfmsub213ps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFMSUB213PS_AA_256,
      "vfmsub213ps %%ymm7, %%ymm8, %%ymm24",
      "vfmsub213ps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFMSUB213PS_AA_128,
      "vfmsub213ps %%xmm7, %%xmm8, %%xmm24",
      "vfmsub213ps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMADD213SD_A9_128,
      "vfmadd213sd %%xmm7, %%xmm8, %%xmm24",
      "vfmadd213sd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMADD213SS_A9_128,
      "vfmadd213ss %%xmm7, %%xmm8, %%xmm24",
      "vfmadd213ss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMADD213PD_A8_512,
      "vfmadd213pd %%zmm7, %%zmm8, %%zmm24",
      "vfmadd213pd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFMADD213PD_A8_256,
      "vfmadd213pd %%ymm7, %%ymm8, %%ymm24",
      "vfmadd213pd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFMADD213PD_A8_128,
      "vfmadd213pd %%xmm7, %%xmm8, %%xmm24",
      "vfmadd213pd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMADD213PS_A8_512,
      "vfmadd213ps %%zmm7, %%zmm8, %%zmm24",
      "vfmadd213ps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFMADD213PS_A8_256,
      "vfmadd213ps %%ymm7, %%ymm8, %%ymm24",
      "vfmadd213ps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFMADD213PS_A8_128,
      "vfmadd213ps %%xmm7, %%xmm8, %%xmm24",
      "vfmadd213ps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMSUBADD213PD_A7_512,
      "vfmsubadd213pd %%zmm7, %%zmm8, %%zmm24",
      "vfmsubadd213pd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFMSUBADD213PD_A7_256,
      "vfmsubadd213pd %%ymm7, %%ymm8, %%ymm24",
      "vfmsubadd213pd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFMSUBADD213PD_A7_128,
      "vfmsubadd213pd %%xmm7, %%xmm8, %%xmm24",
      "vfmsubadd213pd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMSUBADD213PS_A7_512,
      "vfmsubadd213ps %%zmm7, %%zmm8, %%zmm24",
      "vfmsubadd213ps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFMSUBADD213PS_A7_256,
      "vfmsubadd213ps %%ymm7, %%ymm8, %%ymm24",
      "vfmsubadd213ps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFMSUBADD213PS_A7_128,
      "vfmsubadd213ps %%xmm7, %%xmm8, %%xmm24",
      "vfmsubadd213ps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMADDSUB213PD_A6_512,
      "vfmaddsub213pd %%zmm7, %%zmm8, %%zmm24",
      "vfmaddsub213pd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFMADDSUB213PD_A6_256,
      "vfmaddsub213pd %%ymm7, %%ymm8, %%ymm24",
      "vfmaddsub213pd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFMADDSUB213PD_A6_128,
      "vfmaddsub213pd %%xmm7, %%xmm8, %%xmm24",
      "vfmaddsub213pd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFMADDSUB213PS_A6_512,
      "vfmaddsub213ps %%zmm7, %%zmm8, %%zmm24",
      "vfmaddsub213ps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFMADDSUB213PS_A6_256,
      "vfmaddsub213ps %%ymm7, %%ymm8, %%ymm24",
      "vfmaddsub213ps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFMADDSUB213PS_A6_128,
      "vfmaddsub213ps %%xmm7, %%xmm8, %%xmm24",
      "vfmaddsub213ps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMSUB132SD_9F_128,
      "vfnmsub132sd %%xmm8, %%xmm7, %%xmm24",
      "vfnmsub132sd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFNMSUB132SS_9F_128,
      "vfnmsub132ss %%xmm8, %%xmm7, %%xmm24",
      "vfnmsub132ss 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFNMSUB132PD_9E_512,
      "vfnmsub132pd %%zmm8, %%zmm7, %%zmm24",
      "vfnmsub132pd 0x40(%%rax), %%zmm7, %%zmm24")
GEN_test_AllMaskModes(VFNMSUB132PD_9E_256,
      "vfnmsub132pd %%ymm8, %%ymm7, %%ymm24",
      "vfnmsub132pd 0x40(%%rax), %%ymm7, %%ymm24")
GEN_test_AllMaskModes(VFNMSUB132PD_9E_128,
      "vfnmsub132pd %%xmm8, %%xmm7, %%xmm24",
      "vfnmsub132pd 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFNMSUB132PS_9E_512,
      "vfnmsub132ps %%zmm8, %%zmm7, %%zmm24",
      "vfnmsub132ps 0x40(%%rax), %%zmm7, %%zmm24")
GEN_test_AllMaskModes(VFNMSUB132PS_9E_256,
      "vfnmsub132ps %%ymm8, %%ymm7, %%ymm24",
      "vfnmsub132ps 0x40(%%rax), %%ymm7, %%ymm24")
GEN_test_AllMaskModes(VFNMSUB132PS_9E_128,
      "vfnmsub132ps %%xmm8, %%xmm7, %%xmm24",
      "vfnmsub132ps 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFNMADD132SD_9D_128,
      "vfnmadd132sd %%xmm8, %%xmm7, %%xmm24",
      "vfnmadd132sd 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFNMADD132SS_9D_128,
      "vfnmadd132ss %%xmm8, %%xmm7, %%xmm24",
      "vfnmadd132ss 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFNMADD132PD_9C_512,
      "vfnmadd132pd %%zmm8, %%zmm7, %%zmm24",
      "vfnmadd132pd 0x40(%%rax), %%zmm7, %%zmm24")
GEN_test_AllMaskModes(VFNMADD132PD_9C_256,
      "vfnmadd132pd %%ymm8, %%ymm7, %%ymm24",
      "vfnmadd132pd 0x40(%%rax), %%ymm7, %%ymm24")
GEN_test_AllMaskModes(VFNMADD132PD_9C_128,
      "vfnmadd132pd %%xmm8, %%xmm7, %%xmm24",
      "vfnmadd132pd 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFNMADD132PS_9C_512,
      "vfnmadd132ps %%zmm8, %%zmm7, %%zmm24",
      "vfnmadd132ps 0x40(%%rax), %%zmm7, %%zmm24")
GEN_test_AllMaskModes(VFNMADD132PS_9C_256,
      "vfnmadd132ps %%ymm8, %%ymm7, %%ymm24",
      "vfnmadd132ps 0x40(%%rax), %%ymm7, %%ymm24")
GEN_test_AllMaskModes(VFNMADD132PS_9C_128,
      "vfnmadd132ps %%xmm8, %%xmm7, %%xmm24",
      "vfnmadd132ps 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFMSUB132SD_9B_128,
      "vfmsub132sd %%xmm8, %%xmm7, %%xmm24",
      "vfmsub132sd 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFMSUB132SS_9B_128,
      "vfmsub132ss %%xmm8, %%xmm7, %%xmm24",
      "vfmsub132ss 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFMSUB132PD_9A_512,
      "vfmsub132pd %%zmm8, %%zmm7, %%zmm24",
      "vfmsub132pd 0x40(%%rax), %%zmm7, %%zmm24")
GEN_test_AllMaskModes(VFMSUB132PD_9A_256,
      "vfmsub132pd %%ymm8, %%ymm7, %%ymm24",
      "vfmsub132pd 0x40(%%rax), %%ymm7, %%ymm24")
GEN_test_AllMaskModes(VFMSUB132PD_9A_128,
      "vfmsub132pd %%xmm8, %%xmm7, %%xmm24",
      "vfmsub132pd 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFMSUB132PS_9A_512,
      "vfmsub132ps %%zmm8, %%zmm7, %%zmm24",
      "vfmsub132ps 0x40(%%rax), %%zmm7, %%zmm24")
GEN_test_AllMaskModes(VFMSUB132PS_9A_256,
      "vfmsub132ps %%ymm8, %%ymm7, %%ymm24",
      "vfmsub132ps 0x40(%%rax), %%ymm7, %%ymm24")
GEN_test_AllMaskModes(VFMSUB132PS_9A_128,
      "vfmsub132ps %%xmm8, %%xmm7, %%xmm24",
      "vfmsub132ps 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFMADD132SD_99_128,
      "vfmadd132sd %%xmm8, %%xmm7, %%xmm24",
      "vfmadd132sd 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFMADD132SS_99_128,
      "vfmadd132ss %%xmm8, %%xmm7, %%xmm24",
      "vfmadd132ss 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFMADD132PD_98_512,
      "vfmadd132pd %%zmm8, %%zmm7, %%zmm24",
      "vfmadd132pd 0x40(%%rax), %%zmm7, %%zmm24")
GEN_test_AllMaskModes(VFMADD132PD_98_256,
      "vfmadd132pd %%ymm8, %%ymm7, %%ymm24",
      "vfmadd132pd 0x40(%%rax), %%ymm7, %%ymm24")
GEN_test_AllMaskModes(VFMADD132PD_98_128,
      "vfmadd132pd %%xmm8, %%xmm7, %%xmm24",
      "vfmadd132pd 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFMADD132PS_98_512,
      "vfmadd132ps %%zmm8, %%zmm7, %%zmm24",
      "vfmadd132ps 0x40(%%rax), %%zmm7, %%zmm24")
GEN_test_AllMaskModes(VFMADD132PS_98_256,
      "vfmadd132ps %%ymm8, %%ymm7, %%ymm24",
      "vfmadd132ps 0x40(%%rax), %%ymm7, %%ymm24")
GEN_test_AllMaskModes(VFMADD132PS_98_128,
      "vfmadd132ps %%xmm8, %%xmm7, %%xmm24",
      "vfmadd132ps 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFMSUBADD132PD_97_512,
      "vfmsubadd132pd %%zmm8, %%zmm7, %%zmm24",
      "vfmsubadd132pd 0x40(%%rax), %%zmm7, %%zmm24")
GEN_test_AllMaskModes(VFMSUBADD132PD_97_256,
      "vfmsubadd132pd %%ymm8, %%ymm7, %%ymm24",
      "vfmsubadd132pd 0x40(%%rax), %%ymm7, %%ymm24")
GEN_test_AllMaskModes(VFMSUBADD132PD_97_128,
      "vfmsubadd132pd %%xmm8, %%xmm7, %%xmm24",
      "vfmsubadd132pd 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFMSUBADD132PS_97_512,
      "vfmsubadd132ps %%zmm8, %%zmm7, %%zmm24",
      "vfmsubadd132ps 0x40(%%rax), %%zmm7, %%zmm24")
GEN_test_AllMaskModes(VFMSUBADD132PS_97_256,
      "vfmsubadd132ps %%ymm8, %%ymm7, %%ymm24",
      "vfmsubadd132ps 0x40(%%rax), %%ymm7, %%ymm24")
GEN_test_AllMaskModes(VFMSUBADD132PS_97_128,
      "vfmsubadd132ps %%xmm8, %%xmm7, %%xmm24",
      "vfmsubadd132ps 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFMADDSUB132PD_96_512,
      "vfmaddsub132pd %%zmm8, %%zmm7, %%zmm24",
      "vfmaddsub132pd 0x40(%%rax), %%zmm7, %%zmm24")
GEN_test_AllMaskModes(VFMADDSUB132PD_96_256,
      "vfmaddsub132pd %%ymm8, %%ymm7, %%ymm24",
      "vfmaddsub132pd 0x40(%%rax), %%ymm7, %%ymm24")
GEN_test_AllMaskModes(VFMADDSUB132PD_96_128,
      "vfmaddsub132pd %%xmm8, %%xmm7, %%xmm24",
      "vfmaddsub132pd 0x40(%%rax), %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VFMADDSUB132PS_96_512,
      "vfmaddsub132ps %%zmm8, %%zmm7, %%zmm24",
      "vfmaddsub132ps 0x40(%%rax), %%zmm7, %%zmm24")
GEN_test_AllMaskModes(VFMADDSUB132PS_96_256,
      "vfmaddsub132ps %%ymm8, %%ymm7, %%ymm24",
      "vfmaddsub132ps 0x40(%%rax), %%ymm7, %%ymm24")
GEN_test_AllMaskModes(VFMADDSUB132PS_96_128,
      "vfmaddsub132ps %%xmm8, %%xmm7, %%xmm24",
      "vfmaddsub132ps 0x40(%%rax), %%xmm7, %%xmm24")

/* Scalar compare */
GEN_test_OneMaskModeOnly(VCMPSD_128_0x0,
      "vcmpsd $0x0, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x0, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x1,
      "vcmpsd $0x1, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x1, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x2,
      "vcmpsd $0x2, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x2, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x3,
      "vcmpsd $0x3, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x3, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x4,
      "vcmpsd $0x4, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x4, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x5,
      "vcmpsd $0x5, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x5, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x6,
      "vcmpsd $0x6, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x6, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x7,
      "vcmpsd $0x7, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x7, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x8,
      "vcmpsd $0x8, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x8, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0xA,
      "vcmpsd $0xA, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0xA, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0xB,
      "vcmpsd $0xB, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0xB, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0xC,
      "vcmpsd $0xC, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0xC, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0xD,
      "vcmpsd $0xD, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0xD, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0xE,
      "vcmpsd $0xE, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0xE, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0xF,
      "vcmpsd $0xF, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0xF, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x10,
      "vcmpsd $0x10, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x10, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x11,
      "vcmpsd $0x11, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x11, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x12,
      "vcmpsd $0x12, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x12, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x13,
      "vcmpsd $0x13, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x13, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x14,
      "vcmpsd $0x14, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x14, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x15,
      "vcmpsd $0x15, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x15, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x16,
      "vcmpsd $0x16, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x16, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x17,
      "vcmpsd $0x17, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x17, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x18,
      "vcmpsd $0x18, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x18, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x19,
      "vcmpsd $0x19, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x19, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x1A,
      "vcmpsd $0x1A, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x1A, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x1B,
      "vcmpsd $0x1B, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x1B, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x1C,
      "vcmpsd $0x1C, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x1C, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x1D,
      "vcmpsd $0x1D, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x1D, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSD_128_0x1F,
      "vcmpsd $0x1F, %%xmm8,  %%xmm24, %%k4",
      "vcmpsd $0x1F, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x0,
      "vcmpss $0x0, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x0, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x1,
      "vcmpss $0x1, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x1, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x2,
      "vcmpss $0x2, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x2, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x3,
      "vcmpss $0x3, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x3, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x4,
      "vcmpss $0x4, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x4, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x5,
      "vcmpss $0x5, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x5, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x6,
      "vcmpss $0x6, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x6, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x7,
      "vcmpss $0x7, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x7, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x8,
      "vcmpss $0x8, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x8, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0xA,
      "vcmpss $0xA, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0xA, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0xB,
      "vcmpss $0xB, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0xB, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0xC,
      "vcmpss $0xC, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0xC, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0xD,
      "vcmpss $0xD, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0xD, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0xE,
      "vcmpss $0xE, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0xE, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0xF,
      "vcmpss $0xF, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0xF, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x10,
      "vcmpss $0x10, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x10, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x11,
      "vcmpss $0x11, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x11, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x12,
      "vcmpss $0x12, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x12, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x13,
      "vcmpss $0x13, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x13, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x14,
      "vcmpss $0x14, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x14, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x15,
      "vcmpss $0x15, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x15, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x16,
      "vcmpss $0x16, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x16, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x17,
      "vcmpss $0x17, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x17, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x18,
      "vcmpss $0x18, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x18, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x19,
      "vcmpss $0x19, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x19, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x1A,
      "vcmpss $0x1A, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x1A, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x1B,
      "vcmpss $0x1B, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x1B, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x1C,
      "vcmpss $0x1C, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x1C, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x1D,
      "vcmpss $0x1D, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x1D, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_OneMaskModeOnly(VCMPSS_128_0x1F,
      "vcmpss $0x1F, %%xmm8,  %%xmm24, %%k4",
      "vcmpss $0x1F, 0x40(%%rax), %%xmm24, %%k4")
GEN_test_RandM(VCMPSD_C2_128,
      "vcmpsd $0xAB, %%xmm7, %%xmm8, %%k4",
      "vcmpsd $0xAB, 0x40(%%rax), %%xmm8, %%k4")

/* masks */
GEN_test_Ronly(KMOVD_93_128,
      "kmovd %%k4, %%r14d")
GEN_test_Ronly(KMOVB_93_128,
      "kmovb %%k4, %%r14d")
GEN_test_Ronly(KMOVQ_93_128,
      "kmovq %%k4, %%r14")
GEN_test_Ronly(KMOVW_93_128,
      "kmovw %%k4, %%r14d")
GEN_test_Ronly(KMOVD_92_128,
      "kmovd %%r14d, %%k5")
GEN_test_Ronly(KMOVB_92_128,
      "kmovb %%r14d, %%k5")
GEN_test_Ronly(KMOVQ_92_128,
      "kmovq %%r14, %%k5")
GEN_test_Ronly(KMOVW_92_128,
      "kmovw %%r14d, %%k5")
GEN_test_Monly(KMOVD_91_128,
      "kmovd %%k5, 0x40(%%rax)")
GEN_test_Monly(KMOVB_91_128,
      "kmovb %%k5, 0x40(%%rax)")
GEN_test_Monly(KMOVQ_91_128,
      "kmovq %%k5, 0x40(%%rax)")
GEN_test_Monly(KMOVW_91_128,
      "kmovw %%k5, 0x40(%%rax)")
GEN_test_RandM(KMOVD_90_128,
      "kmovd %%k4, %%k5",
      "kmovd 0x40(%%rax), %%k5")
GEN_test_RandM(KMOVB_90_128,
      "kmovb %%k4, %%k5",
      "kmovb 0x40(%%rax), %%k5")
GEN_test_RandM(KMOVQ_90_128,
      "kmovq %%k4, %%k5",
      "kmovq 0x40(%%rax), %%k5")
GEN_test_RandM(KMOVW_90_128,
      "kmovw %%k4, %%k5",
      "kmovw 0x40(%%rax), %%k5")
GEN_test_Ronly(KUNPCKBW_4B_128,
      "kunpckbw %%k4, %%k5, %%k5")
GEN_test_Ronly(KUNPCKDQ_4B_128,
      "kunpckdq  %%k4, %%k5, %%k5")
GEN_test_Ronly(KUNPCKWD_4B_128,
      "kunpckwd %%k4, %%k5, %%k5")
GEN_test_Ronly(KADDD_4A_128,
      "kaddd %%k4, %%k5, %%k5")
GEN_test_Ronly(KADDB_4A_128,
      "kaddb %%k4, %%k5, %%k5")
GEN_test_Ronly(KADDQ_4A_128,
      "kaddq %%k4, %%k5, %%k5")
GEN_test_Ronly(KADDW_4A_128,
      "kaddw %%k4, %%k5, %%k5")
GEN_test_Ronly(KXORD_47_128,
      "kxord %%k4, %%k5, %%k5")
GEN_test_Ronly(KXORB_47_128,
      "kxorb %%k4, %%k5, %%k5")
GEN_test_Ronly(KXORQ_47_128,
      "kxorq %%k4, %%k5, %%k5")
GEN_test_Ronly(KXORW_47_128,
      "kxorw %%k4, %%k5, %%k5")
GEN_test_Ronly(KXNORD_46_128,
      "kxnord %%k4, %%k5, %%k5")
GEN_test_Ronly(KXNORB_46_128,
      "kxnorb %%k4, %%k5, %%k5")
GEN_test_Ronly(KXNORQ_46_128,
      "kxnorq %%k4, %%k5, %%k5")
GEN_test_Ronly(KXNORW_46_128,
      "kxnorw %%k4, %%k5, %%k5")
GEN_test_Ronly(KORD_45_128,
      "kord %%k4, %%k5, %%k5")
GEN_test_Ronly(KORB_45_128,
      "korb %%k4, %%k5, %%k5")
GEN_test_Ronly(KORQ_45_128,
      "korq %%k4, %%k5, %%k5")
GEN_test_Ronly(KORW_45_128,
      "korw %%k4, %%k5, %%k5")
GEN_test_Ronly(KNOTD_44_128,
      "knotd %%k4, %%k5")
GEN_test_Ronly(KNOTB_44_128,
      "knotb %%k4, %%k5")
GEN_test_Ronly(KNOTQ_44_128,
      "knotq %%k4, %%k5")
GEN_test_Ronly(KNOTW_44_128,
      "knotw %%k4, %%k5")
GEN_test_Ronly(KANDND_42_128,
      "kandnd %%k4, %%k5, %%k5")
GEN_test_Ronly(KANDNB_42_128,
      "kandnb %%k4, %%k5, %%k5")
GEN_test_Ronly(KANDNQ_42_128,
      "kandnq %%k4, %%k5, %%k5")
GEN_test_Ronly(KANDNW_42_128,
      "kandnw %%k4, %%k5, %%k5")
GEN_test_Ronly(KANDD_41_128,
      "kandd %%k4, %%k5, %%k5")
GEN_test_Ronly(KANDB_41_128,
      "kandb %%k4, %%k5, %%k5")
GEN_test_Ronly(KANDQ_41_128,
      "kandq %%k4, %%k5, %%k5")
GEN_test_Ronly(KANDW_41_128,
      "kandw %%k4, %%k5, %%k5")
GEN_test_Ronly(KSHIFTLQ_33_128,
      "kshiftlq $0xAB, %%k4, %%k5")
GEN_test_Ronly(KSHIFTLD_33_128,
      "kshiftld $0xAB, %%k4, %%k5")
GEN_test_Ronly(KSHIFTLW_32_128,
      "kshiftlw $0xAB, %%k4, %%k5")
GEN_test_Ronly(KSHIFTLB_32_128,
      "kshiftlb $0xAB, %%k4, %%k5")
GEN_test_Ronly(KSHIFTRQ_31_128,
      "kshiftrq $0xAB, %%k4, %%k5")
GEN_test_Ronly(KSHIFTRD_31_128,
      "kshiftrd $0xAB, %%k4, %%k5")
GEN_test_Ronly(KSHIFTRW_30_128,
      "kshiftrw $0xAB, %%k4, %%k5")
GEN_test_Ronly(KSHIFTRB_30_128,
      "kshiftrb $0xAB, %%k4, %%k5")
GEN_test_Ronly(KTESTD_99_128,
      "ktestd %%k4, %%k5")
GEN_test_Ronly(KTESTB_99_128,
      "ktestb %%k4, %%k5")
GEN_test_Ronly(KTESTQ_99_128,
      "ktestq %%k4, %%k5")
GEN_test_Ronly(KTESTW_99_128,
      "ktestw %%k4, %%k5")
GEN_test_Ronly(KORTESTD_98_128,
      "kortestd %%k4, %%k5")
GEN_test_Ronly(KORTESTB_98_128,
      "kortestb %%k4, %%k5")
GEN_test_Ronly(KORTESTQ_98_128,
      "kortestq %%k4, %%k5")
GEN_test_Ronly(KORTESTW_98_128,
      "kortestw %%k4, %%k5")

/* misc */
GEN_test_AllMaskModes(VUNPCKHPD_15_512,
      "vunpckhpd %%zmm24, %%zmm8, %%zmm7",
      "vunpckhpd 0x40(%%rax), %%zmm24, %%zmm8")
GEN_test_AllMaskModes(VUNPCKHPS_15_512,
      "vunpckhps %%zmm24, %%zmm8, %%zmm7",
      "vunpckhps 0x40(%%rax), %%zmm24, %%zmm8")
GEN_test_AllMaskModes(VUNPCKLPD_14_512,
      "vunpcklpd %%zmm24, %%zmm8, %%zmm7",
      "vunpcklpd 0x40(%%rax), %%zmm24, %%zmm8")
GEN_test_AllMaskModes(VUNPCKLPS_14_512,
      "vunpcklps %%zmm24, %%zmm8, %%zmm7",
      "vunpcklps 0x40(%%rax), %%zmm24, %%zmm8")
GEN_test_AllMaskModes(VPUNPCKHDQ_6A_512,
      "vpunpckhdq %%zmm24, %%zmm8, %%zmm7",
      "vpunpckhdq 0x40(%%rax), %%zmm24, %%zmm8")
GEN_test_AllMaskModes(VPUNPCKHQDQ_6D_512,
      "vpunpckhqdq %%zmm24, %%zmm8, %%zmm7",
      "vpunpckhqdq 0x40(%%rax), %%zmm24, %%zmm8")
GEN_test_AllMaskModes(VPUNPCKLDQ_62_512,
      "vpunpckldq %%zmm24, %%zmm8, %%zmm7",
      "vpunpckldq 0x40(%%rax), %%zmm24, %%zmm8")
GEN_test_AllMaskModes(VPUNPCKLQDQ_6C_512,
      "vpunpcklqdq %%zmm24, %%zmm8, %%zmm7",
      "vpunpcklqdq 0x40(%%rax), %%zmm24, %%zmm8")
GEN_test_AllMaskModes(VSHUFF32x4_23_512,
      "vshuff32x4 $0x7C, %%zmm24, %%zmm8, %%zmm7",
      "vshuff32x4 $0xBA, 0x40(%%rax), %%zmm24, %%zmm8")
GEN_test_AllMaskModes(VSHUFF64x2_23_512,
      "vshuff64x2 $0x7C, %%zmm24, %%zmm8, %%zmm7",
      "vshuff64x2 $0xBA, 0x40(%%rax), %%zmm24, %%zmm8")
GEN_test_AllMaskModes(VSHUFI32x4_43_512,
      "vshufi32x4 $0x7C, %%zmm24, %%zmm8, %%zmm7",
      "vshufi32x4 $0xBA, 0x40(%%rax), %%zmm24, %%zmm8")
GEN_test_AllMaskModes(VSHUFI64x2_43_512,
      "vshufi64x2 $0x7C, %%zmm24, %%zmm8, %%zmm7",
      "vshufi64x2 $0xBA, 0x40(%%rax), %%zmm24, %%zmm8")
GEN_test_AllMaskModes(VPUNPCKHQDQ_6D_256,
      "vpunpckhqdq %%ymm7, %%ymm8, %%ymm24",
      "vpunpckhqdq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPUNPCKHQDQ_6D_128,
      "vpunpckhqdq %%xmm7, %%xmm8, %%xmm24",
      "vpunpckhqdq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPUNPCKLQDQ_6C_256,
      "vpunpcklqdq %%ymm7, %%ymm8, %%ymm24",
      "vpunpcklqdq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPUNPCKLQDQ_6C_128,
      "vpunpcklqdq %%xmm7, %%xmm8, %%xmm24",
      "vpunpcklqdq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPUNPCKHDQ_6A_256,
      "vpunpckhdq %%ymm7, %%ymm8, %%ymm24",
      "vpunpckhdq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPUNPCKHDQ_6A_128,
      "vpunpckhdq %%xmm7, %%xmm8, %%xmm24",
      "vpunpckhdq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VBLENDMPS_65_512,
      "vblendmps %%zmm7, %%zmm8, %%zmm24",
      "vblendmps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VBLENDMPS_65_256,
      "vblendmps %%ymm7, %%ymm8, %%ymm24",
      "vblendmps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VBLENDMPS_65_128,
      "vblendmps %%xmm7, %%xmm8, %%xmm24",
      "vblendmps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VBLENDMPD_65_512,
      "vblendmpd %%zmm7, %%zmm8, %%zmm24",
      "vblendmpd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VBLENDMPD_65_256,
      "vblendmpd %%ymm7, %%ymm8, %%ymm24",
      "vblendmpd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VBLENDMPD_65_128,
      "vblendmpd %%xmm7, %%xmm8, %%xmm24",
      "vblendmpd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPBLENDMQ_64_512,
      "vpblendmq %%zmm7, %%zmm8, %%zmm24",
      "vpblendmq 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPBLENDMQ_64_256,
      "vpblendmq %%ymm7, %%ymm8, %%ymm24",
      "vpblendmq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPBLENDMQ_64_128,
      "vpblendmq %%xmm7, %%xmm8, %%xmm24",
      "vpblendmq 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPBLENDMD_64_512,
      "vpblendmd %%zmm7, %%zmm8, %%zmm24",
      "vpblendmd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPBLENDMD_64_256,
      "vpblendmd %%ymm7, %%ymm8, %%ymm24",
      "vpblendmd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPBLENDMD_64_128,
      "vpblendmd %%xmm7, %%xmm8, %%xmm24",
      "vpblendmd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPUNPCKLDQ_62_256,
      "vpunpckldq %%ymm7, %%ymm8, %%ymm24",
      "vpunpckldq 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPUNPCKLDQ_62_128,
      "vpunpckldq %%xmm7, %%xmm8, %%xmm24",
      "vpunpckldq 0x40(%%rax), %%xmm8, %%xmm24")

GEN_test_Mdisp(VADDSD_md,
      "vaddsd 0x40(%%r14,%%r12,4), %%xmm8, %%xmm24")
GEN_test_Mdisp(VDIVSD_md,
      "vdivsd 0x40(%%r14,%%r12,4), %%xmm8, %%xmm24")
GEN_test_Mdisp(VFMADD213SD_md,
      "vfmadd213sd 0x40(%%r14,%%r12,4), %%xmm8, %%xmm24")
GEN_test_Mdisp(VFMADD231SD_md,
      "vfmadd231sd 0x40(%%r14,%%r12,4), %%xmm8, %%xmm24")
GEN_test_Mdisp(VFMSUB213SD_md,
      "vfmsub213sd 0x40(%%r14,%%r12,4), %%xmm8, %%xmm24")
GEN_test_Mdisp(VFMSUB231SD_md,
      "vfmsub231sd 0x40(%%r14,%%r12,4), %%xmm8, %%xmm24")
GEN_test_Mdisp(VFNMADD213SD_md,
      "vfnmadd213sd 0x40(%%r14,%%r12,4), %%xmm8, %%xmm24")
GEN_test_Mdisp(VFNMADD231SD_md,
      "vfnmadd231sd 0x40(%%r14,%%r12,4), %%xmm8, %%xmm24")
GEN_test_Mdisp(VMOVSDr_md,
      "vmovsd 0x40(%%r14,%%r12,4), %%xmm24")
GEN_test_Mdisp(VMOVSDl_md,
      "vmovsd %%xmm24, 0x40(%%r14,%%r12,4)")

GEN_test_AllMaskModes(VINSERTI64x4_3A_512,
      "vinserti64x4 $0xAB, %%ymm7, %%zmm8, %%zmm24",
      "vinserti64x4 $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VINSERTI32x4_38_512,
      "vinserti32x4 $0xAB, %%xmm7, %%zmm8, %%zmm24",
      "vinserti32x4 $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VINSERTI32x4_38_256,
      "vinserti32x4 $0xAB, %%xmm7, %%ymm8, %%ymm24",
      "vinserti32x4 $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VINSERTF64x4_1A_512,
      "vinsertf64x4 $0xAB, %%ymm7, %%zmm8, %%zmm24",
      "vinsertf64x4 $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VINSERTF32x4_18_512,
      "vinsertf32x4 $0xAB, %%xmm7, %%zmm8, %%zmm24",
      "vinsertf32x4 $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VINSERTF32x4_18_256,
      "vinsertf32x4 $0xAB, %%xmm7, %%ymm8, %%ymm24",
      "vinsertf32x4 $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VUNPCKLPD_14_256,
      "vunpcklpd %%ymm7, %%ymm8, %%ymm24",
      "vunpcklpd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VUNPCKLPD_14_128,
      "vunpcklpd %%xmm7, %%xmm8, %%xmm24",
      "vunpcklpd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VUNPCKLPS_14_256,
      "vunpcklps %%ymm7, %%ymm8, %%ymm24",
      "vunpcklps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VUNPCKLPS_14_128,
      "vunpcklps %%xmm7, %%xmm8, %%xmm24",
      "vunpcklps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VALIGNQ_03_512,
      "valignq $0xAB, %%zmm7, %%zmm8, %%zmm24",
      "valignq $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VALIGNQ_03_256,
      "valignq $0xAB, %%ymm7, %%ymm8, %%ymm24",
      "valignq $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VALIGNQ_03_128,
      "valignq $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "valignq $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VALIGND_03_512,
      "valignd $0xAB, %%zmm7, %%zmm8, %%zmm24",
      "valignd $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VALIGND_03_256,
      "valignd $0xAB, %%ymm7, %%ymm8, %%ymm24",
      "valignd $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VALIGND_03_128,
      "valignd $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "valignd $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFIXUPIMMSD_55_128,
      "vfixupimmsd $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vfixupimmsd $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VFIXUPIMMPD_54_512,
      "vfixupimmpd $0xAB, %%zmm7, %%zmm8, %%zmm24",
      "vfixupimmpd $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VRCP14PD_4C_512,
      "vrcp14pd %%zmm7, %%zmm24",
      "vrcp14pd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VRCP14PD_4C_256,
      "vrcp14pd %%ymm7, %%ymm24",
      "vrcp14pd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VRCP14PD_4C_128,
      "vrcp14pd %%xmm7, %%xmm24",
      "vrcp14pd 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VRCP14PS_4C_512,
      "vrcp14ps %%zmm7, %%zmm24",
      "vrcp14ps 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VRCP14PS_4C_256,
      "vrcp14ps %%ymm7, %%ymm24",
      "vrcp14ps 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VRCP14PS_4C_128,
      "vrcp14ps %%xmm7, %%xmm24",
      "vrcp14ps 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VGETEXPPS_42_512,
      "vgetexpps %%zmm7, %%zmm24",
      "vgetexpps 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VGETEXPPS_42_256,
      "vgetexpps %%ymm7, %%ymm24",
      "vgetexpps 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VGETEXPPS_42_128,
      "vgetexpps %%xmm7, %%xmm24",
      "vgetexpps 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VGETEXPPD_42_512,
      "vgetexppd %%zmm7, %%zmm24",
      "vgetexppd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VGETEXPPD_42_256,
      "vgetexppd %%ymm7, %%ymm24",
      "vgetexppd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VGETEXPPD_42_128,
      "vgetexppd %%xmm7, %%xmm24",
      "vgetexppd 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VSCALEFPS_2C_512,
      "vscalefps %%zmm7, %%zmm8, %%zmm24",
      "vscalefps 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VSCALEFPS_2C_256,
      "vscalefps %%ymm7, %%ymm8, %%ymm24",
      "vscalefps 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VSCALEFPS_2C_128,
      "vscalefps %%xmm7, %%xmm8, %%xmm24",
      "vscalefps 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VSCALEFPD_2C_512,
      "vscalefpd %%zmm7, %%zmm8, %%zmm24",
      "vscalefpd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VSCALEFPD_2C_256,
      "vscalefpd %%ymm7, %%ymm8, %%ymm24",
      "vscalefpd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VSCALEFPD_2C_128,
      "vscalefpd %%xmm7, %%xmm8, %%xmm24",
      "vscalefpd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VGETMANTPS_26_512,
      "vgetmantps $0xAB, %%zmm7, %%zmm24",
      "vgetmantps $0xAB, 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VGETMANTPS_26_256,
      "vgetmantps $0xAB, %%ymm7, %%ymm24",
      "vgetmantps $0xAB, 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VGETMANTPS_26_128,
      "vgetmantps $0xAB, %%xmm7, %%xmm24",
      "vgetmantps $0xAB, 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VGETMANTPD_26_512,
      "vgetmantpd $0xAB, %%zmm7, %%zmm24",
      "vgetmantpd $0xAB, 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VGETMANTPD_26_256,
      "vgetmantpd $0xAB, %%ymm7, %%ymm24",
      "vgetmantpd $0xAB, 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VGETMANTPD_26_128,
      "vgetmantpd $0xAB, %%xmm7, %%xmm24",
      "vgetmantpd $0xAB, 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VRNDSCALEPD_09_512,
      "vrndscalepd $0xAB, %%zmm7, %%zmm24",
      "vrndscalepd $0xAB, 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VRNDSCALEPD_09_256,
      "vrndscalepd $0xAB, %%ymm7, %%ymm24",
      "vrndscalepd $0xAB, 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VRNDSCALEPD_09_128,
      "vrndscalepd $0xAB, %%xmm7, %%xmm24",
      "vrndscalepd $0xAB, 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VRNDSCALEPS_08_512,
      "vrndscaleps $0xAB, %%zmm7, %%zmm24",
      "vrndscaleps $0xAB, 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VRNDSCALEPS_08_256,
      "vrndscaleps $0xAB, %%ymm7, %%ymm24",
      "vrndscaleps $0xAB, 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VRNDSCALEPS_08_128,
      "vrndscaleps $0xAB, %%xmm7, %%xmm24",
      "vrndscaleps $0xAB, 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VFIXUPIMMPD_54_256,
      "vfixupimmpd $0xAB, %%ymm7, %%ymm8, %%ymm24",
      "vfixupimmpd $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFIXUPIMMPD_54_128,
      "vfixupimmpd $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vfixupimmpd $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VSHUFF64x2_23_256,
      "vshuff64x2 $0xAB, %%ymm7, %%ymm8, %%ymm24",
      "vshuff64x2 $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VSHUFF32x4_23_256,
      "vshuff32x4 $0xAB, %%ymm7, %%ymm8, %%ymm24",
      "vshuff32x4 $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFIXUPIMMPS_54_512,
      "vfixupimmps $0xAB, %%zmm7, %%zmm8, %%zmm24",
      "vfixupimmps $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VFIXUPIMMPS_54_256,
      "vfixupimmps $0xAB, %%ymm7, %%ymm8, %%ymm24",
      "vfixupimmps $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VFIXUPIMMPS_54_128,
      "vfixupimmps $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vfixupimmps $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VRCP14SS_4D_128,
      "vrcp14ss %%xmm7, %%xmm8, %%xmm24",
      "vrcp14ss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VSHUFI64x2_43_256,
      "vshufi64x2 $0xAB, %%ymm7, %%ymm8, %%ymm24",
      "vshufi64x2 $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VSHUFI32x4_43_256,
      "vshufi32x4 $0xAB, %%ymm7, %%ymm8, %%ymm24",
      "vshufi32x4 $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VGETEXPSD_43_128,
      "vgetexpsd %%xmm7, %%xmm8, %%xmm24",
      "vgetexpsd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VGETEXPSS_43_128,
      "vgetexpss %%xmm7, %%xmm8, %%xmm24",
      "vgetexpss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_RandM(VCOMISD_2F_128,
      "vcomisd %%xmm7, %%xmm24",
      "vcomisd 0x40(%%rax), %%xmm24")
GEN_test_RandM(VCOMISS_2F_128,
      "vcomiss %%xmm7, %%xmm24",
      "vcomiss 0x40(%%rax), %%xmm24")
GEN_test_RandM(VUCOMISD_2E_128,
      "vucomisd %%xmm6,  %%xmm18; pushfq; popq %%r14; andq $0x8D5, %%r14",
      "vucomisd 0x40(%%rax), %%xmm18; pushfq; popq %%r14; andq $0x8D5, %%r14")
GEN_test_RandM(VUCOMISS_2E_128,
      "vucomiss %%xmm7, %%xmm24",
      "vucomiss 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VGETMANTSD_27_128,
      "vgetmantsd $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vgetmantsd $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VGETMANTSS_27_128,
      "vgetmantss $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vgetmantss $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VINSERTF64x2_18_512,
      "vinsertf64x2 $0xAB, %%xmm7, %%zmm8, %%zmm24",
      "vinsertf64x2 $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VINSERTF64x2_18_256,
      "vinsertf64x2 $0xAB, %%xmm7, %%ymm8, %%ymm24",
      "vinsertf64x2 $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VRNDSCALESD_0B_128,
      "vrndscalesd $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vrndscalesd $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VRNDSCALESS_0A_128,
      "vrndscaless $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vrndscaless $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPAVGW_E3_512,
      "vpavgw %%zmm7, %%zmm8, %%zmm24",
      "vpavgw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPAVGW_E3_256,
      "vpavgw %%ymm7, %%ymm8, %%ymm24",
      "vpavgw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPAVGW_E3_128,
      "vpavgw %%xmm7, %%xmm8, %%xmm24",
      "vpavgw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPAVGB_E0_512,
      "vpavgb %%zmm7, %%zmm8, %%zmm24",
      "vpavgb 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPAVGB_E0_256,
      "vpavgb %%ymm7, %%ymm8, %%ymm24",
      "vpavgb 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPAVGB_E0_128,
      "vpavgb %%xmm7, %%xmm8, %%xmm24",
      "vpavgb 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_RandM(VUCOMISD_2F_128,
      "vucomisd %%xmm7, %%xmm24",
      "vucomisd 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VSHUFPD_C6_512,
      "vshufpd $0xAB, %%zmm7, %%zmm8, %%zmm24",
      "vshufpd $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VSHUFPD_C6_256,
      "vshufpd $0xAB, %%ymm7, %%ymm8, %%ymm24",
      "vshufpd $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VSHUFPD_C6_128,
      "vshufpd $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vshufpd $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VSHUFPS_C6_512,
      "vshufps $0xAB, %%zmm7, %%zmm8, %%zmm24",
      "vshufps $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VSHUFPS_C6_256,
      "vshufps $0xAB, %%ymm7, %%ymm8, %%ymm24",
      "vshufps $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VSHUFPS_C6_128,
      "vshufps $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vshufps $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_RandM(VPINSRW_C4_128,
      "vpinsrw $0x5, %%r14, %%xmm8, %%xmm24",
      "vpinsrw $0x6, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_RandM(VPINSRQ_22_128,
      "vpinsrq $0x0, %%r14, %%xmm8, %%xmm24",
      "vpinsrq $0x1, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_RandM(VPINSRD_22_128,
      "vpinsrd $0x2, %%r14d, %%xmm8, %%xmm24",
      "vpinsrd $0x3, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_RandM(VPINSRB_20_128,
      "vpinsrb $0xA, %%r14, %%xmm8, %%xmm24",
      "vpinsrb $0x4, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSHUFB_00_512,
      "vpshufb %%zmm7, %%zmm8, %%zmm24",
      "vpshufb 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPSHUFB_00_256,
      "vpshufb %%ymm7, %%ymm8, %%ymm24",
      "vpshufb 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPSHUFB_00_128,
      "vpshufb %%xmm7, %%xmm8, %%xmm24",
      "vpshufb 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPACKSSDW_6B_512,
      "vpackssdw %%zmm7, %%zmm8, %%zmm24",
      "vpackssdw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPACKSSDW_6B_256,
      "vpackssdw %%ymm7, %%ymm8, %%ymm24",
      "vpackssdw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPACKSSDW_6B_128,
      "vpackssdw %%xmm7, %%xmm8, %%xmm24",
      "vpackssdw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPUNPCKHWD_69_512,
      "vpunpckhwd %%zmm7, %%zmm8, %%zmm24",
      "vpunpckhwd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPUNPCKHWD_69_256,
      "vpunpckhwd %%ymm7, %%ymm8, %%ymm24",
      "vpunpckhwd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPUNPCKHWD_69_128,
      "vpunpckhwd %%xmm7, %%xmm8, %%xmm24",
      "vpunpckhwd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPUNPCKHBW_68_512,
      "vpunpckhbw %%zmm7, %%zmm8, %%zmm24",
      "vpunpckhbw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPUNPCKHBW_68_256,
      "vpunpckhbw %%ymm7, %%ymm8, %%ymm24",
      "vpunpckhbw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPUNPCKHBW_68_128,
      "vpunpckhbw %%xmm7, %%xmm8, %%xmm24",
      "vpunpckhbw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPACKUSWB_67_512,
      "vpackuswb %%zmm7, %%zmm8, %%zmm24",
      "vpackuswb 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPACKUSWB_67_256,
      "vpackuswb %%ymm7, %%ymm8, %%ymm24",
      "vpackuswb 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPACKUSWB_67_128,
      "vpackuswb %%xmm7, %%xmm8, %%xmm24",
      "vpackuswb 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPACKSSWB_63_512,
      "vpacksswb %%zmm7, %%zmm8, %%zmm24",
      "vpacksswb 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPACKSSWB_63_256,
      "vpacksswb %%ymm7, %%ymm8, %%ymm24",
      "vpacksswb 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPACKSSWB_63_128,
      "vpacksswb %%xmm7, %%xmm8, %%xmm24",
      "vpacksswb 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPUNPCKLWD_61_512,
      "vpunpcklwd %%zmm7, %%zmm8, %%zmm24",
      "vpunpcklwd 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPUNPCKLWD_61_256,
      "vpunpcklwd %%ymm7, %%ymm8, %%ymm24",
      "vpunpcklwd 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPUNPCKLWD_61_128,
      "vpunpcklwd %%xmm7, %%xmm8, %%xmm24",
      "vpunpcklwd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPUNPCKLBW_60_512,
      "vpunpcklbw %%zmm7, %%zmm8, %%zmm24",
      "vpunpcklbw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPUNPCKLBW_60_256,
      "vpunpcklbw %%ymm7, %%ymm8, %%ymm24",
      "vpunpcklbw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPUNPCKLBW_60_128,
      "vpunpcklbw %%xmm7, %%xmm8, %%xmm24",
      "vpunpcklbw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VRSQRT14SD_4F_128,
      "vrsqrt14sd %%xmm7, %%xmm8, %%xmm24",
      "vrsqrt14sd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VRSQRT14SS_4F_128,
      "vrsqrt14ss %%xmm7, %%xmm8, %%xmm24",
      "vrsqrt14ss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VRSQRT14PD_4E_512,
      "vrsqrt14pd %%zmm7, %%zmm24",
      "vrsqrt14pd 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VRSQRT14PD_4E_256,
      "vrsqrt14pd %%ymm7, %%ymm24",
      "vrsqrt14pd 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VRSQRT14PD_4E_128,
      "vrsqrt14pd %%xmm7, %%xmm24",
      "vrsqrt14pd 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VRSQRT14PS_4E_512,
      "vrsqrt14ps %%zmm7, %%zmm24",
      "vrsqrt14ps 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VRSQRT14PS_4E_256,
      "vrsqrt14ps %%ymm7, %%ymm24",
      "vrsqrt14ps 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VRSQRT14PS_4E_128,
      "vrsqrt14ps %%xmm7, %%xmm24",
      "vrsqrt14ps 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VPACKUSDW_2B_512,
      "vpackusdw %%zmm7, %%zmm8, %%zmm24",
      "vpackusdw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPACKUSDW_2B_256,
      "vpackusdw %%ymm7, %%ymm8, %%ymm24",
      "vpackusdw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPACKUSDW_2B_128,
      "vpackusdw %%xmm7, %%xmm8, %%xmm24",
      "vpackusdw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPALIGNR_0F_512,
      "vpalignr $0xAB, %%zmm7, %%zmm8, %%zmm24",
      "vpalignr $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPALIGNR_0F_256,
      "vpalignr $0xAB, %%ymm7, %%ymm8, %%ymm24",
      "vpalignr $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPALIGNR_0F_128,
      "vpalignr $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vpalignr $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPSHUFHW_70_512,
      "vpshufhw $0xAB, %%zmm7, %%zmm24",
      "vpshufhw $0xAB, 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPSHUFHW_70_256,
      "vpshufhw $0xAB, %%ymm7, %%ymm24",
      "vpshufhw $0xAB, 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPSHUFHW_70_128,
      "vpshufhw $0xAB, %%xmm7, %%xmm24",
      "vpshufhw $0xAB, 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VPSHUFLW_70_512,
      "vpshuflw $0xAB, %%zmm7, %%zmm24",
      "vpshuflw $0xAB, 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPSHUFLW_70_256,
      "vpshuflw $0xAB, %%ymm7, %%ymm24",
      "vpshuflw $0xAB, 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPSHUFLW_70_128,
      "vpshuflw $0xAB, %%xmm7, %%xmm24",
      "vpshuflw $0xAB, 0x40(%%rax), %%xmm24")
GEN_test_MemMaskModes(VPCOMPRESSD_8B_512,
      "vpcompressd %%zmm24, %%zmm7",
      "vpcompressd %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPCOMPRESSD_8B_256,
      "vpcompressd %%ymm24, %%ymm7",
      "vpcompressd %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPCOMPRESSD_8B_128,
      "vpcompressd %%xmm24, %%xmm7",
      "vpcompressd %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VCOMPRESSPD_8A_512,
      "vcompresspd %%zmm24, %%zmm7",
      "vcompresspd %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VCOMPRESSPD_8A_256,
      "vcompresspd %%ymm24, %%ymm7",
      "vcompresspd %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VCOMPRESSPD_8A_128,
      "vcompresspd %%xmm24, %%xmm7",
      "vcompresspd %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VCOMPRESSPS_8A_512,
      "vcompressps %%zmm24, %%zmm7",
      "vcompressps %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VCOMPRESSPS_8A_256,
      "vcompressps %%ymm24, %%ymm7",
      "vcompressps %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VCOMPRESSPS_8A_128,
      "vcompressps %%xmm24, %%xmm7",
      "vcompressps %%xmm24, 0x40(%%rax)")
GEN_test_AllMaskModes(VPBLENDMW_66_512,
      "vpblendmw %%zmm7, %%zmm8, %%zmm24",
      "vpblendmw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPBLENDMW_66_256,
      "vpblendmw %%ymm7, %%ymm8, %%ymm24",
      "vpblendmw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPBLENDMW_66_128,
      "vpblendmw %%xmm7, %%xmm8, %%xmm24",
      "vpblendmw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VPBLENDMB_66_512,
      "vpblendmb %%zmm7, %%zmm8, %%zmm24",
      "vpblendmb 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPBLENDMB_66_256,
      "vpblendmb %%ymm7, %%ymm8, %%ymm24",
      "vpblendmb 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPBLENDMB_66_128,
      "vpblendmb %%xmm7, %%xmm8, %%xmm24",
      "vpblendmb 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VRCP14SD_4D_128,
      "vrcp14sd %%xmm7, %%xmm8, %%xmm24",
      "vrcp14sd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_MemMaskModes(VPCOMPRESSQ_8B_512,
      "vpcompressq %%zmm24, %%zmm7",
      "vpcompressq %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPCOMPRESSQ_8B_256,
      "vpcompressq %%ymm24, %%ymm7",
      "vpcompressq %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VPCOMPRESSQ_8B_128,
      "vpcompressq %%xmm24, %%xmm7",
      "vpcompressq %%xmm24, 0x40(%%rax)")
GEN_test_OneMaskModeOnly(VFPCLASSSD_67_128,
      "vfpclasssd $0xAB, %%xmm7, %%k5",
      "vfpclasssd $0xAB, 0x40(%%rax), %%k5")
GEN_test_OneMaskModeOnly(VFPCLASSSS_67_128,
      "vfpclassss $0xAB, %%xmm7, %%k5",
      "vfpclassss $0xAB, 0x40(%%rax), %%k5")
GEN_test_Ronly(VFPCLASSPD_66_512,
      "vfpclasspd $0xAB, %%zmm7, %%k5")
GEN_test_Ronly(VFPCLASSPD_66_256,
      "vfpclasspd $0xAB, %%ymm7, %%k5")
GEN_test_Ronly(VFPCLASSPD_66_128,
      "vfpclasspd $0xAB, %%xmm7, %%k5")
GEN_test_Ronly(VFPCLASSPS_66_512,
      "vfpclassps $0xAB, %%zmm7, %%k5")
GEN_test_Ronly(VFPCLASSPS_66_256,
      "vfpclassps $0xAB, %%ymm7, %%k5")
GEN_test_Ronly(VFPCLASSPS_66_128,
      "vfpclassps $0xAB, %%xmm7, %%k5")
GEN_test_AllMaskModes(VFIXUPIMMSS_55_128,
      "vfixupimmss $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vfixupimmss $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VSCALEFSD_2D_128,
      "vscalefsd %%xmm7, %%xmm8, %%xmm24",
      "vscalefsd 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VSCALEFSS_2D_128,
      "vscalefss %%xmm7, %%xmm8, %%xmm24",
      "vscalefss 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_RandM(VINSERTPS_21_128,
      "vinsertps $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vinsertps $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_RandM(VPEXTRB_14_128,
      "vpextrb $0xAB, %%xmm24, %%r14d",
      "vpextrb $0xAB, %%xmm24, 0x40(%%rax)")
GEN_test_RandM(VPEXTRQ_16_128,
      "vpextrq $0xAB, %%xmm24, %%r14",
      "vpextrq $0xAB, %%xmm24, 0x40(%%rax)")
GEN_test_RandM(VPEXTRD_16_128,
      "vpextrd $0xAB, %%xmm24, %%r14d",
      "vpextrd $0xAB, %%xmm24, 0x40(%%rax)")
GEN_test_RandM(VPEXTRW_15_128,
      "vpextrw $0xAB, %%xmm24, %%r14", 
      "vpextrw $0xAB,  %%xmm24, 0x40(%%rax)")
GEN_test_Ronly(VPEXTRW_C5_128,
      "vpextrw $0xAB, %%xmm24, %%r14d")
GEN_test_AllMaskModes(VREDUCESD_57_128,
      "vreducesd $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vreducesd $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VREDUCESS_57_128,
      "vreducess $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vreducess $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VREDUCEPD_56_512,
      "vreducepd $0xAB, %%zmm7, %%zmm24",
      "vreducepd $0xAB, 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VREDUCEPD_56_256,
      "vreducepd $0xAB, %%ymm7, %%ymm24",
      "vreducepd $0xAB, 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VREDUCEPD_56_128,
      "vreducepd $0xAB, %%xmm7, %%xmm24",
      "vreducepd $0xAB, 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VREDUCEPS_56_512,
      "vreduceps $0xAB, %%zmm7, %%zmm24",
      "vreduceps $0xAB, 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VREDUCEPS_56_256,
      "vreduceps $0xAB, %%ymm7, %%ymm24",
      "vreduceps $0xAB, 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VREDUCEPS_56_128,
      "vreduceps $0xAB, %%xmm7, %%xmm24",
      "vreduceps $0xAB, 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VDBPSADBW_42_512,
      "vdbpsadbw $0xAB, %%zmm7, %%zmm8, %%zmm24",
      "vdbpsadbw $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VDBPSADBW_42_256,
      "vdbpsadbw $0xAB, %%ymm7, %%ymm8, %%ymm24",
      "vdbpsadbw $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VDBPSADBW_42_128,
      "vdbpsadbw $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vdbpsadbw $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VRANGESD_51_128,
      "vrangesd $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vrangesd $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VRANGESS_51_128,
      "vrangess $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vrangess $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VRANGEPD_50_512,
      "vrangepd $0xAB, %%zmm7, %%zmm8, %%zmm24",
      "vrangepd $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VRANGEPD_50_256,
      "vrangepd $0xAB, %%ymm7, %%ymm8, %%ymm24",
      "vrangepd $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VRANGEPD_50_128,
      "vrangepd $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vrangepd $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_AllMaskModes(VRANGEPS_50_512,
      "vrangeps $0xAB, %%zmm7, %%zmm8, %%zmm24",
      "vrangeps $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VRANGEPS_50_256,
      "vrangeps $0xAB, %%ymm7, %%ymm8, %%ymm24",
      "vrangeps $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VRANGEPS_50_128,
      "vrangeps $0xAB, %%xmm7, %%xmm8, %%xmm24",
      "vrangeps $0xAB, 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_RandM(VPSLLDQ_73_512,
      "vpslldq $0xAB, %%zmm7, %%zmm8",
      "vpslldq $0xAB, 0x40(%%rax), %%zmm8")
GEN_test_RandM(VPSLLDQ_73_256,
      "vpslldq $0xAB, %%ymm7, %%ymm8",
      "vpslldq $0xAB, 0x40(%%rax), %%ymm8")
GEN_test_RandM(VPSLLDQ_73_128,
      "vpslldq $0xAB, %%xmm7, %%xmm8",
      "vpslldq $0xAB, 0x40(%%rax), %%xmm8")
GEN_test_AllMaskModes(VINSERTI32x8_3A_512,
      "vinserti32x8 $0xAB, %%ymm7, %%zmm8, %%zmm24",
      "vinserti32x8 $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VINSERTI64x2_38_512,
      "vinserti64x2 $0xAB, %%xmm7, %%zmm8, %%zmm24",
      "vinserti64x2 $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VINSERTI64x2_38_256,
      "vinserti64x2 $0xAB, %%xmm7, %%ymm8, %%ymm24",
      "vinserti64x2 $0xAB, 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VINSERTF32x8_1A_512,
      "vinsertf32x8 $0xAB, %%ymm7, %%zmm8, %%zmm24",
      "vinsertf32x8 $0xAB, 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_R_AllMaskModes(VPSHUFD_70_512,
      "vpshufd $0xAB, %%zmm7, %%zmm24")
GEN_test_R_AllMaskModes(VPSHUFD_70_256,
      "vpshufd $0xAB, %%ymm7, %%ymm24")
GEN_test_R_AllMaskModes(VPSHUFD_70_128,
      "vpshufd $0xAB, %%xmm7, %%xmm24")
GEN_test_AllMaskModes(VPBROADCASTW_79_512,
      "vpbroadcastw %%xmm7, %%zmm24",
      "vpbroadcastw 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPBROADCASTW_79_256,
      "vpbroadcastw %%xmm7, %%ymm24",
      "vpbroadcastw 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPBROADCASTW_79_128,
      "vpbroadcastw %%xmm7, %%xmm24",
      "vpbroadcastw 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VPBROADCASTB_78_512,
      "vpbroadcastb %%xmm7, %%zmm24",
      "vpbroadcastb 0x40(%%rax), %%zmm24")
GEN_test_AllMaskModes(VPBROADCASTB_78_256,
      "vpbroadcastb %%xmm7, %%ymm24",
      "vpbroadcastb 0x40(%%rax), %%ymm24")
GEN_test_AllMaskModes(VPBROADCASTB_78_128,
      "vpbroadcastb %%xmm7, %%xmm24",
      "vpbroadcastb 0x40(%%rax), %%xmm24")
GEN_test_AllMaskModes(VPADDUSW_DD_512,
      "vpaddusw %%zmm7, %%zmm8, %%zmm24",
      "vpaddusw 0x40(%%rax), %%zmm8, %%zmm24")
GEN_test_AllMaskModes(VPADDUSW_DD_256,
      "vpaddusw %%ymm7, %%ymm8, %%ymm24",
      "vpaddusw 0x40(%%rax), %%ymm8, %%ymm24")
GEN_test_AllMaskModes(VPADDUSW_DD_128,
      "vpaddusw %%xmm7, %%xmm8, %%xmm24",
      "vpaddusw 0x40(%%rax), %%xmm8, %%xmm24")
GEN_test_MemMaskModes(VMOVDQU16_7F_512,
      "vmovdqu16 %%zmm24, %%zmm7",
      "vmovdqu16 %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVDQU16_7F_256,
      "vmovdqu16 %%ymm24, %%ymm7",
      "vmovdqu16 %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVDQU16_7F_128,
      "vmovdqu16 %%xmm24, %%xmm7",
      "vmovdqu16 %%xmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVDQU8_7F_512,
      "vmovdqu8 %%zmm24, %%zmm7",
      "vmovdqu8 %%zmm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVDQU8_7F_256,
      "vmovdqu8 %%ymm24, %%ymm7",
      "vmovdqu8 %%ymm24, 0x40(%%rax)")
GEN_test_MemMaskModes(VMOVDQU8_7F_128,
      "vmovdqu8 %%xmm24, %%xmm7",
      "vmovdqu8 %%xmm24, 0x40(%%rax)")

/* Comment duplicated above, for convenient reference:
   Allowed operands in test insns:
   Reg form:  %zmm18,  %zmm7, %zmm8, %zmm24 and %r14.
   Mem form:  (%rax), %zmm7, %zmm8, %zmm24 and %r14.
   Imm8 etc fields are also allowed, where they make sense.
   Both forms may use zmm0 as scratch.  Mem form may also use
   zmm18 as scratch.
   */

// Do the specified test some number of times
#define DO_N(_iters, _testfn) \
   do { int i; for (i = 0; i < (_iters); i++) { test_##_testfn(); } } while (0)

#define DO_NM(_iters, _testfn) \
   do { int i; for (i = 0; i < (_iters); i++) { \
      test_nomask_##_testfn(); \
      test_merge_##_testfn(); \
      test_zero_##_testfn(); \
   } } while (0)

// Do the specified test the default number of times
#define DO_D(_testfn) DO_N(N_DEFAULT_ITERS, _testfn)
// Do the specified test the default number of times without mask and with merge and zeroing masks
#define DO_DM(_testfn) DO_NM(N_DEFAULT_ITERS, _testfn)

/* START REGRESSION TESTS */
void test_fma() {
   DO_DM( VFMADD132PD_98_512 );
   DO_DM( VFMADD132PS_98_512 );
   DO_DM( VFMADD213PD_A8_512 );
   DO_DM( VFMADD213PS_A8_512 );
   DO_DM( VFMADD231PD_B8_512 );
   DO_DM( VFMADD231PS_B8_512 );
   DO_DM( VFMADDSUB132PD_96_512 );
   DO_DM( VFMADDSUB132PS_96_512 );
   DO_DM( VFMADDSUB213PD_A6_512 );
   DO_DM( VFMADDSUB213PS_A6_512 );
   DO_DM( VFMADDSUB231PD_B6_512 );
   DO_DM( VFMADDSUB231PS_B6_512 );
   DO_DM( VFMSUB132PD_9A_512 );
   DO_DM( VFMSUB132PS_9A_512 );
   DO_DM( VFMSUB213PD_AA_512 );
   DO_DM( VFMSUB213PS_AA_512 );
   DO_DM( VFMSUB231PD_BA_512 );
   DO_DM( VFMSUB231PS_BA_512 );
   DO_DM( VFMSUBADD132PD_97_512 );
   DO_DM( VFMSUBADD132PS_97_512 );
   DO_DM( VFMSUBADD213PD_A7_512 );
   DO_DM( VFMSUBADD213PS_A7_512 );
   DO_DM( VFMSUBADD231PD_B7_512 );
   DO_DM( VFMSUBADD231PS_B7_512 );
   DO_DM( VFNMADD132PD_9C_512 );
   DO_DM( VFNMADD132PS_9C_512 );
   DO_DM( VFNMADD213PD_AC_512 );
   DO_DM( VFNMADD213PS_AC_512 );
   DO_DM( VFNMADD231PD_BC_512 );
   DO_DM( VFNMADD231PS_BC_512 );
   DO_DM( VFNMSUB132PD_9E_512 );
   DO_DM( VFNMSUB132PS_9E_512 );
   DO_DM( VFNMSUB213PD_AE_512 );
   DO_DM( VFNMSUB213PS_AE_512 );
   DO_DM( VFNMSUB231PD_BE_512 );
   DO_DM( VFNMSUB231PS_BE_512 );
   DO_DM( VFMADD132SD_99_128 );
   DO_DM( VFMADD132SS_99_128 );
   DO_DM( VFMADD213SD_A9_128 );
   DO_DM( VFMADD213SS_A9_128 );
   DO_DM( VFMADD231SD_B9_128 );
   DO_DM( VFMADD231SS_B9_128 );
   DO_DM( VFMSUB132SD_9B_128 );
   DO_DM( VFMSUB132SS_9B_128 );
   DO_DM( VFMSUB213SD_AB_128 );
   DO_DM( VFMSUB213SS_AB_128 );
   DO_DM( VFMSUB231SD_BB_128 );
   DO_DM( VFMSUB231SS_BB_128 );
   DO_DM( VFNMADD132SD_9D_128 );
   DO_DM( VFNMADD132SS_9D_128 );
   DO_DM( VFNMADD213SD_AD_128 );
   DO_DM( VFNMADD213SS_AD_128 );
   DO_DM( VFNMADD231SD_BD_128 );
   DO_DM( VFNMADD231SS_BD_128 );
   DO_DM( VFNMSUB132SD_9F_128 );
   DO_DM( VFNMSUB132SS_9F_128 );
   DO_DM( VFNMSUB213SD_AF_128 );
   DO_DM( VFNMSUB213SS_AF_128 );
   DO_DM( VFNMSUB231SD_BF_128 );
   DO_DM( VFNMSUB231SS_BF_128 );
}
void test_vec_comparisons() {
   DO_DM( VCMPSD_128_0x0 );
   DO_DM( VCMPSD_128_0x1 );
   DO_DM( VCMPSD_128_0x2 );
   DO_DM( VCMPSD_128_0x3 );
   DO_DM( VCMPSD_128_0x4 );
   DO_DM( VCMPSD_128_0x5 );
   DO_DM( VCMPSD_128_0x6 );
   DO_DM( VCMPSD_128_0x7 );
   DO_DM( VCMPSD_128_0x8 );
   DO_DM( VCMPSD_128_0xA );
   DO_DM( VCMPSD_128_0xB );
   DO_DM( VCMPSD_128_0xC );
   DO_DM( VCMPSD_128_0xD );
   DO_DM( VCMPSD_128_0xE );
   DO_DM( VCMPSD_128_0xF );
   DO_DM( VCMPSD_128_0x10 );
   DO_DM( VCMPSD_128_0x11 );
   DO_DM( VCMPSD_128_0x12 );
   DO_DM( VCMPSD_128_0x13 );
   DO_DM( VCMPSD_128_0x14 );
   DO_DM( VCMPSD_128_0x15 );
   DO_DM( VCMPSD_128_0x16 );
   DO_DM( VCMPSD_128_0x17 );
   DO_DM( VCMPSD_128_0x18 );
   DO_DM( VCMPSD_128_0x19 );
   DO_DM( VCMPSD_128_0x1A );
   DO_DM( VCMPSD_128_0x1B );
   DO_DM( VCMPSD_128_0x1C );
   DO_DM( VCMPSD_128_0x1D );
   DO_DM( VCMPSD_128_0x1F );
   DO_DM( VCMPSS_128_0x0 );
   DO_DM( VCMPSS_128_0x1 );
   DO_DM( VCMPSS_128_0x2 );
   DO_DM( VCMPSS_128_0x3 );
   DO_DM( VCMPSS_128_0x4 );
   DO_DM( VCMPSS_128_0x5 );
   DO_DM( VCMPSS_128_0x6 );
   DO_DM( VCMPSS_128_0x7 );
   DO_DM( VCMPSS_128_0x8 );
   DO_DM( VCMPSS_128_0xA );
   DO_DM( VCMPSS_128_0xB );
   DO_DM( VCMPSS_128_0xC );
   DO_DM( VCMPSS_128_0xD );
   DO_DM( VCMPSS_128_0xE );
   DO_DM( VCMPSS_128_0xF );
   DO_DM( VCMPSS_128_0x10 );
   DO_DM( VCMPSS_128_0x11 );
   DO_DM( VCMPSS_128_0x12 );
   DO_DM( VCMPSS_128_0x13 );
   DO_DM( VCMPSS_128_0x14 );
   DO_DM( VCMPSS_128_0x15 );
   DO_DM( VCMPSS_128_0x16 );
   DO_DM( VCMPSS_128_0x17 );
   DO_DM( VCMPSS_128_0x18 );
   DO_DM( VCMPSS_128_0x19 );
   DO_DM( VCMPSS_128_0x1A );
   DO_DM( VCMPSS_128_0x1F );
   DO_DM( VCMPSS_128_0x1C );
   DO_DM( VCMPSS_128_0x1D );
   DO_DM( VCMPSS_128_0x1F );
   DO_DM( VCMPPD_512_0x0 );
   DO_DM( VCMPPD_512_0x1 );
   DO_DM( VCMPPD_512_0x2 );
   DO_DM( VCMPPD_512_0x3 );
   DO_DM( VCMPPD_512_0x4 );
   DO_DM( VCMPPD_512_0x5 );
   DO_DM( VCMPPD_512_0x6 );
   DO_DM( VCMPPD_512_0x7 );
   DO_DM( VCMPPD_512_0x8 );
   DO_DM( VCMPPD_512_0x9 );
   DO_DM( VCMPPD_512_0xA );
   DO_DM( VCMPPD_512_0xB );
   DO_DM( VCMPPD_512_0xC );
   DO_DM( VCMPPD_512_0xD );
   DO_DM( VCMPPD_512_0xE );
   DO_DM( VCMPPD_512_0xF );
   DO_DM( VCMPPD_512_0x10 );
   DO_DM( VCMPPD_512_0x11 );
   DO_DM( VCMPPD_512_0x12 );
   DO_DM( VCMPPD_512_0x13 );
   DO_DM( VCMPPD_512_0x14 );
   DO_DM( VCMPPD_512_0x15 );
   DO_DM( VCMPPD_512_0x16 );
   DO_DM( VCMPPD_512_0x17 );
   DO_DM( VCMPPD_512_0x18 );
   DO_DM( VCMPPD_512_0x19 );
   DO_DM( VCMPPD_512_0x1A );
   DO_DM( VCMPPD_512_0x1B );
   DO_DM( VCMPPD_512_0x1C );
   DO_DM( VCMPPD_512_0x1D );
   DO_DM( VCMPPD_512_0x1E );
   DO_DM( VCMPPD_512_0x1F );
   DO_DM( VCMPPS_512_0x0 );
   DO_DM( VCMPPS_512_0x1 );
   DO_DM( VCMPPS_512_0x2 );
   DO_DM( VCMPPS_512_0x3 );
   DO_DM( VCMPPS_512_0x4 );
   DO_DM( VCMPPS_512_0x5 );
   DO_DM( VCMPPS_512_0x6 );
   DO_DM( VCMPPS_512_0x7 );
   DO_DM( VCMPPS_512_0x8 );
   DO_DM( VCMPPS_512_0x9 );
   DO_DM( VCMPPS_512_0xA );
   DO_DM( VCMPPS_512_0xB );
   DO_DM( VCMPPS_512_0xC );
   DO_DM( VCMPPS_512_0xD );
   DO_DM( VCMPPS_512_0xE );
   DO_DM( VCMPPS_512_0xF );
   DO_DM( VCMPPS_512_0x10 );
   DO_DM( VCMPPS_512_0x11 );
   DO_DM( VCMPPS_512_0x12 );
   DO_DM( VCMPPS_512_0x13 );
   DO_DM( VCMPPS_512_0x14 );
   DO_DM( VCMPPS_512_0x15 );
   DO_DM( VCMPPS_512_0x16 );
   DO_DM( VCMPPS_512_0x17 );
   DO_DM( VCMPPS_512_0x18 );
   DO_DM( VCMPPS_512_0x19 );
   DO_DM( VCMPPS_512_0x1A );
   DO_DM( VCMPPS_512_0x1B );
   DO_DM( VCMPPS_512_0x1C );
   DO_DM( VCMPPS_512_0x1D );
   DO_DM( VCMPPS_512_0x1E );
}

int main ( void ) {
   {
      int i;
      for (i = 0; i < sizeof(_randArray); i++)
         _randArray[i] = randUChar();
   }
   
#if REGTEST
   /* working tests section */

#ifdef AVX512F
   DO_D( KANDNW_42_128 );
   DO_D( KANDW_41_128 );
   DO_D( KMOVW_90_128 );
   DO_D( KMOVW_91_128 );
   DO_D( KMOVW_92_128 );
   DO_D( KMOVW_93_128 );
   DO_D( KUNPCKBW_4B_128 );
   DO_D( KNOTW_44_128 );
   DO_D( KORW_45_128 );
   DO_D( KORTESTW_98_128 );
   DO_D( KSHIFTLW_32_128 );
   DO_D( KSHIFTRW_30_128 );
   DO_D( KXNORW_46_128 );
   DO_D( KXORW_47_128 );
   DO_DM( VADDPD_58_512 );
   DO_DM( VADDPS_58_512 );
   DO_DM( VALIGND_03_512 );
   DO_DM( VALIGNQ_03_512 );
   DO_DM( VBLENDMPD_65_512 );
   DO_DM( VBLENDMPS_65_512 );
   DO_DM( VBROADCASTF32x4_1A_512 );
   DO_DM( VBROADCASTF64x4_1B_512 );
   DO_DM( VBROADCASTI32x4_5A_512 );
   DO_DM( VBROADCASTI64x4_5B_512 );
   DO_DM( VBROADCASTSD_19_512 );
   DO_DM( VBROADCASTSS_18_512 );
   DO_DM( VCMPEQ_UQPD_512 );
   DO_DM( VCMPNEQPD_512 );
   DO_DM( VCMPPD_C2_512 );
   DO_DM( VCVTDQ2PD_E6_512 );
   DO_DM( VCVTDQ2PS_5B_512 );
   DO_DM( VCVTPD2UDQ_79_512 );
   DO_DM( VCVTPS2DQ_5B_512 );
   DO_DM( VCVTTPD2DQ_E6_512 );
   DO_DM( VCVTTPD2UDQ_78_512 );
   DO_DM( VCVTUDQ2PD_7A_512 );
   DO_DM( VCVTUDQ2PS_7A_512 );
   DO_DM( VDIVPD_5E_512 );
   DO_DM( VDIVPS_5E_512 );
   DO_DM( VEXPANDPD_88_512 );
   DO_DM( VEXPANDPS_88_512 );
   DO_DM( VEXTRACTF32x4_19_512 );
   DO_DM( VEXTRACTF64x4_1B_512 );
   DO_DM( VEXTRACTI32x4_39_512 );
   DO_DM( VEXTRACTI64x4_3B_512 );
   DO_DM( VFIXUPIMMPD_54_512 );
   DO_DM( VFIXUPIMMPS_54_512 );
   DO_DM( VGETEXPPD_42_512 );
   DO_DM( VGETEXPPS_42_512 );
   DO_DM( VGETMANTPD_26_512 );
   DO_DM( VGETMANTPS_26_512 );
   DO_DM( VINSERTF32x4_18_512 );
   DO_DM( VINSERTF64x4_1A_512 );
   DO_DM( VINSERTI32x4_38_512 );
   DO_DM( VINSERTI64x4_3A_512 );
   DO_DM( VMAXPD_5F_512 );
   DO_DM( VMAXPS_5F_512 );
   DO_DM( VMINPD_5D_512 );
   DO_DM( VMINPS_5D_512 );
   DO_DM( VMOVAPD_28_512 );
   DO_DM( VMOVAPD_29_512 );
   DO_DM( VMOVAPS_28_512 );
   DO_DM( VMOVAPS_29_512 );
   DO_DM( VMOVDDUP_12_512 );
   DO_DM( VMOVDQA32_6F_512 );
   DO_DM( VMOVDQA32_7F_512 );
   DO_DM( VMOVDQA64_6F_512 );
   DO_DM( VMOVDQA64_7F_512 );
   DO_DM( VMOVDQU32_6F_512 );
   DO_DM( VMOVDQU32_7F_512 );
   DO_DM( VMOVDQU64_6F_512 );
   DO_DM( VMOVDQU64_7F_512 );
   DO_DM( VMOVSHDUP_16_512 );
   DO_DM( VMOVSLDUP_12_512 );
   DO_DM( VMOVUPD_10_512 );
   DO_DM( VMOVUPD_11_512 );
   DO_DM( VMOVUPS_10_512 );
   DO_DM( VMOVUPS_11_512 );
   DO_DM( VMULPD_59_512 );
   DO_DM( VMULPS_59_512 );
   DO_DM( VPABSD_1E_512 );
   DO_DM( VPABSQ_1F_512 );
   DO_DM( VPADDD_FE_512 );
   DO_DM( VPADDQ_D4_512 );
   DO_DM( VPANDD_DB_512 );
   DO_DM( VPANDND_DF_512 );
   DO_DM( VPANDNQ_DF_512 );
   DO_DM( VPANDQ_DB_512 );
   DO_DM( VPBLENDMD_64_512 );
   DO_DM( VPBLENDMQ_64_512 );
   DO_DM( VPBROADCASTD_58_512 );
   DO_DM( VPBROADCASTQ_59_512 );
   DO_DM( VPCMPD_1F_512 );
   DO_DM( VPCMPEQD_76_512 );
   DO_DM( VPCMPEQQ_29_512 );
   DO_DM( VPCMPEQUD_512 );
   DO_DM( VPCMPGTD_512 );
   DO_DM( VPCMPGTQ_37_512 );
   DO_DM( VPCMPLEUD_512 );
   DO_DM( VPCMPLTUD_512 );
   DO_DM( VPCMPQ_1F_512 );
   DO_DM( VPCMPUD_1E_512 );
   DO_DM( VPCMPUQ_1E_512 );
   DO_DM( VPERMD_36_512 );
   DO_DM( VPERMI2D_76_512 );
   DO_DM( VPERMI2PD_77_512 );
   DO_DM( VPERMI2PS_77_512 );
   DO_DM( VPERMI2Q_76_512 );
   DO_DM( VPERMILPD_05_512 );
   DO_DM( VPERMILPD_0D_512 );
   DO_DM( VPERMILPS_04_512 );
   DO_DM( VPERMILPS_0C_512 );
   DO_DM( VPERMPD_01_512 );
   DO_DM( VPERMPD_16_512 );
   DO_DM( VPERMPS_16_512 );
   DO_DM( VPERMQ_00_512 );
   DO_DM( VPERMQ_36_512 );
   DO_DM( VPERMT2D_7E_512 );
   DO_DM( VPERMT2PD_7F_512 );
   DO_DM( VPERMT2PS_7F_512 );
   DO_DM( VPERMT2Q_7E_512 );
   DO_DM( VPEXPANDD_89_512 );
   DO_DM( VPEXPANDQ_89_512 );
   DO_DM( VPMAXSD_3D_512 );
   DO_DM( VPMAXSQ_3D_512 );
   DO_DM( VPMAXUD_3F_512 );
   DO_DM( VPMAXUQ_3F_512 );
   DO_DM( VPMINSD_39_512 );
   DO_DM( VPMINSQ_39_512 );
   DO_DM( VPMINUD_3B_512 );
   DO_DM( VPMINUQ_3B_512 );
   DO_DM( VPMOVDB_31_512 );
   DO_DM( VPMOVDW_33_512 );
   DO_DM( VPMOVQB_32_512 );
   DO_DM( VPMOVQD_35_512 );
   DO_DM( VPMOVQW_34_512 );
   DO_DM( VPMOVSDB_21_512 );
   DO_DM( VPMOVSDW_23_512 );
   DO_DM( VPMOVSQB_22_512 );
   DO_DM( VPMOVSQD_25_512 );
   DO_DM( VPMOVSQW_24_512 );
   DO_DM( VPMOVSXBQ_22_512 );
   DO_DM( VPMOVSXDQ_25_512 );
   DO_DM( VPMOVSXWD_23_512 );
   DO_DM( VPMOVSXWQ_24_512 );
   DO_DM( VPMOVUSDB_11_512 );
   DO_DM( VPMOVUSDW_13_512 );
   DO_DM( VPMOVUSQB_12_512 );
   DO_DM( VPMOVUSQD_15_512 );
   DO_DM( VPMOVUSQW_14_512 );
   DO_DM( VPMOVZXBD_31_512 );
   DO_DM( VPMOVZXBQ_32_512 );
   DO_DM( VPMOVZXDQ_35_512 );
   DO_DM( VPMOVZXWD_33_512 );
   DO_DM( VPMOVZXWQ_34_512 );
   DO_DM( VPMULDQ_28_512 );
   DO_DM( VPMULLD_40_512 );
   DO_DM( VPMULUDQ_F4_512 );
   DO_DM( VPORD_EB_512 );
   DO_DM( VPORQ_EB_512 );
   DO_DM( VPROLD_72_512 );
   DO_DM( VPROLQ_72_512 );
   DO_DM( VPROLVD_15_512 );
   DO_DM( VPROLVQ_15_512 );
   DO_DM( VPRORD_72_512 );
   DO_DM( VPRORQ_72_512 );
   DO_DM( VPRORVD_14_512 );
   DO_DM( VPRORVQ_14_512 );
   DO_DM( VPSLLQ_F3_512 );
   DO_DM( VPSRAD_E2_512 );
   DO_DM( VPSRAVD_46_512 );
   DO_DM( VPSRAVD_47_512 );
   DO_DM( VPSRAVQ_46_512 );
   DO_DM( VPSRAVQ_47_512 );
   DO_DM( VPSRLQ_D3_512 );
   DO_DM( VPSRLVD_45_512 );
   DO_DM( VPSRLVQ_45_512 );
   DO_DM( VPSUBD_FA_512 );
   DO_DM( VPSUBQ_FB_512 );
   DO_DM( VPTERNLOGD_25_512 );
   DO_DM( VPTERNLOGQ_25_512 );
   DO_DM( VPTESTMD_27_512 );
   DO_DM( VPTESTMQ_27_512 );
   DO_DM( VPUNPCKHDQ_6A_512 );
   DO_DM( VPUNPCKHQDQ_6D_512 );
   DO_DM( VPUNPCKLDQ_62_512 );
   DO_DM( VPUNPCKLQDQ_6C_512 );
   DO_DM( VPXORD_EF_512 );
   DO_DM( VPXORQ_EF_512 );
   DO_DM( VRCP14PD_4C_512 );
   DO_DM( VRCP14PS_4C_512 );
   DO_DM( VRNDSCALEPD_09_512 );
   DO_DM( VRNDSCALEPS_08_512 );
   DO_DM( VRSQRT14PD_4E_512 );
   DO_DM( VRSQRT14PS_4E_512 );
   DO_DM( VSHUFF32x4_23_512 );
   DO_DM( VSHUFF64x2_23_512 );
   DO_DM( VSHUFI32x4_43_512 );
   DO_DM( VSHUFI64x2_43_512 );
   DO_DM( VSHUFPD_C6_512 );
   DO_DM( VSHUFPS_C6_512 );
   DO_DM( VSQRTPD_51_512 );
   DO_DM( VSQRTPS_51_512 );
   DO_DM( VSUBPD_5C_512 );
   DO_DM( VSUBPS_5C_512 );
   DO_DM( VUNPCKHPD_15_512 );
   DO_DM( VUNPCKHPS_15_512 );
   DO_DM( VUNPCKLPD_14_512 );
   DO_DM( VUNPCKLPS_14_512 );
   DO_D( VCMPPD_C4_512 );
   DO_D( VCMPPS_C2_512 );
   DO_D( VGATHERDPD_512 );
   DO_D( VGATHERDPS_512 );
   DO_D( VGATHERQPD_512 );
   DO_D( VGATHERQPS_512 );
   DO_D( VMOVNTDQA_2A_512 );
   DO_D( VMOVNTDQ_E7_512 );
   DO_D( VMOVNTPD_2B_512 );
   DO_D( VMOVNTPS_2B_512 );
   DO_D( VPBROADCASTD_7C_512 );
   DO_D( VPBROADCASTQ_7C_512 );
   DO_D( VPCMPGTD_66_512 );
   DO_D( VPGATHERDD_512 );
   DO_D( VPGATHERDQ_512 );
   DO_D( VPGATHERQD_512 );
   DO_D( VPGATHERQQ_512 );
   DO_D( VPSCATTERDD_512 );
   DO_D( VPSCATTERDQ_512 );
   DO_D( VPSCATTERQD_512 );
   DO_D( VPSCATTERQQ_512 );
   DO_D( VSCATTERDPD_512 );
   DO_D( VSCATTERDPS_512 );
   DO_D( VSCATTERQPD_512 );
   DO_D( VSCATTERQPS_512 );
   DO_DM( VCVTPD2DQ_E6_512 );
   DO_DM( VPCOMPRESSQ_8B_512 );
   DO_DM( VPCOMPRESSD_8B_512 );
   DO_DM( VCOMPRESSPD_8A_512 );
   DO_DM( VCOMPRESSPS_8A_512 );
   DO_DM( VCVTPD2PS_5A_512 );
   DO_DM( VCVTPS2PD_5A_512 );
   DO_DM( VPTESTNMQ_27_512 );
   DO_DM( VPTESTNMD_27_512 );
   DO_DM( VSCALEFPD_2C_512 );
   DO_DM( VSCALEFPS_2C_512 );
   DO_DM( VCVTPS2PH_1D_512 );
   DO_DM( VCVTPH2PS_13_512 );
   DO_DM( VPSHUFD_70_512 );
   DO_DM( VADDSD_58_128 );
   DO_DM( VADDSS_58_128 );
   DO_DM( VCMPSS_C2_128 );
   DO_DM( VDIVSD_5E_128 );
   DO_DM( VDIVSS_5E_128 );
   DO_DM( VCVTSD2SS_5A_128 );
   DO_DM( VCVTSS2SD_5A_128 );
   DO_DM( VGETEXPSD_43_128 );
   DO_DM( VGETEXPSS_43_128 );
   DO_DM( VGETMANTSD_27_128 );
   DO_DM( VGETMANTSS_27_128 );
   DO_DM( VMAXSD_5F_128 );
   DO_DM( VMAXSS_5F_128 );
   DO_DM( VMINSD_5D_128 );
   DO_DM( VMINSS_5D_128 );
   DO_DM( VMOVSD_10_128 );
   DO_DM( VMOVSD_11_128 );
   DO_DM( VMOVSS_10_128 );
   DO_DM( VMOVSS_11_128 );
   DO_DM( VMULSD_59_128 );
   DO_DM( VMULSS_59_128 );
   DO_DM( VRCP14SS_4D_128 );
   DO_DM( VRNDSCALESD_0B_128 );
   DO_DM( VRNDSCALESS_0A_128 );
   DO_DM( VRSQRT14SD_4F_128 );
   DO_DM( VRSQRT14SS_4F_128 );
   DO_DM( VSQRTSD_51_128 );
   DO_DM( VSQRTSS_51_128 );
   DO_DM( VSUBSD_5C_128 );
   DO_DM( VSUBSS_5C_128 );
   DO_D( VCMPSD_C2_128 );
   DO_D( VCOMISD_2F_128 );
   DO_D( VCOMISS_2F_128 );
   DO_D( VCVTSI2SD_2A_128 );
   DO_D( VCVTSI2SS_2A_128 );
   DO_D( VCVTUSI2SD_7B_128 );
   DO_D( VCVTUSI2SS_7B_128 );
   DO_D( VUCOMISD_2E_128 );
   DO_D( VUCOMISD_2F_128 );
   DO_D( VUCOMISS_2E_128 );
   DO_DM( VRCP14SD_4D_128 );
   DO_DM( VFIXUPIMMSD_55_128 );
   DO_DM( VFIXUPIMMSS_55_128 );
   DO_DM( VSCALEFSD_2D_128 );
   DO_DM( VSCALEFSS_2D_128 );
#endif

#ifdef AVX512CD
   DO_D( VPBROADCASTMB2Q_2A_512 );
   DO_D( VPBROADCASTMW2D_3A_512 );
   DO_DM( VPCONFLICTD_C4_512 );
   DO_DM( VPCONFLICTQ_C4_512 );
#endif 

#ifdef AVX512PF
   DO_D(VGATHERPF0DPS_512);
   DO_D(VGATHERPF0DPD_512);
   DO_D(VGATHERPF1DPS_512);
   DO_D(VGATHERPF1DPD_512);
   DO_D(VGATHERPF0QPS_512);
   DO_D(VGATHERPF0QPD_512);
   DO_D(VGATHERPF1QPS_512);
   DO_D(VGATHERPF1QPD_512);
   DO_D(VSCATTERPF0DPS_512);
   DO_D(VSCATTERPF0DPD_512);
   DO_D(VSCATTERPF1DPS_512);
   DO_D(VSCATTERPF1DPD_512);
   DO_D(VSCATTERPF0QPS_512);
   DO_D(VSCATTERPF0QPD_512);
   DO_D(VSCATTERPF1QPS_512);
   DO_D(VSCATTERPF1QPD_512);
#endif

#ifdef AVX512ER
   DO_DM( VRSQRT28SD_CD_128 );
   DO_DM( VRSQRT28SS_CD_128 );
   DO_DM( VRSQRT28PS_CC_512 );
   DO_DM( VRCP28SD_CB_128 );
   DO_DM( VRCP28SS_CB_128 );
   DO_DM( VRCP28PD_CA_512 );
   DO_DM( VRCP28PS_CA_512 );
   DO_DM( VEXP2PD_C8_512 );
   DO_DM( VEXP2PS_C8_512 );
#endif
   /* END REGRESSION TESTS */

#else
   /* in-progress tests */

#endif
   return 0;
}
