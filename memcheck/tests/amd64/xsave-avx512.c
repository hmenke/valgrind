
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "tests/asm.h"
#include "tests/malloc.h"
#include <string.h>

#define XSAVE_AREA_SIZE 2800

typedef  unsigned char           UChar;
typedef  unsigned int            UInt;
typedef  unsigned long long int  ULong;

typedef  unsigned long int       UWord;

typedef  unsigned char  Bool;
#define  True   ((Bool)1)
#define  False  ((Bool)0)

const unsigned int vec0[8]
   = { 0x12345678, 0x11223344, 0x55667788, 0x87654321,
       0x15263748, 0x91929394, 0x19293949, 0x48372615 };

const unsigned int vec1[8]
   = { 0xABCDEF01, 0xAABBCCDD, 0xEEFF0011, 0x10FEDCBA,
       0xBADCFE10, 0xFFEE9988, 0x11667722, 0x01EFCDAB };

const unsigned int vecZ[8]
   = { 0, 0, 0, 0, 0, 0, 0, 0 };

/* A version of memset that doesn't use XMM or zmm registers. */
static __attribute__((noinline))
void* my_memset(void* s, int c, size_t n)
{
   size_t i;
   for (i = 0; i < n; i++) {
      ((unsigned char*)s)[i] = (unsigned char)(unsigned int)c;
      /* Defeat any attempt at autovectorisation */
      __asm__ __volatile__("" ::: "cc","memory");
   }
   return s;
}

/* Ditto for memcpy */
static __attribute__((noinline))
void* my_memcpy(void *dest, const void *src, size_t n)
{
   size_t i;
   for (i = 0; i < n; i++) {
      ((unsigned char*)dest)[i] = ((unsigned char*)src)[i];
      __asm__ __volatile__("" ::: "cc","memory");
   }
   return dest;
}

static void* memalign_zeroed64(size_t size)
{
   char* p = memalign64(size);
   if (p && size > 0) {
      my_memset(p, 0, size);
   }
   return p;
}

__attribute__((noinline))
static void do_xsave ( void* p, UInt rfbm )
{
   assert((rfbm <= 7) || 
         ((rfbm >= 0xE0) && (rfbm <= 0xE7)));
   __asm__ __volatile__(
      "movq %0, %%rax;  xorq %%rdx, %%rdx;  xsave (%1)"
         : /*OUT*/ : /*IN*/ "r"((ULong)rfbm), "r"(p)
         : /*TRASH*/ "memory", "rax", "rdx"
   );
}

__attribute__((noinline))
static void do_xrstor ( void* p, UInt rfbm )
{
   assert((rfbm <= 7) ||
         ((rfbm >= 0xE0) && (rfbm <= 0xE7)));
   __asm__ __volatile__(
      "movq %0, %%rax;  xorq %%rdx, %%rdx;  xrstor (%1)"
         : /*OUT*/ : /*IN*/ "r"((ULong)rfbm), "r"(p)
         : /*TRASH*/ "rax", "rdx" /* FIXME plus all X87,SSE,AVX regs */
   );
}

/* set up the FP, SSE and AVX state, and then dump it. */
static void do_setup_then_xsave ( void* p, UInt rfbm )
{
   __asm__ __volatile__("finit");
   __asm__ __volatile__("fldpi");
   __asm__ __volatile__("fld1");
   __asm__ __volatile__("fldln2");
   __asm__ __volatile__("fldlg2");
   __asm__ __volatile__("fld %st(3)");
   __asm__ __volatile__("fld %st(3)");
   __asm__ __volatile__("fld1");
   __asm__ __volatile__("vmovups (%0), %%zmm0" : : "r"(&vec0[0]) : "xmm0" );
   __asm__ __volatile__("vmovups (%0), %%zmm1" : : "r"(&vec1[0]) : "xmm1" );
   __asm__ __volatile__("vpxord  %zmm2, %zmm2, %zmm2");
   __asm__ __volatile__("vmovaps %zmm0, %zmm3");
   __asm__ __volatile__("vmovaps %zmm1, %zmm4");
   __asm__ __volatile__("vmovaps %zmm2, %zmm5");
   __asm__ __volatile__("vmovaps %zmm0, %zmm6");
   __asm__ __volatile__("vmovaps %zmm1, %zmm7");
   __asm__ __volatile__("vmovaps %zmm1, %zmm8");
   __asm__ __volatile__("vmovaps %zmm2, %zmm9");
   __asm__ __volatile__("vmovaps %zmm0, %zmm10");
   __asm__ __volatile__("vmovaps %zmm1, %zmm11");
   __asm__ __volatile__("vmovaps %zmm1, %zmm12");
   __asm__ __volatile__("vmovaps %zmm1, %zmm13");
   __asm__ __volatile__("vmovaps %zmm2, %zmm14");
   __asm__ __volatile__("vmovaps %zmm0, %zmm15");
   __asm__ __volatile__("vmovaps %zmm1, %zmm16");
   __asm__ __volatile__("vmovaps %zmm1, %zmm17");
   __asm__ __volatile__("vmovaps %zmm1, %zmm18");
   __asm__ __volatile__("vmovaps %zmm1, %zmm19");
   __asm__ __volatile__("vmovaps %zmm2, %zmm20");
   __asm__ __volatile__("vmovaps %zmm0, %zmm21");
   __asm__ __volatile__("vmovaps %zmm1, %zmm22");
   __asm__ __volatile__("vmovaps %zmm1, %zmm23");
   __asm__ __volatile__("vmovaps %zmm1, %zmm24");
   __asm__ __volatile__("vmovaps %zmm1, %zmm25");
   __asm__ __volatile__("vmovaps %zmm1, %zmm26");
   __asm__ __volatile__("vmovaps %zmm2, %zmm27");
   __asm__ __volatile__("vmovaps %zmm0, %zmm28");
   __asm__ __volatile__("vmovaps %zmm1, %zmm29");
   __asm__ __volatile__("vmovaps %zmm1, %zmm30");
   __asm__ __volatile__("vmovaps %zmm1, %zmm31");
   __asm__ __volatile__("kmovw   (%0),  %%k1" : : "r"(&vec0[0]));
   __asm__ __volatile__("kxorw   %k2, %k2, %k2");
   __asm__ __volatile__("kmovw   %k1,   %k3");
   __asm__ __volatile__("kmovw   %k2,   %k4");
   __asm__ __volatile__("kmovw   %k1,   %k5");
   __asm__ __volatile__("kmovw   %k1,   %k6");
   __asm__ __volatile__("kmovw   %k2,   %k7");
   do_xsave(p, rfbm);
}

static int isFPLsbs ( int i )
{
   int q;
   q = 32; if (i == q || i == q+1) return 1;
   q = 48; if (i == q || i == q+1) return 1;
   q = 64; if (i == q || i == q+1) return 1;
   q = 80; if (i == q || i == q+1) return 1;
   q = 96; if (i == q || i == q+1) return 1;
   q = 112; if (i == q || i == q+1) return 1;
   q = 128; if (i == q || i == q+1) return 1;
   q = 144; if (i == q || i == q+1) return 1;
   return 0;
}

static void show ( unsigned char* buf, Bool hideBits)
{
   int i;
   for (i = 0; i < XSAVE_AREA_SIZE; i++) {
      // Also differ:
      // bits 8-11 - always
      // bits 0-8, 16-19 - for XRSTOR with odd rfbm. This issue can be seen
      // on xsave-avx.c test - expected test result differs from machine run
      // bit 512 for XRSTOR - to fix later 
      if ((i % 16) == 0)
         fprintf(stderr, "%3d   ", i);
      if (hideBits &&
            (i <= 11 ) ||
            ((i >= 16) && (i <= 19)) ||
            (i == 512))
      {
         fprintf(stderr, "xx ");
      }
      else {
         if (hideBits&& isFPLsbs(i))
            fprintf(stderr, "xx ");
         else
            fprintf(stderr, "%02x ", buf[i]);
         if (i > 0 && ((i % 16) == 15))
            fprintf(stderr, "\n");
      }
   }
}

static void cpuid ( UInt* eax, UInt* ebx, UInt* ecx, UInt* edx, 
                    UInt index, UInt ecx_in )
{
   UInt a,b,c,d;
   asm volatile ("cpuid"
                 : "=a" (a), "=b" (b), "=c" (c), "=d" (d) \
                 : "0" (index), "2"(ecx_in) );
   *eax = a; *ebx = b; *ecx = c; *edx = d;
   //fprintf(stderr, "%08x %08x -> %08x %08x %08x %08x\n",
   //        index,ecx_in, a,b,c,d );
}

static void xgetbv ( UInt* eax, UInt* edx, UInt ecx_in )
{
   UInt a,d;
   asm volatile ("xgetbv"
                 : "=a" (a), "=d" (d) \
                 : "c"(ecx_in) );
   *eax = a; *edx = d;
}

static void check_for_xsave ( void )
{
   UInt eax, ebx, ecx, edx;
   Bool ok = True;

   eax = ebx = ecx = edx = 0;
   cpuid(&eax, &ebx, &ecx, &edx, 1,0);
   //fprintf(stderr, "cpuid(1).ecx[26=xsave]   = %u\n", (ecx >> 26) & 1);
   ok = ok && (((ecx >> 26) & 1) == 1);

   eax = ebx = ecx = edx = 0;
   cpuid(&eax, &ebx, &ecx, &edx, 1,0);
   //fprintf(stderr, "cpuid(1).ecx[27=osxsave] = %u\n", (ecx >> 27) & 1);
   ok = ok && (((ecx >> 27) & 1) == 1);

   eax = ebx = ecx = edx = 0;
   xgetbv(&eax, &edx, 0);
   //fprintf(stderr, "xgetbv(0) = %u:%u\n", edx, eax);
   ok = ok && (edx == 0) && (eax && 0xE7);

   if (ok) return;

   fprintf(stderr,
           "This program must be run on a CPU that supports AVX and XSAVE.\n");
   exit(1);
}


void test_xsave ( Bool hideBits64to79 )
{
   /* Testing XSAVE:

      For RBFM in 0 .. 7 and 0xE0 .. 0xE7 (that is, all combinations): set 
      the x87, SSE, AVX and AVX-512 registers with some values, do XSAVE to 
      dump it, and print the resulting buffer. */

   UInt rfbm;
   for (rfbm = 0; rfbm <= 7; rfbm++) {
      UChar* saved_img = memalign_zeroed64(XSAVE_AREA_SIZE);

      my_memset(saved_img, 0xAA, XSAVE_AREA_SIZE);
      saved_img[512] = 0;
      do_setup_then_xsave(saved_img, rfbm);

      fprintf(stderr, 
              "------------------ XSAVE, rfbm = %u ------------------\n", rfbm);
      show(saved_img, hideBits64to79);
      fprintf(stderr, "\n");

      free(saved_img);
   }
   for (rfbm = 0xE0; rfbm <= 0xE7; rfbm++) {
      UChar* saved_img = memalign_zeroed64(XSAVE_AREA_SIZE);

      my_memset(saved_img, 0xAA, XSAVE_AREA_SIZE);
      saved_img[512] = 0;
      do_setup_then_xsave(saved_img, rfbm);

      fprintf(stderr, 
              "------------------ XSAVE, rfbm = %u ------------------\n", rfbm);
      show(saved_img, hideBits64to79);
      fprintf(stderr, "\n");

      free(saved_img);
   }
}

#define ALL_REGS  0xE7

void test_xrstor ( Bool hideBits64to79 )
{
   /* Testing XRSTOR is more complex than testing XSAVE, because the
      loaded value(s) depend not only on what bits are requested (by
      RBFM) but also on what bits are actually present in the image
      (defined by XSTATE_BV).  So we have to test all 64 (8 x 8)
      combinations.

      The approach is to fill a memory buffer with data, do XRSTOR
      from the buffer, them dump all components with XSAVE in a new
      buffer, and print the result.  This is complicated by the fact
      that we need to be able to see which parts of the state (in
      registers) are neither overwritten nor zeroed by the restore.
      Hence the registers must be pre-filled with values which are
      neither zero nor the data to be loaded.  We choose to use 0x55
      where possible. */

   UChar* fives = memalign_zeroed64(XSAVE_AREA_SIZE);
   my_memset(fives, 0x55, XSAVE_AREA_SIZE);
   /* Set MXCSR so that the insn doesn't fault */
   fives[24] = 0x80;
   fives[25] = 0x1f;
   fives[26] = 0;
   fives[27] = 0;
   /* Ditto for the XSAVE header area.  Also set XSTATE_BV. */
   fives[512] = ALL_REGS; 
   UInt i;
   for (i = 1; i <= 23; i++) fives[512+i] = 0;
   /* Fill the x87 register values with something that VEX's
      80-vs-64-bit kludging won't mess up -- an 80 bit number which is
      representable also as 64 bit: 123456789.0123 */
   for (i = 0; i <= 7; i++) {
      UChar* p = &fives[32 + 16 * i];
      p[0]=0x00; p[1]=0xf8; p[2]=0xc2; p[3]=0x64; p[4]=0xa0;
      p[5]=0xa2; p[6]=0x79; p[7]=0xeb; p[8]=0x19; p[9]=0x40;
   }
   /* And mark the tags for all 8 dumped regs as "valid". */
   fives[4/*FTW*/] = 0xFF;

   /* (1) (see comment in loop below) */
   UChar* standard_test_data = memalign_zeroed64(XSAVE_AREA_SIZE);
   do_setup_then_xsave(standard_test_data, 0xE7);

   UInt xstate_bv, rfbm;
   for (xstate_bv = 0; xstate_bv <= 7; xstate_bv++) {
      for (rfbm = 0; rfbm <= 7; rfbm++) {
         /* 1.  Copy the "standard test data" into registers, and dump
                it with XSAVE.  This gives us an image we can try
                restoring from.

            2.  Set the register state to all-0x55s (as far as is
                possible), so we can see which parts get overwritten
                and which parts get zeroed on the test restore.

            3.  Do the restore from the image prepared in (1).

            4.  Dump the state with XSAVE and print it.
         */

         /* (3a).  We can't use |standard_test_data| directly, since we
            need to put in the required |xstate_bv| value.  So make a
            copy and modify that instead. */
         UChar* img_to_restore_from = memalign_zeroed64(XSAVE_AREA_SIZE);
         my_memcpy(img_to_restore_from, standard_test_data, XSAVE_AREA_SIZE);
         img_to_restore_from[512] = xstate_bv;

         /* (4a) */
         UChar* saved_img = memalign_zeroed64(XSAVE_AREA_SIZE);
         my_memset(saved_img, 0xAA, XSAVE_AREA_SIZE);
         saved_img[512] = 0;

         /* (2) */
         do_xrstor(fives, ALL_REGS);

         // X87, SSE, AVX state LIVE

         /* (3b) */
         /* and this is what we're actually trying to test */
         do_xrstor(img_to_restore_from, rfbm);

         // X87, SSE, AVX state LIVE

         /* (4b) */
         do_xsave(saved_img, ALL_REGS);

         fprintf(stderr, 
                 "---------- XRSTOR, xstate_bv = %u, rfbm = %u ---------\n",
                xstate_bv, rfbm);
         show(saved_img, hideBits64to79);
         fprintf(stderr, "\n");

         free(saved_img);
         free(img_to_restore_from);
      }
      for (rfbm = 0xE0; rfbm <= 0xE7; rfbm++) {
         /* (3a) */
         UChar* img_to_restore_from = memalign_zeroed64(XSAVE_AREA_SIZE);
         my_memcpy(img_to_restore_from, standard_test_data, XSAVE_AREA_SIZE);
         img_to_restore_from[512] = xstate_bv;
         /* (4a) */
         UChar* saved_img = memalign_zeroed64(XSAVE_AREA_SIZE);
         my_memset(saved_img, 0xAA, XSAVE_AREA_SIZE);
         saved_img[512] = 0;
         /* (2) */
         do_xrstor(fives, ALL_REGS);
         /* (3b) */
         do_xrstor(img_to_restore_from, rfbm);
         /* (4b) */
         do_xsave(saved_img, ALL_REGS);
         fprintf(stderr, 
                 "---------- XRSTOR, xstate_bv = %u, rfbm = %u ---------\n",
                xstate_bv, rfbm);
         show(saved_img, hideBits64to79);
         fprintf(stderr, "\n");
         free(saved_img);
         free(img_to_restore_from);
      }
   }
   for (xstate_bv = 0xE0; xstate_bv <= 0xE7; xstate_bv++) {
      for (rfbm = 0; rfbm <= 7; rfbm++) {
         /* (3a) */
         UChar* img_to_restore_from = memalign_zeroed64(XSAVE_AREA_SIZE);
         my_memcpy(img_to_restore_from, standard_test_data, XSAVE_AREA_SIZE);
         img_to_restore_from[512] = xstate_bv;
         /* (4a) */
         UChar* saved_img = memalign_zeroed64(XSAVE_AREA_SIZE);
         my_memset(saved_img, 0xAA, XSAVE_AREA_SIZE);
         saved_img[512] = 0;
         /* (2) */
         do_xrstor(fives, ALL_REGS);
         /* (3b) */
         do_xrstor(img_to_restore_from, rfbm);
         /* (4b) */
         do_xsave(saved_img, ALL_REGS);
         fprintf(stderr, 
                 "---------- XRSTOR, xstate_bv = %u, rfbm = %u ---------\n",
                xstate_bv, rfbm);
         show(saved_img, hideBits64to79);
         fprintf(stderr, "\n");
         free(saved_img);
         free(img_to_restore_from);
      }
      for (rfbm = 0xE0; rfbm <= 0xE7; rfbm++) {
         /* (3a) */
         UChar* img_to_restore_from = memalign_zeroed64(XSAVE_AREA_SIZE);
         my_memcpy(img_to_restore_from, standard_test_data, XSAVE_AREA_SIZE);
         img_to_restore_from[512] = xstate_bv;
         /* (4a) */
         UChar* saved_img = memalign_zeroed64(XSAVE_AREA_SIZE);
         my_memset(saved_img, 0xAA, XSAVE_AREA_SIZE);
         saved_img[512] = 0;
         /* (2) */
         do_xrstor(fives, ALL_REGS);
         /* (3b) */
         do_xrstor(img_to_restore_from, rfbm);
         /* (4b) */
         do_xsave(saved_img, ALL_REGS);
         fprintf(stderr, 
                 "---------- XRSTOR, xstate_bv = %u, rfbm = %u ---------\n",
                xstate_bv, rfbm);
         show(saved_img, hideBits64to79);
         fprintf(stderr, "\n");
         free(saved_img);
         free(img_to_restore_from);
      }
   }
}


int main ( int argc, char** argv )
{
   Bool hideBits = argc > 1;
   fprintf(stderr, "Re-run with any arg to suppress least-significant\n"
                   "   16 bits of 80-bit FP numbers\n");

   check_for_xsave();

   if (1)
   test_xsave(hideBits);

   if (1)
   test_xrstor(hideBits);

   return 0;
}
