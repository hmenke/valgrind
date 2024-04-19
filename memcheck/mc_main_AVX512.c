/*--------------------------------------------------------------------*/
/*--- MemCheck: Maintain bitmaps of memory, tracking the           ---*/
/*--- accessibility (A) and validity (V) status of each byte.      ---*/
/*---                                             mc_main_AVX512.c ---*/
/*--------------------------------------------------------------------*/

/*
   This file is part of MemCheck, a heavyweight Valgrind tool for
   detecting memory errors.

   Copyright (C) 2000-2017 Julian Seward
      jseward@acm.org

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <http://www.gnu.org/licenses/>.

   The GNU General Public License is contained in the file COPYING.
*/

#ifdef AVX_512
#include "mc_include_AVX512.h"

/*------------------------------------------------------------*/
/*--- Functions called directly from generated code:       ---*/
/*--- Load/store handlers.                                 ---*/
/*------------------------------------------------------------*/
/* Based on mc_LOADV_128_or_256_slow */
__attribute__((noinline))
void mc_LOADV_512_slow ( /*OUT*/ULong* res, Addr a, SizeT nBits )
{
   ULong pessim[8] = {0}; /* only used when p-l-ok=yes */
   SSizeT szB = nBits / 8;
   SSizeT szL = szB / 8; /* Size in Longs (64-bit units) */
   SSizeT i=0, j=0; /* Must be signed. */
   SizeT  n_addrs_bad = 0;
   Addr ai = 0;
   UChar vbits8 =0 ;
   Bool ok = False;

   /* Code below assumes load size is a power of two and at least 64 bits. */
   tl_assert((szB & (szB-1)) == 0 && szL > 0);

   /* If this triggers, you probably just need to increase the size of the
    * pessim array. */
   tl_assert(szL <= sizeof(pessim) / sizeof(pessim[0]));

   for (j = 0; j < szL; j++) {
      pessim[j] = V_BITS64_DEFINED;
      res[j] = V_BITS64_UNDEFINED;
   }

   /* Make up a result V word, which contains the loaded data for
      valid addresses and Defined for invalid addresses.  Iterate over
      the bytes in the word, from the most significant down to the
      least.  The vbits to return are calculated into vbits128.  Also
      compute the pessimising value to be used when
      --partial-loads-ok=yes.  n_addrs_bad is redundant (the relevant
      info can be gleaned from the pessim array) but is used as a
      cross-check. */
   for (j = szL-1; j >= 0; j--) {
      ULong vbits64    = V_BITS64_UNDEFINED;
      ULong pessim64   = V_BITS64_DEFINED;
      UWord long_index = byte_offset_w(szL, False, j);
      for (i = 8-1; i >= 0; i--) {
         PROF_EVENT(MCPE_LOADV_512_SLOW_LOOP);
         ai = a + 8*long_index + byte_offset_w(8, False, i);
         ok = get_vbits8(ai, &vbits8);
         vbits64 <<= 8;
         vbits64 |= (ULong) vbits8;
         if (!ok) n_addrs_bad++;
         pessim64 <<= 8;
         pessim64 |= (ULong) (ok ? V_BITS8_DEFINED : V_BITS8_UNDEFINED);
      }
      res[long_index] = vbits64;
      pessim[long_index] = pessim64;
   }

   /* In the common case, all the addresses involved are valid, so we
      just return the computed V bits and have done. */
   if (LIKELY(n_addrs_bad == 0))
      return;

   /* If there's no possibility of getting a partial-loads-ok
      exemption, report the error and quit. */
   if (!MC_(clo_partial_loads_ok)) {
      MC_(record_address_error)( VG_(get_running_tid)(), a, szB, False );
      return;
   }

   /* The partial-loads-ok excemption might apply.  Find out if it
      does.  If so, don't report an addressing error, but do return
      Undefined for the bytes that are out of range, so as to avoid
      false negatives.  If it doesn't apply, just report an addressing
      error in the usual way. */

   /* Some code steps along byte strings in aligned chunks
      even when there is only a partially defined word at the end (eg,
      optimised strlen).  This is allowed by the memory model of
      modern machines, since an aligned load cannot span two pages and
      thus cannot "partially fault".

      Therefore, a load from a partially-addressible place is allowed
      if all of the following hold:
      - the command-line flag is set [by default, it isn't]
      - it's an aligned load
      - at least one of the addresses in the word *is* valid

      Since this suppresses the addressing error, we avoid false
      negatives by marking bytes undefined when they come from an
      invalid address.
   */

   /* "at least one of the addresses is invalid" */
   ok = False;
   for (j = 0; j < szL; j++)
      ok |= pessim[j] != V_BITS64_DEFINED;
   tl_assert(ok);

   if (0 == (a & (szB - 1)) && n_addrs_bad < szB) {
      /* Exemption applies.  Use the previously computed pessimising
         value and return the combined result, but don't flag an
         addressing error.  The pessimising value is Defined for valid
         addresses and Undefined for invalid addresses. */
      /* for assumption that doing bitwise or implements UifU */
      tl_assert(V_BIT_UNDEFINED == 1 && V_BIT_DEFINED == 0);
      /* (really need "UifU" here...)
         vbits[j] UifU= pessim[j]  (is pessimised by it, iow) */
      for (j = szL-1; j >= 0; j--)
         res[j] |= pessim[j];
      return;
   }

   /* Exemption doesn't apply.  Flag an addressing error in the normal
      way. */
   MC_(record_address_error)( VG_(get_running_tid)(), a, szB, False );
}


INLINE void mc_LOADV_512 ( /*OUT*/ULong* res, Addr a, SizeT nBits )
{
   PROF_EVENT(MCPE_LOADV_512);

#ifndef PERF_FAST_LOADV
   mc_LOADV_512_slow( res, a, nBits );
   return;
#else
   {
      UWord   sm_off16=0, vabits16=0, j=0;
      UWord   nBytes  = nBits / 8;
      UWord   nULongs = nBytes / 8;
      SecMap* sm = NULL;

      if (UNLIKELY( UNALIGNED_OR_HIGH(a,nBits) )) {
         PROF_EVENT(MCPE_LOADV_512_SLOW1);
         mc_LOADV_512_slow( res, a, nBits );
         return;
      }

      /* Handle common cases quickly: a (and a+8 and a+16 etc.) is
         suitably aligned, is mapped, and addressible. */
      for (j = 0; j < nULongs; j++) {
         sm       = get_secmap_for_reading_low(a + 8*j);
         sm_off16 = SM_OFF_16(a + 8*j);
         vabits16 = sm->vabits16[sm_off16];

         // Convert V bits from compact memory form to expanded
         // register form.
         if (LIKELY(vabits16 == VA_BITS16_DEFINED)) {
            res[j] = V_BITS64_DEFINED;
         } else if (LIKELY(vabits16 == VA_BITS16_UNDEFINED)) {
            res[j] = V_BITS64_UNDEFINED;
         } else {
            /* Slow case: some block of 8 bytes are not all-defined or
               all-undefined. */
            PROF_EVENT(MCPE_LOADV_512_SLOW2);
            mc_LOADV_512_slow( res, a, nBits );
            return;
         }
      }
      return;
   }
#endif
}

VG_REGPARM(2) void MC_(helperc_LOADV512) ( /*OUT*/V512* res, Addr a ) {
   mc_LOADV_512(&res->w64[0], a, 512);
}

void init_prof_mem_evex(void) {
#ifdef MC_PROFILE_MEMORY
   MC_(event_ctr_name)[MCPE_LOADV_512_SLOW_LOOP] = "LOADV_512_slow(loop)";
   MC_(event_ctr_name)[MCPE_LOADV_512]           = "LOADV_512";
   MC_(event_ctr_name)[MCPE_LOADV_512_SLOW1]     = "LOADV_512-slow1";
   MC_(event_ctr_name)[MCPE_LOADV_512_SLOW2]     = "LOADV_512-slow2";
#endif
}

UWord VG_REGPARM(1) MC_(helperc_b_load64)( Addr a ) {
   UInt oLo = (UInt)MC_(helperc_b_load32)( a + 0 );
   UInt oHi = (UInt)MC_(helperc_b_load32)( a + 32 );
   UInt oBoth = merge_origins(oLo, oHi);
   return (UWord)oBoth;
}

void VG_REGPARM(2) MC_(helperc_b_store64)( Addr a, UWord d32 ) {
   MC_(helperc_b_store32)( a +  0, d32 );
   MC_(helperc_b_store32)( a + 32, d32 );
}

#endif /* ndef AVX_512 */
/*--------------------------------------------------------------------*/
/*--- end                                         mc_main_AVX512.c ---*/
/*--------------------------------------------------------------------*/
