/*---------------------------------------------------------------*/
/*--- begin                         libvex_guest_amd_AVX512.h ---*/
/*---------------------------------------------------------------*/
/*
   This file is part of Valgrind, a dynamic binary instrumentation
   framework.

   Copyright (C) 2021 Intel Corporation
      tatyana.a.volnina@intel.com

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

#ifndef __LIBVEX_PUB_GUEST_AMD512_H
#define __LIBVEX_PUB_GUEST_AMD512_H

#include "libvex_basictypes.h"
#include "libvex_emnote.h"


/*---------------------------------------------------------------*/
/*--- EVEX's representation of the AMD64 CPU state.            ---*/
/*---------------------------------------------------------------*/

/* See detailed comments at the top of libvex_guest_x86.h for
   further info.  This representation closely follows the
   x86 representation.
*/


typedef
   struct {
      /* Event check fail addr, counter, and padding to make RAX 16
         aligned. */
      /*   0 */ ULong  host_EvC_FAILADDR;
      /*   8 */ UInt   host_EvC_COUNTER;
      /*  12 */ UInt   pad0;
      /*  16 */ ULong  guest_RAX;
      /*  24 */ ULong  guest_RCX;
      /*  32 */ ULong  guest_RDX;
      /*  40 */ ULong  guest_RBX;
      /*  48 */ ULong  guest_RSP;
      /*  56 */ ULong  guest_RBP;
      /*  64 */ ULong  guest_RSI;
      /*  72 */ ULong  guest_RDI;
      /*  80 */ ULong  guest_R8;
      /*  88 */ ULong  guest_R9;
      /*  96 */ ULong  guest_R10;
      /* 104 */ ULong  guest_R11;
      /* 112 */ ULong  guest_R12;
      /* 120 */ ULong  guest_R13;
      /* 128 */ ULong  guest_R14;
      /* 136 */ ULong  guest_R15;
      /* 4-word thunk used to calculate O S Z A C P flags. */
      /* 144 */ ULong  guest_CC_OP;
      /* 152 */ ULong  guest_CC_DEP1;
      /* 160 */ ULong  guest_CC_DEP2;
      /* 168 */ ULong  guest_CC_NDEP;
      /* The D flag is stored here, encoded as either -1 or +1 */
      /* 176 */ ULong  guest_DFLAG;
      /* 184 */ ULong  guest_RIP;
      /* Bit 18 (AC) of eflags stored here, as either 0 or 1. */
      /* ... */ ULong  guest_ACFLAG;
      /* Bit 21 (ID) of eflags stored here, as either 0 or 1. */
      /* 192 */ ULong guest_IDFLAG;
      /* Probably a lot more stuff too.
         D,ID flags
         16  128-bit SSE registers
         all the old x87 FPU gunk
         segment registers */

      /* HACK to e.g. make tls on amd64-linux/solaris work.  %fs only ever seems
         to hold a constant value (zero on linux main thread, 0x63 in other
         threads), and so guest_FS_CONST holds
         the 64-bit offset associated with this constant %fs value. */
      /* 200 */ ULong guest_FS_CONST;

      /* ZMM registers.  Note that these must be allocated
         consecutively in order that the SSE4.2 PCMP{E,I}STR{I,M}
         helpers can treat them as an array.
         ZMM32 is a fake reg used as an intermediary */
      /* 208 */ULong guest_SSEROUND;
      /* 216 */
      U512 guest_ZMM0;
      U512 guest_ZMM1;
      U512 guest_ZMM2;
      U512 guest_ZMM3;
      U512 guest_ZMM4;
      U512 guest_ZMM5;
      U512 guest_ZMM6;
      U512 guest_ZMM7;
      U512 guest_ZMM8;
      U512 guest_ZMM9;
      U512 guest_ZMM10;
      U512 guest_ZMM11;
      U512 guest_ZMM12;
      U512 guest_ZMM13;
      U512 guest_ZMM14;
      U512 guest_ZMM15;
      U512 guest_ZMM16;
      U512 guest_ZMM17;
      U512 guest_ZMM18;
      U512 guest_ZMM19;
      U512 guest_ZMM20;
      U512 guest_ZMM21;
      U512 guest_ZMM22;
      U512 guest_ZMM23;
      U512 guest_ZMM24;
      U512 guest_ZMM25;
      U512 guest_ZMM26;
      U512 guest_ZMM27;
      U512 guest_ZMM28;
      U512 guest_ZMM29;
      U512 guest_ZMM30;
      U512 guest_ZMM31;
      U512 guest_ZMM32; //NULL
      ULong guest_MASKREG[8];

      /* FPU */
      /* Note.  Setting guest_FTOP to be ULong messes up the
         delicately-balanced PutI/GetI optimisation machinery.
         Therefore best to leave it as a UInt. */
      UInt  guest_FTOP;
      UInt  pad1;
      ULong guest_FPREG[8];
      UChar guest_FPTAG[8];
      ULong guest_FPROUND;
      ULong guest_FC3210;

      /* Emulation notes */
      UInt  guest_EMNOTE;
      /* Used by Darwin and FreeBSD when setting the carry flag from
       * ML_(do_syscall_for_client_WRK). Needed to determine how
       * to restart interrupted syscalls. */
      UInt guest_SETC;

      /* Translation-invalidation area description.  Not used on amd64
         (there is no invalidate-icache insn), but needed so as to
         allow users of the library to uniformly assume that the guest
         state contains these two fields -- otherwise there is
         compilation breakage.  On amd64, these two fields are set to
         zero by LibVEX_GuestAMD64_initialise and then should be
         ignored forever thereafter. */
      ULong guest_CMSTART;
      ULong guest_CMLEN;

      /* Used to record the unredirected guest address at the start of
         a translation whose start has been redirected.  By reading
         this pseudo-register shortly afterwards, the translation can
         find out what the corresponding no-redirection address was.
         Note, this is only set for wrap-style redirects, not for
         replace-style ones. */
      ULong guest_NRADDR;

      /* Used for Darwin syscall dispatching. */
      ULong guest_SC_CLASS;

      /* HACK to make e.g. tls on darwin work, wine on linux work, ...
         %gs only ever seems to hold a constant value (e.g. 0x60 on darwin,
         0x6b on linux), and so guest_GS_CONST holds the 64-bit offset
         associated with this constant %gs value.  (A direct analogue
         of the %fs-const hack for amd64-linux/solaris). */
      ULong guest_GS_CONST;

      /* Needed for Darwin (but mandated for all guest architectures):
         RIP at the last syscall insn (int 0x80/81/82, sysenter,
         syscall).  Used when backing up to restart a syscall that has
         been interrupted by a signal. */
      ULong guest_IP_AT_SYSCALL;

      /* Padding to make it have an 16-aligned size */
      ULong pad3;
   }
   VexGuestAMD64State;

#define guest_YMM0 guest_ZMM0
#define guest_YMM1 guest_ZMM1
#define guest_YMM2 guest_ZMM2
#define guest_YMM3 guest_ZMM3
#define guest_YMM4 guest_ZMM4
#define guest_YMM5 guest_ZMM5
#define guest_YMM6 guest_ZMM6
#define guest_YMM7 guest_ZMM7
#define guest_YMM8 guest_ZMM8
#define guest_YMM9 guest_ZMM9
#define guest_YMM10 guest_ZMM10
#define guest_YMM11 guest_ZMM11
#define guest_YMM12 guest_ZMM12
#define guest_YMM13 guest_ZMM13
#define guest_YMM14 guest_ZMM14
#define guest_YMM15 guest_ZMM15
#define guest_YMM16 guest_ZMM16

/*---------------------------------------------------------------*/
/*--- Utility functions for amd64 guest stuff.                ---*/
/*---------------------------------------------------------------*/

/* ALL THE FOLLOWING ARE VISIBLE TO LIBRARY CLIENT */

/* Initialise all guest amd64 state.  The FPU is put in default  mode. */
extern void LibVEX_GuestAMD64_initialise_ZMM ( /*OUT*/VexGuestAMD64State* vex_state );

#endif /* ndef __LIBVEX_PUB_GUEST_AMD512_H */
#endif /* ndef AVX_512 */
/*---------------------------------------------------------------*/
/*---                               libvex_guest_amd_AVX512.h ---*/
/*---------------------------------------------------------------*/

