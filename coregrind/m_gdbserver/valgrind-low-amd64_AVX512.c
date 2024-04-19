/* Low level interface to valgrind, for the remote server for GDB integrated
   in valgrind. Access to ZMM and opmask (k) registers for AVX-512
   Copyright (C) 2011
   Free Software Foundation, Inc.

   This file is part of VALGRIND.
   It has been inspired from a file from gdbserver in gdb 6.6.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

#ifdef AVX_512

#include "server.h"
#include "target.h"
#include "regdef.h"
#include "regcache.h"

#include "pub_core_machine.h"
#include "pub_core_threadstate.h"
#include "pub_core_transtab.h"
#include "pub_core_gdbserver.h"

#include "valgrind_low.h"

#include "libvex_guest_amd64.h"
#include "../../VEX/priv/guest_generic_x87.h"

void transfer_register_AVX512 (VexGuestAMD64State* amd64, int regno, void * buf,
                        transfer_direction dir, int size, Bool *mod)
{
   switch (regno) {
   case 40: VG_(transfer) (&amd64->guest_ZMM0[0],  buf, dir, size, mod); break;
   case 41: VG_(transfer) (&amd64->guest_ZMM1[0],  buf, dir, size, mod); break;
   case 42: VG_(transfer) (&amd64->guest_ZMM2[0],  buf, dir, size, mod); break;
   case 43: VG_(transfer) (&amd64->guest_ZMM3[0],  buf, dir, size, mod); break;
   case 44: VG_(transfer) (&amd64->guest_ZMM4[0],  buf, dir, size, mod); break;
   case 45: VG_(transfer) (&amd64->guest_ZMM5[0],  buf, dir, size, mod); break;
   case 46: VG_(transfer) (&amd64->guest_ZMM6[0],  buf, dir, size, mod); break;
   case 47: VG_(transfer) (&amd64->guest_ZMM7[0],  buf, dir, size, mod); break;
   case 48: VG_(transfer) (&amd64->guest_ZMM8[0],  buf, dir, size, mod); break;
   case 49: VG_(transfer) (&amd64->guest_ZMM9[0],  buf, dir, size, mod); break;
   case 50: VG_(transfer) (&amd64->guest_ZMM10[0], buf, dir, size, mod); break;
   case 51: VG_(transfer) (&amd64->guest_ZMM11[0], buf, dir, size, mod); break;
   case 52: VG_(transfer) (&amd64->guest_ZMM12[0], buf, dir, size, mod); break;
   case 53: VG_(transfer) (&amd64->guest_ZMM13[0], buf, dir, size, mod); break;
   case 54: VG_(transfer) (&amd64->guest_ZMM14[0], buf, dir, size, mod); break;
   case 55: VG_(transfer) (&amd64->guest_ZMM15[0], buf, dir, size, mod); break;
   case 56:
      if (dir == valgrind_to_gdbserver) {
         // vex only models the rounding bits (see libvex_guest_x86.h)
         UWord value = 0x1f80;
         value |= amd64->guest_SSEROUND << 13;
         VG_(transfer)(&value, buf, dir, size, mod);
      } else {
         *mod = False;  // GDBTD???? VEX equivalent mxcsr
      }
      break;
   case 57: *mod = False; break; // GDBTD???? VEX equivalent { "orig_rax"},
    case 58: VG_(transfer) (&amd64->guest_ZMM0[4],  buf, dir, size, mod); break;
   case 59: VG_(transfer) (&amd64->guest_ZMM1[4],  buf, dir, size, mod); break;
   case 60: VG_(transfer) (&amd64->guest_ZMM2[4],  buf, dir, size, mod); break;
   case 61: VG_(transfer) (&amd64->guest_ZMM3[4],  buf, dir, size, mod); break;
   case 62: VG_(transfer) (&amd64->guest_ZMM4[4],  buf, dir, size, mod); break;
   case 63: VG_(transfer) (&amd64->guest_ZMM5[4],  buf, dir, size, mod); break;
   case 64: VG_(transfer) (&amd64->guest_ZMM6[4],  buf, dir, size, mod); break;
   case 65: VG_(transfer) (&amd64->guest_ZMM7[4],  buf, dir, size, mod); break;
   case 66: VG_(transfer) (&amd64->guest_ZMM8[4],  buf, dir, size, mod); break;
   case 67: VG_(transfer) (&amd64->guest_ZMM9[4],  buf, dir, size, mod); break;
   case 68: VG_(transfer) (&amd64->guest_ZMM10[4], buf, dir, size, mod); break;
   case 69: VG_(transfer) (&amd64->guest_ZMM11[4], buf, dir, size, mod); break;
   case 70: VG_(transfer) (&amd64->guest_ZMM12[4], buf, dir, size, mod); break;
   case 71: VG_(transfer) (&amd64->guest_ZMM13[4], buf, dir, size, mod); break;
   case 72: VG_(transfer) (&amd64->guest_ZMM14[4], buf, dir, size, mod); break;
   case 73: VG_(transfer) (&amd64->guest_ZMM15[4], buf, dir, size, mod); break;
   /* xmm16-31 */
   case 74: VG_(transfer) (&amd64->guest_ZMM16[0], buf, dir, size, mod); break;
   case 75: VG_(transfer) (&amd64->guest_ZMM17[0], buf, dir, size, mod); break;
   case 76: VG_(transfer) (&amd64->guest_ZMM18[0], buf, dir, size, mod); break;
   case 77: VG_(transfer) (&amd64->guest_ZMM19[0], buf, dir, size, mod); break;
   case 78: VG_(transfer) (&amd64->guest_ZMM20[0], buf, dir, size, mod); break;
   case 79: VG_(transfer) (&amd64->guest_ZMM21[0], buf, dir, size, mod); break;
   case 80: VG_(transfer) (&amd64->guest_ZMM22[0], buf, dir, size, mod); break;
   case 81: VG_(transfer) (&amd64->guest_ZMM23[0], buf, dir, size, mod); break;
   case 82: VG_(transfer) (&amd64->guest_ZMM24[0], buf, dir, size, mod); break;
   case 83: VG_(transfer) (&amd64->guest_ZMM25[0], buf, dir, size, mod); break;
   case 84: VG_(transfer) (&amd64->guest_ZMM26[0], buf, dir, size, mod); break;
   case 85: VG_(transfer) (&amd64->guest_ZMM27[0], buf, dir, size, mod); break;
   case 86: VG_(transfer) (&amd64->guest_ZMM28[0], buf, dir, size, mod); break;
   case 87: VG_(transfer) (&amd64->guest_ZMM29[0], buf, dir, size, mod); break;
   case 88: VG_(transfer) (&amd64->guest_ZMM30[0], buf, dir, size, mod); break;
   case 89: VG_(transfer) (&amd64->guest_ZMM31[0], buf, dir, size, mod); break;
   /* ymm16-31 */
   case 90: VG_(transfer) (&amd64->guest_ZMM16[4], buf, dir, size, mod); break;
   case 91: VG_(transfer) (&amd64->guest_ZMM17[4], buf, dir, size, mod); break;
   case 92: VG_(transfer) (&amd64->guest_ZMM18[4], buf, dir, size, mod); break;
   case 93: VG_(transfer) (&amd64->guest_ZMM19[4], buf, dir, size, mod); break;
   case 94: VG_(transfer) (&amd64->guest_ZMM20[4], buf, dir, size, mod); break;
   case 95: VG_(transfer) (&amd64->guest_ZMM21[4], buf, dir, size, mod); break;
   case 96: VG_(transfer) (&amd64->guest_ZMM22[4], buf, dir, size, mod); break;
   case 97: VG_(transfer) (&amd64->guest_ZMM23[4], buf, dir, size, mod); break;
   case 98: VG_(transfer) (&amd64->guest_ZMM24[4], buf, dir, size, mod); break;
   case 99: VG_(transfer) (&amd64->guest_ZMM25[4], buf, dir, size, mod); break;
   case 100: VG_(transfer) (&amd64->guest_ZMM26[4], buf, dir, size, mod); break;
   case 101: VG_(transfer) (&amd64->guest_ZMM27[4], buf, dir, size, mod); break;
   case 102: VG_(transfer) (&amd64->guest_ZMM28[4], buf, dir, size, mod); break;
   case 103: VG_(transfer) (&amd64->guest_ZMM29[4], buf, dir, size, mod); break;
   case 104: VG_(transfer) (&amd64->guest_ZMM30[4], buf, dir, size, mod); break;
   case 105: VG_(transfer) (&amd64->guest_ZMM31[4], buf, dir, size, mod); break;
   /* k */
   case 106: VG_(transfer) (&amd64->guest_MASKREG[0], buf, dir, size, mod); break;
   case 107: VG_(transfer) (&amd64->guest_MASKREG[1], buf, dir, size, mod); break;
   case 108: VG_(transfer) (&amd64->guest_MASKREG[2], buf, dir, size, mod); break;
   case 109: VG_(transfer) (&amd64->guest_MASKREG[3], buf, dir, size, mod); break;
   case 110: VG_(transfer) (&amd64->guest_MASKREG[4], buf, dir, size, mod); break;
   case 111: VG_(transfer) (&amd64->guest_MASKREG[5], buf, dir, size, mod); break;
   case 112: VG_(transfer) (&amd64->guest_MASKREG[6], buf, dir, size, mod); break;
   case 113: VG_(transfer) (&amd64->guest_MASKREG[7], buf, dir, size, mod); break;
   /* zmm */
   case 114: VG_(transfer) (&amd64->guest_ZMM0[8], buf, dir, size, mod); break;
   case 115: VG_(transfer) (&amd64->guest_ZMM1[8], buf, dir, size, mod); break;
   case 116: VG_(transfer) (&amd64->guest_ZMM2[8], buf, dir, size, mod); break;
   case 117: VG_(transfer) (&amd64->guest_ZMM3[8], buf, dir, size, mod); break;
   case 118: VG_(transfer) (&amd64->guest_ZMM4[8], buf, dir, size, mod); break;
   case 119: VG_(transfer) (&amd64->guest_ZMM5[8], buf, dir, size, mod); break;
   case 120: VG_(transfer) (&amd64->guest_ZMM6[8], buf, dir, size, mod); break;
   case 121: VG_(transfer) (&amd64->guest_ZMM7[8], buf, dir, size, mod); break;
   case 122: VG_(transfer) (&amd64->guest_ZMM8[8], buf, dir, size, mod); break;
   case 123: VG_(transfer) (&amd64->guest_ZMM9[8], buf, dir, size, mod); break;
   case 124: VG_(transfer) (&amd64->guest_ZMM10[8], buf, dir, size, mod); break;
   case 125: VG_(transfer) (&amd64->guest_ZMM11[8], buf, dir, size, mod); break;
   case 126: VG_(transfer) (&amd64->guest_ZMM12[8], buf, dir, size, mod); break;
   case 127: VG_(transfer) (&amd64->guest_ZMM13[8], buf, dir, size, mod); break;
   case 128: VG_(transfer) (&amd64->guest_ZMM14[8], buf, dir, size, mod); break;
   case 129: VG_(transfer) (&amd64->guest_ZMM15[8], buf, dir, size, mod); break;
   case 130: VG_(transfer) (&amd64->guest_ZMM16[8], buf, dir, size, mod); break;
   case 131: VG_(transfer) (&amd64->guest_ZMM17[8], buf, dir, size, mod); break;
   case 132: VG_(transfer) (&amd64->guest_ZMM18[8], buf, dir, size, mod); break;
   case 133: VG_(transfer) (&amd64->guest_ZMM19[8], buf, dir, size, mod); break;
   case 134: VG_(transfer) (&amd64->guest_ZMM20[8], buf, dir, size, mod); break;
   case 135: VG_(transfer) (&amd64->guest_ZMM21[8], buf, dir, size, mod); break;
   case 136: VG_(transfer) (&amd64->guest_ZMM22[8], buf, dir, size, mod); break;
   case 137: VG_(transfer) (&amd64->guest_ZMM23[8], buf, dir, size, mod); break;
   case 138: VG_(transfer) (&amd64->guest_ZMM24[8], buf, dir, size, mod); break;
   case 139: VG_(transfer) (&amd64->guest_ZMM25[8], buf, dir, size, mod); break;
   case 140: VG_(transfer) (&amd64->guest_ZMM26[8], buf, dir, size, mod); break;
   case 141: VG_(transfer) (&amd64->guest_ZMM27[8], buf, dir, size, mod); break;
   case 142: VG_(transfer) (&amd64->guest_ZMM28[8], buf, dir, size, mod); break;
   case 143: VG_(transfer) (&amd64->guest_ZMM29[8], buf, dir, size, mod); break;
   case 144: VG_(transfer) (&amd64->guest_ZMM30[8], buf, dir, size, mod); break;
   case 145: VG_(transfer) (&amd64->guest_ZMM31[8], buf, dir, size, mod); break;
   default: vg_assert(0);
   }
}

Bool have_avx512(void) {
   VexArch va;
   VexArchInfo vai;
   VG_(machine_get_VexArchInfo) (&va, &vai);
   Bool knl = (vai.hwcaps & VEX_HWCAPS_AMD64_AVX512_KNL) ? True : False;
   Bool skx = (vai.hwcaps & VEX_HWCAPS_AMD64_AVX512_SKX) ? True : False;
   return (knl || skx);
 }


#endif /* ndef AVX_512 */
