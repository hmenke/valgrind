/* Low level interface to valgrind, for the remote server for GDB integrated
   in valgrind.
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
#ifndef __VALGRIND_LOW_AMD64_AVX512_H
#define __VALGRIND_LOW_AMD64_AVX512_H

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
/* GDBTD: ??? have a cleaner way to get the f80 <> f64 conversion functions */
/* below include needed for conversion f80 <> f64 */
#include "../../VEX/priv/guest_generic_x87.h"

struct reg regs[] = {
  { "rax", 0, 64 },
  { "rbx", 64, 64 },
  { "rcx", 128, 64 },
  { "rdx", 192, 64 },
  { "rsi", 256, 64 },
  { "rdi", 320, 64 },
  { "rbp", 384, 64 },
  { "rsp", 448, 64 },
  { "r8", 512, 64 },
  { "r9", 576, 64 },
  { "r10", 640, 64 },
  { "r11", 704, 64 },
  { "r12", 768, 64 },
  { "r13", 832, 64 },
  { "r14", 896, 64 },
  { "r15", 960, 64 },
  { "rip", 1024, 64 },
  { "eflags", 1088, 32 },
  { "cs", 1120, 32 },
  { "ss", 1152, 32 },
  { "ds", 1184, 32 },
  { "es", 1216, 32 },
  { "fs", 1248, 32 },
  { "gs", 1280, 32 },
  { "st0", 1312, 80 },
  { "st1", 1392, 80 },
  { "st2", 1472, 80 },
  { "st3", 1552, 80 },
  { "st4", 1632, 80 },
  { "st5", 1712, 80 },
  { "st6", 1792, 80 },
  { "st7", 1872, 80 },
  { "fctrl", 1952, 32 },
  { "fstat", 1984, 32 },
  { "ftag", 2016, 32 },
  { "fiseg", 2048, 32 },
  { "fioff", 2080, 32 },
  { "foseg", 2112, 32 },
  { "fooff", 2144, 32 },
  { "fop", 2176, 32 },
  { "xmm0", 2208, 128 },
  { "xmm1", 2336, 128 },
  { "xmm2", 2464, 128 },
  { "xmm3", 2592, 128 },
  { "xmm4", 2720, 128 },
  { "xmm5", 2848, 128 },
  { "xmm6", 2976, 128 },
  { "xmm7", 3104, 128 },
  { "xmm8", 3232, 128 },
  { "xmm9", 3360, 128 },
  { "xmm10", 3488, 128 },
  { "xmm11", 3616, 128 },
  { "xmm12", 3744, 128 },
  { "xmm13", 3872, 128 },
  { "xmm14", 4000, 128 },
  { "xmm15", 4128, 128 },
  { "mxcsr", 4256, 32  },
#if defined(VGO_linux)
  { "orig_rax", 4288, 64 },
#endif
  /*
  { "fs_base", 4352, 64 },
  { "gs_base", 4416, 64 },
  */
  { "ymm0h", 4480, 128 },
  { "ymm1h", 4608, 128 },
  { "ymm2h", 4736, 128 },
  { "ymm3h", 4864, 128 },
  { "ymm4h", 4992, 128 },
  { "ymm5h", 5120, 128 },
  { "ymm6h", 5248, 128 },
  { "ymm7h", 5376, 128 },
  { "ymm8h", 5504, 128 },
  { "ymm9h", 5632, 128 },
  { "ymm10h", 5760, 128 },
  { "ymm11h", 5888, 128 },
  { "ymm12h", 6016, 128 },
  { "ymm13h", 6144, 128 },
  { "ymm14h", 6272, 128 },
  { "ymm15h", 6400, 128 },
  { "xmm16", 6528, 128 },
  { "xmm17", 6656, 128 },
  { "xmm18", 6784, 128 },
  { "xmm19", 6912, 128 },
  { "xmm20", 7040, 128 },
  { "xmm21", 7168, 128 },
  { "xmm22", 7296, 128 },
  { "xmm23", 7424, 128 },
  { "xmm24", 7552, 128 },
  { "xmm25", 7680, 128 },
  { "xmm26", 7808, 128 },
  { "xmm27", 7936, 128 },
  { "xmm28", 8064, 128 },
  { "xmm29", 8192, 128 },
  { "xmm30", 8320, 128 },
  { "xmm31", 8448, 128 },
  { "ymm16h", 8576, 128 },
  { "ymm17h", 8704, 128 },
  { "ymm18h", 8832, 128 },
  { "ymm19h", 8960, 128 },
  { "ymm20h", 9088, 128 },
  { "ymm21h", 9216, 128 },
  { "ymm22h", 9344, 128 },
  { "ymm23h", 9472, 128 },
  { "ymm24h", 9600, 128 },
  { "ymm25h", 9728, 128 },
  { "ymm26h", 9856, 128 },
  { "ymm27h", 9984, 128 },
  { "ymm28h", 10112, 128 },
  { "ymm29h", 10240, 128 },
  { "ymm30h", 10368, 128 },
  { "ymm31h", 10496, 128 },
  { "k0", 10624, 64 },
  { "k1", 10688, 64 },
  { "k2", 10752, 64 },
  { "k3", 10816, 64 },
  { "k4", 10880, 64 },
  { "k5", 10944, 64 },
  { "k6", 11008, 64 },
  { "k7", 11072, 64 },
  { "zmm0h", 11136, 256 },
  { "zmm1h", 11392, 256 },
  { "zmm2h", 11648, 256 },
  { "zmm3h", 11904, 256 },
  { "zmm4h", 12160, 256 },
  { "zmm5h", 12416, 256 },
  { "zmm6h", 12672, 256 },
  { "zmm7h", 12928, 256 },
  { "zmm8h", 13184, 256 },
  { "zmm9h", 13440, 256 },
  { "zmm10h", 13696, 256 },
  { "zmm11h", 13952, 256 },
  { "zmm12h", 14208, 256 },
  { "zmm13h", 14464, 256 },
  { "zmm14h", 14720, 256 },
  { "zmm15h", 14976, 256 },
  { "zmm16h", 15232, 256 },
  { "zmm17h", 15488, 256 },
  { "zmm18h", 15744, 256 },
  { "zmm19h", 16000, 256 },
  { "zmm20h", 16256, 256 },
  { "zmm21h", 16512, 256 },
  { "zmm22h", 16768, 256 },
  { "zmm23h", 17024, 256 },
  { "zmm24h", 17280, 256 },
  { "zmm25h", 17536, 256 },
  { "zmm26h", 17792, 256 },
  { "zmm27h", 18048, 256 },
  { "zmm28h", 18304, 256 },
  { "zmm29h", 18560, 256 },
  { "zmm30h", 18816, 256 },
  { "zmm31h", 19072, 256 }
};

void transfer_register_AVX512 (VexGuestAMD64State* amd64, int regno, void * buf,
                        transfer_direction dir, int size, Bool *mod);
#endif
#endif /* ndef AVX_512 */
