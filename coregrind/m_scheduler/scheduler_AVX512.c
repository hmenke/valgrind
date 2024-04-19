
/*--------------------------------------------------------------------*/
/*--- Thread scheduling.                        scheduler_AVX512.c ---*/
/*--------------------------------------------------------------------*/

/*
   This file is part of Valgrind, a dynamic binary instrumentation
   framework.

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
#include "pub_core_threadstate.h"
#include "libvex_guest_amd_AVX512.h"

static void do_pre_run_checks_AVX512 ( volatile ThreadState* tst ){
   vg_assert(
      (offsetof(VexGuestAMD64State,guest_ZMM32)
       - offsetof(VexGuestAMD64State,guest_ZMM0))
      == (33/*#regs*/-1) * 64/*bytes per reg*/
   );

   vg_assert(VG_IS_16_ALIGNED(offsetof(VexGuestAMD64State,guest_ZMM0)));
}

#endif /* ndef AVX_512 */
/*--------------------------------------------------------------------*/
/*--- end                                                          ---*/
/*--------------------------------------------------------------------*/

