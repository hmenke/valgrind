#ifdef AVX_512
/*--------------------------------------------------------------------*/
/*--- A header file for AVX512 part of the MemCheck tool.           ---*/
/*---                                             mc_include_512.h ---*/
/*--------------------------------------------------------------------*/

/*
   This file is part of MemCheck, a heavyweight Valgrind tool for
   detecting memory errors.

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

#ifndef __MC_INCLUDE_512_H
#define __MC_INCLUDE_512_H

#ifdef MC_PROFILE_MEMORY
/* Profiling of memory events. Order of enumerators does not matter,
 * but MCPE_LAST has to be the last entry */
enum {
   MCPE_LOADV_512 = MCPE_LAST_NOT_EVEX,
   MCPE_LOADV_512_SLOW_LOOP,
   MCPE_LOADV_512_SLOW1,
   MCPE_LOADV_512_SLOW2,
   MCPE_LAST
};
#endif

VG_REGPARM(2) void  MC_(helperc_b_store64)( Addr a, UWord d32 );
VG_REGPARM(1) UWord MC_(helperc_b_load64)( Addr a );

/* V-bits load/store helpers */
VG_REGPARM(2) void  MC_(helperc_LOADV512) ( /*OUT*/V512*, Addr );

#endif /* ndef __MC_INCLUDE_512_H */
#endif /* ndef AVX_512 */
/*--------------------------------------------------------------------*/
/*--- end                                                          ---*/
/*--------------------------------------------------------------------*/
