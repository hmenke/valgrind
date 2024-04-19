
/*---------------------------------------------------------------*/
/*--- begin                        libvex_basictypes_AVX512.h ---*/
/*---------------------------------------------------------------*/

/*
   This file is part of Valgrind, a dynamic binary instrumentation
   framework.

   Copyright (C) 2004-2017 OpenWorks LLP
      info@open-works.net

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

   Neither the names of the U.S. Department of Energy nor the
   University of California nor the names of its contributors may be
   used to endorse or promote products derived from this software
   without prior written permission.
*/
#ifdef AVX_512
#ifndef __LIBVEX_BASICTYPES512_H
#define __LIBVEX_BASICTYPES512_H

/* Always 512 bits. */
typedef  UInt  U512[16];

/* A union for doing 512-bit vector primitives conveniently. */
typedef union {
   UChar  w8[64];
   UShort w16[32];
   UInt   w32[16];
   ULong  w64[8];
}
V512;

#endif /* ndef __LIBVEX_BASICTYPES512_H */
#endif /* ndef AVX_512 */
/*---------------------------------------------------------------*/
/*--- end                          libvex_basictypes_AVX512.h ---*/
/*---------------------------------------------------------------*/
