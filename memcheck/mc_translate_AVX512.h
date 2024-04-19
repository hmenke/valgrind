/*--------------------------------------------------------------------*/
/*---                                        mc_translate_AVX512.h ---*/
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

#ifndef __MC_TRANSLATE_512_H
#define __MC_TRANSLATE_512_H

IRAtom* mkDifDV512 ( MCEnv* mce, IRAtom* a1, IRAtom* a2 );
IRAtom* mkUifUV512 ( MCEnv* mce, IRAtom* a1, IRAtom* a2 );
IRAtom* CollapseTo1( MCEnv* mce, IRAtom* vbits );
IRAtom* Widen1to512(MCEnv* mce, IRType dst_ty, IRAtom* tmp1);

IRExpr* expr2vbits_Unop_AVX512 ( MCEnv* mce, IROp_EVEX op, IRAtom* vatom );
IRAtom* expr2vbits_Binop_AVX512 ( MCEnv* mce, IROp_EVEX op,
      IRAtom* atom1, IRAtom* atom2,
      IRAtom* vatom1, IRAtom* vatom2);
IRAtom* expr2vbits_Triop_AVX512 ( MCEnv* mce, IROp_EVEX op,
      IRAtom* atom1, IRAtom* atom2, IRAtom* atom3,
      IRAtom* vatom1, IRAtom* vatom2, IRAtom* vatom3 );
IRAtom* expr2vbits_Qop_AVX512 ( MCEnv* mce, IROp_EVEX op,
      IRAtom* atom1, IRAtom* atom2, IRAtom* atom3, IRAtom* atom4,
      IRAtom* vatom1, IRAtom* vatom2, IRAtom* vatom3, IRAtom* vatom4 );

void do_shadow_Store_512 ( MCEnv* mce,
                       IREndness end,
                       IRAtom* addr, UInt bias,
                       IRAtom* data, IRAtom* vdata,
                       IRAtom* guard );


#endif /* ndef __MC_TRANSLATE_512_H */
#endif /* ndef AVX_512 */
/*--------------------------------------------------------------------*/
/*--- end                                    mc_translate_AVX512.c ---*/
/*--------------------------------------------------------------------*/
