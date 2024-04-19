/*--------------------------------------------------------------------*/
/*--- Instrument IR to perform memory checking operations.         ---*/
/*---                                        mc_translate_AVX512.c ---*/
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

/*------------------------------------------------------------*/
/*--- Constructing definedness primitive ops               ---*/
/*------------------------------------------------------------*/

IRAtom* mkDifDV512 ( MCEnv* mce, IRAtom* a1, IRAtom* a2 ) {
   tl_assert(isShadowAtom(mce,a1));
   tl_assert(isShadowAtom(mce,a2));
   return assignNew('V', mce, Ity_V512, binop(Iop_AndV512, a1, a2));
}

IRAtom* mkUifUV512 ( MCEnv* mce, IRAtom* a1, IRAtom* a2 ) {
   tl_assert(isShadowAtom(mce,a1));
   tl_assert(isShadowAtom(mce,a2));
   return assignNew('V', mce, Ity_V512, binop(Iop_OrV512, a1, a2));
}

/* --------- 'Improvement' functions for AND/OR. --------- */

IRAtom* mkImproveANDV512 ( MCEnv* mce, IRAtom* data, IRAtom* vbits ) {
   tl_assert(isOriginalAtom(mce, data));
   tl_assert(isShadowAtom(mce, vbits));
   tl_assert(sameKindedAtoms(data, vbits));
   return assignNew('V', mce, Ity_V512, binop(Iop_OrV512, data, vbits));
}

IRAtom* mkImproveORV512 ( MCEnv* mce, IRAtom* data, IRAtom* vbits ) {
   tl_assert(isOriginalAtom(mce, data));
   tl_assert(isShadowAtom(mce, vbits));
   tl_assert(sameKindedAtoms(data, vbits));
   return assignNew(
         'V', mce, Ity_V512,
         binop(Iop_OrV512,
            assignNew('V', mce, Ity_V512, unop(Iop_NotV512, data)),
            vbits) );
}

/* --------- Pessimising casts. --------- */

/* The function returns an expression of type DST_TY. If any of the VBITS
   is undefined (value == 1) the resulting expression has all bits set to 1.
   Otherwise, all bits are 0. */
IRAtom* CollapseTo1( MCEnv* mce, IRAtom* vbits )
{
   IRType src_ty = typeOfIRExpr(mce->sb->tyenv, vbits);
   switch (src_ty) {
      case Ity_V256: {
         IRAtom* tmp1 = NULL;
         IRAtom* tmp_128_hi = assignNew('V', mce, Ity_V128, unop(Iop_V256toV128_1, vbits));
         IRAtom* tmp_128_lo = assignNew('V', mce, Ity_V128, unop(Iop_V256toV128_0, vbits));
         IRAtom* tmp_128    = assignNew('V', mce, Ity_V128, binop(Iop_OrV128, tmp_128_hi, tmp_128_lo));
         IRAtom* tmp_64_hi  = assignNew('V', mce,  Ity_I64, unop(Iop_V128HIto64, tmp_128));
         IRAtom* tmp_64_lo  = assignNew('V', mce,  Ity_I64, unop(Iop_V128to64, tmp_128));
         IRAtom* tmp_64 = assignNew('V', mce, Ity_I64, binop(Iop_Or64, tmp_64_hi, tmp_64_lo));
         tmp1 = assignNew('V', mce, Ity_I1, unop(Iop_CmpNEZ64, tmp_64));
         return tmp1;
      }
      case Ity_V512: {
         IRAtom* tmp1 = NULL;
         IRAtom* tmp_256_hi = assignNew('V', mce, Ity_V256, unop(Iop_V512toV256_1, vbits));
         IRAtom* tmp_256_lo = assignNew('V', mce, Ity_V256, unop(Iop_V512toV256_0, vbits));
         IRAtom* tmp_256    = assignNew('V', mce, Ity_V256, binop(Iop_OrV256, tmp_256_hi, tmp_256_lo));
         IRAtom* tmp_128_hi = assignNew('V', mce, Ity_V128, unop(Iop_V256toV128_1, tmp_256));
         IRAtom* tmp_128_lo = assignNew('V', mce, Ity_V128, unop(Iop_V256toV128_0, tmp_256));
         IRAtom* tmp_128    = assignNew('V', mce, Ity_V128, binop(Iop_OrV128, tmp_128_hi, tmp_128_lo));
         IRAtom* tmp_64_hi  = assignNew('V', mce,  Ity_I64, unop(Iop_V128HIto64, tmp_128));
         IRAtom* tmp_64_lo  = assignNew('V', mce,  Ity_I64, unop(Iop_V128to64, tmp_128));
         IRAtom* tmp_64 = assignNew('V', mce, Ity_I64, binop(Iop_Or64, tmp_64_hi, tmp_64_lo));
         tmp1 = assignNew('V', mce, Ity_I1, unop(Iop_CmpNEZ64, tmp_64));
         return tmp1;
      }
      default: break;
   }
   ppIRType(src_ty);
   VG_(tool_panic)("CollapseTo1");
}

IRAtom* Widen1to512(MCEnv* mce, IRType dst_ty, IRAtom* tmp1) {
   if( dst_ty == Ity_V512 ) {
      tmp1 = assignNew('V', mce, Ity_I64,  unop(Iop_1Sto64, tmp1));
      tmp1 = assignNew('V', mce, Ity_V128, binop(Iop_64HLtoV128, tmp1, tmp1));
      tmp1 = assignNew('V', mce, Ity_V256, binop(Iop_V128HLtoV256, tmp1, tmp1));
      tmp1 = assignNew('V', mce, Ity_V512, binop(Iop_V256HLtoV512, tmp1, tmp1));
      return tmp1;
   }
   ppIRType(dst_ty);
   VG_(tool_panic)("Widen1to512");
}

IRAtom* mkPCast64x8 ( MCEnv* mce, IRAtom* at ) {
   return assignNew('V', mce, Ity_V512, unop(Iop_CmpNEZ64x8, at));
}

IRAtom* mkPCast32x16 ( MCEnv* mce, IRAtom* at ) {
   return assignNew('V', mce, Ity_V512, unop(Iop_CmpNEZ32x16, at));
}

static
IRAtom* unary64Fx8 ( MCEnv* mce, IRAtom* vatomX )
{
   IRAtom* at;
   tl_assert(isShadowAtom(mce, vatomX));
   at = assignNew('V', mce, Ity_V512, mkPCast64x8(mce, vatomX));
   return at;
}
static
IRAtom* unary32Fx16 ( MCEnv* mce, IRAtom* vatomX )
{
   IRAtom* at;
   tl_assert(isShadowAtom(mce, vatomX));
   at = assignNew('V', mce, Ity_V512, mkPCast32x16(mce, vatomX));
   return at;
}

static
IRAtom* binary64Fx8 ( MCEnv* mce, IRAtom* vatomX, IRAtom* vatomY ) {
   IRAtom* at;
   tl_assert(isShadowAtom(mce, vatomX));
   tl_assert(isShadowAtom(mce, vatomY));
   at = mkUifUV512(mce, vatomX, vatomY);
   at = assignNew('V', mce, Ity_V512, mkPCast64x8(mce, at));
   return at;
}

static
IRAtom* binary32Fx16 ( MCEnv* mce, IRAtom* vatomX, IRAtom* vatomY )
{
   IRAtom* at;
   tl_assert(isShadowAtom(mce, vatomX));
   tl_assert(isShadowAtom(mce, vatomY));
   at = mkUifUV512(mce, vatomX, vatomY);
   at = assignNew('V', mce, Ity_V512, mkPCast32x16(mce, at));
   return at;
}

/*------------------------------------------------------------*/
/*--- Generate shadow values from all kinds of IRExprs.    ---*/
/*------------------------------------------------------------*/

IRExpr* expr2vbits_Unop_AVX512 ( MCEnv* mce, IROp_EVEX op, IRAtom* vatom )
{
   tl_assert(isShadowAtom(mce,vatom));
   IRType t_dst, t_arg1, t_arg2, t_arg3, t_arg4;
   typeOfPrimop(op, &t_dst, &t_arg1, &t_arg2, &t_arg3, &t_arg4);

   switch (op) {
      case Iop_NotV512:
      case Iop_PrintI64:
         return vatom;

      // repeat the operation on shadow values
      case Iop_V512toV256_0: case Iop_V512toV256_1:
      case Iop_V512to64_0: case Iop_V512to64_1:
      case Iop_V512to64_2: case Iop_V512to64_3:
      case Iop_V512to64_4: case Iop_V512to64_5:
      case Iop_V512to64_6: case Iop_V512to64_7:
         return assignNew('V', mce, t_dst, unop(op, vatom));

      // If any source bit is undefined, entire result is undefined
      // TODO: actually, only its part closer to the least significant element
      case Iop_CfD32x4: case Iop_CfD32x8: case Iop_CfD32x16:
      case Iop_CfD64x2: case Iop_CfD64x4: case Iop_CfD64x8:
      case Iop_16Sto8x8: // TODO fix
         return mkPCastTo(mce, t_dst, vatom);

      // simply pessimize elements
      case Iop_Recip14_32x16:
      case Iop_RSqrt14_32x16:
      case Iop_ExtractExp32x16: // TODO ignore undefined sign or mantissa
      case Iop_Exp32x16:
      case Iop_Recip28_32x16:
      case Iop_RSqrt28_32x16:
      case Iop_CmpNEZ32x16:
         return unary32Fx16(mce, vatom);
      case Iop_Recip14_64x8:
      case Iop_RSqrt14_64x8:
      case Iop_ExtractExp64x8: // TODO ignore undefined sign or mantissa
      case Iop_Exp64x8:
      case Iop_Recip28_64x8:
      case Iop_RSqrt28_64x8:
      case Iop_CmpNEZ64x8:
         return unary64Fx8(mce, vatom);

      default:
         ppIROp(op);
         VG_(tool_panic)("memcheck:expr2vbits_Unop_AVX512");
   }
}

IRAtom* expr2vbits_Binop_AVX512 ( MCEnv* mce, IROp_EVEX op,
                           IRAtom* atom1, IRAtom* atom2,
                           IRAtom* vatom1, IRAtom* vatom2)
{
   tl_assert(isOriginalAtom(mce, atom1));
   tl_assert(isOriginalAtom(mce, atom2));
   tl_assert(isShadowAtom(mce, vatom1));
   tl_assert(isShadowAtom(mce, vatom2));

   IRAtom* (*uifu)    (MCEnv*, IRAtom*, IRAtom*);
   IRAtom* (*difd)    (MCEnv*, IRAtom*, IRAtom*);
   IRAtom* (*improve) (MCEnv*, IRAtom*, IRAtom*);
   uifu = mkUifUV512;
   difd = mkDifDV512;
   IRType t_dst, t_arg1, t_arg2, t_arg3, t_arg4;
   typeOfPrimop(op, &t_dst, &t_arg1, &t_arg2, &t_arg3, &t_arg4);

   switch (op) {
      // repeat on shadows
      case Iop_V256HLtoV512:
         return assignNew('V', mce, t_dst, binop(op, vatom1, vatom2));

      // repeat the operation on shadow 1st source with same width (2nd src)
      case Iop_ExpandBitsToV128:
      case Iop_ExpandBitsToV256:
      case Iop_ExpandBitsToV512:
         return assignNew('V', mce, t_dst, binop(op, vatom1, atom2));

      case Iop_AndV512:
         improve = mkImproveANDV512;
         return
            assignNew( 'V', mce, Ity_V512,
                  difd(mce, uifu(mce, vatom1, vatom2),
                     difd(mce, improve(mce, atom1, vatom1),
                        improve(mce, atom2, vatom2) ) ) );
      case Iop_OrV512:
         improve = mkImproveORV512;
         return
            assignNew( 'V', mce, Ity_V512,
                  difd(mce, uifu(mce, vatom1, vatom2),
                     difd(mce, improve(mce, atom1, vatom1),
                        improve(mce, atom2, vatom2) ) ) );

      case Iop_Perm8x32: case Iop_Perm8x64:
      case Iop_Perm16x8: case Iop_Perm16x16: case Iop_Perm16x32:
      case Iop_Perm32x16:
      case Iop_Perm64x4:  case Iop_Perm64x8:
         // TODO Check validity of entire second arg and permute by it
         return assignNew('V', mce, t_dst, binop(op, atom1, vatom2));

      // ignore imm8
      case Iop_Reduce32x4:
         return unary32Fx4(mce, vatom1);
      case Iop_Reduce64x2:
         return unary64Fx2(mce, vatom1);
      case Iop_RoundScale32x16:
      case Iop_GetMant32x16: // TODO ignore sign and exponent
         return unary32Fx16(mce, vatom1);
      case Iop_RoundScale64x8:
      case Iop_GetMant64x8: // TODO ignore sign and exponent
         return unary64Fx8(mce, vatom1);
      case Iop_Max64Sx8: case Iop_Max64Ux8:
      case Iop_Min64Sx8: case Iop_Min64Ux8:
      case Iop_Scale64x8:
         return binary64Fx8(mce, vatom1, vatom2);
      case Iop_Scale32x16:
         return binary32Fx16(mce, vatom1, vatom2);

      // pseudo-serial cases
      case Iop_ExtractExp32F0x4:
      case Iop_Recip14_32F0x4:
      case Iop_RSqrt14_32F0x4:
      case Iop_Recip28_32F0x4:
      case Iop_RSqrt28_32F0x4:
      case Iop_Scale32F0x4:
         return binary32F0x4(mce, vatom1, vatom2);
      case Iop_ExtractExp64F0x2:
      case Iop_Recip14_64F0x2:
      case Iop_RSqrt14_64F0x2:
      case Iop_Recip28_64F0x2:
      case Iop_RSqrt28_64F0x2:
      case Iop_Scale64F0x2:
         return binary64F0x2(mce, vatom1, vatom2);

      case Iop_I32UtoF32_SKX:
      case Iop_I64UtoF64_SKX:
      case Iop_F32toI32U_SKX:
      case Iop_F64toI32U_SKX:
      case Iop_F64toI64U_SKX:
         // vatom1 is rmode, TODO check if defined
         return mkPCastTo(mce, t_dst, vatom2);

      case Iop_ExpandBitsToInt:
         // repeat the operation on shadow values; vatom1 is dummy
         return assignNew('V', mce, t_dst, binop(op, vatom1, vatom2));

      default:
         ppIROp(op);
         VG_(tool_panic)("memcheck:expr2vbits_Binop_AVX512");
   }
}

#define qop(_op, _arg1, _arg2, _arg3, _arg4) \
                                 IRExpr_Qop((_op),(_arg1),(_arg2),(_arg3),(_arg4))

IRAtom* expr2vbits_Triop_AVX512 ( MCEnv* mce, IROp_EVEX op,
      IRAtom* atom1, IRAtom* atom2, IRAtom* atom3,
      IRAtom* vatom1, IRAtom* vatom2, IRAtom* vatom3 )
{
   tl_assert(isOriginalAtom(mce, atom1));
   tl_assert(isOriginalAtom(mce, atom2));
   tl_assert(isOriginalAtom(mce, atom3));
   tl_assert(isShadowAtom(mce, vatom1));
   tl_assert(isShadowAtom(mce, vatom2));
   tl_assert(isShadowAtom(mce, vatom3));

   IRType t_dst, t_arg1, t_arg2, t_arg3, t_arg4;
   typeOfPrimop(op, &t_dst, &t_arg1, &t_arg2, &t_arg3, &t_arg4);
   IRAtom* (*uifu)    (MCEnv*, IRAtom*, IRAtom*);
   IRAtom* (*difd)    (MCEnv*, IRAtom*, IRAtom*);
   IRAtom* (*improve) (MCEnv*, IRAtom*, IRAtom*);
   IRAtom* at;

   switch (op) {
      // simple case, atom 3 is imm8
      case Iop_Range32x4:
         return binary32Fx4(mce, vatom1, vatom2);
      case Iop_Range64x2:
         return binary64Fx2(mce, vatom1, vatom2);

      // repeat op on vatoms using the same imm8
      case Iop_PermI8x16: case Iop_PermI8x32:  case Iop_PermI8x64:
      case Iop_PermI16x8: case Iop_PermI16x16: case Iop_PermI16x32:
      case Iop_PermI32x4: case Iop_PermI32x8:  case Iop_PermI32x16:
      case Iop_PermI64x2: case Iop_PermI64x4:  case Iop_PermI64x8:
         return assignNew('V', mce, t_dst, triop(op, vatom1, vatom2, atom3));

      // vatom1 is dummy, vatom 3 is imm8 and irrelevant
      // TODO properly compress into mask?
      case Iop_Classify32x4:
      case Iop_Classify32x8:
      case Iop_Classify32x16:
      case Iop_Classify64x2:
      case Iop_Classify64x4:
      case Iop_Classify64x8:
      case Iop_Classify32F0x4: // only check and set lowest 32 bits of vatom2
      case Iop_Classify64F0x2: // only check and set lowest 64 bits of vatom2
         return mkPCastTo(mce, t_dst, vatom2);

      // Approximate: AND and undef mask if anything stays undefined
      // TODO Exact: AND and compress into mask
      case Iop_Test16x32:  case Iop_TestN16x32:
      case Iop_Test8x64:   case Iop_TestN8x64:
      case Iop_Test32x16:  case Iop_TestN32x16:
      case Iop_Test64x8:   case Iop_TestN64x8:
         uifu = mkUifUV512;
         difd = mkDifDV512;
         improve = mkImproveANDV512;
         goto do_Test_approx;
      case Iop_Test16x16:  case Iop_TestN16x16:
      case Iop_Test8x32:   case Iop_TestN8x32:
      case Iop_Test32x8:   case Iop_TestN32x8:
      case Iop_Test64x4:   case Iop_TestN64x4:
         uifu = mkUifUV256;
         difd = mkDifDV256;
         improve = mkImproveANDV256;
         goto do_Test_approx;
      case Iop_Test16x8:   case Iop_TestN16x8:
      case Iop_Test8x16:   case Iop_TestN8x16:
      case Iop_Test32x4:   case Iop_TestN32x4:
      case Iop_Test64x2:   case Iop_TestN64x2:
         uifu = mkUifUV128;
         difd = mkDifDV128;
         improve = mkImproveANDV128;
         goto do_Test_approx;
      do_Test_approx:
         at = difd(mce, uifu(mce, vatom2, vatom3),
                        difd(mce, improve(mce, atom2, vatom2),
                                  improve(mce, atom3, vatom3)));
         return mkPCastTo(mce, t_dst, at);

      case Iop_GetMant32F0x4: // TODO ignore exp and sign?
      case Iop_Range64F0x2:
      case Iop_Reduce32F0x4:
      case Iop_RoundScale32F0x4:
         return binary32F0x4(mce, vatom1, vatom2);
      case Iop_Range32F0x4:
      case Iop_GetMant64F0x2: // TODO ignore exp and sign?
      case Iop_Reduce64F0x2:
      case Iop_RoundScale64F0x2:
         return binary64F0x2(mce, vatom1, vatom2);

      case Iop_VDBPSADBW: // Exact - just awful, TODO
         return binary64F0x2(mce, vatom1, vatom2);

      default:
         ppIROp(op);
         VG_(tool_panic)("memcheck:expr2vbits_Triop_AVX512");
   }
}


#define qop(_op, _arg1, _arg2, _arg3, _arg4) \
   IRExpr_Qop((_op),(_arg1),(_arg2),(_arg3),(_arg4))

IRAtom* expr2vbits_Qop_AVX512 ( MCEnv* mce, IROp_EVEX op,
      IRAtom* atom1, IRAtom* atom2, IRAtom* atom3, IRAtom* atom4,
      IRAtom* vatom1, IRAtom* vatom2, IRAtom* vatom3, IRAtom* vatom4 )
{
   tl_assert(isOriginalAtom(mce, atom3));
   tl_assert(isOriginalAtom(mce, atom4));
   tl_assert(isShadowAtom(mce, vatom1));
   tl_assert(isShadowAtom(mce, vatom2));
   tl_assert(isShadowAtom(mce, vatom3));
   tl_assert(isShadowAtom(mce, vatom4));

   IRType t_dst, t_arg1, t_arg2, t_arg3, t_arg4;
   typeOfPrimop(op, &t_dst, &t_arg1, &t_arg2, &t_arg3, &t_arg4);
   IRAtom* at;
   IRAtom* (*pessim_u)  (MCEnv*, IRAtom*);

   switch (op) {
      case Iop_Compress32x4: case Iop_Compress32x8: case Iop_Compress32x16:
      case Iop_Compress64x2: case Iop_Compress64x4: case Iop_Compress64x8:
         return assignNew('V', mce, t_dst,
               qop(op, vatom1, vatom2, atom3, atom4));

      // TODO handle "TRUE" and "FALSE" comparators separately
      case Iop_Cmp8Sx16: case Iop_Cmp8Sx32:  case Iop_Cmp8Sx64:
      case Iop_Cmp16Sx8: case Iop_Cmp16Sx16: case Iop_Cmp16Sx32:
      case Iop_Cmp32Sx4: case Iop_Cmp32Sx8:  case Iop_Cmp32Sx16:
      case Iop_Cmp64Sx2: case Iop_Cmp64Sx4:  case Iop_Cmp64Sx8:
      case Iop_Cmp8Ux16: case Iop_Cmp8Ux32:  case Iop_Cmp8Ux64:
      case Iop_Cmp16Ux8: case Iop_Cmp16Ux16: case Iop_Cmp16Ux32:
      case Iop_Cmp32Ux4: case Iop_Cmp32Ux8:  case Iop_Cmp32Ux16:
      case Iop_Cmp64Ux2: case Iop_Cmp64Ux4:  case Iop_Cmp64Ux8:
      case Iop_Cmp32Fx4: case Iop_Cmp32Fx8:  case Iop_Cmp32Fx16:
      case Iop_Cmp64Fx2: case Iop_Cmp64Fx4:  case Iop_Cmp64Fx8:
         at = mkPCastTo(mce, Ity_I1, vatom2);
         at = assignNew('V', mce, Ity_I1, binop(Iop_Or1, at, mkPCastTo(mce, Ity_I1, vatom3)));
         return mkPCastTo(mce, Ity_I64, at);
      case Iop_FixupImm32F0x4:
         return binary32F0x4(mce, vatom1, vatom2);
      case Iop_FixupImm64F0x2:
         return binary64F0x2(mce, vatom1, vatom2);
      case Iop_Scale32x16:
      case Iop_FixupImm32x16:
      case Iop_Ternlog32x16: // TODO
         return binary32Fx16(mce, vatom1, vatom2);
      case Iop_Scale64x8:
      case Iop_FixupImm64x8:
      case Iop_Ternlog64x8: // TODO
         return binary64Fx8(mce, vatom1, vatom2);
      
      // Tricky! TODO Check that 1st is all defined
      // pessimize second and expand using the 1st
      case Iop_Expand32x4:  pessim_u = unary32Fx4;  goto do_Expand;
      case Iop_Expand32x8:  pessim_u = unary32Fx8;  goto do_Expand;
      case Iop_Expand32x16: pessim_u = unary32Fx16; goto do_Expand;
      case Iop_Expand64x2:  pessim_u = unary64Fx2;  goto do_Expand;
      case Iop_Expand64x4:  pessim_u = unary64Fx4;  goto do_Expand;
      case Iop_Expand64x8:  pessim_u = unary64Fx8;  goto do_Expand;
      do_Expand:
         at = pessim_u(mce, vatom2);
         return assignNew('V', mce, t_dst, qop(op, atom1, at, atom3, atom4));
      default: 
         ppIROp(op);
         VG_(tool_panic)("memcheck:expr2vbits_Qop_AVX512");
   }
}


void do_shadow_Store_512 ( MCEnv* mce,
                       IREndness end,
                       IRAtom* addr, UInt bias,
                       IRAtom* data, IRAtom* vdata,
                       IRAtom* guard )
{
   tl_assert(end == Iend_LE);
   if (MC_(clo_mc_level) == 1) {
      vdata = IRExpr_Const( IRConst_V512(V_BITS32_DEFINED) );
   }
   complainIfUndefined( mce, addr, guard );

   IRType tyAddr = mce->hWordTy;
   IROp mkAdd = (tyAddr==Ity_I32) ? Iop_Add32 : Iop_Add64;
   void* helper = &MC_(helperc_STOREV64le);
   const HChar* hname = "MC_(helperc_STOREV64le)";

   IRDirty *diQ[8] = {NULL};
   IRAtom  *addrQ[8] = {NULL}, *vdataQ[8] = {NULL}, *eBiasQ[8] = {NULL};
   for (Int i = 0; i < 8; i++) {
      eBiasQ[i] = tyAddr==Ity_I32 ? mkU32(bias + i*8 ) : mkU64(bias + i*8);
      addrQ[i]  = assignNew('V', mce, tyAddr, binop(mkAdd, addr, eBiasQ[i]) );
      vdataQ[i] = assignNew('V', mce, Ity_I64, unop(Iop_V512to64_0+i, vdata));
      diQ[i]    = unsafeIRDirty_0_N(
            1/*regparms*/,
            hname, VG_(fnptr_to_fnentry)( helper ),
            mkIRExprVec_2( addrQ[i], vdataQ[i] )
            );
      if (guard)
         diQ[i]->guard = guard;
      setHelperAnns( mce, diQ[i] );
      stmt( 'V', mce, IRStmt_Dirty(diQ[i]) );
   }
   return;
}

#endif /* ndef AVX_512 */
/*--------------------------------------------------------------------*/
/*--- end                                    mc_translate_AVX512.c ---*/
/*--------------------------------------------------------------------*/
