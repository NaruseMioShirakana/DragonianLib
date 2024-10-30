#pragma once

#define _D_Dragonian_Lib_Operator_Loop_Single(_Info_1, _Dim, _NDim, _BeginIdx, _FrontSize, _Function)							\
for (SizeType _Loop_Value_##_NDim = (_BeginIdx); _Loop_Value_##_NDim < (_Info_1).Shape[_Dim]; ++_Loop_Value_##_NDim)		\
{																														\
	const auto IndexAxis##_NDim##A = ((_Loop_Value_##_NDim * (_Info_1).ViewStride[_Dim])									\
		+ (_Info_1).ViewLeft[_Dim]) * (_Info_1).ViewStep[_Dim] + (_FrontSize);											\
	{ _Function }																										\
}


#define _D_Dragonian_Lib_Operator_Loop_Double(_Info_1, _Info_2, _Dim, _NDim, _BeginIdx, _FS1, _FS2, _Function)					\
for (SizeType _Loop_Value_##_NDim = (_BeginIdx); _Loop_Value_##_NDim < (_Info_1).Shape[_Dim]; ++_Loop_Value_##_NDim)		\
{																														\
	const auto IndexAxis##_NDim##A = ((_Loop_Value_##_NDim * (_Info_1).ViewStride[_Dim])									\
		+ (_Info_1).ViewLeft[_Dim]) * (_Info_1).ViewStep[_Dim] + (_FS1);											\
	const auto IndexAxis##_NDim##B = ((_Loop_Value_##_NDim * (_Info_2).ViewStride[_Dim])									\
		+ (_Info_2).ViewLeft[_Dim]) * (_Info_2).ViewStep[_Dim] + (_FS2);											\
	{ _Function }																										\
}


#define _D_Dragonian_Lib_Operator_Loop_Triple(_Info_1, _Info_2, _Info_3, _Dim, _NDim, _BeginIdx, _FS1, _FS2, _FS3, _Function)	\
for (SizeType _Loop_Value_##_NDim = (_BeginIdx); _Loop_Value_##_NDim < (_Info_1).Shape[_Dim]; ++_Loop_Value_##_NDim)		\
{																														\
	const auto IndexAxis##_NDim##A = ((_Loop_Value_##_NDim * (_Info_1).ViewStride[_Dim])									\
		+ (_Info_1).ViewLeft[_Dim]) * (_Info_1).ViewStep[_Dim] + (_FS1);											\
	const auto IndexAxis##_NDim##B = ((_Loop_Value_##_NDim * (_Info_2).ViewStride[_Dim])									\
		+ (_Info_2).ViewLeft[_Dim]) * (_Info_2).ViewStep[_Dim] + (_FS2);											\
	const auto IndexAxis##_NDim##C = ((_Loop_Value_##_NDim * (_Info_3).ViewStride[_Dim])									\
		+ (_Info_3).ViewLeft[_Dim]) * (_Info_3).ViewStep[_Dim] + (_FS3);											\
	{ _Function }																										\
}


#define _D_Dragonian_Lib_Operator_Loop_Quad(_Info_1, _Info_2, _Info_3, _Info_4, _Dim, _NDim, _BeginIdx, _FS1, _FS2, _FS3, _FS4, _Function)	\
for (SizeType _Loop_Value_##_NDim = (_BeginIdx); _Loop_Value_##_NDim < (_Info_1).Shape[_Dim]; ++_Loop_Value_##_NDim)		\
{																														\
	const auto IndexAxis##_NDim##A = ((_Loop_Value_##_NDim * (_Info_1).ViewStride[_Dim])									\
		+ (_Info_1).ViewLeft[_Dim]) * (_Info_1).ViewStep[_Dim] + (_FS1);											\
	const auto IndexAxis##_NDim##B = ((_Loop_Value_##_NDim * (_Info_2).ViewStride[_Dim])									\
		+ (_Info_2).ViewLeft[_Dim]) * (_Info_2).ViewStep[_Dim] + (_FS2);											\
	const auto IndexAxis##_NDim##C = ((_Loop_Value_##_NDim * (_Info_3).ViewStride[_Dim])									\
		+ (_Info_3).ViewLeft[_Dim]) * (_Info_3).ViewStep[_Dim] + (_FS3);											\
	const auto IndexAxis##_NDim##D = ((_Loop_Value_##_NDim * (_Info_4).ViewStride[_Dim])									\
		+ (_Info_4).ViewLeft[_Dim]) * (_Info_4).ViewStep[_Dim] + (_FS4);											\
	{ _Function }																										\
}


#define _D_Dragonian_Lib_Operator_Loop_Reversed_Single(_Info_1, _Dim, _NDim, _BeginIdx, _FrontSize, _Function)					\
for (SizeType _Loop_Value_##_NDim = (_Info_1).Shape[_Dim]; _Loop_Value_##_NDim >= (_BeginIdx); --_Loop_Value_##_NDim)		\
{																														\
	const auto IndexAxis##_NDim##A = ((_Loop_Value_##_NDim * (_Info_1).ViewStride[_Dim])									\
		+ (_Info_1).ViewLeft[_Dim]) * (_Info_1).ViewStep[_Dim] + (_FrontSize);											\
	{ _Function }																										\
}


#define _D_Dragonian_Lib_Operator_Loop_Reversed_Double(_Info_1, _Info_2, _Dim, _NDim, _BeginIdx, _FS1, _FS2, _Function)			\
for (SizeType _Loop_Value_##_NDim = (_Info_1).Shape[_Dim]; _Loop_Value_##_NDim >= (_BeginIdx); --_Loop_Value_##_NDim)		\
{																														\
	const auto IndexAxis##_NDim##A = ((_Loop_Value_##_NDim * (_Info_1).ViewStride[_Dim])									\
		+ (_Info_1).ViewLeft[_Dim]) * (_Info_1).ViewStep[_Dim] + (_FS1);											\
	const auto IndexAxis##_NDim##B = ((_Loop_Value_##_NDim * (_Info_2).ViewStride[_Dim])									\
		+ (_Info_2).ViewLeft[_Dim]) * (_Info_2).ViewStep[_Dim] + (_FS2);											\
	{ _Function }																										\
}


#define _D_Dragonian_Lib_Operator_Loop_Reversed_Triple(_Info_1, _Info_2, _Info_3, _Dim, _NDim, _BeginIdx, _FS1, _FS2, _FS3, _Function)\
for (SizeType _Loop_Value_##_NDim = (_Info_1).Shape[_Dim]; _Loop_Value_##_NDim >= (_BeginIdx); --_Loop_Value_##_NDim)		\
{																														\
	const auto IndexAxis##_NDim##A = ((_Loop_Value_##_NDim * (_Info_1).ViewStride[_Dim])									\
		+ (_Info_1).ViewLeft[_Dim]) * (_Info_1).ViewStep[_Dim] + (_FS1);											\
	const auto IndexAxis##_NDim##B = ((_Loop_Value_##_NDim * (_Info_2).ViewStride[_Dim])									\
		+ (_Info_2).ViewLeft[_Dim]) * (_Info_2).ViewStep[_Dim] + (_FS2);											\
	const auto IndexAxis##_NDim##C = ((_Loop_Value_##_NDim * (_Info_3).ViewStride[_Dim])									\
		+ (_Info_3).ViewLeft[_Dim]) * (_Info_3).ViewStep[_Dim] + (_FS3);											\
	{ _Function }																										\
}


#define _D_Dragonian_Lib_Operator_Loop_S_0(_I1, _D, _DN, _B0, _FS, _F)															\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, _D, 0##_DN, _B0, _FS, _F)													\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_S_1(_I1, _D, _DN, _B0, _B1, _FS, _F)														\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, _D, 0##_DN, _B0, _FS,														\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A, _F))								\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_S_2(_I1, _D, _DN, _B0, _B1, _B2, _FS, _F)												\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, _D, 0##_DN, _B0, _FS,														\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A,									\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, (_D) + 2, 2##_DN, _B2, IndexAxis1##_DN##A, _F)))								\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_S_3(_I1, _D, _DN, _B0, _B1, _B2, _B3, _FS, _F)											\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, _D, 0##_DN, _B0, _FS,														\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A,									\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, (_D) + 2, 2##_DN, _B2, IndexAxis1##_DN##A,									\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, (_D) + 3, 3##_DN, _B3, IndexAxis2##_DN##A, _F))))							\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_S_4(_I1, _D, _DN, _B0, _B1, _B2, _B3, _B4, _FS, _F)										\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, _D, 0##_DN, _B0, _FS,														\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A,									\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, (_D) + 2, 2##_DN, _B2, IndexAxis1##_DN##A,									\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, (_D) + 3, 3##_DN, _B3, IndexAxis2##_DN##A,									\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, (_D) + 4, 4##_DN, _B4, IndexAxis3##_DN##A, _F)))))							\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_S_5(_I1, _D, _DN, _B0, _B1, _B2, _B3, _B4, _B5, _FS, _F)									\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, _D, 0##_DN, _B0, _FS,														\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A,									\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, (_D) + 2, 2##_DN, _B2, IndexAxis1##_DN##A,									\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, (_D) + 3, 3##_DN, _B3, IndexAxis2##_DN##A,									\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, (_D) + 4, 4##_DN, _B4, IndexAxis3##_DN##A,									\
		_D_Dragonian_Lib_Operator_Loop_Single(_I1, (_D) + 5, 5##_DN, _B5, IndexAxis4##_DN##A, _F))))))							\
	} while (0)


#define _D_Dragonian_Lib_Operator_Loop_D_0(_I1, _I2, _D, _DN, _B0, _FS1, _FS2, _F)														\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, _D, 0##_DN, _B0, _FS1, _FS2, _F)										\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_D_1(_I1, _I2, _D, _DN, _B0, _B1, _FS1, _FS2, _F)												\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, _D, 0##_DN, _B0, _FS1, _FS2,												\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A, IndexAxis0##_DN##B, _F))		\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_D_2(_I1, _I2, _D, _DN, _B0, _B1, _B2, _FS1, _FS2, _F)									\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, _D, 0##_DN, _B0, _FS1, _FS2,												\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A, IndexAxis0##_DN##B,				\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, (_D) + 2, 2##_DN, _B2, IndexAxis1##_DN##A, IndexAxis1##_DN##B, _F)))		\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_D_3(_I1, _I2, _D, _DN, _B0, _B1, _B2, _B3, _FS1, _FS2, _F)						\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, _D, 0##_DN, _B0, _FS1, _FS2,												\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A, IndexAxis0##_DN##B,				\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, (_D) + 2, 2##_DN, _B2, IndexAxis1##_DN##A, IndexAxis1##_DN##B,				\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, (_D) + 3, 3##_DN, _B3, IndexAxis2##_DN##A, IndexAxis2##_DN##B, _F))))		\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_D_4(_I1, _I2, _D, _DN, _B0, _B1, _B2, _B3, _B4, _FS1, _FS2, _F)			\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, _D, 0##_DN, _B0, _FS1, _FS2,												\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A, IndexAxis0##_DN##B,				\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, (_D) + 2, 2##_DN, _B2, IndexAxis1##_DN##A, IndexAxis1##_DN##B,				\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, (_D) + 3, 3##_DN, _B3, IndexAxis2##_DN##A, IndexAxis2##_DN##B,				\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, (_D) + 4, 4##_DN, _B4, IndexAxis3##_DN##A, IndexAxis3##_DN##B, _F)))))	\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_D_5(_I1, _I2, _D, _DN, _B0, _B1, _B2, _B3, _B4, _B5, _FS1, _FS2, _F)	\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, _D, 0##_DN, _B0, _FS1, _FS2,												\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A, IndexAxis0##_DN##B,				\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, (_D) + 2, 2##_DN, _B2, IndexAxis1##_DN##A, IndexAxis1##_DN##B,				\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, (_D) + 3, 3##_DN, _B3, IndexAxis2##_DN##A, IndexAxis2##_DN##B,				\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, (_D) + 4, 4##_DN, _B4, IndexAxis3##_DN##A, IndexAxis3##_DN##B,				\
		_D_Dragonian_Lib_Operator_Loop_Double(_I1, _I2, (_D) + 5, 5##_DN, _B5, IndexAxis4##_DN##A, IndexAxis4##_DN##B, _F))))))	\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_T_0(_I1, _I2, _I3, _D, _DN, _B0, _FS1, _FS2, _FS3, _F)											\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, _D, 0##_DN, _B0, _FS1, _FS2, _FS3, _F)								\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_T_1(_I1, _I2, _I3, _D, _DN, _B0, _B1, _FS1, _FS2, _FS3, _F)									\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, _D, 0##_DN, _B0, _FS1, _FS2, _FS3,									\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A, IndexAxis0##_DN##B, IndexAxis0##_DN##C, _F))	\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_T_2(_I1, _I2, _I3, _D, _DN, _B0, _B1, _B2, _FS1, _FS2, _FS3, _F)							\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, _D, 0##_DN, _B0, _FS1, _FS2, _FS3,									\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A, IndexAxis0##_DN##B, IndexAxis0##_DN##C,	\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, (_D) + 2, 2##_DN, _B2, IndexAxis1##_DN##A, IndexAxis1##_DN##B, IndexAxis1##_DN##C, _F)))	\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_T_3(_I1, _I2, _I3, _D, _DN, _B0, _B1, _B2, _B3, _FS1, _FS2, _FS3, _F)				\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, _D, 0##_DN, _B0, _FS1, _FS2, _FS3,									\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A, IndexAxis0##_DN##B, IndexAxis0##_DN##C,	\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, (_D) + 2, 2##_DN, _B2, IndexAxis1##_DN##A, IndexAxis1##_DN##B, IndexAxis1##_DN##C,	\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, (_D) + 3, 3##_DN, _B3, IndexAxis2##_DN##A, IndexAxis2##_DN##B, IndexAxis2##_DN##C, _F))))	\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_T_4(_I1, _I2, _I3, _D, _DN, _B0, _B1, _B2, _B3, _B4, _FS1, _FS2, _FS3, _F)	\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, _D, 0##_DN, _B0, _FS1, _FS2, _FS3,									\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A, IndexAxis0##_DN##B, IndexAxis0##_DN##C,	\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, (_D) + 2, 2##_DN, _B2, IndexAxis1##_DN##A, IndexAxis1##_DN##B, IndexAxis1##_DN##C,	\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, (_D) + 3, 3##_DN, _B3, IndexAxis2##_DN##A, IndexAxis2##_DN##B, IndexAxis2##_DN##C,	\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, (_D) + 4, 4##_DN, _B4, IndexAxis3##_DN##A, IndexAxis3##_DN##B, IndexAxis3##_DN##C, _F)))))	\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_T_5(_I1, _I2, _I3, _D, _DN, _B0, _B1, _B2, _B3, _B4, _B5, _FS1, _FS2, _FS3, _F)	\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, _D, 0##_DN, _B0, _FS1, _FS2, _FS3,									\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A, IndexAxis0##_DN##B, IndexAxis0##_DN##C,	\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, (_D) + 2, 2##_DN, _B2, IndexAxis1##_DN##A, IndexAxis1##_DN##B, IndexAxis1##_DN##C,	\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, (_D) + 3, 3##_DN, _B3, IndexAxis2##_DN##A, IndexAxis2##_DN##B, IndexAxis2##_DN##C,	\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, (_D) + 4, 4##_DN, _B4, IndexAxis3##_DN##A, IndexAxis3##_DN##B, IndexAxis3##_DN##C,	\
		_D_Dragonian_Lib_Operator_Loop_Triple(_I1, _I2, _I3, (_D) + 5, 5##_DN, _B5, IndexAxis4##_DN##A, IndexAxis4##_DN##B, IndexAxis4##_DN##C, _F))))))	\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_Q_0(_I1, _I2, _I3, _I4, _D, _DN, _B0, _FS1, _FS2, _FS3, _FS4, _F)							\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, _D, 0##_DN, _B0, _FS1, _FS2, _FS3, _FS4, _F)						\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_Q_1(_I1, _I2, _I3, _I4, _D, _DN, _B0, _B1, _FS1, _FS2, _FS3, _FS4, _F)						\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, _D, 0##_DN, _B0, _FS1, _FS2, _FS3, _FS4,							\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A, IndexAxis0##_DN##B, IndexAxis0##_DN##C, IndexAxis0##_DN##D, _F))	\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_Q_2(_I1, _I2, _I3, _I4, _D, _DN, _B0, _B1, _B2, _FS1, _FS2, _FS3, _FS4, _F)				\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, _D, 0##_DN, _B0, _FS1, _FS2, _FS3, _FS4,							\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A, IndexAxis0##_DN##B, IndexAxis0##_DN##C, IndexAxis0##_DN##D,	\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, (_D) + 2, 2##_DN, _B2, IndexAxis1##_DN##A, IndexAxis1##_DN##B, IndexAxis1##_DN##C, IndexAxis1##_DN##D, _F)))	\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_Q_3(_I1, _I2, _I3, _I4, _D, _DN, _B0, _B1, _B2, _B3, _FS1, _FS2, _FS3, _FS4, _F)	\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, _D, 0##_DN, _B0, _FS1, _FS2, _FS3, _FS4,							\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A, IndexAxis0##_DN##B, IndexAxis0##_DN##C, IndexAxis0##_DN##D,	\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, (_D) + 2, 2##_DN, _B2, IndexAxis1##_DN##A, IndexAxis1##_DN##B, IndexAxis1##_DN##C, IndexAxis1##_DN##D,	\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, (_D) + 3, 3##_DN, _B3, IndexAxis2##_DN##A, IndexAxis2##_DN##B, IndexAxis2##_DN##C, IndexAxis2##_DN##D, _F))))	\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_Q_4(_I1, _I2, _I3, _I4, _D, _DN, _B0, _B1, _B2, _B3, _B4, _FS1, _FS2, _FS3, _FS4, _F)	\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, _D, 0##_DN, _B0, _FS1, _FS2, _FS3, _FS4,							\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A, IndexAxis0##_DN##B, IndexAxis0##_DN##C, IndexAxis0##_DN##D,	\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, (_D) + 2, 2##_DN, _B2, IndexAxis1##_DN##A, IndexAxis1##_DN##B, IndexAxis1##_DN##C, IndexAxis1##_DN##D,	\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, (_D) + 3, 3##_DN, _B3, IndexAxis2##_DN##A, IndexAxis2##_DN##B, IndexAxis2##_DN##C, IndexAxis2##_DN##D,	\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, (_D) + 4, 4##_DN, _B4, IndexAxis3##_DN##A, IndexAxis3##_DN##B, IndexAxis3##_DN##C, IndexAxis3##_DN##D, _F)))))	\
	} while (0)

#define _D_Dragonian_Lib_Operator_Loop_Q_5(_I1, _I2, _I3, _I4, _D, _DN, _B0, _B1, _B2, _B3, _B4, _B5, _FS1, _FS2, _FS3, _FS4, _F)	\
	do{																															\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, _D, 0##_DN, _B0, _FS1, _FS2, _FS3, _FS4,							\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, (_D) + 1, 1##_DN, _B1, IndexAxis0##_DN##A, IndexAxis0##_DN##B, IndexAxis0##_DN##C, IndexAxis0##_DN##D,	\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, (_D) + 2, 2##_DN, _B2, IndexAxis1##_DN##A, IndexAxis1##_DN##B, IndexAxis1##_DN##C, IndexAxis1##_DN##D,	\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, (_D) + 3, 3##_DN, _B3, IndexAxis2##_DN##A, IndexAxis2##_DN##B, IndexAxis2##_DN##C, IndexAxis2##_DN##D,	\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, (_D) + 4, 4##_DN, _B4, IndexAxis3##_DN##A, IndexAxis3##_DN##B, IndexAxis3##_DN##C, IndexAxis3##_DN##D,	\
		_D_Dragonian_Lib_Operator_Loop_Quad(_I1, _I2, _I3, _I4, (_D) + 5, 5##_DN, _B5, IndexAxis4##_DN##A, IndexAxis4##_DN##B, IndexAxis4##_DN##C, IndexAxis4##_DN##D, _F))))))	\
 	} while (0)

