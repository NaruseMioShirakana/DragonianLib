#include "TensorLib/Include/Base/Tensor/Operators/CPU/Linear.h"
#include <cblas.h>

_D_Dragonian_Lib_Operator_Space_Begin

namespace Linear
{

	void Gemm(
		Int ATrans, Int BTrans,
		Int InFeature, Int OutFeature, Int CommonDim,
		const Float32* InData, const Float32* WeightData, Float32* OutData,
		const Float32* BiasData, bool BroadcastBias,
		Float32 Alpha, Float32 Beta, Float32 AlphaB
	)
	{
		CBLAS_TRANSPOSE TransA = CBLAS_TRANSPOSE(ATrans), TransB = CBLAS_TRANSPOSE(BTrans);
		const auto M = CommonDim, N = OutFeature, K = InFeature;
		const blasint lda = TransA == CblasTrans || TransA == CblasConjTrans ? blasint(M) : blasint(K);
		const blasint ldb = TransB == CblasTrans || TransB == CblasConjTrans ? blasint(K) : blasint(N);
		const blasint ldc = blasint(N);
		
		cblas_sgemm(
			CblasRowMajor, TransA, TransB,
			M, N, K,
			Alpha, InData, lda,
			WeightData, ldb, Beta,
			OutData, ldc
		);
		
		if (BiasData)
		{
			if (!BroadcastBias)
			{
				//Bias [M, N]
				cblas_saxpy(
					M * N,
					AlphaB, BiasData, 1,
					OutData, 1
				);
			}
			else
			{
				//Bias [1, N]
				for (Int i = 0; i < M; ++i)
				{
					cblas_saxpy(
						N,
						AlphaB, BiasData, 1,
						OutData + Int64(i) * N, 1
					);
				}
			}
		}
	}

	void Gemm(
		Int ATrans, Int BTrans,
		Int InFeature, Int OutFeature, Int CommonDim,
		const Float64* InData, const Float64* WeightData, Float64* OutData,
		const Float64* BiasData, bool BroadcastBias,
		Float64 Alpha, Float64 Beta, Float64 AlphaB
	)

	{
		CBLAS_TRANSPOSE TransA = CBLAS_TRANSPOSE(ATrans), TransB = CBLAS_TRANSPOSE(BTrans);
		const auto M = CommonDim, N = OutFeature, K = InFeature;
		const blasint lda = TransA == CblasTrans || TransA == CblasConjTrans ? blasint(M) : blasint(K);
		const blasint ldb = TransB == CblasTrans || TransB == CblasConjTrans ? blasint(K) : blasint(N);
		const blasint ldc = blasint(N);

		cblas_dgemm(
			CblasRowMajor, TransA, TransB,
			M, N, K,
			Alpha, InData, lda,
			WeightData, ldb, Beta,
			OutData, ldc
		);

		if (BiasData)
		{
			if (!BroadcastBias)
			{
				//Bias [M, N]
				cblas_daxpy(
					M * N,
					AlphaB, BiasData, 1,
					OutData, 1
				);
			}
			else
			{
				//Bias [1, N]
				for (Int i = 0; i < M; ++i)
				{
					cblas_daxpy(
						N,
						AlphaB, BiasData, 1,
						OutData + Int64(i) * N, 1
					);
				}
			}
		}
	}

	void Gemm(
		Int ATrans, Int BTrans,
		Int InFeature, Int OutFeature, Int CommonDim,
		const Complex32* InData, const Complex32* WeightData, Complex32* OutData,
		const Complex32* BiasData, bool BroadcastBias,
		Complex32 Alpha, Complex32 Beta, Complex32 AlphaB
	)
	{
		CBLAS_TRANSPOSE TransA = CBLAS_TRANSPOSE(ATrans), TransB = CBLAS_TRANSPOSE(BTrans);
		const auto M = CommonDim, N = OutFeature, K = InFeature;
		const blasint lda = TransA == CblasTrans || TransA == CblasConjTrans ? blasint(M) : blasint(K);
		const blasint ldb = TransB == CblasTrans || TransB == CblasConjTrans ? blasint(K) : blasint(N);
		const blasint ldc = blasint(N);

		cblas_cgemm(
			CblasRowMajor, TransA, TransB,
			M, N, K,
			&Alpha, InData, lda,
			WeightData, ldb, &Beta,
			OutData, ldc
		);

		if (BiasData)
		{
			if (!BroadcastBias)
			{
				//Bias [M, N]
				cblas_caxpy(
					M * N,
					&AlphaB, BiasData, 1,
					OutData, 1
				);
			}
			else
			{
				//Bias [1, N]
				for (Int i = 0; i < M; ++i)
				{
					cblas_caxpy(
						N,
						&AlphaB, BiasData, 1,
						OutData + Int64(i) * N, 1
					);
				}
			}
		}
	}

	void Gemm(
		Int ATrans, Int BTrans,
		Int InFeature, Int OutFeature, Int CommonDim,
		const Complex64* InData, const Complex64* WeightData, Complex64* OutData,
		const Complex64* BiasData, bool BroadcastBias,
		Complex64 Alpha, Complex64 Beta, Complex64 AlphaB
	)
	{
		CBLAS_TRANSPOSE TransA = CBLAS_TRANSPOSE(ATrans), TransB = CBLAS_TRANSPOSE(BTrans);
		const auto M = CommonDim, N = OutFeature, K = InFeature;
		const blasint lda = TransA == CblasTrans || TransA == CblasConjTrans ? blasint(M) : blasint(K);
		const blasint ldb = TransB == CblasTrans || TransB == CblasConjTrans ? blasint(K) : blasint(N);
		const blasint ldc = blasint(N);

		cblas_zgemm(
			CblasRowMajor, TransA, TransB,
			M, N, K,
			&Alpha, InData, lda,
			WeightData, ldb, &Beta,
			OutData, ldc
		);

		if (BiasData)
		{
			if (!BroadcastBias)
			{
				//Bias [M, N]
				cblas_zaxpy(
					M * N,
					&AlphaB, BiasData, 1,
					OutData, 1
				);
			}
			else
			{
				//Bias [1, N]
				for (Int i = 0; i < M; ++i)
				{
					cblas_zaxpy(
						N,
						&AlphaB, BiasData, 1,
						OutData + Int64(i) * N, 1
					);
				}
			}
		}
	}

}

_D_Dragonian_Lib_Operator_Space_End