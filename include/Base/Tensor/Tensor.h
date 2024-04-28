#pragma once
#include <deque>
#include <ranges>
#include <random>
#include "Tensor/TensorBase.h"
#include "Util/ThreadPool.h"
#include "Tensor/Macro.h"
#include "Util/Avx256.h"

LibSvcBegin
using ShapeType = Vector<SizeType>;
using ShapeIterator = ShapeType::iterator;

struct Range
{
	SizeType Begin = 0;
	SizeType Step = 1;
	SizeType End = 0;
	bool IsVal = false;
	bool IsNone = false;

	Range(SizeType _Val) :Begin(_Val), Step(_Val), End(_Val), IsVal(true) {}
	Range(NoneType _NoneVal) :IsNone(true) { UNUSED(_NoneVal); }
	Range(SizeType _Begin, SizeType _Step, SizeType _End) :Begin(_Begin), Step(_Step), End(_End) {}
	Range(NoneType _NoneVal, SizeType _Step, SizeType _End) :Begin(_Step > 0 ? 0 : -1), Step(_Step), End(_End) { UNUSED(_NoneVal); }
	Range(SizeType _Begin, SizeType _Step, NoneType _NoneVal) :Begin(_Begin), Step(_Step), End(_Step > 0 ? -1 : 0) { UNUSED(_NoneVal); }
	Range(SizeType _Begin, SizeType _End) :Begin(_Begin), End(_End) {}
	Range(NoneType _NoneVal, SizeType _End) :End(_End) { UNUSED(_NoneVal); }
	Range(SizeType _Begin, NoneType _NoneVal) :Begin(_Begin), End(-1) { UNUSED(_NoneVal); }
	void Reverse() { std::swap(Begin, End); Step = -Step; }
	bool operator==(const NoneType& _NoneVal) const { UNUSED(_NoneVal); return IsNone; }
};

enum class PaddingType
{
	Zero,
	Constant,
	Reflect,
	Cicular,
	Replicate
};

using SliceOptions = Vector<Range>;

SizeType VectorMul(const ShapeType& _Input);

bool RangeIsAllNone(const Vector<Range>& _Input);

class Tensor : public TensorBase
{
public:
	Tensor() = delete;
	~Tensor() override;
	Tensor(TensorType _DType) :TensorBase(_DType) {}
	Tensor(const ShapeType& _Shape, TensorType _DType);
	Tensor(const Tensor& _Left);
	Tensor(Tensor&& _Right) noexcept;
	using InvokeFnType = void(*)(Tensor&);
	static void SetThreadCount(SizeType _Count);

protected:
	byte* DataPtr_ = nullptr;
	Tensor* ViewParent_ = nullptr;

	ShapeType ShapeBack_;
	ShapeType StepFront_, StepBack_;
	ShapeType SliceBegin_;
	ShapeType DimStride_;
	ShapeType CurIndices_;
	int64_t AlignSize_ = 4;
	bool IsBroadCasted_ = false;

	std::deque<Tensor*> ViewChild_;
	mutable std::mutex ViewMx_;

public:
	//ʹ��Indice��ȡһ������
	template <typename Ref>
	Ref& Item(const ShapeType& _Indices)
	{
		if (sizeof(Ref) != AlignSize_)
			LibSvcThrow("TypeError!");
		return *(Ref*)(Data(_Indices));
	}

	//��ȡ��ǰ��һ������
	template <typename Ref>
	Ref& Item()
	{
		if (sizeof(Ref) != AlignSize_)
			LibSvcThrow("TypeError!");
		return *(Ref*)(GetPtr());
	}

	//ʹ��_Valָ���_Type���������������Tensor
	void Assign(const void* _Val, TensorType _Type, ThreadPool* _ThreadPool = nullptr) const;

	//ʹ��һ��Buffer������Tensor��ֵ����������������˳��ֱ��Buffer���������Tensor������ֵ���
	void Assign(const void* _Buffer, SizeType _BufferSize, ThreadPool* _ThreadPool = nullptr) const;

	//ʹ��_Val�������Tensor
	void Assign(int64 _Val, ThreadPool* _ThreadPool = nullptr) const;

	//ʹ��_Val�������Tensor
	void Assign(float64 _Val, ThreadPool* _ThreadPool = nullptr) const;

	//ʹ��һ��Tensor������Tensor��ֵ�������Shape������ͬ����Թ㲥
	void Assign(const Tensor& _Val, ThreadPool* _ThreadPool = nullptr) const;

	//����ǰTensor�滻ά��һ��Tensor��View�����´�������������ΪView������һ��View��ViewԴ����Ϊ��ǰTensor������뿽��Tensor����ʹ��Clone������
	Tensor& operator=(const Tensor& _Left);

	//�ƶ���ֵ
	Tensor& operator=(Tensor&& _Right) noexcept;

	//ʹ��_Val�������Tensor
	Tensor& operator=(int64 _Val);

	//ʹ��_Val�������Tensor
	Tensor& operator=(float64 _Val);

	Tensor operator+(const Tensor& _Right) const;

	Tensor operator-(const Tensor& _Right) const;

	Tensor operator*(const Tensor& _Right) const;

	Tensor operator/(const Tensor& _Right) const;

	Tensor operator+(int64 _Right) const;

	Tensor operator-(int64 _Right) const;

	Tensor operator*(int64 _Right) const;

	Tensor operator/(int64 _Right) const;

	Tensor operator+(float64 _Right) const;

	Tensor operator-(float64 _Right) const;

	Tensor operator*(float64 _Right) const;

	Tensor operator/(float64 _Right) const;

	Tensor& operator+=(const Tensor& _Right);

	Tensor& operator-=(const Tensor& _Right);

	Tensor& operator*=(const Tensor& _Right);

	Tensor& operator/=(const Tensor& _Right);

	Tensor& operator+=(int64 _Right);

	Tensor& operator-=(int64 _Right);

	Tensor& operator*=(int64 _Right);

	Tensor& operator/=(int64 _Right);

	Tensor& operator+=(float64 _Right);

	Tensor& operator-=(float64 _Right);

	Tensor& operator*=(float64 _Right);

	Tensor& operator/=(float64 _Right);

	Tensor operator[](SizeType _Index) const;

	Tensor operator[](const SliceOptions& _SliceOptions) const;

	Tensor operator[](const ShapeType& _Indice) const;

private:
	void Free();
	void ClearViewChilds();
	void ThrowOnNotEnabled() const;
	void RemoveSelfViewPtr();
	bool HasChild(const Tensor* _Child) const;
	void Assign1D(const void* _Val) const;
	void ReCalcViewInfo();
	Tensor GatherRef(SizeType _Index) const;

public:
	//ʹLoopIterator������ + 1
	void IteratorAdd(ShapeType& _Indices) const;

	//ʹLoopIterator������ - 1
	void IteratorSub(ShapeType& _Indices) const;

	//����Index
	static SizeType CalcIndex(SizeType _Index, SizeType _Max);

	//����Range
	static SizeType CalcRange(SizeType _Index, SizeType _Max);

	//Ceil����Int64����
	static SizeType Ceil(SizeType _Left, SizeType _Right);

	//��ǰTensor�Ƿ񡰲�����������
	bool IsEnabled() const;

	//��ǰTensor�Ƿ�Ϊ����
	bool IsScalar() const;

	//��ǰTensor�Ƿ����View��Tensor������
	bool HasViewedFeature() const;

	//��ǰTensor������˳���Ƿ��ϸ��ڴ�����
	bool IsContinuous() const;

	//��ǰTensor�Ƿ����ͨ��Permute�������˳���ڴ�����
	bool IsTranSposedContinuous() const;

	//��ǰTensor�Ƿ����ViewԴ
	bool IsView() const;

	//��¡��ǰTensor������һ�����ж����ڴ����Tensor���½�Tensor��
	Tensor Clone(ThreadPool* _ThreadPool = nullptr) const;

	//����һ����ǰTensor��View��View�����ж����ڴ棬ֻ�����������ԣ���ǰTensor���Ǹ�View��ViewԴ
	Tensor CreateView() const;

	//����һ����ǰTensor��View�����������Ƭ�����ظ�View
	Tensor Slice(const SliceOptions& _SliceOptions) const;

	//����һ����ǰTensor��View�����������Ƭ�����ظ�View
	Tensor ReversedSlice(const SliceOptions& _SliceOptions) const;

	//����һ����ǰTensor��View���ı���������˳�򣬷��ظ�View
	Tensor Permute(const ShapeType& _DPremute) const;

	//��һ��Tensor��ָ������ú���
	static void Invoke(Tensor& _Tensor, SizeType InvokedDim, InvokeFnType _Fn);

	//�ڵ�ǰTensor��ָ������ú���
	void Invoke(SizeType InvokedDim, InvokeFnType _Fn);

	//��ȡShape
	const ShapeType& Shape() const;

	//��ȡָ�����Shape
	SizeType Shape(SizeType _Index) const;

	//��ȡShape
	const ShapeType& Size() const;

	//��ȡָ�����Shape
	SizeType Size(SizeType _Index) const;

	//��ȡ�������Stride��Ϣ
	const ShapeType& Strides() const;

	//��ȡ�������Step��Ϣ
	const ShapeType& StepsBack() const;

	//��ȡ���������Ƭ��ʼλ��
	const ShapeType& SliceBegins() const;

	//ʹ�� 1 �������Tensor
	void FixOnes(ThreadPool* _ThreadPool = nullptr) const;

	//ʹ�� 0 �������Tensor
	void FixZeros(ThreadPool* _ThreadPool = nullptr) const;

	//ʹ��_Val�������Tensor
	void Fix(double _Val, ThreadPool* _ThreadPool = nullptr) const;

	//ʹ��_Val�������Tensor
	void Fix(int64 _Val, ThreadPool* _ThreadPool = nullptr) const;

	//ʹ��������������Tensor
	void RandFix(uint64 _Seed = 114514, ThreadPool* _ThreadPool = nullptr) const;

	//ʹ����̬���ɵ�������������Tensor
	void RandnFix(uint64 _Seed = 114514, double _Mean = 0., double _Sigma = 1., ThreadPool* _ThreadPool = nullptr) const;

	//��ȡ��ǰTensor�Ļ������׵�ַ�����TensorΪView�򷵻�ViewԴ�ģ�
	byte* Buffer() const;

	//��ȡ��ǰTensor�����ַ
	byte* Data() const;

	//��ȡ��ǰTensorָ��Indiceλ�����ݵĵ�ַ
	byte* Data(const ShapeType& _Indices) const;

	//����һ����ǰTensor��View��������ֱ�ӿ���ShapeΪViewShape��Tensor�����ظ�View
	Tensor View(const ShapeType& _ViewShape) const;

	//ʹ��ǰTensor�ڴ����������ص�ǰTensor����ֵ����
	Tensor& Continuous(ThreadPool* _ThreadPool = nullptr);

	//����һ����ǰTensor��View������ָ��λ�ò���һ���ᣬ���ظ�View
	Tensor UnSqueeze(SizeType Dim) const;

	//����һ����ǰTensor��View������ָ�����SizeΪ 1����ɾ�����ᣬ���ظ�View
	Tensor Squeeze(SizeType Dim) const;

	//����һ����ǰTensor��View��������SizeΪ 1 ����ɾ�������ظ�View
	Tensor Squeeze() const;

	//��������Tensor��View���������ǹ㲥Ϊ����Shapeһ�µ�View������������View
	static std::pair<Tensor, Tensor> BroadCast(const Tensor& _A, const Tensor& _B);

	//��������Tensor��View������㲥Ϊ�뵱ǰTensor��Shapeһ�µ�View�����ظ�View
	Tensor BroadCast(const Tensor& _Other) const;

	//��ǰTensor�Ƿ�Ϊ���㲥��Tensor
	bool IsBroadCasted() const;

	//��ȡ��ǰTensor������
	SizeType DimCount() const;

	//�жϵ�ǰTensor�Ƿ�Ϊ����
	bool IsVector() const;

	//��ȡ��ǰTensor���������׵�ַ
	byte* GetPtr() const;

	//��ȡ����������������С������
	ShapeType CalcContinuous() const;

	//����Axis[0]����ȡ��ǰTensor��Indices����Tensor���½�Tensor��
	Tensor Gather(
		const Tensor& _Indices,
		ThreadPool* _ThreadPool = nullptr
	) const;

	//ת��Tensor�����ͣ��½�Tensor��
	Tensor Cast(
		TensorType _Dtype,
		ThreadPool* _ThreadPool = nullptr
	) const;

	//�������Tensor����Padding��˳��Ϊ���򣩣�����Padding���Tensor���½�Tensor��
	static Tensor Padding(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type,
		TensorType _ValueType,
		lpvoid _Val,
		ThreadPool* _ThreadPool
	);

	//�������Tensor����Padding��˳��Ϊ���򣬼�Torch��˳�򣩣�����Padding���Tensor���½�Tensor��
	static Tensor Pad(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type,
		TensorType _ValueType,
		lpvoid _Val,
		ThreadPool* _ThreadPool
	);

	//�������Tensor����Padding��˳��Ϊ���򣩣�����Padding���Tensor���½�Tensor��
	static Tensor Padding(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type = PaddingType::Zero,
		ThreadPool* _ThreadPool = nullptr
	);

	//�������Tensor����Padding��˳��Ϊ���򣬼�Torch��˳�򣩣�����Padding���Tensor���½�Tensor��
	static Tensor Pad(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		PaddingType _Type = PaddingType::Zero,
		ThreadPool* _ThreadPool = nullptr
	);

	//�������Tensor����Padding��˳��Ϊ���򣩣�����Padding���Tensor���½�Tensor��
	static Tensor Padding(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		float64 _Val,
		ThreadPool* _ThreadPool = nullptr
	);

	//�������Tensor����Padding��˳��Ϊ���򣬼�Torch��˳�򣩣�����Padding���Tensor���½�Tensor��
	static Tensor Pad(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		float64 _Val,
		ThreadPool* _ThreadPool = nullptr
	);

	//�������Tensor����Padding��˳��Ϊ���򣩣�����Padding���Tensor���½�Tensor��
	static Tensor Padding(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		int64 _Val,
		ThreadPool* _ThreadPool = nullptr
	);

	//�������Tensor����Padding��˳��Ϊ���򣬼�Torch��˳�򣩣�����Padding���Tensor���½�Tensor��
	static Tensor Pad(
		const Tensor& _Input,
		const Vector<Range>& _Pad,
		int64 _Val,
		ThreadPool* _ThreadPool = nullptr
	);

	//�������Tensor�����ظ��������½�Tensor��
	static Tensor Repeat(
		const Tensor& _Input,
		const Vector<std::pair<SizeType, SizeType>>& _Repeat,
		ThreadPool* _ThreadPool = nullptr
	);

	//������Tensor��ShapeΪģ�壬��_Dim������һ���µ�Shape��������_Dim����п������½�Tensor�������뱣֤����Tensor��Shape��ȫ��ͬ
	static Tensor Stack(
		const Vector<Tensor>& _Inputs,
		SizeType _Dim = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	//������Tensor��ShapeΪģ�壬������Tensor��_Dim���ϲ����½�Tensor�������뱣֤����Tensor��Shape�ڳ���_Dim������ȫ��ͬ
	static Tensor Cat(
		const Vector<Tensor>& _Inputs,
		SizeType _Dim = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	//����Axis[0]����ȡInput��Indices����Tensor���½�Tensor��
	static Tensor Gather(
		const Tensor& _Input,
		const Tensor& _Indices,
		ThreadPool* _ThreadPool = nullptr
	);

	//ת��Tensor�����ͣ��½�Tensor��
	static Tensor Cast(
		const Tensor& _Input,
		TensorType _Dtype,
		ThreadPool* _ThreadPool = nullptr
	);

public:
	static Tensor Pow(const Tensor& _InputA, const Tensor& _InputB, ThreadPool* _ThreadPool = nullptr);
	static Tensor Pow(const Tensor& _InputA, float64 _Val, ThreadPool* _ThreadPool = nullptr);
	Tensor Pow(const Tensor& _InputB, ThreadPool* _ThreadPool = nullptr) const;
	Tensor Pow(float64 _Val, ThreadPool* _ThreadPool = nullptr) const;
	Tensor& Pow_(const Tensor& _InputB, ThreadPool* _ThreadPool = nullptr);
	Tensor& Pow_(float64 _Val, ThreadPool* _ThreadPool = nullptr);

	LibSvcTensorFnDef(Abs);
	LibSvcTensorFnDef(Sin);
	LibSvcTensorFnDef(Sinh);
	LibSvcTensorFnDef(Cos);
	LibSvcTensorFnDef(Cosh);
	LibSvcTensorFnDef(Tan);
	LibSvcTensorFnDef(Tanh);
	LibSvcTensorFnDef(ASin);
	LibSvcTensorFnDef(ACos);
	LibSvcTensorFnDef(ATan);
	LibSvcTensorFnDef(ASinh);
	LibSvcTensorFnDef(ACosh);
	LibSvcTensorFnDef(ATanh);
	LibSvcTensorFnDef(Exp);
	LibSvcTensorFnDef(Exp2);
	LibSvcTensorFnDef(Exp10);
	LibSvcTensorFnDef(Log);
	LibSvcTensorFnDef(Log2);
	LibSvcTensorFnDef(Log10);
};

template<typename _1Ty, typename _2Ty>
void CastFrom(const Tensor& _InputA, const Tensor& _InputB, const SizeType CurDims)
{
	_1Ty* DataPtr1 = (_1Ty*)_InputA.Data();
	const _2Ty* DataPtr2 = (_2Ty*)_InputB.Data();

	if (!_InputA.IsBroadCasted() && !_InputB.IsBroadCasted() && _InputA.IsContinuous() && _InputB.IsContinuous())
	{
		DataPtr1 = (_1Ty*)_InputA.GetPtr();
		DataPtr2 = (_2Ty*)_InputB.GetPtr();
		const size_t BufferSize = VectorMul(_InputA.Shape());
		const auto DataEnd = DataPtr1 + BufferSize;
		while (DataPtr1 != DataEnd)
			*(DataPtr1++) = (_1Ty)(*(DataPtr2++));
		return;
	}

	auto Steps1 = _InputA.StepsBack();
	for (auto& i : Steps1)
		i /= sizeof(_1Ty);
	auto Steps2 = _InputB.StepsBack();
	for (auto& i : Steps2)
		i /= sizeof(_2Ty);
	const SizeType* __restrict ShapePtr = _InputA.Shape().data();
	const SizeType* __restrict StepPtr1 = Steps1.data();
	const SizeType* __restrict StepPtr2 = Steps2.data();
	const SizeType* __restrict BeginsPtr1 = _InputA.SliceBegins().data();
	const SizeType* __restrict BeginsPtr2 = _InputB.SliceBegins().data();
	const SizeType* __restrict StridesPtr1 = _InputA.Strides().data();
	const SizeType* __restrict StridesPtr2 = _InputB.Strides().data();

	if (CurDims > 5)
	{
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		LibSvcCycle(
			IndicesPtr,
			ShapePtr,
			CurDims,
			{
				SizeType Index1 = 0;
				SizeType Index2 = 0;
				for (SizeType i = 0; i < CurDims; ++i)
				{
					Index1 += ((IndicesPtr[i] * StridesPtr1[i]) + BeginsPtr1[i]) * StepPtr1[i];
					Index2 += ((IndicesPtr[i] * StridesPtr2[i]) + BeginsPtr2[i]) * StepPtr2[i];
				}
				DataPtr1[Index1] = (_1Ty)DataPtr2[Index2];
			}
		);

		return;
	}

	auto Cont = _InputA.CalcContinuous();
	Cont.resize(5);
	const SizeType* __restrict ContPtr = Cont.data();
	const SizeType Axis0 = ContPtr[0], Axis1 = ContPtr[1], Axis2 = ContPtr[2], Axis3 = ContPtr[3], Axis4 = ContPtr[4];

	if (CurDims == 5)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						for (SizeType m = 0; l < ShapePtr[Axis4]; ++m)
						{
							const auto IndexAxis4A = IndexAxis3A +
								((m * StridesPtr1[Axis4]) + BeginsPtr1[Axis4]) * StepPtr1[Axis4];
							const auto IndexAxis4B = IndexAxis3B +
								((m * StridesPtr2[Axis4]) + BeginsPtr2[Axis4]) * StepPtr2[Axis4];
							DataPtr1[IndexAxis4A] = (_1Ty)DataPtr2[IndexAxis4B];
						}
					}
				}
			}
		}
	}
	else if (CurDims == 4)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						DataPtr1[IndexAxis3A] = (_1Ty)DataPtr2[IndexAxis3B];
					}
				}
			}
		}
	}
	else if (CurDims == 3)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					DataPtr1[IndexAxis2A] = (_1Ty)DataPtr2[IndexAxis2B];
				}
			}
		}
	}
	else if (CurDims == 2)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				DataPtr1[IndexAxis1A] = (_1Ty)DataPtr2[IndexAxis1B];
			}
		}
	}
	else if (CurDims == 1)
	{
		for (SizeType i = 0; i < ShapePtr[0]; ++i)
		{
			DataPtr1[((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0]] =
				(_1Ty)DataPtr2[((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0]];
		}
	}
}

template<typename _Type, typename _Fn, typename _AvxFn>
void MonoOperators(
	const Tensor& _Dst,
	const Tensor& _Src,
	const SizeType CurDims,
	_Fn _Func = nullptr,
	_AvxFn _AvxFunc = nullptr
)
{
	_Type* DataPtr1 = (_Type*)_Dst.Data();
	const _Type* DataPtr2 = (_Type*)_Src.Data();

	if (!_Dst.IsBroadCasted() && !_Src.IsBroadCasted() && _Dst.IsContinuous() && _Src.IsContinuous())
	{
		DataPtr1 = (_Type*)_Dst.GetPtr();
		DataPtr2 = (_Type*)_Src.GetPtr();
		const size_t DataSize = VectorMul(_Dst.Shape());
		const auto DataEnd = DataPtr1 + DataSize;
		if (_AvxFunc)
			_AvxFunc(DataPtr1, DataPtr2, DataSize);
		else
			while (DataPtr1 != DataEnd)
				*(DataPtr1++) = (_Type)_Func(*(DataPtr2++));
		return;
	}

	auto Steps1 = _Dst.StepsBack();
	for (auto& i : Steps1)
		i /= sizeof(_Type);
	auto Steps2 = _Src.StepsBack();
	for (auto& i : Steps2)
		i /= sizeof(_Type);
	const SizeType* __restrict ShapePtr = _Dst.Shape().data();
	const SizeType* __restrict StepPtr1 = Steps1.data();
	const SizeType* __restrict StepPtr2 = Steps2.data();
	const SizeType* __restrict BeginsPtr1 = _Dst.SliceBegins().data();
	const SizeType* __restrict BeginsPtr2 = _Src.SliceBegins().data();
	const SizeType* __restrict StridesPtr1 = _Dst.Strides().data();
	const SizeType* __restrict StridesPtr2 = _Src.Strides().data();

	if (CurDims > 5)
	{
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		LibSvcCycle(
			IndicesPtr,
			ShapePtr,
			CurDims,
			{
				SizeType Index1 = 0;
				SizeType Index2 = 0;
				for (SizeType i = 0; i < CurDims; ++i)
				{
					Index1 += ((IndicesPtr[i] * StridesPtr1[i]) + BeginsPtr1[i]) * StepPtr1[i];
					Index2 += ((IndicesPtr[i] * StridesPtr2[i]) + BeginsPtr2[i]) * StepPtr2[i];
				}
				DataPtr1[Index1] = (_Type)_Func(DataPtr2[Index2]);
			}
		);

		return;
	}

	auto Cont = _Dst.CalcContinuous();
	Cont.resize(5);
	const SizeType* __restrict ContPtr = Cont.data();
	const SizeType Axis0 = ContPtr[0], Axis1 = ContPtr[1], Axis2 = ContPtr[2], Axis3 = ContPtr[3], Axis4 = ContPtr[4];

	if (CurDims == 5)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						for (SizeType m = 0; l < ShapePtr[Axis4]; ++m)
						{
							const auto IndexAxis4A = IndexAxis3A +
								((m * StridesPtr1[Axis4]) + BeginsPtr1[Axis4]) * StepPtr1[Axis4];
							const auto IndexAxis4B = IndexAxis3B +
								((m * StridesPtr2[Axis4]) + BeginsPtr2[Axis4]) * StepPtr2[Axis4];
							DataPtr1[IndexAxis4A] = (_Type)_Func(DataPtr2[IndexAxis4B]);
						}
					}
				}
			}
		}
	}
	else if (CurDims == 4)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						DataPtr1[IndexAxis3A] = (_Type)_Func(DataPtr2[IndexAxis3B]);
					}
				}
			}
		}
	}
	else if (CurDims == 3)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					DataPtr1[IndexAxis2A] = (_Type)_Func(DataPtr2[IndexAxis2B]);
				}
			}
		}
	}
	else if (CurDims == 2)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				DataPtr1[IndexAxis1A] = (_Type)_Func(DataPtr2[IndexAxis1B]);
			}
		}
	}
	else if (CurDims == 1)
	{
		for (SizeType i = 0; i < ShapePtr[0]; ++i)
		{
			DataPtr1[((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0]] =
				(_Type)_Func(DataPtr2[((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0]]);
		}
	}
}

template<typename _Type, typename _Fn, typename _AvxFn>
void MultiOperators(
	const Tensor& _Dst,
	const Tensor& _Src1,
	const Tensor& _Src2,
	const SizeType CurDims,
	_Fn _Func = nullptr,
	_AvxFn _AvxFunc = nullptr
)
{
	_Type* DataPtr1 = (_Type*)_Dst.Data();
	const _Type* DataPtr2 = (_Type*)_Src1.Data();
	const _Type* DataPtr3 = (_Type*)_Src2.Data();

	if (!_Dst.IsBroadCasted() && !_Src1.IsBroadCasted() && !_Src2.IsBroadCasted() && _Dst.IsContinuous() && _Src1.IsContinuous() && _Src2.IsContinuous())
	{
		DataPtr1 = (_Type*)_Dst.GetPtr();
		DataPtr2 = (_Type*)_Src1.GetPtr();
		DataPtr3 = (_Type*)_Src2.GetPtr();
		const size_t DataSize = VectorMul(_Dst.Shape());
		const auto DataEnd = DataPtr1 + DataSize;
		if (_AvxFunc)
			_AvxFunc(DataPtr1, DataPtr2, DataPtr3, DataSize);
		else
			while (DataPtr1 != DataEnd)
				*(DataPtr1++) = (_Type)_Func(*(DataPtr2++), *(DataPtr3++));
		return;
	}

	auto Steps1 = _Dst.StepsBack();
	for (auto& i : Steps1)
		i /= sizeof(_Type);
	auto Steps2 = _Src1.StepsBack();
	for (auto& i : Steps2)
		i /= sizeof(_Type);
	auto Steps3 = _Src2.StepsBack();
	for (auto& i : Steps3)
		i /= sizeof(_Type);
	const SizeType* __restrict ShapePtr = _Dst.Shape().data();
	const SizeType* __restrict StepPtr1 = Steps1.data();
	const SizeType* __restrict StepPtr2 = Steps2.data();
	const SizeType* __restrict StepPtr3 = Steps3.data();
	const SizeType* __restrict BeginsPtr1 = _Dst.SliceBegins().data();
	const SizeType* __restrict BeginsPtr2 = _Src1.SliceBegins().data();
	const SizeType* __restrict BeginsPtr3 = _Src2.SliceBegins().data();
	const SizeType* __restrict StridesPtr1 = _Dst.Strides().data();
	const SizeType* __restrict StridesPtr2 = _Src1.Strides().data();
	const SizeType* __restrict StridesPtr3 = _Src2.Strides().data();

	if (CurDims > 5)
	{
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		LibSvcCycle(
			IndicesPtr,
			ShapePtr,
			CurDims,
			{
				SizeType Index1 = 0;
				SizeType Index2 = 0;
				SizeType Index3 = 0;
				for (SizeType i = 0; i < CurDims; ++i)
				{
					Index1 += ((IndicesPtr[i] * StridesPtr1[i]) + BeginsPtr1[i]) * StepPtr1[i];
					Index2 += ((IndicesPtr[i] * StridesPtr2[i]) + BeginsPtr2[i]) * StepPtr2[i];
					Index3 += ((IndicesPtr[i] * StridesPtr3[i]) + BeginsPtr3[i]) * StepPtr3[i];
				}
				DataPtr1[Index1] = (_Type)_Func(DataPtr2[Index2], DataPtr3[Index3]);
			}
		);

		return;
	}

	auto Cont = _Dst.CalcContinuous();
	Cont.resize(5);
	const SizeType* __restrict ContPtr = Cont.data();
	const SizeType Axis0 = ContPtr[0], Axis1 = ContPtr[1], Axis2 = ContPtr[2], Axis3 = ContPtr[3], Axis4 = ContPtr[4];

	if (CurDims == 5)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			const auto IndexAxis0C = ((i * StridesPtr3[Axis0]) + BeginsPtr3[Axis0]) * StepPtr3[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				const auto IndexAxis1C = IndexAxis0C +
					((j * StridesPtr3[Axis1]) + BeginsPtr3[Axis1]) * StepPtr3[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					const auto IndexAxis2C = IndexAxis1C +
						((k * StridesPtr3[Axis2]) + BeginsPtr3[Axis2]) * StepPtr3[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						const auto IndexAxis3C = IndexAxis2C +
							((l * StridesPtr3[Axis3]) + BeginsPtr3[Axis3]) * StepPtr3[Axis3];
						for (SizeType m = 0; l < ShapePtr[Axis4]; ++m)
						{
							const auto IndexAxis4A = IndexAxis3A +
								((m * StridesPtr1[Axis4]) + BeginsPtr1[Axis4]) * StepPtr1[Axis4];
							const auto IndexAxis4B = IndexAxis3B +
								((m * StridesPtr2[Axis4]) + BeginsPtr2[Axis4]) * StepPtr2[Axis4];
							const auto IndexAxis4C = IndexAxis3C +
								((m * StridesPtr3[Axis4]) + BeginsPtr3[Axis4]) * StepPtr3[Axis4];
							DataPtr1[IndexAxis4A] = (_Type)_Func(DataPtr2[IndexAxis4B], DataPtr3[IndexAxis4C]);
						}
					}
				}
			}
		}
	}
	else if (CurDims == 4)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			const auto IndexAxis0C = ((i * StridesPtr3[Axis0]) + BeginsPtr3[Axis0]) * StepPtr3[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				const auto IndexAxis1C = IndexAxis0C +
					((j * StridesPtr3[Axis1]) + BeginsPtr3[Axis1]) * StepPtr3[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					const auto IndexAxis2C = IndexAxis1C +
						((k * StridesPtr3[Axis2]) + BeginsPtr3[Axis2]) * StepPtr3[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						const auto IndexAxis3C = IndexAxis2C +
							((l * StridesPtr3[Axis3]) + BeginsPtr3[Axis3]) * StepPtr3[Axis3];
						DataPtr1[IndexAxis3A] = (_Type)_Func(DataPtr2[IndexAxis3B], DataPtr3[IndexAxis3C]);
					}
				}
			}
		}
	}
	else if (CurDims == 3)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			const auto IndexAxis0C = ((i * StridesPtr3[Axis0]) + BeginsPtr3[Axis0]) * StepPtr3[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				const auto IndexAxis1C = IndexAxis0C +
					((j * StridesPtr3[Axis1]) + BeginsPtr3[Axis1]) * StepPtr3[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					const auto IndexAxis2C = IndexAxis1C +
						((k * StridesPtr3[Axis2]) + BeginsPtr3[Axis2]) * StepPtr3[Axis2];
					DataPtr1[IndexAxis2A] = (_Type)_Func(DataPtr2[IndexAxis2B], DataPtr3[IndexAxis2C]);
				}
			}
		}
	}
	else if (CurDims == 2)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			const auto IndexAxis0C = ((i * StridesPtr3[Axis0]) + BeginsPtr3[Axis0]) * StepPtr3[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				const auto IndexAxis1C = IndexAxis0C +
					((j * StridesPtr3[Axis1]) + BeginsPtr3[Axis1]) * StepPtr3[Axis1];
				DataPtr1[IndexAxis1A] = (_Type)_Func(DataPtr2[IndexAxis1B], DataPtr3[IndexAxis1C]);
			}
		}
	}
	else if (CurDims == 1)
	{
		for (SizeType i = 0; i < ShapePtr[0]; ++i)
		{
			DataPtr1[((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0]] =
				(_Type)_Func(DataPtr2[((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0]],
					DataPtr3[((i * StridesPtr3[0]) + BeginsPtr3[0]) * StepPtr3[0]]);
		}
	}
}

template<typename _Type, typename _Fn, typename _AvxFn>
void MultiOperatorsScalar(
	const Tensor& _Dst,
	const Tensor& _Src1,
	const _Type& _Src2,
	const SizeType CurDims,
	_Fn _Func = nullptr,
	_AvxFn _AvxFunc = nullptr
)
{
	_Type* DataPtr1 = (_Type*)_Dst.Data();
	const _Type* DataPtr2 = (_Type*)_Src1.Data();

	if (!_Dst.IsBroadCasted() && !_Src1.IsBroadCasted() && _Dst.IsContinuous() && _Src1.IsContinuous())
	{
		DataPtr1 = (_Type*)_Dst.GetPtr();
		DataPtr2 = (_Type*)_Src1.GetPtr();
		const size_t DataSize = VectorMul(_Dst.Shape());
		const auto DataEnd = DataPtr1 + DataSize;
		if (_AvxFunc)
			_AvxFunc(DataPtr1, DataPtr2, _Src2, DataSize);
		else
			while (DataPtr1 != DataEnd)
				*(DataPtr1++) = (_Type)_Func(*(DataPtr2++), _Src2);
		return;
	}

	auto Steps1 = _Dst.StepsBack();
	for (auto& i : Steps1)
		i /= sizeof(_Type);
	auto Steps2 = _Src1.StepsBack();
	for (auto& i : Steps2)
		i /= sizeof(_Type);
	const SizeType* __restrict ShapePtr = _Dst.Shape().data();
	const SizeType* __restrict StepPtr1 = Steps1.data();
	const SizeType* __restrict StepPtr2 = Steps2.data();
	const SizeType* __restrict BeginsPtr1 = _Dst.SliceBegins().data();
	const SizeType* __restrict BeginsPtr2 = _Src1.SliceBegins().data();
	const SizeType* __restrict StridesPtr1 = _Dst.Strides().data();
	const SizeType* __restrict StridesPtr2 = _Src1.Strides().data();

	if (CurDims > 5)
	{
		ShapeType CurIndice(CurDims, 0);
		SizeType* __restrict IndicesPtr = CurIndice.data();
		LibSvcCycle(
			IndicesPtr,
			ShapePtr,
			CurDims,
			{
				SizeType Index1 = 0;
				SizeType Index2 = 0;
				for (SizeType i = 0; i < CurDims; ++i)
				{
					Index1 += ((IndicesPtr[i] * StridesPtr1[i]) + BeginsPtr1[i]) * StepPtr1[i];
					Index2 += ((IndicesPtr[i] * StridesPtr2[i]) + BeginsPtr2[i]) * StepPtr2[i];
				}
				DataPtr1[Index1] = (_Type)_Func(DataPtr2[Index2], _Src2);
			}
		);

		return;
	}

	auto Cont = _Dst.CalcContinuous();
	Cont.resize(5);
	const SizeType* __restrict ContPtr = Cont.data();
	const SizeType Axis0 = ContPtr[0], Axis1 = ContPtr[1], Axis2 = ContPtr[2], Axis3 = ContPtr[3], Axis4 = ContPtr[4];

	if (CurDims == 5)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						for (SizeType m = 0; l < ShapePtr[Axis4]; ++m)
						{
							const auto IndexAxis4A = IndexAxis3A +
								((m * StridesPtr1[Axis4]) + BeginsPtr1[Axis4]) * StepPtr1[Axis4];
							const auto IndexAxis4B = IndexAxis3B +
								((m * StridesPtr2[Axis4]) + BeginsPtr2[Axis4]) * StepPtr2[Axis4];
							DataPtr1[IndexAxis4A] = (_Type)_Func(DataPtr2[IndexAxis4B], _Src2);
						}
					}
				}
			}
		}
	}
	else if (CurDims == 4)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					for (SizeType l = 0; l < ShapePtr[Axis3]; ++l)
					{
						const auto IndexAxis3A = IndexAxis2A +
							((l * StridesPtr1[Axis3]) + BeginsPtr1[Axis3]) * StepPtr1[Axis3];
						const auto IndexAxis3B = IndexAxis2B +
							((l * StridesPtr2[Axis3]) + BeginsPtr2[Axis3]) * StepPtr2[Axis3];
						DataPtr1[IndexAxis3A] = (_Type)_Func(DataPtr2[IndexAxis3B], _Src2);
					}
				}
			}
		}
	}
	else if (CurDims == 3)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				for (SizeType k = 0; k < ShapePtr[Axis2]; ++k)
				{
					const auto IndexAxis2A = IndexAxis1A +
						((k * StridesPtr1[Axis2]) + BeginsPtr1[Axis2]) * StepPtr1[Axis2];
					const auto IndexAxis2B = IndexAxis1B +
						((k * StridesPtr2[Axis2]) + BeginsPtr2[Axis2]) * StepPtr2[Axis2];
					DataPtr1[IndexAxis2A] = (_Type)_Func(DataPtr2[IndexAxis2B], _Src2);
				}
			}
		}
	}
	else if (CurDims == 2)
	{
		for (SizeType i = 0; i < ShapePtr[Axis0]; ++i)
		{
			const auto IndexAxis0A = ((i * StridesPtr1[Axis0]) + BeginsPtr1[Axis0]) * StepPtr1[Axis0];
			const auto IndexAxis0B = ((i * StridesPtr2[Axis0]) + BeginsPtr2[Axis0]) * StepPtr2[Axis0];
			for (SizeType j = 0; j < ShapePtr[Axis1]; ++j)
			{
				const auto IndexAxis1A = IndexAxis0A +
					((j * StridesPtr1[Axis1]) + BeginsPtr1[Axis1]) * StepPtr1[Axis1];
				const auto IndexAxis1B = IndexAxis0B +
					((j * StridesPtr2[Axis1]) + BeginsPtr2[Axis1]) * StepPtr2[Axis1];
				DataPtr1[IndexAxis1A] = (_Type)_Func(DataPtr2[IndexAxis1B], _Src2);
			}
		}
	}
	else if (CurDims == 1)
	{
		for (SizeType i = 0; i < ShapePtr[0]; ++i)
		{
			DataPtr1[((i * StridesPtr1[0]) + BeginsPtr1[0]) * StepPtr1[0]] =
				(_Type)_Func(DataPtr2[((i * StridesPtr2[0]) + BeginsPtr2[0]) * StepPtr2[0]], _Src2);
		}
	}
}
LibSvcEnd