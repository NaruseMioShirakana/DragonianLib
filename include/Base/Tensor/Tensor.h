#pragma once
#include <deque>
#include <ranges>
#include <random>
#include "Tensor/TensorBase.h"
#include "Util/ThreadPool.h"
#include "Tensor/Macro.h"
#include "Util/Avx256.h"
#include "Util/SpecialOperator.h"

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

enum class InterpolateType
{
	Nearest,
	Linear,
	Bilinear,
	Bicubic,
	Trilinear,
	Area,
};

using SliceOptions = Vector<Range>;

SizeType VectorMul(const ShapeType& _Input);

bool RangeIsAllNone(const Vector<Range>& _Input);

class Tensor : public TensorBase
{
public:
	using InvokeFnType = void(*)(Tensor&);

	Tensor() = delete;
	~Tensor() override;
	Tensor(TensorType _DType) :TensorBase(_DType) {}
	Tensor(const ShapeType& _Shape, TensorType _DType);
	Tensor(const Tensor& _Left);
	Tensor(Tensor&& _Right) noexcept;
	static Tensor FloatTensor(const Vector<float32>& _Array, ThreadPool* _ThreadPool = nullptr);
	static Tensor LongTensor(const Vector<int64>& _Array, ThreadPool* _ThreadPool = nullptr);
	static Tensor Ones(const ShapeType& _Shape, TensorType _Type = TensorType::Float32, ThreadPool* _ThreadPool = nullptr);
	static Tensor Zeros(const ShapeType& _Shape, TensorType _Type = TensorType::Float32, ThreadPool* _ThreadPool = nullptr);
	static Tensor ConstantOf(const ShapeType& _Shape, double _Val, TensorType _Type = TensorType::Float32, ThreadPool* _ThreadPool = nullptr);
	static Tensor ConstantOf(const ShapeType& _Shape, int64 _Val, TensorType _Type = TensorType::Float32, ThreadPool* _ThreadPool = nullptr);
	static Tensor Rand(const ShapeType& _Shape, TensorType _Type = TensorType::Float32, int64_t _Seed = 1919810, ThreadPool* _ThreadPool = nullptr);
	static Tensor Randn(const ShapeType& _Shape, TensorType _Type = TensorType::Float32, int64_t _Seed = 1919810, double _Mean = 0., double _Sigma = 1., ThreadPool* _ThreadPool = nullptr);
	static Tensor OnesLike(const Tensor& _Shape, ThreadPool* _ThreadPool = nullptr);
	static Tensor ZerosLike(const Tensor& _Shape, ThreadPool* _ThreadPool = nullptr);
	static Tensor ConstantLike(const Tensor& _Shape, double _Val, ThreadPool* _ThreadPool = nullptr);
	static Tensor ConstantLike(const Tensor& _Shape, int64 _Val, ThreadPool* _ThreadPool = nullptr);
	static Tensor RandLike(const Tensor& _Shape, int64_t _Seed = 1919810, ThreadPool* _ThreadPool = nullptr);
	static Tensor RandnLike(const Tensor& _Shape, int64_t _Seed = 1919810, double _Mean = 0., double _Sigma = 1., ThreadPool* _ThreadPool = nullptr);
	static Tensor Arange(float64 _Begin, float64 _End, float64 _Step, TensorType _Dtype = TensorType::Float32, ThreadPool* _ThreadPool = nullptr);
	static Tensor Arange(int64 _Begin, int64 _End, int64 _Step, TensorType _Dtype = TensorType::Int64, ThreadPool* _ThreadPool = nullptr);

	static void SetThreadCount(SizeType _Count);
	static void EnableTimeLogger(bool _Enabled);
	SizeType GetAlignSize() const
	{
		return AlignSize_;
	}
	std::mutex& GetRelMx() const
	{
		return RelMx_;
	}

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
	mutable std::mutex ViewMx_, RelMx_;
	
public:
	//�����������Ҫʹ���̳߳�
	Tensor& DoNotUseThreadPool()
	{
		UseThreadPool_ = false;
		return *this;
	}

	//���������ʹ���̳߳�
	Tensor& PlUseThreadPool()
	{
		UseThreadPool_ = true;
		return *this;
	}

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

	//ʹ��Indice��ȡһ������
	template <typename Ref>
	const Ref& Item(const ShapeType& _Indices) const
	{
		if (sizeof(Ref) != AlignSize_)
			LibSvcThrow("TypeError!");
		return *(Ref*)(Data(_Indices));
	}

	//��ȡ��ǰ��һ������
	template <typename Ref>
	const Ref& Item() const
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

	Tensor operator!=(const Tensor& _Right) const;
	Tensor operator==(const Tensor& _Right) const;
	Tensor operator<(const Tensor& _Right) const;
	Tensor operator>(const Tensor& _Right) const;
	Tensor operator<=(const Tensor& _Right) const;
	Tensor operator>=(const Tensor& _Right) const;
	Tensor operator!=(float64 _Right) const;
	Tensor operator==(float64 _Right) const;
	Tensor operator<(float64 _Right) const;
	Tensor operator>(float64 _Right) const;
	Tensor operator<=(float64 _Right) const;
	Tensor operator>=(float64 _Right) const;
	Tensor operator!=(int64 _Right) const;
	Tensor operator==(int64 _Right) const;
	Tensor operator<(int64 _Right) const;
	Tensor operator>(int64 _Right) const;
	Tensor operator<=(int64 _Right) const;
	Tensor operator>=(int64 _Right) const;

private:
	void Free();
	void ClearViewChilds();
	void ThrowOnNotEnabled() const;
	void RemoveSelfViewPtr();
	bool HasChild(const Tensor* _Child) const;
	void Assign1D(const void* _Val) const;
	void ReCalcViewInfo();

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

	//����һ����ǰTensor��View������_Dim����Last�ụ������˳�򣬷��ظ�View
	Tensor SwapLastDim(SizeType _Dim) const;

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
		SizeType _Axis = 0,
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
		SizeType _Axis = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	//ת��Tensor�����ͣ��½�Tensor��
	static Tensor Cast(
		const Tensor& _Input,
		TensorType _Dtype,
		ThreadPool* _ThreadPool = nullptr
	);

	static Tensor Sum(
		const Tensor& _Input,
		SizeType _Axis = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	static Tensor CumSum(
		const Tensor& _Input,
		SizeType _Axis = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	static Tensor Diff(
		const Tensor& _Input,
		SizeType _Axis = 0,
		ThreadPool* _ThreadPool = nullptr
	);

	static Tensor CumProd(
		const Tensor& _Input,
		SizeType _Axis = 0,
		ThreadPool* _ThreadPool = nullptr
	);

protected:
	Tensor GatherRef(SizeType _Index) const;

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
	LibSvcTensorFnDef(Floor);
	LibSvcTensorFnDef(Ceil);
	LibSvcTensorFnDef(Round);

private:
	bool UseThreadPool_ = true;
};

LibSvcEnd