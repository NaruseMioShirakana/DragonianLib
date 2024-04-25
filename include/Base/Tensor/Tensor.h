#pragma once
#include <deque>
#include <ranges>
#include <random>
#include "Tensor/TensorBase.h"
#include "Util/ThreadPool.h"
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

	Tensor operator[](SizeType _Index) const;

	Tensor operator[](const SliceOptions& _SliceOptions) const;

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

	//����������������ظ��������½�Tensor��
	static Tensor Repeat(
		const Tensor& _Input,
		const Vector<std::pair<SizeType, SizeType>>& _Repeat,
		ThreadPool* _ThreadPool = nullptr
	);
};
LibSvcEnd