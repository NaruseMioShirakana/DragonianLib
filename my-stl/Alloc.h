#pragma once
#define LIBSVCSTLBEGIN namespace libsvcstd {
#define LIBSVCSTLEND }

#ifndef LIBSVC_EMPTY_CAPACITY
#define LIBSVC_EMPTY_CAPACITY 16
#endif
#ifndef LIBSVC_ALLOC_ALIG
#define LIBSVC_ALLOC_ALIG 16
#ifndef LIBSVC_DEBUG
#define LIBSVC_DEBUG
#endif
#endif

LIBSVCSTLBEGIN

class BaseAllocator;
using Allocator = BaseAllocator*;

LIBSVCSTLEND

namespace libsvc
{
	static inline size_t NopID = size_t(-1);
	using libsvcstd::Allocator;

	enum class Device
	{
		CPU = 0,
		CUDA,
		HIP,
		DIRECTX,
		CPUMP,
		CUDAMP,
		HIPMP,
		DIRECTXMP
	};

	class MemoryProvider
	{
	public:
		~MemoryProvider();
		friend Allocator GetMemoryProvider(Device _Device);
		MemoryProvider(const MemoryProvider&) = delete;
		MemoryProvider(MemoryProvider&&) = delete;
		MemoryProvider& operator=(const MemoryProvider&) = delete;
		MemoryProvider& operator=(MemoryProvider&&) = delete;

	protected:
		Allocator _Provider[8];
	private:
		MemoryProvider();
	};

	Allocator GetMemoryProvider(Device _Device);
}

LIBSVCSTLBEGIN
using libsvc::Device;
using libsvc::MemoryProvider;
using libsvc::GetMemoryProvider;
using libsvc::NopID;

class BaseAllocator
{
public:
	friend class MemoryProvider;
	virtual ~BaseAllocator() {}
	virtual unsigned char* Allocate(size_t _Size);
	virtual void Free(void* _Block);
	
	Device GetDevice() const;
protected:
	BaseAllocator(Device _Type) : Type_(_Type) {}
	BaseAllocator(const BaseAllocator&) = delete;
	BaseAllocator(BaseAllocator&&) = delete;
	BaseAllocator& operator=(const BaseAllocator&) = delete;
	BaseAllocator& operator=(BaseAllocator&&) = delete;
	Device Type_;
};

class CPUAllocator : public BaseAllocator
{
public:
	friend class MemoryProvider;
	~CPUAllocator() override {}
	unsigned char* Allocate(size_t _Size) override;
	void Free(void* _Block) override;
protected:
	CPUAllocator() : BaseAllocator(Device::CPU) {}
	CPUAllocator(const CPUAllocator&) = delete;
	CPUAllocator(CPUAllocator&&) = delete;
	CPUAllocator& operator=(const CPUAllocator&) = delete;
	CPUAllocator& operator=(CPUAllocator&&) = delete;
};

LIBSVCSTLEND