#pragma once
#include <mutex>
#include "Libraries/Base.h"
#include "Libraries/MyTemplateLibrary/Vector.h"
#define _D_Dragonian_Lib_G2P_Header _D_Dragonian_Lib_Space_Begin namespace G2P {
#define _D_Dragonian_Lib_G2P_End _D_Dragonian_Lib_Space_End }

_D_Dragonian_Lib_G2P_Header

using namespace DragonianLibSTL;

class G2PBase
{
public:
	
	G2PBase() = default;
	virtual ~G2PBase() = default;

	virtual std::pair<Vector<std::wstring>, Vector<Int64>> Convert(
		const std::wstring& InputText,
		const std::string& LanguageID,
		const void* UserParameter = nullptr
	) = 0;

	virtual std::pair<std::unique_lock<std::mutex>, void*> GetExtraInfo() = 0;

protected:
	void Construct(const void* Parameter);
	void Destory();
	virtual void Initialize(const void* Parameter) = 0;
	virtual void Release() = 0;
	std::mutex _MyMutex;

private:
	G2PBase(const G2PBase&) = delete;
	G2PBase(G2PBase&&) = delete;
	G2PBase& operator=(G2PBase&&) noexcept = delete;
	G2PBase& operator=(const G2PBase&) = delete;
};

_D_Dragonian_Lib_G2P_End