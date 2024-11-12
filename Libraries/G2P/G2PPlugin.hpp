#pragma once
#include <mutex>
#include "G2PBase.hpp"
#include "PluginBase/PluginBase.h"

_D_Dragonian_Lib_G2P_Header

class BasicG2P : public G2PBase
{
public:
	using G2PApiType = void** (*)(void*, const wchar_t*, const char*, const void*);
	using G2PGetInstanceType = void* (*)(const void*);

	BasicG2P() = delete;
	BasicG2P(const void* UserParameter, Plugin::Plugin PluginInp);
	~BasicG2P() override;

	std::pair<Vector<std::wstring>, Vector<Int64>> Convert(
		const std::wstring& InputText,
		const std::string& LanguageID,
		const void* UserParameter = nullptr
	) override;

private:
	void* _MyInstance = nullptr;
	Plugin::Plugin _MyPlugin = nullptr;
	G2PApiType _MyConvert = nullptr;
	std::mutex _MyMutex;

protected:
	void Initialize(const void* Parameter) override;
	void Release() override;

private:
	BasicG2P(const BasicG2P&) = delete;
	BasicG2P(BasicG2P&&) = delete;
	BasicG2P& operator=(BasicG2P&&) noexcept = delete;
	BasicG2P& operator=(const BasicG2P&) = delete;
};

_D_Dragonian_Lib_G2P_End