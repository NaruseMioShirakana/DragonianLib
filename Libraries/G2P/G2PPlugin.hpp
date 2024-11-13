﻿#pragma once
#include "G2PBase.hpp"
#include "PluginBase/PluginBase.h"

_D_Dragonian_Lib_G2P_Header

namespace PreDefinedExtra
{
	struct _MyPhoneme { wchar_t P_2[8]; wchar_t P_1[8]; wchar_t P0[8]; wchar_t P1[8]; wchar_t P2[8]; };
	struct _MyA { int8_t A1; uint8_t A2; uint8_t A3; };
	struct _MyB { wchar_t B1[8]; wchar_t B2[8]; wchar_t B3[8]; };
	struct _MyC { wchar_t C1[8]; wchar_t C2[8]; wchar_t C3[8]; };
	struct _MyD { wchar_t D1[8]; wchar_t D2[8]; wchar_t D3[8]; };
	struct _MyE { uint8_t E1; uint8_t E2; bool E3; wchar_t E4[8]; bool E5; };
	struct _MyF { uint8_t F1; uint8_t F2; bool F3; wchar_t F4[8]; uint8_t F5; uint8_t F6; uint8_t F7; uint8_t F8; };
	struct _MyG { uint8_t G1; uint8_t G2; bool G3; wchar_t G4[8]; bool G5; };
	struct _MyH { uint8_t H1; uint8_t H2; };
	struct _MyI { uint8_t I1; uint8_t I2; uint8_t I3; uint8_t I4; uint8_t I5; uint8_t I6; uint8_t I7; uint8_t I8; };
	struct _MyJ { uint8_t J1; uint8_t J2; };
	struct _MyK { uint8_t K1; uint8_t K2; uint8_t K3; };
	struct FullContext
	{
		_MyPhoneme Phoneme; _MyA A;
		_MyB B; _MyC C; _MyD D; _MyE E; _MyF F;
		_MyG G; _MyH H; _MyI I; _MyJ J; _MyK K;
	};
}

class BasicG2P : public G2PBase
{
public:
	using G2PApiType = struct { wchar_t Phoneme[8]; int64_t Tone; }*(*)(void*, const wchar_t*, const char*, const void*);
	using G2PGetExtraInfoType = void* (*)(void*);

	BasicG2P() = delete;
	BasicG2P(const void* UserParameter, Plugin::Plugin PluginInp);
	~BasicG2P() override;

	std::pair<Vector<std::wstring>, Vector<Int64>> Convert(
		const std::wstring& InputText,
		const std::string& LanguageID,
		const void* UserParameter = nullptr
	) override;

	std::pair<std::unique_lock<std::mutex>, void*> GetExtraInfo() override
	{
		if (!_MyGetExtraInfo)
			_D_Dragonian_Lib_Not_Implemented_Error;
		return { std::unique_lock(_MyMutex) ,_MyGetExtraInfo(_MyInstance)};
	}

private:
	void* _MyInstance = nullptr;
	Plugin::Plugin _MyPlugin = nullptr;
	G2PApiType _MyConvert = nullptr;
	G2PGetExtraInfoType _MyGetExtraInfo = nullptr;

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