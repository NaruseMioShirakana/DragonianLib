#include "../G2PW.hpp"

_D_Dragonian_Lib_G2P_Header

G2PWModel::G2PWModel(
	const void* Parameter
) : OnnxModelBase(*(const OnnxRuntime::OnnxRuntimeEnvironment*)((const G2PWModelHParams*)Parameter)->Enviroment,
	((const G2PWModelHParams*)Parameter)->ModelPath,
	*(const DLogger*)(((const G2PWModelHParams*)Parameter)->Logger))
{
	
}

std::pair<Vector<std::wstring>, Vector<Int64>> G2PWModel::Convert(
	const std::wstring& InputText,
	const std::string& LanguageID,
	const void* UserParameter
)
{
	return Forward(InputText, LanguageID, UserParameter);
}

std::pair<Vector<std::wstring>, Vector<Int64>> G2PWModel::Forward(
	const std::wstring& InputText,
	const std::string& LanguageID,
	const void* UserParameter
)
{
	return {};
}

_D_Dragonian_Lib_G2P_End