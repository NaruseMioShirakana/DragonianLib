#pragma once
#include <functional>
#include "G2PPlugin.hpp"

_D_Dragonian_Lib_G2P_Header

using G2PModule = std::shared_ptr<BasicG2P>;

/**
 * @brief Get a G2PModule
 * @param Name Name of the G2PModule
 * @param Parameter User parameter
 * @return G2PModule
 */
G2PModule GetG2P(
	const std::wstring& Name,
	const void* Parameter = nullptr
);

/**
 * @brief Register All G2PModules in the directory
 * @param _PluginRootDirectory Root directory of the G2PModules
 */
void RegisterG2PModule(const std::wstring& _PluginRootDirectory);

/**
 * @brief Get a list of G2PModule names
 * @return List of G2PModule names
 */
const std::vector<std::wstring>& GetG2PModuleList();

_D_Dragonian_Lib_G2P_End