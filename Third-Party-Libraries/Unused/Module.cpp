#include <ranges>
#include "Module/Module.h"

#include "Util/StringPreprocess.h"

DragonianLibSpaceBegin
DragonianLibGraphBegin

Module::Module(Module* _Parent, const std::wstring& _Name)
{
	if (_Parent != nullptr)
	{
		if (!_Parent->RegName_.empty())
			RegName_ = _Parent->RegName_ + L"." + _Name;
		else
			RegName_ = _Name;
		_Parent->Layers_[RegName_] = this;
	}
	else
		RegName_ = _Name;
}

void Module::loadData(const DictType& _WeightDict, bool _Strict)
{
	for (const auto& it : Layers_ | std::views::values)
		it->loadData(_WeightDict, _Strict);
}

void Module::saveData(FileGuard& _File)
{
	for (const auto& it : Layers_ | std::views::values)
		it->saveData(_File);
}

std::any Module::operator()(const std::any&, ThreadPool*, bool _Inplace)
{
	DragonianLibNotImplementedError;
}

void Module::ReCalcLayerName(const std::wstring& _Parent)
{
	std::unordered_map<std::wstring, Module*> Lay;
	for (auto i : Layers_ | std::ranges::views::values)
	{
		i->ReCalcLayerName(_Parent);
		i->RegName_ = _Parent + L"." + i->RegName_;
		Lay[i->RegName_] = i;
	}
	Layers_ = std::move(Lay);
}

std::wstring Module::DumpLayerNameInfo()
{
	std::wstring Ret;
	DumpLayerNameInfoImpl(Ret, 0);
	return Ret;
}

void Module::DumpLayerNameInfoImpl(std::wstring& _Tmp, int TabCount)
{
	for (int i = 0; i < TabCount; ++i) _Tmp += L"  ";
	DumpCurrentLayerInfo(_Tmp);
	if (!Layers_.empty())
	{
		if (!_Tmp.empty())
			_Tmp += L" = {";
		else
			_Tmp += L"{";
		_Tmp += '\n';
		for (auto i : Layers_ | std::ranges::views::values)
			i->DumpLayerNameInfoImpl(_Tmp, TabCount + 1);
		for (int i = 0; i < TabCount; ++i) _Tmp += L"  ";
		_Tmp += L'}';
		_Tmp += '\n';
	}
	else
		_Tmp += '\n';
}

void Module::DumpCurrentLayerInfo(std::wstring& _Tmp)
{
	_Tmp += RegName_;
}

Parameter::Parameter(Module* _Parent, const std::wstring& _Name, const std::vector<SizeType>& _Shape, TensorType _DType) :
	Module(_Parent, _Name),
	Weight_(_Shape, _DType, Device::CPU)
{
	
}

Parameter::operator const Tensor& () const
{
	return Weight_;
}

void Parameter::ChangeShape(const std::vector<SizeType>& _Shape)
{
	Weight_ = { _Shape, Weight_.DType(), Weight_.GetDevice() };
}

void Parameter::DumpCurrentLayerInfo(std::wstring& _Tmp)
{
	_Tmp += (RegName_ + L" (Size[");
	for (auto i : Weight_.Shape())
		_Tmp += (std::to_wstring(i) + L", ");
	if (!Weight_.Shape().empty())
	{
		_Tmp.pop_back();
		_Tmp.pop_back();
	}
	_Tmp += L"])";
}

bool operator==(const ShapeType& _L, const ShapeType& _R)
{
	if (_L.size() != _R.size())
		return false;
	for (size_t i = 0; i < _L.size(); ++i)
		if (_L[i] != _R[i])
			return false;
	return true;
}

void Parameter::loadData(const DictType& _WeightDict, bool _Strict)
{
	try
	{
		auto Iter = _WeightDict.find(WideStringToUTF8(RegName_));
		if (Iter == _WeightDict.end() && _Strict)
			DragonianLibThrow("Missing Weight \"" + WideStringToUTF8(RegName_) + '\"');
		if (Weight_.Shape() != Iter->second.Shape_)
			DragonianLibThrow(WideStringToUTF8(L"Size Of \"" + RegName_ + L"\" ≠ Size Of Pretrained Model!"));
	}
	catch (std::exception& e)
	{
		DragonianLibThrow(e.what());
	}
}

void Parameter::saveData(FileGuard& _File)
{
	DragonianLibNotImplementedError;
}

Sequential::Sequential(Module* _Parent, const std::wstring& _Name) : Module(_Parent, _Name)
{
}

Sequential::~Sequential()
{
	for (auto i : _Items)
		delete i;
	_Items.clear();
}

std::any Sequential::operator()(const std::any& _Input, ThreadPool* _Thp, bool _Inplace)
{
	if (_Items.empty())
		return {};
	auto Output = (*_Items[0])(_Input, _Thp, _Inplace);
	for (size_t i = 1; i < _Items.size(); ++i)
		Output = (*_Items[i])(Output, _Thp, _Inplace);
	return Output;
}

void Sequential::Append(Module* _Module)
{
	_Module->Name() = RegName_ + L"." + std::to_wstring(_Items.size());
	_Module->ReCalcLayerName(_Module->Name());
	_Items.emplace_back(_Module);
	Layers_[_Module->Name()] = _Module;
}

Sequential& Sequential::operator=(const std::initializer_list<Module*>& _Input)
{
	for (auto i : _Input)
		Append(i);
	return *this;
}

ModuleList::ModuleList(Module* _Parent, const std::wstring& _Name) : Sequential(_Parent, _Name)
{

}

ModuleList::ModuleList(Module* _Parent, const std::wstring& _Name, const std::initializer_list<Module*>& _Input) : Sequential(_Parent, _Name)
{
	for (auto i : _Input)
		Append(i);
}

ModuleList::~ModuleList()
{

}

std::any ModuleList::operator()(const std::any& _Input, ThreadPool* _Thp, bool _Inplace)
{
	DragonianLibNotImplementedError;
}

ModuleList& ModuleList::operator=(const std::initializer_list<Module*>& _Input)
{
	for (auto i : _Input)
		Append(i);
	return *this;
}

Conv1D::Conv1D(Module* _Parent, const std::wstring& _Name, const ConvParam& _Params) :
	Module(_Parent, _Name),
	DragonianLibRegisterLayer(weight, { _Params.OutChannels, _Params.InChannels / _Params.Groups, _Params.KernelSize }),
	DragonianLibRegisterLayer(bias, { _Params.OutChannels }),
	params(_Params)
{
	if (!_Params.Bias)
		Layers_.erase(bias.Name());
}

void Conv1D::DumpCurrentLayerInfo(std::wstring& _Tmp)
{
	_Tmp += std::format(
		L"{} Conv1D(in_channel[{}], out_channel[{}], kernel_size[{}])",
		RegName_,
		params.InChannels,
		params.OutChannels,
		params.KernelSize
	);
}

std::any Conv1D::operator()(const std::any& _Input, ThreadPool* _Thp, bool _Inplace)
{
	DragonianLibNotImplementedError;
}

void Conv1D::ChangeParam(const ConvParam& _Params)
{
	params = _Params;
	weight.ChangeShape({ _Params.OutChannels, _Params.InChannels / _Params.Groups, _Params.KernelSize });
	bias.ChangeShape({ _Params.OutChannels });
}

Conv2D::Conv2D(Module* _Parent, const std::wstring& _Name, const ConvParam& _Params) :
	Module(_Parent, _Name),
	DragonianLibRegisterLayer(
		weight,
		{ _Params.OutChannels, _Params.InChannels / _Params.Groups, _Params.KernelSize[0], _Params.KernelSize[1] }
	),
	DragonianLibRegisterLayer(bias, { _Params.OutChannels }),
	params(_Params)
{
	if (!_Params.Bias)
		Layers_.erase(bias.Name());
}

void Conv2D::DumpCurrentLayerInfo(std::wstring& _Tmp)
{
	_Tmp += std::format(
		L"{} Conv2D(in_channel[{}], out_channel[{}], kernel_size[{}, {}])",
		RegName_,
		params.InChannels,
		params.OutChannels,
		params.KernelSize[0],
		params.KernelSize[1]
	);
}

std::any Conv2D::operator()(const std::any& _Input, ThreadPool* _Thp, bool _Inplace)
{
	DragonianLibNotImplementedError;
}

void Conv2D::ChangeParam(const ConvParam& _Params)
{
	params = _Params;
	weight.ChangeShape({ _Params.OutChannels, _Params.InChannels / _Params.Groups, _Params.KernelSize[0], _Params.KernelSize[1] });
	bias.ChangeShape({ _Params.OutChannels });
}

ConvTranspose1D::ConvTranspose1D(Module* _Parent, const std::wstring& _Name, const ConvParam& _Params) :
	Module(_Parent, _Name),
	DragonianLibRegisterLayer(weight, { _Params.InChannels, _Params.OutChannels / _Params.Groups, _Params.KernelSize }),
	DragonianLibRegisterLayer(bias, { _Params.OutChannels }),
	params(_Params)
{
	if (_Params.PaddingMode != PaddingType::Zero)
		DragonianLibThrow("Only `Zeros` Padding Mode Is Supported For ConvTranspose1D");
	if (!_Params.Bias)
		Layers_.erase(bias.Name());
}

void ConvTranspose1D::DumpCurrentLayerInfo(std::wstring& _Tmp)
{
	_Tmp += std::format(
		L"{} ConvTranspose1D(in_channel[{}], out_channel[{}], kernel_size[{}])",
		RegName_,
		params.InChannels,
		params.OutChannels,
		params.KernelSize
	);
}

std::any ConvTranspose1D::operator()(const std::any& _Input, ThreadPool* _Thp, bool _Inplace)
{
	DragonianLibNotImplementedError;
}

void ConvTranspose1D::ChangeParam(const ConvParam& _Params)
{
	params = _Params;
	weight.ChangeShape({ _Params.InChannels, _Params.OutChannels / _Params.Groups, _Params.KernelSize });
	bias.ChangeShape({ _Params.OutChannels });
}

ConvTranspose2D::ConvTranspose2D(Module* _Parent, const std::wstring& _Name, const ConvParam& _Params) :
	Module(_Parent, _Name),
	DragonianLibRegisterLayer(
		weight,
		{ _Params.InChannels, _Params.OutChannels / _Params.Groups, _Params.KernelSize[0], _Params.KernelSize[1] }
	),
	DragonianLibRegisterLayer(bias, { _Params.OutChannels }),
	params(_Params)
{
	if (!_Params.Bias)
		Layers_.erase(bias.Name());
}

void ConvTranspose2D::DumpCurrentLayerInfo(std::wstring& _Tmp)
{
	_Tmp += std::format(
		L"{} ConvTranspose2D(in_channel[{}], out_channel[{}], kernel_size[{}, {}])",
		RegName_,
		params.InChannels,
		params.OutChannels,
		params.KernelSize[0],
		params.KernelSize[1]
	);
}

std::any ConvTranspose2D::operator()(const std::any& _Input, ThreadPool* _Thp, bool _Inplace)
{
	DragonianLibNotImplementedError; //TODO
}

void ConvTranspose2D::ChangeParam(const ConvParam& _Params)
{
	params = _Params;
	weight.ChangeShape({ _Params.InChannels, _Params.OutChannels / _Params.Groups, _Params.KernelSize[0], _Params.KernelSize[1] });
	bias.ChangeShape({ _Params.OutChannels });
}

Linear::Linear(Module* _Parent, const std::wstring& _Name, const LinearParam& _Params) :
	Module(_Parent, _Name),
	DragonianLibRegisterLayer(weight, { _Params.OutFeatures, _Params.InFeatures }),
	DragonianLibRegisterLayer(bias, { _Params.OutFeatures }),
	params(_Params)
{
	if (!_Params.Bias)
		Layers_.erase(bias.Name());
}

void Linear::DumpCurrentLayerInfo(std::wstring& _Tmp)
{
	_Tmp += std::format(
		L"{} Linear(in_feature[{}], out_feature[{}])",
		RegName_,
		params.InFeatures,
		params.OutFeatures
	);
}

std::any Linear::operator()(const std::any& _Input, ThreadPool* _Thp, bool _Inplace)
{
	DragonianLibNotImplementedError;
}

void Linear::ChangeParam(const LinearParam& _Params)
{
	params = _Params;
	weight.ChangeShape({ _Params.OutFeatures, _Params.InFeatures });
	bias.ChangeShape({ _Params.OutFeatures });
}

Embedding::Embedding(Module* _Parent, const std::wstring& _Name, const EmbeddingParam& _Params) :
	Module(_Parent, _Name),
	DragonianLibRegisterLayer(weight, { _Params.NumEmbeddings, _Params.EmbeddingDim }),
	params(_Params)
{

}

void Embedding::DumpCurrentLayerInfo(std::wstring& _Tmp)
{
	_Tmp += std::format(
		L"{} Embedding(num_embeddings[{}], embedding_dim[{}])",
		RegName_,
		params.NumEmbeddings,
		params.EmbeddingDim
	);
}

std::any Embedding::operator()(const std::any& _Input, ThreadPool* _Thp, bool _Inplace)
{
	return weight->Gather(std::any_cast<Tensor>(_Input), _Thp);
}

void Embedding::ChangeParam(const EmbeddingParam& _Params)
{
	params = _Params;
	weight.ChangeShape({ _Params.NumEmbeddings, _Params.EmbeddingDim });
}

DragonianLibGraphEnd
DragonianLibSpaceEnd