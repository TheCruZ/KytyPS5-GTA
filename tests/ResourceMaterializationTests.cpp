#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace {

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "ResourceMaterializationTests: failed: %s\n", text);
    std::abort();
  }
}

bool RejectSpecializationRead(void *userdata, uint64_t, std::span<uint32_t>) {
  ++*static_cast<uint32_t *>(userdata);
  return false;
}

Libs::Graphics::ShaderRecompiler::IR::Block &
AddValueBlock(Libs::Graphics::ShaderRecompiler::IR::Program &program) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto block = std::make_unique<Block>();
  auto *result = block.get();
  program.blocks.push_back(result);
  result->id = static_cast<uint32_t>(program.blocks.size() - 1u);
  program.block_storage.push_back(std::move(block));
  return *result;
}

Libs::Graphics::ShaderRecompiler::IR::Program SrtProgram(uint64_t address) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  program.memory_info.push_back(memory);
  const auto low = Value(static_cast<uint32_t>(address));
  const auto high = Value(static_cast<uint32_t>(address >> 32u));
  auto &handle =
      value_block.AppendNewInst(ValueOpcode::GetAddressResource, {low, high});
  auto &raw = value_block.AppendNewInst(
      ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(0u), Value(0u), Value(true)});
  raw.SetFlags(MemoryFlags{.index = 0, .pc = 0x40});
  program.srt_reads.push_back({Value(&raw), 0});

  auto &srt = value_block.AppendNewInst(ValueOpcode::GetSrtResource);
  auto &flat = value_block.AppendNewInst(ValueOpcode::ReadConst,
                                         {Value(&srt), Value(0u)});
  DescriptorSource source;
  source.dwords[0] = Value(&flat);
  source.dwords[1] = Value(0u);
  source.dword_count = 2;
  program.descriptor_sources.push_back(source);
  return program;
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UnbasedFlatPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  AddValueBlock(program);
  program.info.uses_dma = true;
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UserDataBufferPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  auto &user_data = value_block.AppendNewInst(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(0))});
  DescriptorSource source;
  source.dwords[0] = Value(&user_data);
  source.dwords[1] = Value(0u);
  source.dwords[2] = Value(0u);
  source.dwords[3] = Value(0u);
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0});
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::Program MixedSamplerProgram() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);

  const auto AddSource = [&program](uint32_t dword_count) {
    DescriptorSource source;
    source.dword_count = dword_count;
    for (uint32_t i = 0; i < dword_count; i++) {
      source.dwords[i] = Value(0u);
    }
    program.descriptor_sources.push_back(source);
    return static_cast<uint32_t>(program.descriptor_sources.size() - 1u);
  };

  const auto image0 = AddSource(8);
  const auto image1 = AddSource(8);
  const auto sampler0 = AddSource(4);
  const auto sampler1 = AddSource(4);
  program.descriptor_sources[image1].dwords[0] = Value(1u);
  program.descriptor_sources[image1].dwords[1] = Value(static_cast<uint32_t>(
      Libs::Graphics::Prospero::BufferFormat::k11_11_10UInt) << 20u);
  program.descriptor_sources[image1].dwords[3] = Value(static_cast<uint32_t>(
      Libs::Graphics::Prospero::ImageType::kColor2D) << 28u);
  for (uint32_t index = 0; index < 2; ++index) {
    auto &value = block.AppendNewInst(
        ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(index))});
    program.descriptor_sources[sampler0 + index].dwords[0] = Value(&value);
  }
  program.info.images.push_back(
      {.source = image0,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.images.push_back(
      {.source = image1,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.samplers.push_back({.source = sampler0});
  program.info.samplers.push_back({.source = sampler1});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 0});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 1});
  program.info.sampled_pairs.push_back({.image = 1, .sampler = 1});
  return program;
}

void TestMappedSrtUsesDirectReaderByDefault() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const uint32_t dword = 0x12345678;
  auto plan = ExtractResourcePlan(SrtProgram(reinterpret_cast<uint64_t>(&dword)));
  uint32_t specialization_reads = 0;
  const SrtRuntime runtime{.userdata = &specialization_reads,
                           .read_specialization_memory =
                               RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mapped SRT stage materialization failed");
  Check(specialization_reads == 0,
        "ordinary SRT read used the specialization reader");
  Check(snapshot.flattened_srt.size() == 1 &&
            snapshot.flattened_srt[0] == dword,
        "cache rematerialization did not use the direct reader by default");
}

void TestNullPageSrtReadsZero() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = ExtractResourcePlan(SrtProgram(0x18u));
  uint32_t specialization_reads = 0;
  const SrtRuntime runtime{.userdata = &specialization_reads,
                           .read_specialization_memory =
                               RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            snapshot.flattened_srt == std::vector<uint32_t>{0u},
        "direct SRT read from a null table pointer did not read zero");
  // Captured reads go through the ordinary reader instead of the direct one.
  plan.source_reads = {{0x1000u, 4u}};
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            snapshot.flattened_srt == std::vector<uint32_t>{0u} &&
            snapshot.specialization_reads == plan.source_reads,
        "captured SRT read from a null table pointer did not read zero untracked");
}

void TestNativeNestedSrtRefresh() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  std::array<uint32_t, 3> first{0u, 0x12345678u, 0xabcdef01u};
  std::array<uint32_t, 3> second{0u, 0x87654321u, 0x10fedcbau};
  const auto pointer_words = [](const void *pointer) {
    const auto address = reinterpret_cast<uint64_t>(pointer);
    return std::array<uint32_t, 2>{static_cast<uint32_t>(address) | 3u,
                                  static_cast<uint32_t>(address >> 32u) | 0xabcd0000u};
  };
  auto table = pointer_words(first.data());
  const auto user_data = pointer_words(table.data());
  auto program = SrtProgram(0u);
  program.user_data_base = 4;
  program.user_data_count = 2;
  auto &block = *program.blocks[0];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto low = emit(ValueOpcode::GetUserData, {Value(ScalarReg(4))});
  const auto high = emit(ValueOpcode::GetUserData, {Value(ScalarReg(5))});
  const auto root = emit(ValueOpcode::GetAddressResource, {low, high});
  const auto read = [&](Value address, uint32_t offset) {
    const auto value = emit(ValueOpcode::LoadAddressU32,
                            {address, Value(offset), Value(0u), Value(true)});
    value.Instruction()->SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back(
        {value, static_cast<uint32_t>(program.srt_reads.size())});
    return value;
  };
  const auto pointer_low = read(root, 0u);
  const auto pointer_high = read(root, 4u);
  auto *leaf = program.srt_reads[0].value.Instruction();
  leaf->Arg(0).Instruction()->SetArg(0, pointer_low);
  leaf->Arg(0).Instruction()->SetArg(1, pointer_high);
  leaf->SetArg(1, Value(7u));
  read(leaf->Arg(0), 11u);
  auto plan = ExtractResourcePlan(program);
  Check(plan.srt_code != nullptr, "nested user-data tree did not compile an x86 walker");
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  const SrtRuntime runtime{.user_data = user_data};
  {
    SrtWalker walker(plan, runtime);
    Check(walker.RefreshFlatBuffer(snapshot.flattened_srt),
          "native SRT refresh failed");
    const auto loaded = first[1];
    first[1] ^= 0xffffffffu;
    uint32_t value = 0;
    Check(walker.Evaluate(plan.srt_reads[0].value, value) && value == loaded,
          "descriptor evaluation reloaded a value already copied by the native walker");
    first[1] = loaded;
  }
  for (const auto *data : {&first, &second}) {
    table = pointer_words(data->data());
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.flattened_srt ==
                  std::vector<uint32_t>{(*data)[1], table[0], table[1], (*data)[2]},
          "native SRT refresh lost pointer masking, alignment, or current values");
  }
  table = pointer_words(nullptr);
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            snapshot.flattened_srt == std::vector<uint32_t>{0u, table[0], table[1], 0u},
        "native SRT did not read zero through a null nested table");
  Check(!MaterializeResources(plan, {.user_data = std::span(user_data).first(1)},
                              snapshot, specialization),
        "native SRT walk accepted truncated user data");
  const std::array<uint32_t, 2> fake_user_data{0x1003u, 0xabcd0000u};
  uint32_t reads = 0;
  const SrtRuntime callback_runtime{
      .user_data = fake_user_data,
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        ++*static_cast<uint32_t *>(data);
        switch (address) {
          case 0x1000u: words[0] = 0x2003u; break;
          case 0x1004u: words[0] = 0xabcd0000u; break;
          case 0x2004u: words[0] = 0x1234u; break;
          case 0x2008u: words[0] = 0x5678u; break;
          default: return false;
        }
        return true;
      },
      .userdata = &reads};
  Check(MaterializeResources(plan, callback_runtime, snapshot, specialization) && reads == 4 &&
            snapshot.flattened_srt ==
                std::vector<uint32_t>{0x1234u, 0x2003u, 0xabcd0000u, 0x5678u},
        "native-eligible SRT bypassed its callback or repeated nested pointer reads");
  leaf->SetArg(0, root);
  leaf->SetArg(1, Value(0u));
  program.srt_reads.resize(1);
  for (const int32_t offset : {-1, 7}) {
    program.memory_info[0].offset = static_cast<uint32_t>(offset);
    auto bounded = ExtractResourcePlan(program);
    auto base = pointer_words(second.data() + 1);
    Check(bounded.srt_code != nullptr &&
              MaterializeResources(bounded, {.user_data = base}, snapshot, specialization) &&
              snapshot.flattened_srt[0] == second[offset < 0 ? 0 : 2],
          "native SRT applied a signed immediate offset incorrectly");
    base = offset < 0 ? std::array<uint32_t, 2>{0u, 0u}
                      : std::array<uint32_t, 2>{0xfffffffcu, 0xffffu};
    Check(!MaterializeResources(bounded, {.user_data = base}, snapshot, specialization),
          "native SRT accepted address underflow or 48-bit overflow");
    if (offset > 0) {
      Check(MaterializeResources(bounded, {.user_data = std::array<uint32_t, 2>{0u, 0u}},
                                 snapshot, specialization) &&
                snapshot.flattened_srt[0] == 0u,
            "native SRT did not read zero through a null table pointer");
    }
  }
  high.Instruction()->SetArg(0, Value(ScalarReg(6)));
  Check(ExtractResourcePlan(program).srt_code == nullptr,
        "native SRT accepted nonadjacent user-data pointer words");
  high.Instruction()->SetArg(0, Value(ScalarReg(5)));
  program.memory_info[0].offset = 0;
  program.srt_reads.push_back({pointer_low, 1});
  program.srt_reads.push_back({pointer_high, 2});
  leaf->SetArg(0, emit(ValueOpcode::GetAddressResource, {pointer_low, pointer_high}));
  pointer_high.Instruction()->SetArg(1, Value(8u));
  Check(ExtractResourcePlan(program).srt_code == nullptr,
        "native SRT accepted nonadjacent nested pointer words");
}

void TestIntegerRuntimeValueFollowsSrtReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = SrtProgram(0x10000);
  const auto root = plan.descriptor_sources.front().dwords[0];
  Check(ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT read was rejected");

  Block values;
  auto &comparison = values.AppendNewInst(ValueOpcode::FPOrdLessThanEqual32,
                                          {Value::F32(1.f), Value::F32(0.f)});
  auto &selection = values.AppendNewInst(
      ValueOpcode::SelectU32, {Value(&comparison), Value(1u), Value(0u)});
  plan.srt_reads[0].value = Value(&selection);
  Check(ValidateRuntimeValue(plan, root),
        "ordinary SRT validation rejected a floating-point dependency");
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT validation missed a hidden floating-point dependency");

  auto &first =
      values.AppendNewInst(ValueOpcode::ReadFirstLane, {root, Value(true)});
  Check(!ValidateRuntimeValue(plan, Value(&first), RuntimeValueType::Integer),
        "read-first-lane lost integer-only SRT validation");

  auto &active = values.AppendNewInst(ValueOpcode::ReadFirstLane,
                                      {Value(&selection), Value(&comparison)});
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point execution mask was accepted as integer-only");

  auto &lane = values.AppendNewInst(
      ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)),
       Value(0u)});
  auto &mask =
      values.AppendNewInst(ValueOpcode::INotEqual32, {Value(&lane), Value(0u)});
  selection.SetArg(0, Value(&mask));
  active.SetArg(1, Value(&mask));
  Check(ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "nonuniform integer execution mask was rejected");
  auto &float_value =
      values.AppendNewInst(ValueOpcode::BitCastU32F32, {Value::F32(1.f)});
  selection.SetArg(2, Value(&float_value));
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point inactive arm was accepted as integer-only");

  plan.srt_reads[0].value = Value(&first);
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "cyclic SRT read-first-lane dependency was accepted");
}

void TestSrtAliasesRetainReadPolicy() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto program = SrtProgram(0x1000u);
  auto &block = *program.blocks[0];
  const auto append_read = [&](uint32_t address) {
    auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                       {Value(address), Value(0u)});
    auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(0u), Value(0u), Value(true)});
    // Different reads deliberately share MemoryInfo but have different guest PCs.
    read.SetFlags(MemoryFlags{.index = 0, .pc = address});
    const auto slot = static_cast<uint32_t>(program.srt_reads.size());
    program.srt_reads.push_back({Value(&read), slot});
    auto &srt = block.AppendNewInst(ValueOpcode::GetSrtResource);
    return Value(&block.AppendNewInst(ValueOpcode::ReadConst,
                                      {Value(&srt), Value(slot)}));
  };
  const auto ordinary = append_read(0x2000u);
  const auto pointer = append_read(0x3000u);
  program.srt_reads[0].value.Instruction()->Arg(0).Instruction()->SetArg(0, pointer);
  auto &mask = block.AppendNewInst(ValueOpcode::INotEqual32, {ordinary, Value(0u)});
  auto &first = block.AppendNewInst(ValueOpcode::ReadFirstLane,
                                    {Value(9u), Value(&mask)});
  program.descriptor_sources.push_back({.dwords = {Value(&first), Value(0u)},
                                       .dword_count = 2});
  DescriptorSource indirect;
  indirect.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1};
  program.descriptor_sources.push_back(indirect);
  auto plan = ExtractResourcePlan(program);
  for (const auto &inst : plan.value_storage) {
    Check(inst.GetOpcode() != ValueOpcode::ReadConst &&
              inst.GetOpcode() != ValueOpcode::GetSrtResource,
          "retained plan still contains flattened-read aliases");
  }
  Check(plan.descriptor_sources[0].dwords[0] == plan.srt_reads[0].value &&
            plan.srt_reads[0].value.Instruction()->Flags<SrtReadFlags>().clean == 1u &&
            plan.srt_reads[1].value.Instruction()->Flags<SrtReadFlags>().clean == 0u &&
            plan.srt_reads[2].value.Instruction()->Flags<SrtReadFlags>().clean == 1u,
        "alias normalization lost read identity or retained discarded EXEC provenance");
  Check(plan.srt_code == nullptr,
        "captured strict reads compiled an unused x86 walker");
  struct Reads { uint32_t strict = 0; uint32_t ordinary = 0; bool dirty = false; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        if (address == 0x2000u) words[0] = 0x2222u;
        else if (address == 0x3000u) words[0] = 0x8000u;
        else if (address == 0x8000u) words[0] = 0xeeeeu;
        else return false;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address,
                                        std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        if (reads.dirty) return false;
        if (address == 0x3000u) words[0] = 0x1000u;
        else if (address == 0x1000u) words[0] = 0x1111u;
        else return false;
        return true;
      }};
  std::vector<uint32_t> flat;
  DescriptorValue descriptor;
  {
    SrtWalker clean(plan, CleanRuntime(runtime));
    SrtWalker walker(plan, runtime, &clean);
    Check(walker.RefreshFlatBuffer(flat) &&
              flat == std::vector<uint32_t>{0x1111u, 0x2222u, 0x1000u} &&
              walker.EvaluateDescriptor(0, descriptor) && descriptor.dwords[0] == 0x1111u &&
              reads.strict == 2 && reads.ordinary == 1,
          "normalized aliases changed nested strict reads or repeated a shared read");
  }
  Check(!SrtWalker(plan, runtime).RefreshFlatBuffer(flat),
        "strict flat read accepted a missing clean evaluator");
  auto no_reader = runtime;
  no_reader.read_specialization_memory = nullptr;
  {
    SrtWalker clean(plan, CleanRuntime(no_reader));
    Check(!SrtWalker(plan, no_reader, &clean).RefreshFlatBuffer(flat),
          "strict flat read accepted a missing strict reader");
  }
  Check(SrtWalker(plan, runtime).EvaluateDescriptor(0, descriptor) &&
            descriptor.dwords[0] == 0xeeeeu && reads.strict == 2 && reads.ordinary == 3,
        "direct descriptor evaluation without a clean evaluator changed read domains");
  reads.dirty = true;
  {
    SrtWalker clean(plan, CleanRuntime(runtime));
    Check(!SrtWalker(plan, runtime, &clean).RefreshFlatBuffer(flat) &&
              reads.strict == 3 && reads.ordinary == 3,
          "dirty strict pointer fell back to an ordinary read");
  }
}

void TestUniformVectorDescriptorRead() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  auto &handle = block.AppendNewInst(ValueOpcode::GetBufferResource,
      {Value(0x1000u), Value(4u << 16u), Value(1u), Value(0x16204u)});
  program.memory_info.push_back({.kind = ResourceKind::Buffer});
  auto &count = block.AppendNewInst(ValueOpcode::LoadBufferU32,
      {Value(&handle), Value(0u), Value(0u), Value(0u), Value(true)});
  count.SetFlags(MemoryFlags{.index = 0});
  // The captured indirect kernel shares one read across sibling scalar lane reads,
  // enclosed by a different EXEC mask. Its resource plan must share that read too.
  auto &lane = block.AppendNewInst(ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
  auto &active = block.AppendNewInst(ValueOpcode::ULessThan32, {Value(&lane), Value(32u)});
  auto &first = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&count), Value(true)});
  auto &next = block.AppendNewInst(ValueOpcode::IAdd32, {Value(&count), Value(1u)});
  auto &second = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&next), Value(true)});
  auto &sum = block.AppendNewInst(ValueOpcode::IAdd32, {Value(&first), Value(&second)});
  auto &small = block.AppendNewInst(ValueOpcode::ULessThanEqual32, {Value(&count), Value(72u)});
  auto &selected = block.AppendNewInst(ValueOpcode::SelectU32, {Value(&small), Value(&sum), Value(&count)});
  auto &masked = block.AppendNewInst(ValueOpcode::SelectU32, {Value(&active), Value(&selected), Value(0u)});
  auto &records = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&masked), Value(&active)});
  DescriptorSource source;
  source.dword_count = 4;
  source.dwords = {Value(0x2000u), Value(4u << 16u), Value(&records), Value(0x16204u)};
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .written = true});
  Check(ValidateRuntimeValue(program, Value(&count)), "uniform DWORD count was rejected");
  struct Reads { uint32_t value = 72; uint32_t strict = 0; uint32_t ordinary = 0; bool clean = true; } reads;
  const SrtRuntime runtime{
      .read_memory = [](void *data, uint64_t, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        words[0] = 999;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = [](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        if (!reads.clean || address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.value;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const bool written : {false, true}) {
    program.info.buffers[0].written = written;
    auto plan = ExtractResourcePlan(program);
    Check(plan.control_flow.empty() && plan.requires_specialization_memory &&
              plan.capture_specialization_reads,
          "vector descriptor read depended on incidental control-flow capture");
    reads = {};
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.buffers[0].dwords[2] == 145 && reads.strict == 1 && reads.ordinary == 0 &&
              snapshot.specialization_reads ==
                  std::vector<std::pair<uint64_t, uint64_t>>{{0x1000u, 4u}},
          "vector descriptor input was not read and captured exactly once");
    reads.clean = false;
    reads.strict = 0;
    Check(!MaterializeResources(plan, runtime, snapshot, specialization) &&
              reads.strict == 1 && reads.ordinary == 0,
          "dirty vector descriptor input fell back to an ordinary memory read");
  }
  count.SetArg(4, Value(false));
  auto &inactive = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&count), Value(false)});
  program.descriptor_sources.push_back({.dwords = {Value(&inactive)}, .dword_count = 1});
  reads.strict = 0;
  uint32_t result = 99;
  {
    const auto plan = ExtractResourcePlan(program);
    Check(SrtWalker(plan, runtime).Evaluate(plan.descriptor_sources.back().dwords[0], result) &&
              result == 0 && reads.strict == 0 && reads.ordinary == 0,
          "literal false EXEC read vector memory");
  }
  count.SetArg(4, Value(true));
  handle.SetArg(3, Value(0x204u));
  program.descriptor_sources.back().dwords[0] = Value(&count);
  {
    const auto plan = ExtractResourcePlan(program);
    Check(SrtWalker(plan, runtime).Evaluate(plan.descriptor_sources.back().dwords[0], result) &&
              result == 0 && reads.strict == 0 && reads.ordinary == 0,
          "invalid vector buffer format read memory");
  }
  handle.SetArg(3, Value(0x16204u));
  count.SetArg(1, Value(&lane));
  Check(!ValidateRuntimeValue(program, Value(&count)),
        "varying vector address was treated as a uniform descriptor read");
}

void TestExactReciprocalDescriptorArithmetic() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  auto &block = AddValueBlock(program);
  for (const float divisor : {64.f, 128.f, 256.f, 512.f,
                              std::numeric_limits<float>::min(),
                              std::bit_cast<float>(253u << 23u)}) {
    auto &reciprocal = block.AppendNewInst(ValueOpcode::FPRecipIFlag32, {Value::F32(divisor)});
    uint32_t result = 0;
    Check(SrtWalker(program, {}).Evaluate(Value(&reciprocal), result) &&
              result == std::bit_cast<uint32_t>(1.f / divisor),
          "power-of-two reciprocal was not exact");
  }
  for (const float divisor : {0.f, 3.f, -64.f, std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::denorm_min(),
                              std::bit_cast<float>(254u << 23u)}) {
    auto &reciprocal = block.AppendNewInst(ValueOpcode::FPRecipIFlag32, {Value::F32(divisor)});
    uint32_t result = 0;
    Check(!SrtWalker(program, {}).Evaluate(Value(&reciprocal), result),
          "unsupported reciprocal rounding or exceptional input was accepted");
  }
}

void TestUnbasedFlatCacheHitMaterializes() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UnbasedFlatPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, {}, snapshot, specialization),
        "unbased FLAT stage materialization failed");
  Check(snapshot.buffers.empty() && snapshot.images.empty(),
        "unbased FLAT plan produced unexpected descriptors");
}

void TestWrittenDescriptorPredicateReads(bool memory_condition) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.user_data_count = 1;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                     {Value(0x1000u), Value(0u)});
  auto &offset = block.AppendNewInst(ValueOpcode::GetUserData,
                                     {Value(static_cast<ScalarReg>(0))});
  auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(&offset), Value(0u), Value(false)});
  read.SetFlags(MemoryFlags{.index = 0});
  DescriptorSource source;
  source.dwords = {Value(&read), Value(0u), Value(4u), Value(0u)};
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .written = true});
  // Only memory-derived host decisions need clean, disjoint descriptor reads.
  auto &condition = block.AppendNewInst(ValueOpcode::IEqual32,
      {memory_condition ? Value(&read) : Value(&offset),
       Value(memory_condition ? 0x8000u : 4u)});
  auto &store_block = AddValueBlock(program);
  AddValueBlock(program);
  program.blocks[0]->condition = Value(&condition);
  program.blocks[0]->terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::ConditionalBranch;
  program.blocks[0]->terminator.true_block = program.blocks[1];
  program.blocks[0]->terminator.false_block = program.blocks[2];
  program.blocks[1]->id = 1;
  program.blocks[1]->terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
  program.blocks[2]->id = 2;
  program.blocks[2]->terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
  program.memory_info.push_back({.kind = ResourceKind::Buffer, .resource = 0});
  auto &output = store_block.AppendNewInst(ValueOpcode::GetBufferResource,
      {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]});
  store_block.AppendNewInst(ValueOpcode::StoreBufferU32,
      {Value(&output), Value(0u), Value(0u), Value(0u), Value(1u), Value(true)})
      .SetFlags(MemoryFlags{.index = 1});
  auto plan = ExtractResourcePlan(program);
  Check(plan.capture_specialization_reads == memory_condition && !plan.control_flow.empty(),
        "conditional writable descriptor used the wrong memory dependency policy");
  struct Reads { uint32_t ordinary = 0; uint32_t strict = 0; bool clean = false; } reads;
  const std::array<uint32_t, 1> user_data{4u};
  const SrtRuntime runtime{
      .user_data = user_data,
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        words[0] = 0x8000u;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        if (!reads.clean || address != 0x1004u) return false;
        words[0] = 0x8000u;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const bool clean : {false, true}) {
    reads = {.clean = clean};
    // A failed predicate explores both edges; its dirty read also rejects the writable descriptor.
    const bool expected = !memory_condition || clean;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) == expected &&
              reads.ordinary == (memory_condition ? 0u : 1u) &&
              reads.strict == (memory_condition ? 1u : 0u),
          "writable descriptor changed reader policy or bypassed a dirty memory predicate");
    if (!expected) continue;
    Check(snapshot.buffers[0].dwords[0] == 0x8000u &&
              snapshot.specialization_reads == (memory_condition
                  ? std::vector<std::pair<uint64_t, uint64_t>>{{0x1004u, 4u}}
                  : std::vector<std::pair<uint64_t, uint64_t>>{}),
          "writable descriptor lost its address or captured an immutable predicate");
  }
}

void TestFailedMaterializationRejectsStage() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UserDataBufferPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(!MaterializeResources(plan, {}, snapshot, specialization),
        "missing runtime user data did not reject the cached stage");
}

void TestSharedControlPredicates() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  constexpr uint32_t depth = 8;
  for (const auto operation : {ValueOpcode::LogicalAnd, ValueOpcode::LogicalOr}) {
    Program program;
    program.user_data_count = 1;
    program.srt_plan_complete = program.resource_tracking_complete = true;
    auto &block = AddValueBlock(program);
    const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
      return Value(&block.AppendNewInst(opcode, args));
    };
    const auto user = emit(ValueOpcode::GetUserData, {Value(ScalarReg(0))});
    const auto uniform = emit(ValueOpcode::INotEqual32, {user, Value(0u)});
    const auto lane = emit(ValueOpcode::GetBuiltin,
        {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
    auto varying = emit(ValueOpcode::INotEqual32, {lane, Value(0u)});
    // Both failed and partially uniform subgraphs share their operands exponentially.
    for (uint32_t i = 0; i < depth; ++i)
      varying = emit(ValueOpcode::LogicalOr, {varying, varying});
    auto condition = emit(operation, {varying, uniform});
    for (uint32_t i = 0; i < depth; ++i)
      condition = emit(ValueOpcode::LogicalOr, {condition, condition});
    condition = emit(ValueOpcode::LogicalNot, {condition});
    condition = emit(ValueOpcode::ConditionRef, {condition});
    for (uint32_t i = 0; i < 3; ++i) AddValueBlock(program);
    for (uint32_t i = 0; i < 2; ++i) {
      program.blocks[i]->condition = condition;
      program.blocks[i]->terminator.kind = TerminatorKind::ConditionalBranch;
      program.blocks[i]->terminator.true_block = program.blocks[i == 0 ? 1 : 3];
      program.blocks[i]->terminator.false_block = program.blocks[2];
    }
    auto plan = ExtractResourcePlan(program);
    Check(plan.value_storage.size() <= 2u * depth + 12u &&
              plan.control_flow.size() == 4 &&
              plan.control_flow[0].condition == plan.control_flow[1].condition &&
              !plan.capture_specialization_reads,
          "shared predicates expanded into trees or retained varying dependencies");
    for (const uint32_t flag : {0u, 1u}) {
      const std::array<uint32_t, 1> user_data{flag};
      SrtWalker walker(plan, {.user_data = user_data});
      uint32_t result = 99;
      const bool conjunction = operation == ValueOpcode::LogicalAnd;
      const bool known = conjunction ? flag == 0 : flag != 0;
      Check(walker.Evaluate(plan.control_flow[0].condition, result) == known &&
                (!known || result == uint32_t{conjunction}),
            "shared predicate changed its uniform factor or cached runtime values");
      std::vector<uint32_t> flat;
      Check(walker.RefreshFlatBuffer(flat) &&
                plan.visited_blocks == (known
                    ? (conjunction ? std::vector<uint8_t>{1, 1, 0, 1}
                                   : std::vector<uint8_t>{1, 0, 1, 0})
                    : std::vector<uint8_t>{1, 1, 1, 1}),
            "shared predicate skipped an unknown branch or followed a rejected branch");
    }
  }
}

void TestNestedExecPredicatesReuseReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  constexpr uint32_t depth = 12;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  for (uint32_t i = 0; i < 3; ++i) AddValueBlock(program);
  auto &entry = *program.blocks[0];
  auto &use = *program.blocks[1];
  auto &exit = *program.blocks[2];
  const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&entry.AppendNewInst(opcode, args));
  };
  const auto lane = emit(ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  const auto address = emit(ValueOpcode::GetAddressResource, {Value(0x1000u), Value(0u)});
  const auto count = emit(ValueOpcode::LoadAddressU32,
      {address, Value(0u), Value(0u), Value(true)});
  count.Instruction()->SetFlags(MemoryFlags{.index = 0});
  auto condition = emit(ValueOpcode::LogicalNot,
      {emit(ValueOpcode::ULessThanEqual32, {count, lane})});
  // A compact shared DAG must not become a tree of fresh EXEC evaluations.
  for (uint32_t i = 0; i < depth; ++i) {
    const auto mask = emit(ValueOpcode::INotEqual32, {lane, Value(i)});
    condition = emit(ValueOpcode::LogicalAnd,
        {mask, emit(ValueOpcode::LogicalOr, {condition, condition})});
  }
  entry.condition = condition;
  entry.terminator.kind = TerminatorKind::ConditionalBranch;
  entry.terminator.true_block = &use;
  entry.terminator.false_block = &exit;
  entry.AddBranch(&use);
  entry.AddBranch(&exit);
  use.terminator.kind = TerminatorKind::Branch;
  use.terminator.true_block = &exit;
  use.AddBranch(&exit);
  DescriptorSource source;
  source.dwords = {Value(0x8000u), Value(1u << 16u), Value(16u), Value(3u << 28u)};
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .read = true});
  program.memory_info.push_back({.kind = ResourceKind::Buffer, .resource = 0});
  const auto buffer = Value(&use.AppendNewInst(ValueOpcode::GetBufferResource,
      {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]}));
  use.AppendNewInst(ValueOpcode::LoadBufferU32,
      {buffer, Value(0u), Value(0u), Value(0u), Value(true)})
      .SetFlags(MemoryFlags{.index = 1});
  auto plan = ExtractResourcePlan(program);
  struct Reads { uint32_t value = 0; bool clean = true; uint32_t strict = 0, ordinary = 0; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t>) {
        ++static_cast<Reads *>(data)->ordinary;
        return false;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        Check(reads.strict <= depth + 1,
              "shared EXEC predicates repeated scalar reads beyond their distinct masks");
        if (!reads.clean || address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.value;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const auto state : {Reads{64u, true}, Reads{0u, true}, Reads{0u, false}, Reads{64u, true}}) {
    reads = state;
    Check(MaterializeResources(plan, runtime, snapshot, specialization),
          "nested EXEC predicates rejected a valid resource snapshot");
    const bool inactive = reads.clean && reads.value == 0;
    Check(snapshot.buffers[0].dwords[0] == (inactive ? 0u : 0x8000u),
          "nested EXEC predicates lost an unknown branch or retained stale runtime facts");
    Check(reads.strict > 0 && reads.ordinary == 0 &&
              snapshot.specialization_reads.size() <= depth + 1,
          "shared EXEC predicates repeated scalar reads beyond their distinct masks");
  }
}

void TestFiniteImageRefreshReusesScalarReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                     {Value(0x1000u), Value(0u)});
  auto &srt = block.AppendNewInst(ValueOpcode::GetSrtResource);
  for (uint32_t index = 0; index < 3; ++index) {
    auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(index * 4u), Value(0u), Value(true)});
    read.SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back({Value(&read), index});
    auto &flat = block.AppendNewInst(ValueOpcode::ReadConst,
                                     {Value(&srt), Value(index)});
    DescriptorSource source;
    source.dword_count = 8;
    source.dwords.fill(Value(0u));
    source.dwords[0] = Value(&flat);
    source.dwords[1] = Value(static_cast<uint32_t>(
        Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float) << 20u);
    source.dwords[3] = Value(Libs::Graphics::DstSel(4, 5, 6, 7) |
        (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D) << 28u));
    program.descriptor_sources.push_back(source);
  }
  DescriptorSource root;
  root.dword_count = 8;
  root.dwords.fill(Value(0u));
  root.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1, 2, 1};
  program.descriptor_sources.push_back(root);
  program.info.images.push_back({
      .source = 3,
      .resource_class = ImageResourceClass::Sampled,
      .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
      .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  auto plan = ExtractResourcePlan(program);
  struct Reads {
    std::array<uint32_t, 3> words{0x100u, 0x200u, 0x200u};
    std::array<uint32_t, 3> counts{};
    uint32_t ordinary = 0;
  } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t>) {
        ++static_cast<Reads *>(data)->ordinary;
        return false;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address,
                                        std::span<uint32_t> words) {
        if (words.size() != 1 || address < 0x1000u || address >= 0x100cu ||
            (address & 3u) != 0) return false;
        auto &reads = *static_cast<Reads *>(data);
        const auto index = (address - 0x1000u) / 4u;
        ++reads.counts[index];
        words[0] = reads.words[index];
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  const auto capacities = [&] {
    return std::array{snapshot.images.capacity(), snapshot.flattened_srt.capacity(),
                      snapshot.specialization_reads.capacity(), specialization.images.capacity()};
  };
  const auto check_mapping = [&](std::array<uint32_t, 4> ordinals) {
    const auto offset = specialization.images[0].indirect_mapping_offset;
    Check(snapshot.images.size() == 2 && snapshot.flattened_srt[offset] == 4,
          "finite image candidates were not deduplicated");
    for (uint32_t key = 0; key < ordinals.size(); ++key) {
      Check(snapshot.flattened_srt[offset + 1u + key * 2u] == key &&
                snapshot.flattened_srt[offset + 2u + key * 2u] == ordinals[key],
            "finite image selector mapping is stale or incorrect");
    }
  };
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "finite image materialization failed");
  Check(reads.ordinary == 0 && reads.counts == std::array<uint32_t, 3>{1, 1, 1} &&
            snapshot.specialization_reads.size() == 3,
        "finite image candidates repeated scalar reads or bypassed clean provenance");
  check_mapping({0, 1, 1, 1});
  const auto warm_capacities = capacities();
  reads.words = {0x300u, 0x300u, 0x400u};
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "finite image refresh failed after descriptor changes");
  Check(reads.ordinary == 0 && reads.counts == std::array<uint32_t, 3>{2, 2, 2} &&
            snapshot.specialization_reads.size() == 3 &&
            snapshot.images[0].dwords[0] == 0x300u && snapshot.images[1].dwords[0] == 0x400u,
        "finite image refresh retained old scalar values or repeated reads");
  check_mapping({0, 0, 1, 0});
  Check(capacities() == warm_capacities,
        "finite image refresh grew reusable resource storage after warmup");
}

void TestEmptyTraversalSkipsUnmappedDescriptor(bool independent_mask, bool counted_loop = false) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  for (uint32_t i = 0; i < 6; ++i) AddValueBlock(program);
  auto &entry = *program.blocks[0];
  auto &loop = *program.blocks[1];
  auto &body = *program.blocks[2];
  auto &merge = *program.blocks[3];
  auto &use = *program.blocks[4];
  auto &exit = *program.blocks[5];
  const auto emit = [](Block &block, ValueOpcode opcode, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(opcode, args));
  };
  const auto branch = [](Block &block, Block &target, Value condition = {}, Block *other = nullptr) {
    block.terminator.kind = other ? TerminatorKind::ConditionalBranch : TerminatorKind::Branch;
    block.terminator.true_block = &target;
    block.terminator.false_block = other;
    block.condition = condition;
    block.AddBranch(&target);
    if (other) block.AddBranch(other);
  };
  Value lane, active, initial;
  if (!counted_loop) {
    lane = emit(entry, ValueOpcode::GetBuiltin,
        {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
    active = emit(entry, ValueOpcode::INotEqual32, {lane, Value(0u)});
    const auto initial_mask = independent_mask
        ? emit(entry, ValueOpcode::IEqual32, {lane, Value(0u)}) : active;
    initial = emit(entry, ValueOpcode::SelectU32,
        {initial_mask, Value(UINT32_MAX), Value(0u)});
  }
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
  const auto address = emit(entry, ValueOpcode::GetAddressResource, {Value(0x1000u), Value(0u)});
  const auto count = emit(entry, ValueOpcode::LoadAddressU32,
      {address, Value(0u), Value(0u), Value(true)});
  count.Instruction()->SetFlags(MemoryFlags{.index = 0});
  if (counted_loop) {
    branch(entry, loop);
  } else {
    const auto rejected = emit(entry, ValueOpcode::ULessThanEqual32, {count, lane});
    const auto accepted = emit(entry, ValueOpcode::LogicalNot, {rejected});
    branch(entry, loop, emit(entry, ValueOpcode::LogicalAnd, {active, accepted}), &merge);
  }

  // A nonempty traversal contains a varying loop-carried value. Its cycle must
  // remain unknown; only the proven empty traversal can select the bypass arm.
  auto &carried = loop.AppendNewInst(ValueOpcode::Phi);
  carried.SetFlags(Type::U32);
  const auto next = emit(body, ValueOpcode::IAdd32, {Value(&carried), Value(1u)});
  carried.AddPhiOperand(&entry, counted_loop ? Value(0u) : lane);
  carried.AddPhiOperand(&body, next);
  const auto condition = counted_loop
      ? emit(loop, ValueOpcode::ULessThan32, {Value(&carried), count})
      : emit(loop, ValueOpcode::INotEqual32, {Value(&carried), Value(UINT32_MAX)});
  branch(loop, counted_loop ? use : body, condition, &merge);
  loop.terminator.loop_header = true;
  loop.terminator.continue_block = &body;
  loop.terminator.merge_block = &merge;
  branch(body, loop);
  if (counted_loop) {
    branch(merge, exit);
    branch(use, body);
  } else {
    auto &selected = merge.AppendNewInst(ValueOpcode::Phi);
    selected.SetFlags(Type::U32);
    selected.AddPhiOperand(&entry, initial);
    selected.AddPhiOperand(&loop, Value(&carried));
    const auto valid = emit(merge, ValueOpcode::INotEqual32, {Value(&selected), Value(UINT32_MAX)});
    branch(merge, use, emit(merge, ValueOpcode::LogicalAnd, {active, valid}), &exit);
    branch(use, exit);
  }
  DescriptorSource source;
  const auto descriptor_address = emit(use, ValueOpcode::GetAddressResource,
      {Value(0x2000u), Value(0u)});
  for (uint32_t i = 0; i < 4; ++i) {
    const auto index = static_cast<uint32_t>(program.memory_info.size());
    program.memory_info.push_back({.kind = ResourceKind::ScalarAddress, .planning_only = true});
    source.dwords[i] = emit(use, ValueOpcode::LoadAddressU32,
        {descriptor_address, Value(i * 4u), Value(0u), Value(true)});
    source.dwords[i].Instruction()->SetFlags(MemoryFlags{.index = index});
  }
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .read = true});
  const auto buffer_index = static_cast<uint32_t>(program.memory_info.size());
  program.memory_info.push_back({.kind = ResourceKind::Buffer, .resource = 0});
  const auto buffer = emit(use, ValueOpcode::GetBufferResource,
      {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]});
  emit(use, ValueOpcode::LoadBufferU32, {buffer, Value(0u), Value(0u), Value(0u), Value(true)})
      .Instruction()->SetFlags(MemoryFlags{.index = buffer_index});
  auto plan = ExtractResourcePlan(program);
  struct Reads { uint32_t count = 0; bool clean = true; uint32_t descriptors = 0; } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        constexpr std::array<uint32_t, 4> descriptor{0x8000u, 1u << 16u, 16u,
                                                   (3u << 28u) | 0x4dfacu};
        if (address < 0x2000u || address >= 0x2010u || address % 4u || words.size() != 1)
          return false;
        ++static_cast<Reads *>(data)->descriptors;
        words[0] = descriptor[(address - 0x2000u) / 4u];
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        const auto &reads = *static_cast<Reads *>(data);
        if (!reads.clean || address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.count;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const auto state : {Reads{0u, true}, Reads{4u, true}, Reads{0u, false}, Reads{0u, true}}) {
    reads = state;
    if (!counted_loop) {
      uint32_t condition = 99;
      Check(!SrtWalker(plan, runtime).Evaluate(plan.control_flow[3].condition, condition),
            "direct merge evaluation used missing or previous runtime reachability");
    }
    Check(MaterializeResources(plan, runtime, snapshot, specialization),
          "partial traversal proof rejected a valid resource snapshot");
    const bool inactive = !independent_mask && reads.clean && reads.count == 0u;
    Check(snapshot.buffers[0].dwords[0] == (inactive ? 0u : 0x8000u),
          "empty traversal kept its descriptor or an uncertain traversal lost it");
    Check(specialization.buffers[0].zero_stride_oob == inactive,
          "inactive traversal did not expose dead vector reads to specialization");
    Check(reads.descriptors == (inactive ? 0u : 4u),
          "inactive traversal read its descriptor or an uncertain traversal skipped it");
    if (inactive) {
      Check(!snapshot.specialization_reads.empty(), "empty traversal lost its clean scalar read");
      for (const auto &read : snapshot.specialization_reads)
        Check(read == std::pair<uint64_t, uint64_t>{0x1000u, 4u},
              "empty traversal proof captured an unrelated memory range");
    }
  }
}

void TestCyclicControlPredicatesRemainUnknown() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  for (uint32_t i = 0; i < 4; ++i) AddValueBlock(program);
  auto &entry = *program.blocks[0];
  auto &left = *program.blocks[1];
  auto &right = *program.blocks[2];
  auto &exit = *program.blocks[3];
  const auto lane = Value(&entry.AppendNewInst(ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)}));
  const auto u = Value(&entry.AppendNewInst(ValueOpcode::INotEqual32, {lane, Value(0u)}));
  const auto v = Value(&entry.AppendNewInst(ValueOpcode::INotEqual32, {lane, Value(1u)}));
  auto &p = left.AppendNewInst(ValueOpcode::Phi);
  auto &q = right.AppendNewInst(ValueOpcode::Phi);
  p.SetFlags(Type::U1);
  q.SetFlags(Type::U1);
  const auto a = Value(&left.AppendNewInst(ValueOpcode::LogicalAnd, {u, Value(&p)}));
  const auto b = Value(&right.AppendNewInst(ValueOpcode::LogicalAnd, {v, Value(&q)}));
  p.AddPhiOperand(&entry, Value(true));
  p.AddPhiOperand(&right, b);
  q.AddPhiOperand(&entry, Value(true));
  q.AddPhiOperand(&left, a);
  const auto branch = [](Block &block, Value condition, Block &yes, Block &no) {
    block.condition = condition;
    block.terminator.kind = TerminatorKind::ConditionalBranch;
    block.terminator.true_block = &yes;
    block.terminator.false_block = &no;
    block.AddBranch(&yes);
    block.AddBranch(&no);
  };
  // Two loop entries have no single header whose Phi can be discarded up front.
  branch(entry, u, left, right);
  branch(left, a, right, exit);
  branch(right, b, left, exit);
  auto plan = ExtractResourcePlan(program);
  SrtWalker walker(plan, {});
  uint32_t value = 99;
  Check(!walker.Evaluate(plan.control_flow[1].condition, value) &&
            !walker.Evaluate(plan.control_flow[2].condition, value),
        "cyclic varying control acquired an unsupported uniform value");
  std::vector<uint32_t> flat;
  Check(walker.RefreshFlatBuffer(flat) && plan.visited_blocks == std::vector<uint8_t>({1, 1, 1, 1}),
        "cyclic control lost a possible branch during reachability");
}

void TestEmptyScalarBufferSpecialization() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  struct Fixture {
    Program program;
    std::vector<std::pair<Inst *, Type>> uses;
    Inst *nonnull = nullptr;
  };
  const auto make_fixture = [] {
    Fixture fixture;
    auto &program = fixture.program;
    program.stage = Libs::Graphics::ShaderType::Compute;
    program.srt_plan_complete = program.resource_tracking_complete = true;
    program.user_data_count = 4u;
    auto &block = AddValueBlock(program);
    DescriptorSource source;
    source.dword_count = 4u;
    for (uint32_t word = 0; word < 4u; ++word)
      source.dwords[word] = Value(&block.AppendNewInst(ValueOpcode::GetUserData,
          {Value(static_cast<ScalarReg>(word))}));
    program.descriptor_sources.push_back(source);
    program.info.buffers.push_back({.source = 0u, .read = true, .scalar = true});
    const auto table = Value(&block.AppendNewInst(ValueOpcode::GetBufferResource,
        {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]}));
    std::array<Value, 4> words;
    for (uint32_t word = 0; word < 4u; ++word) {
      program.memory_info.push_back({.kind = ResourceKind::ScalarBuffer, .resource = 0u,
                                     .offset = word * 4u});
      auto &read = block.AppendNewInst(ValueOpcode::ReadConstBuffer, {table, Value(0u)});
      read.SetFlags(MemoryFlags{.index = word});
      words[word] = Value(&read);
    }
    const auto handle = Value(&block.AppendNewInst(ValueOpcode::GetBufferResource,
        {words[0], words[1], words[2], words[3]}));
    for (const auto opcode : {ValueOpcode::ReadConstBuffer, ValueOpcode::LoadBufferU8,
                              ValueOpcode::LoadBufferU16, ValueOpcode::LoadBufferU32,
                              ValueOpcode::LoadBufferU32x2, ValueOpcode::LoadBufferU32x3,
                              ValueOpcode::LoadBufferU32x4}) {
      const auto index = static_cast<uint32_t>(program.memory_info.size());
      program.memory_info.push_back({.kind = ResourceKind::IndirectBuffer, .resource = UINT32_MAX,
          .data_dwords = BufferComponentCount(opcode),
          .data_bits = opcode == ValueOpcode::LoadBufferU8 ? 8u : opcode == ValueOpcode::LoadBufferU16 ? 16u : 32u,
          .formatted = opcode == ValueOpcode::LoadBufferU32});
      auto &read = opcode == ValueOpcode::ReadConstBuffer
          ? block.AppendNewInst(opcode, {handle, Value(0u)})
          : block.AppendNewInst(opcode, {handle, Value(0u), Value(0u), Value(0u), Value(true)});
      read.SetFlags(MemoryFlags{.index = index});
      fixture.uses.emplace_back(&block.AppendNewInst(ValueOpcode::Identity, {Value(&read)}), read.GetType());
    }
    const auto nonnull = Value(&block.AppendNewInst(ValueOpcode::GetBufferResource,
        {words[0], words[1], words[2], Value(1u)}));
    auto &read = block.AppendNewInst(ValueOpcode::ReadConstBuffer, {nonnull, Value(0u)});
    read.SetFlags(MemoryFlags{.index = 4u});
    fixture.nonnull = &block.AppendNewInst(ValueOpcode::Identity, {Value(&read)});
    return fixture;
  };
  auto fixture = make_fixture();
  auto plan = ExtractResourcePlan(fixture.program);
  std::array<uint32_t, 4> descriptor{0x1000u, 8u << 16u, 0u, 3u << 28u};
  uint32_t reads = 0u;
  const SrtRuntime runtime{.user_data = descriptor, .read_memory = RejectSpecializationRead,
                           .userdata = &reads, .read_specialization_memory = RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization, empty, populated;
  for (const uint32_t records : {0u, 1u, 0x20000000u, 0u}) {
    descriptor[2] = records;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) && reads == 0u &&
              specialization.buffers[0].scalar_empty == (records == 0u),
          "scalar emptiness read table contents or truncated its 64-bit byte size");
    if (records == 0u) {
      if (empty.buffers.empty()) empty = specialization;
      else Check(specialization == empty, "returning to an empty scalar table changed its specialization");
    }
    else if (records == 1u) populated = specialization;
    else Check(specialization == populated, "nonempty scalar size created a shader permutation");
    auto current = make_fixture();
    ApplyResourceSpecialization(current.program, specialization);
    for (const auto &[use, type] : current.uses) {
      const auto value = use->Arg(0).Resolve();
      Check(value.GetType() == type, "empty scalar-table folding changed the load result type");
      if (records != 0u) {
        Check(value.TryInstruction() != nullptr && BufferAccessOf(value.Instruction()->GetOpcode()) == BufferAccess::Read,
              "nonempty scalar table lost a runtime buffer load");
      } else if (value.IsImmediate()) {
        Check(value == Value(uint8_t{0}) || value == Value(uint16_t{0}) || value == Value(0u),
              "empty scalar table did not produce a typed zero");
      } else {
        for (size_t component = 0; component < value.Instruction()->NumArgs(); ++component)
          Check(value.Instruction()->Arg(component) == Value(0u),
                "empty scalar table retained a nonzero vector component");
      }
    }
    Check(current.nonnull->Arg(0).Resolve().TryInstruction() != nullptr,
          "a partly nonzero indirect descriptor was treated as unbound");
  }
  Check(empty != populated, "empty-to-populated scalar transition reused the folded specialization");
  plan.info.buffers[0].scalar = false;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            !specialization.buffers[0].scalar_empty,
        "vector-only resource acquired scalar-empty specialization");
  empty.buffers[0].indirect_root = 0u;
  empty.buffers.push_back(empty.buffers[0]);
  empty.buffers[1].scalar_empty = false;
  ApplyResourceSpecialization(fixture.program, empty);
  for (const auto &[use, type] : fixture.uses)
    Check(use->Arg(0).GetType() == type && use->Arg(0).Resolve().TryInstruction() != nullptr &&
              BufferAccessOf(use->Arg(0).ResolveInstruction()->GetOpcode()) == BufferAccess::Read,
          "empty root folded a scalar read with nonempty candidates");
}

void TestMixedSamplerVariantsShareRuntimeDescriptor() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto program = MixedSamplerProgram();
  auto plan = ExtractResourcePlan(program);
  std::array<uint32_t, 2> user_data{0x11111111u, 0x22222222u};
  const SrtRuntime runtime{.user_data = user_data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mixed sampler materialization failed");
  ApplyResourceSpecialization(program, specialization);
  const auto &samplers = program.info.samplers;
  Check(snapshot.samplers.size() == 2 && samplers.size() == 3 &&
            samplers[0].snapshot_index == 0 && samplers[1].snapshot_index == 1 &&
            samplers[2].snapshot_index == 1 &&
            samplers[0].source == plan.info.samplers[0].source &&
            samplers[1].source == plan.info.samplers[1].source &&
            samplers[2].source == samplers[1].source &&
            !samplers[1].force_point_filtering && samplers[2].force_point_filtering &&
            program.info.sampled_pairs[2].sampler == 2,
        "native sampler variants lost their source identity or binding order");
  const auto capacity = snapshot.samplers.capacity();
  user_data[1] = 0x33333333u;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            snapshot.samplers.size() == 2 && snapshot.samplers.capacity() == capacity &&
            snapshot.samplers[samplers[0].snapshot_index].dwords[0] == user_data[0] &&
            snapshot.samplers[samplers[1].snapshot_index].dwords[0] == user_data[1] &&
            snapshot.samplers[samplers[2].snapshot_index].dwords[0] == user_data[1],
        "sampler variants retained stale or duplicated descriptors after refresh");
}

// A T# with format 16_16_16_16 USCALED/SSCALED used to be a fatal "unsupported format"; it now
// materializes as a float image converted from the UNorm/SNorm backing. Formats that were
// already supported keep their class and get no conversion.
void TestScaled16SampledFormats() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using F = Libs::Graphics::Prospero::BufferFormat;
  using N = Libs::Graphics::Prospero::TextureNumericClass;
  struct Case {
    F format;
    N numeric_class;
    bool converted;
  };
  for (const auto &[format, numeric_class, converted] :
       {Case{F::k16_16_16_16UScaled, N::Float, true}, Case{F::k16_16_16_16SScaled, N::Float, true},
        Case{F::k16_16_16_16UNorm, N::Float, false}, Case{F::k16_16_16_16Float, N::Float, false},
        Case{F::k16_16_16_16UInt, N::Uint, false}, Case{F::k8UScaled, N::Float, true}}) {
    auto program = MixedSamplerProgram();
    auto &first = program.descriptor_sources[0];
    first.dwords[0] = Value(0x100u);
    first.dwords[1] = Value(static_cast<uint32_t>(format) << 20u);
    first.dwords[3] = program.descriptor_sources[1].dwords[3];
    auto plan = ExtractResourcePlan(program);
    std::array<uint32_t, 2> user_data{0x11111111u, 0x22222222u};
    const SrtRuntime runtime{.user_data = user_data};
    ResourceSnapshot snapshot;
    ResourceSpecialization specialization;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              specialization.images[0].numeric_class == numeric_class &&
              (specialization.images[0].conversion_format != F::kInvalid) == converted &&
              (!converted || specialization.images[0].conversion_format == format),
          "16_16_16_16 scaled sampled image changed its materialization");
  }
}

void TestSignedStorageSpecialization() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using F = Libs::Graphics::Prospero::BufferFormat;
  using N = Libs::Graphics::Prospero::TextureNumericClass;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  program.user_data_count = 1u;
  auto &block = AddValueBlock(program);
  DescriptorSource source;
  source.dword_count = 8u;
  source.dwords.fill(Value(0u));
  source.dwords[0] = Value(0x100u);
  source.dwords[1] = Value(&block.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(0))}));
  constexpr auto swizzle = Libs::Graphics::DstSel(5, 4, 1, 0);
  source.dwords[3] = Value(swizzle |
      (uint32_t(Libs::Graphics::Prospero::ImageType::kColor2D) << 28u));
  program.descriptor_sources.push_back(source);
  program.info.images.push_back({.source = 0u, .resource_class = ImageResourceClass::Storage,
      .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D, .written = true});
  auto plan = ExtractResourcePlan(program);
  std::array<uint32_t, 1> data{};
  const SrtRuntime runtime{.user_data = data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization, signed16;
  for (const auto format : {F::k16SInt, F::k32SInt, F::k16SInt}) {
    data[0] = uint32_t(format) << 20u;
    Check(MaterializeResources(plan, runtime, snapshot, specialization),
          "write-only signed storage image failed materialization");
    const auto conversion = format == F::k16SInt ? format : F::kInvalid;
    const auto view = format == F::k16SInt ? F::k16UInt : F::k32UInt;
    Check(specialization.images[0].numeric_class == N::Uint &&
              specialization.images[0].conversion_format == conversion &&
              specialization.images[0].shader_swizzle == swizzle &&
              snapshot.images[0].dwords[1] == data[0] &&
              RawSignedStorageFormat(plan.info.images[0], format) == view,
          "signed storage specialization lost guest format, conversion, swizzle or raw view width");
    if (format == F::k16SInt) {
      if (signed16.images.empty()) signed16 = specialization;
      else Check(specialization == signed16, "signed storage format refresh retained a stale specialization");
    } else {
      Check(specialization != signed16, "signed 16/32-bit stores shared incompatible conversion state");
    }
    ApplyResourceSpecialization(program, specialization);
    const auto binding = DescriptorBindingForImage(program.info.images[0]);
    Check(binding && ImageBindingResourceClass(*binding) == ImageResourceClass::Storage,
          "signed storage conversion did not reuse the unsigned storage binding");
  }
  for (const auto format : {F::k8SInt, F::k16SInt, F::k32SInt}) {
    data[0] = uint32_t(format) << 20u;
    auto &image = plan.info.images[0];
    image.read = true;
    Check(RawSignedStorageFormat(image, format) == F::kInvalid &&
              !MaterializeResources(plan, runtime, snapshot, specialization),
          "signed raw storage view accepted a non-atomic read/write image");
    image.atomic = true;
    Check((RawSignedStorageFormat(image, format) != F::kInvalid) == (format == F::k32SInt) &&
              MaterializeResources(plan, runtime, snapshot, specialization) == (format == F::k32SInt),
          "signed storage conversion changed atomic format eligibility");
    image.atomic64 = true;
    Check(RawSignedStorageFormat(image, format) == F::kInvalid &&
              !MaterializeResources(plan, runtime, snapshot, specialization),
          "signed 16/32-bit image acquired a 64-bit atomic raw view");
    image.resource_class = ImageResourceClass::Sampled;
    image.written = image.atomic = image.atomic64 = false;
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              specialization.images[0].numeric_class == N::Sint &&
              specialization.images[0].conversion_format == F::kInvalid &&
              Libs::Graphics::Prospero::NumBytesPerElement(format) ==
                  (format == F::k8SInt ? 1u : format == F::k16SInt ? 2u : 4u) &&
              RawSignedStorageFormat(image, format) == F::kInvalid,
          "storage conversion changed ordinary signed image reads");
    image.resource_class = ImageResourceClass::Storage;
    image.written = true;
    image.read = false;
  }
}

void TestIndirectImageSamplerClasses(bool mixed) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  using Libs::Graphics::Prospero::BufferFormat;
  using Libs::Graphics::Prospero::TextureNumericClass;
  auto program = MixedSamplerProgram();
  auto &first = program.descriptor_sources[0];
  first.dwords[0] = Value(0x100u);
  first.dwords[1] = Value(static_cast<uint32_t>(BufferFormat::k8UNorm) << 20u);
  first.dwords[3] = program.descriptor_sources[1].dwords[3];
  program.descriptor_sources[1].dwords[1] = Value(static_cast<uint32_t>(
      mixed ? BufferFormat::k8UInt : BufferFormat::k8UNorm) << 20u);
  DescriptorSource root;
  root.dword_count = 8;
  root.dwords.fill(Value(0u));
  root.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1, 0};
  program.info.images.resize(1);
  program.info.images[0].source = static_cast<uint32_t>(program.descriptor_sources.size());
  program.descriptor_sources.push_back(root);
  program.info.samplers.resize(1);
  program.info.sampled_pairs.resize(1);
  auto plan = ExtractResourcePlan(program);
  std::array<uint32_t, 1> user_data{0x11111111u};
  uint32_t reads = 0;
  const SrtRuntime runtime{.user_data = user_data, .userdata = &reads,
                           .read_specialization_memory = RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) && reads == 0 &&
            snapshot.images.size() == 2 && snapshot.images[0].dwords[0] == 0x100u &&
            snapshot.images[1].dwords[0] == 1u &&
            specialization.images[0].numeric_class == TextureNumericClass::Float &&
            specialization.images[1].numeric_class ==
                (mixed ? TextureNumericClass::Uint : TextureNumericClass::Float),
        "indirect image materialization rejected or changed a typed candidate");
  const auto offset = specialization.images[0].indirect_mapping_offset;
  Check(snapshot.flattened_srt[offset] == 3u &&
            snapshot.flattened_srt[offset + 2u] == 0u &&
            snapshot.flattened_srt[offset + 4u] == 1u &&
            snapshot.flattened_srt[offset + 6u] == 0u,
        "mixed image classes changed selector ordinals or duplicated candidates");
  ApplyResourceSpecialization(program, specialization);
  Check(program.info.images[0].indirect_resources == std::vector<uint32_t>({0, 1}) &&
            snapshot.samplers.size() == 1 && program.info.samplers.size() == (mixed ? 2u : 1u),
        "indirect images lost candidates or created unnecessary sampler variants");
  user_data[0] = 0x33333333u;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mixed image sampler refresh failed");
  for (uint32_t index = 0; index < program.info.samplers.size(); ++index) {
    const auto &sampler = program.info.samplers[index];
    Check(sampler.snapshot_index == 0u && sampler.source == plan.info.samplers[0].source &&
              sampler.force_point_filtering == (index != 0u) &&
              sampler.integer_border == (index != 0u) &&
              snapshot.samplers[sampler.snapshot_index].dwords[0] == user_data[0],
          "typed sampler variants lost filtering, border class, or shared live descriptor");
  }
}

// Compiled-walker equivalence: random value graphs are materialized through the compiled plan
// and through SrtWalker; results, snapshots and the order of guest reads must be identical.

struct SplitMix {
  uint64_t state;

  uint64_t Next() {
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31u);
  }
  uint32_t Below(uint32_t count) {
    return static_cast<uint32_t>(Next() % count);
  }
  bool Chance(uint32_t percent) { return Below(100) < percent; }
};

constexpr uint64_t FuzzMemoryBase = 0x40000;
constexpr uint32_t FuzzMemoryWords = 1024;

uint32_t FuzzPointer(uint32_t index) {
  return static_cast<uint32_t>(FuzzMemoryBase) + (index % (FuzzMemoryWords - 64u)) * 4u;
}

struct FuzzRead {
  uint64_t address;
  size_t bytes;
  bool strict;
  bool operator==(const FuzzRead &) const = default;
};

struct FuzzMemory {
  std::vector<uint32_t> words = std::vector<uint32_t>(FuzzMemoryWords);
  uint32_t strict_salt = 0;
  uint32_t strict_modulus = 7; // Zero: strict reads never fail.
  std::vector<FuzzRead> *log = nullptr;
};

bool ReadFuzzMemory(FuzzMemory &memory, uint64_t address,
                    std::span<uint32_t> values, bool strict) {
  memory.log->push_back({address, values.size_bytes(), strict});
  if (address < FuzzMemoryBase || (address & 3u) != 0u ||
      address - FuzzMemoryBase >
          FuzzMemoryWords * sizeof(uint32_t) - values.size_bytes()) {
    return false;
  }
  // Strict reads reject some GPU-dirty words.
  if (strict && memory.strict_modulus != 0u &&
      ((address >> 2u) * 2654435761u + memory.strict_salt) % memory.strict_modulus == 0u) {
    return false;
  }
  std::copy_n(memory.words.begin() + (address - FuzzMemoryBase) / 4u,
              values.size(), values.begin());
  return true;
}

bool ReadFuzzOrdinary(void *userdata, uint64_t address,
                      std::span<uint32_t> values) {
  return ReadFuzzMemory(*static_cast<FuzzMemory *>(userdata), address, values,
                        false);
}

bool ReadFuzzStrict(void *userdata, uint64_t address,
                    std::span<uint32_t> values) {
  return ReadFuzzMemory(*static_cast<FuzzMemory *>(userdata), address, values,
                        true);
}

bool SameSnapshot(const Libs::Graphics::ShaderRecompiler::IR::ResourceSnapshot &a,
                  const Libs::Graphics::ShaderRecompiler::IR::ResourceSnapshot &b) {
  if (a.buffers != b.buffers || a.images != b.images ||
      a.samplers != b.samplers || a.flattened_srt != b.flattened_srt ||
      a.user_data != b.user_data || !(a.uniform_fill == b.uniform_fill) ||
      a.specialization_reads != b.specialization_reads ||
      a.specialization_read_slots != b.specialization_read_slots ||
      a.image_tables.size() != b.image_tables.size()) {
    return false;
  }
  for (size_t i = 0; i < a.image_tables.size(); i++) {
    const auto &x = a.image_tables[i];
    const auto &y = b.image_tables[i];
    if (x.base != y.base || x.size != y.size || x.offset != y.offset ||
        x.entry_mask != y.entry_mask || x.raw != y.raw || x.slots != y.slots ||
        x.mapping != y.mapping) {
      return false;
    }
  }
  return true;
}

class FuzzPlanBuilder {
public:
  explicit FuzzPlanBuilder(uint64_t seed) : m_random{seed}, m_safe(m_random.Chance(55)) {
    using namespace Libs::Graphics::ShaderRecompiler::IR;
    m_program.stage = Libs::Graphics::ShaderType::Compute;
    m_program.srt_plan_complete = true;
    m_program.resource_tracking_complete = true;
    m_block = &AddValueBlock(m_program);
  }

  Libs::Graphics::ShaderRecompiler::IR::ResourcePlan Build() {
    using namespace Libs::Graphics::ShaderRecompiler::IR;
    for (uint32_t reg = 0; reg < 12; reg++) {
      m_u32.push_back(Emit(ValueOpcode::GetUserData,
                           {Value(static_cast<ScalarReg>(reg))}));
      if (reg < 6) {
        m_ptr.push_back(m_u32.back());
      }
    }
    m_ptr.push_back(Value(FuzzPointer(m_random.Below(FuzzMemoryWords))));
    if (!m_safe && m_random.Chance(30)) {
      // Beyond the runtime user data of some refreshes.
      m_u32.push_back(Emit(ValueOpcode::GetUserData,
                           {Value(static_cast<ScalarReg>(20))}));
    }
    for (uint32_t i = 0; i < 6; i++) {
      m_u32.push_back(Value(RandomImmediate()));
    }
    m_u1.push_back(Value(true));
    m_u1.push_back(Value(false));
    m_u64.push_back(Emit(ValueOpcode::GetShaderBase, {}));
    const auto steps = 30 + m_random.Below(90);
    for (uint32_t step = 0; step < steps; step++) {
      Step();
      if (m_random.Chance(12)) {
        // Flattened SRT slots hold scalar loads; ReadConst aliases them.
        const auto slot = static_cast<uint32_t>(m_program.srt_reads.size());
        m_program.srt_reads.push_back({RawRead(), slot});
      }
    }
    for (uint32_t i = 0; i < 3; i++) {
      m_program.srt_reads.push_back(
          {RawRead(), static_cast<uint32_t>(m_program.srt_reads.size())});
    }
    AddResources();
    return ExtractResourcePlan(m_program);
  }

  // Values of the extra single-dword sources: [0] control-flow conditions, [1..4] fill words.
  // Safe plans only read valid pointers and avoid failing operations.
  bool Safe() const { return m_safe; }
  uint32_t ConditionSource() const { return m_condition_source; }
  uint32_t FillSource() const { return m_fill_source; }
  uint32_t BufferSource(uint32_t index) const { return m_buffer_sources[index]; }
  uint32_t ImageSource(uint32_t index) const { return m_image_sources[index]; }

private:
  using Value = Libs::Graphics::ShaderRecompiler::IR::Value;
  using ValueOpcode = Libs::Graphics::ShaderRecompiler::IR::ValueOpcode;

  Value Emit(ValueOpcode opcode, std::initializer_list<Value> args,
             uint64_t flags = 0) {
    return Value(&m_block->AppendNewInst(opcode, args, flags));
  }

  uint32_t RandomImmediate() {
    switch (m_random.Below(5)) {
    case 0: return m_random.Below(40);
    case 1: return FuzzPointer(m_random.Below(FuzzMemoryWords));
    case 2: return 0xffffffffu - m_random.Below(4);
    case 3: return std::bit_cast<uint32_t>(static_cast<float>(m_random.Below(1000)) * 0.25f);
    default: return static_cast<uint32_t>(m_random.Next());
    }
  }

  Value Pick(const std::vector<Value> &pool) {
    return pool[m_random.Below(static_cast<uint32_t>(pool.size()))];
  }

  // Prefer recent values so graphs grow deep.
  Value PickRecent(const std::vector<Value> &pool) {
    const auto size = static_cast<uint32_t>(pool.size());
    const auto window = std::min<uint32_t>(size, 8u);
    return m_random.Chance(60) ? pool[size - 1u - m_random.Below(window)]
                               : Pick(pool);
  }

  uint32_t ExtractRange() { return m_safe || m_random.Chance(90) ? 2u : 3u; }

  Value MaybeIdentity(Value value) {
    return m_random.Chance(10) ? Emit(ValueOpcode::Identity, {value}) : value;
  }

  Value RawRead() {
    using namespace Libs::Graphics::ShaderRecompiler::IR;
    const bool buffer = m_random.Chance(30);
    MemoryInfo memory;
    memory.kind = buffer ? ResourceKind::ScalarBuffer : ResourceKind::ScalarAddress;
    const auto choice = m_random.Below(10);
    memory.offset = m_safe        ? m_random.Below(32) * 4u
                    : choice == 0 ? static_cast<uint32_t>(-4 * static_cast<int32_t>(m_random.Below(4)))
                    : choice == 1 ? m_random.Below(64)
                                  : m_random.Below(32) * 4u;
    const auto index = static_cast<uint32_t>(m_program.memory_info.size());
    m_program.memory_info.push_back(memory);
    const auto low = m_safe || m_random.Chance(60) ? PickRecent(m_ptr) : Pick(m_u32);
    const auto high = m_safe || m_random.Chance(85) ? Value(0u) : Pick(m_u32);
    const auto offset =
        m_safe || m_random.Chance(60) ? Value(m_random.Below(16) * 4u) : Pick(m_u32);
    const auto flags = std::bit_cast<uint64_t>(MemoryFlags{.index = index, .pc = index * 4u});
    if (buffer) {
      const auto stride = Value((m_random.Below(3) * 16u) << 16u);
      const auto records = m_safe || m_random.Chance(80) ? Value(0x10000u) : Pick(m_u32);
      const auto handle = Emit(ValueOpcode::GetBufferResource,
                               {MaybeIdentity(low), stride, records, Value(0u)});
      return Emit(ValueOpcode::ReadConstBuffer, {MaybeIdentity(handle), offset}, flags);
    }
    const auto handle = Emit(ValueOpcode::GetAddressResource, {MaybeIdentity(low), high});
    return Emit(ValueOpcode::LoadAddressU32,
                {MaybeIdentity(handle), offset, Value(0u), Value(true)}, flags);
  }

  void Step() {
    using namespace Libs::Graphics::ShaderRecompiler::IR;
    constexpr std::array binary32{
        ValueOpcode::IAdd32, ValueOpcode::ISub32, ValueOpcode::IMul32,
        ValueOpcode::UMin32, ValueOpcode::BitwiseAnd32, ValueOpcode::BitwiseOr32,
        ValueOpcode::BitwiseXor32, ValueOpcode::ShiftLeftLogical32,
        ValueOpcode::ShiftRightLogical32, ValueOpcode::ShiftRightArithmetic32};
    constexpr std::array compare32{
        ValueOpcode::IEqual32, ValueOpcode::INotEqual32, ValueOpcode::ULessThan32,
        ValueOpcode::UGreaterThan32, ValueOpcode::SGreaterThanEqual32};
    constexpr std::array logical{ValueOpcode::LogicalAnd, ValueOpcode::LogicalOr,
                                 ValueOpcode::LogicalXor};
    constexpr std::array binary64{
        ValueOpcode::IAdd64, ValueOpcode::ISub64, ValueOpcode::IMul64,
        ValueOpcode::BitwiseAnd64, ValueOpcode::ShiftLeftLogical64,
        ValueOpcode::ShiftRightLogical64, ValueOpcode::ShiftRightArithmetic64};
    switch (m_random.Below(24)) {
    case 0:
    case 1:
    case 2:
      m_u32.push_back(RawRead());
      m_ptr.push_back(m_u32.back());
      break;
    case 3:
    case 4:
      m_u32.push_back(Emit(binary32[m_random.Below(binary32.size())],
                           {MaybeIdentity(PickRecent(m_u32)), Pick(m_u32)}));
      break;
    case 5:
      m_u1.push_back(Emit(compare32[m_random.Below(compare32.size())],
                          {PickRecent(m_u32), Pick(m_u32)}));
      break;
    case 6:
      m_u1.push_back(m_random.Chance(50)
                         ? Emit(logical[m_random.Below(logical.size())],
                                {Pick(m_u1), Pick(m_u1)})
                         : Emit(m_random.Chance(50) ? ValueOpcode::LogicalNot
                                                    : ValueOpcode::ConditionRef,
                                {Pick(m_u1)}));
      break;
    case 7:
      m_u32.push_back(Emit(ValueOpcode::SelectU32,
                           {PickRecent(m_u1), PickRecent(m_u32), Pick(m_u32)}));
      break;
    case 8: {
      // An EXEC-masked read: selects on the mask take their active operand.
      const auto mask = PickRecent(m_u1);
      const auto active = Emit(ValueOpcode::SelectU32,
                               {mask, PickRecent(m_u32), Pick(m_u32)});
      const auto inner = m_random.Chance(50)
                             ? Emit(ValueOpcode::IAdd32, {active, Pick(m_u32)})
                             : active;
      m_u32.push_back(Emit(ValueOpcode::ReadFirstLane,
                           {inner, m_random.Chance(85) ? mask : Pick(m_u1)}));
      break;
    }
    case 9: {
      const auto packed = Emit(ValueOpcode::CompositeConstructU64,
                               {PickRecent(m_u32), Pick(m_u32)});
      m_u64.push_back(packed);
      m_u32.push_back(Emit(ValueOpcode::CompositeExtractU64,
                           {packed, Value(m_random.Below(ExtractRange()))}));
      break;
    }
    case 10: {
      const auto source = Emit(m_random.Chance(50) ? ValueOpcode::CompositeConstructU32x2
                                                   : ValueOpcode::IAddCarry32,
                               {PickRecent(m_u32), Pick(m_u32)});
      m_u32.push_back(Emit(ValueOpcode::CompositeExtractU32x2,
                           {MaybeIdentity(source),
                            Value(m_random.Below(ExtractRange()))}));
      break;
    }
    case 11:
      m_u64.push_back(Emit(binary64[m_random.Below(binary64.size())],
                           {PickRecent(m_u64), Pick(m_u64)}));
      m_u32.push_back(Emit(ValueOpcode::CompositeExtractU64,
                           {m_u64.back(), Value(m_random.Below(2u))}));
      break;
    case 12: {
      const auto op = m_random.Below(4);
      if (op == 0) {
        m_u32.push_back(Emit(ValueOpcode::ConvertF32U32, {PickRecent(m_u32)}));
      } else if (op == 1 && !m_safe) {
        m_u32.push_back(Emit(ValueOpcode::ConvertU32F32, {PickRecent(m_u32)}));
      } else if (op == 2) {
        m_u32.push_back(Emit(ValueOpcode::FPMul32, {PickRecent(m_u32), Value::F32(0.5f)}));
      } else {
        m_u32.push_back(Emit(ValueOpcode::FPTrunc32, {PickRecent(m_u32)}));
      }
      break;
    }
    case 13: {
      const auto op = m_random.Below(3);
      if (op == 0) {
        m_u1.push_back(Emit(ValueOpcode::FPIsNan32, {PickRecent(m_u32)}));
      } else {
        const auto flush = std::bit_cast<uint8_t>(FPCompareFlags{m_random.Chance(50)});
        m_u1.push_back(Emit(op == 1 ? ValueOpcode::FPOrdLessThanEqual32
                                    : ValueOpcode::FPOrdGreaterThanEqual32,
                            {PickRecent(m_u32), Pick(m_u32)}, flush));
      }
      break;
    }
    case 14: {
      const auto op = m_random.Below(3);
      const auto offset = m_safe              ? Value(m_random.Below(17))
                          : m_random.Chance(80) ? Value(m_random.Below(33))
                                                : Pick(m_u32);
      const auto width = m_safe              ? Value(m_random.Below(16))
                         : m_random.Chance(80) ? Value(m_random.Below(33))
                                               : Pick(m_u32);
      if (op == 2) {
        m_u32.push_back(Emit(ValueOpcode::BitFieldInsert,
                             {PickRecent(m_u32), Pick(m_u32), offset, width}));
      } else {
        m_u32.push_back(Emit(op == 0 ? ValueOpcode::BitFieldUExtract
                                     : ValueOpcode::BitFieldSExtract,
                             {PickRecent(m_u32), offset, width}));
      }
      break;
    }
    case 15:
      m_u32.push_back(Emit(m_random.Chance(50) ? ValueOpcode::BitwiseNot32
                                               : ValueOpcode::BitCastU32F32,
                           {PickRecent(m_u32)}));
      break;
    case 16: {
      // Invariant (removed by the plan) or genuinely divergent phis.
      auto &phi = m_block->AppendNewInst(ValueOpcode::Phi, {},
                                         static_cast<uint64_t>(Type::U32));
      const auto first = PickRecent(m_u32);
      phi.AddPhiOperand(m_block, first);
      phi.AddPhiOperand(m_block, m_safe || m_random.Chance(50) ? first : Pick(m_u32));
      if (m_random.Chance(30)) {
        phi.AddPhiOperand(m_block, Value(&phi));
      }
      m_u32.push_back(Value(&phi));
      break;
    }
    case 17:
    case 18: {
      const auto slots = static_cast<uint32_t>(m_program.srt_reads.size());
      // SSA values cannot reach their own slot.
      if (slots == 0u) {
        m_u32.push_back(RawRead());
        break;
      }
      const auto srt = Emit(ValueOpcode::GetSrtResource, {});
      m_u32.push_back(Emit(ValueOpcode::ReadConst, {srt, Value(m_random.Below(slots))}));
      break;
    }
    case 19:
      if (!m_safe && m_random.Chance(20)) {
        m_u32.push_back(Emit(ValueOpcode::UndefU32, {}));
      } else {
        m_u32.push_back(Value(RandomImmediate()));
      }
      break;
    case 20:
      m_u64.push_back(Emit(ValueOpcode::GetShaderBase, {}));
      m_u32.push_back(Emit(ValueOpcode::CompositeExtractU64, {m_u64.back(), Value(0u)}));
      break;
    default:
      m_u32.push_back(Emit(ValueOpcode::IAdd32,
                           {PickRecent(m_u32), Value(m_random.Below(8) * 4u)}));
      break;
    }
  }

  uint32_t AddSource(std::initializer_list<Value> dwords) {
    using namespace Libs::Graphics::ShaderRecompiler::IR;
    DescriptorSource source;
    for (const auto dword : dwords) {
      source.dwords[source.dword_count++] = dword;
    }
    m_program.descriptor_sources.push_back(source);
    return static_cast<uint32_t>(m_program.descriptor_sources.size() - 1u);
  }

  Value Word() { return m_random.Chance(70) ? PickRecent(m_u32) : Pick(m_u32); }

  void AddResources() {
    using namespace Libs::Graphics::ShaderRecompiler::IR;
    namespace Prospero = Libs::Graphics::Prospero;
    const auto buffers = 1u + m_random.Below(3);
    for (uint32_t i = 0; i < buffers; i++) {
      const auto dword3 = m_random.Chance(70) ? Value(0u) : Word();
      m_buffer_sources.push_back(AddSource({Word(), Word(), Word(), dword3}));
      m_program.info.buffers.push_back(
          {.source = m_buffer_sources.back(), .written = m_random.Chance(40),
           .formatted = m_random.Chance(50)});
    }
    const auto valid_format = Value(
        static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float) << 20u);
    const auto valid_type = Value(
        Libs::Graphics::DstSel(4, 5, 6, 7) |
        (static_cast<uint32_t>(Prospero::ImageType::kColor2D) << 28u));
    const auto images = 1u + m_random.Below(2);
    for (uint32_t i = 0; i < images; i++) {
      const bool valid = m_random.Chance(60);
      m_image_sources.push_back(
          AddSource({Word(), valid ? valid_format : Word(), Value(0u),
                     valid ? valid_type : Word(), Value(0u), Value(0u), Value(0u),
                     Value(0u)}));
      m_program.info.images.push_back(
          {.source = m_image_sources.back(),
           .resource_class = ImageResourceClass::Sampled,
           .numeric_class = Prospero::TextureNumericClass::Float,
           .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
    }
    const auto sampler = AddSource({Word(), Word(), Word(), Word()});
    m_program.info.samplers.push_back({.source = sampler});
    for (uint32_t i = 0; i < images; i++) {
      m_program.info.sampled_pairs.push_back({.image = i, .sampler = 0});
    }
    m_condition_source = AddSource({PickRecent(m_u1)});
    m_fill_source = AddSource({Word(), Word(), Word(), Word()});
  }

  SplitMix m_random;
  bool m_safe = false;
  Libs::Graphics::ShaderRecompiler::IR::Program m_program;
  Libs::Graphics::ShaderRecompiler::IR::Block *m_block = nullptr;
  std::vector<Value> m_u32;
  std::vector<Value> m_u1;
  std::vector<Value> m_u64;
  std::vector<Value> m_ptr;
  std::vector<uint32_t> m_buffer_sources;
  std::vector<uint32_t> m_image_sources;
  uint32_t m_condition_source = 0;
  uint32_t m_fill_source = 0;
};

void TestCompiledWalkerMatchesReference() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  uint32_t successes = 0;
  uint32_t failures = 0;
  for (uint64_t seed = 1; seed <= 1500; seed++) {
    FuzzPlanBuilder builder(seed);
    auto plan = builder.Build();
    SplitMix random{seed * 7919u};
    // Plan attributes that the extraction derives from real shaders.
    if (random.Chance(40)) {
      const auto condition = plan.descriptor_sources[builder.ConditionSource()].dwords[0];
      plan.control_flow.resize(4);
      plan.control_flow[0] = {.condition = condition, .successors = {1, 2}};
      plan.control_flow[1] = {.condition = random.Chance(50) ? condition : Value{},
                              .successors = {3, 2},
                              .sources = {builder.BufferSource(0)}};
      plan.control_flow[2] = {.successors = {3}, .sources = {builder.ImageSource(0)}};
      plan.control_flow[3] = {.sources = {builder.FillSource()}};
    }
    if (random.Chance(30)) {
      const auto &fill = plan.descriptor_sources[builder.FillSource()];
      plan.uniform_fill.fill = {.kind = UniformFillKind::Buffer,
                                .words = 1u + random.Below(4)};
      for (uint32_t i = 0; i < plan.uniform_fill.fill.words; i++) {
        plan.uniform_fill.values[i] = random.Chance(80) ? fill.dwords[0] : fill.dwords[i];
      }
    }
    for (auto &inst : plan.value_storage) {
      if (inst.GetOpcode() == ValueOpcode::LoadAddressU32 ||
          inst.GetOpcode() == ValueOpcode::ReadConstBuffer) {
        auto flags = inst.Flags<SrtReadFlags>();
        flags.clean = random.Chance(15) ? 1u : 0u;
        inst.SetFlags(flags);
      }
    }
    // Extraction captures reads only for plans that require the strict reader.
    plan.capture_specialization_reads = random.Chance(25);
    plan.requires_specialization_memory =
        plan.capture_specialization_reads || random.Chance(10);
    Check(CompileSrtPlan(plan) != nullptr, "fuzz plan was not compiled");

    ResourceSnapshot reference_snapshot;
    ResourceSnapshot compiled_snapshot;
    ResourceSpecialization reference_specialization;
    ResourceSpecialization compiled_specialization;
    for (uint32_t refresh = 0; refresh < 6; refresh++) {
      FuzzMemory memory;
      const bool safe = builder.Safe();
      memory.strict_salt = static_cast<uint32_t>(random.Next());
      memory.strict_modulus = std::array{0u, 7u, 31u}[random.Below(3)];
      for (auto &word : memory.words) {
        word = safe || random.Chance(40) ? FuzzPointer(random.Below(FuzzMemoryWords))
               : random.Chance(50)       ? random.Below(64)
                                         : static_cast<uint32_t>(random.Next());
      }
      std::vector<uint32_t> user_data(safe || random.Chance(85) ? 16u : 8u);
      for (auto &word : user_data) {
        word = safe || random.Chance(60) ? FuzzPointer(random.Below(FuzzMemoryWords))
                                         : random.Below(64);
      }
      const SrtRuntime runtime{
          .user_data = user_data,
          .shader_base = random.Chance(50) ? FuzzMemoryBase : random.Next(),
          .read_memory = ReadFuzzOrdinary,
          .userdata = &memory,
          .read_specialization_memory = random.Chance(85) ? ReadFuzzStrict : nullptr};
      std::vector<FuzzRead> reference_reads;
      std::vector<FuzzRead> compiled_reads;
      memory.log = &reference_reads;
      const bool reference = MaterializeResourcesReference(
          plan, runtime, reference_snapshot, reference_specialization);
      memory.log = &compiled_reads;
      const bool compiled = MaterializeResources(plan, runtime, compiled_snapshot,
                                                 compiled_specialization);
      Check(reference == compiled, "compiled walker changed the refresh result");
      Check(reference_reads == compiled_reads,
            "compiled walker changed the guest reads");
      Check(SameSnapshot(reference_snapshot, compiled_snapshot),
            "compiled walker changed the resource snapshot");
      Check(reference_specialization == compiled_specialization,
            "compiled walker changed the resource specialization");
      (reference ? successes : failures)++;
    }
  }
  // Both outcomes must be common for the comparison to mean anything.
  Check(successes > 2000 && failures > 2000,
        "fuzz plans did not exercise both refresh outcomes");
}

void TestCompiledWalkerDirectReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  // The production reader copies guest memory directly.
  const std::array<uint32_t, 2> words{0x12345678u, 0x9abcdef0u};
  auto plan = ExtractResourcePlan(SrtProgram(reinterpret_cast<uint64_t>(words.data())));
  Check(CompileSrtPlan(plan) != nullptr, "direct SRT plan was not compiled");
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  ResourceSnapshot reference_snapshot;
  ResourceSpecialization reference_specialization;
  uint32_t specialization_reads = 0;
  const SrtRuntime runtime{.userdata = &specialization_reads,
                           .read_specialization_memory = RejectSpecializationRead};
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            MaterializeResourcesReference(plan, runtime, reference_snapshot,
                                          reference_specialization),
        "direct SRT refresh failed");
  Check(snapshot.flattened_srt == std::vector<uint32_t>{words[0]} &&
            SameSnapshot(snapshot, reference_snapshot) && specialization_reads == 0,
        "compiled walker changed a direct SRT read");
}


} // namespace

namespace Common {

int DbgExitHandler(const char *, int, std::string_view) { std::abort(); }

int DbgExitHandler(const char *, int, fmt::text_style, std::string_view) {
  std::abort();
}

int DbgExitIfHandler(const char *, const char *, int) { return 1; }

void DbgExit(int) { std::abort(); }

} // namespace Common

int main() {
  TestMappedSrtUsesDirectReaderByDefault();
  TestNullPageSrtReadsZero();
  TestNativeNestedSrtRefresh();
  TestIntegerRuntimeValueFollowsSrtReads();
  TestSrtAliasesRetainReadPolicy();
  TestUniformVectorDescriptorRead();
  TestExactReciprocalDescriptorArithmetic();
  TestUnbasedFlatCacheHitMaterializes();
  for (const bool memory_condition : {false, true})
    TestWrittenDescriptorPredicateReads(memory_condition);
  TestFailedMaterializationRejectsStage();
  TestSharedControlPredicates();
  TestNestedExecPredicatesReuseReads();
  TestFiniteImageRefreshReusesScalarReads();
  for (const bool independent_mask : {false, true})
    TestEmptyTraversalSkipsUnmappedDescriptor(independent_mask);
  TestEmptyTraversalSkipsUnmappedDescriptor(false, true);
  TestCyclicControlPredicatesRemainUnknown();
  TestEmptyScalarBufferSpecialization();
  TestMixedSamplerVariantsShareRuntimeDescriptor();
  TestSignedStorageSpecialization();
  for (const bool mixed : {false, true}) TestIndirectImageSamplerClasses(mixed);
  TestCompiledWalkerDirectReads();
  TestCompiledWalkerMatchesReference();
  std::puts("ResourceMaterializationTests: all cases passed");
  return 0;
}

// Keep this focused standalone target self-contained by amalgamating its small
// typed-IR implementation set.
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
