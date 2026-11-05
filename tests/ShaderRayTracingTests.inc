void TestTrinityStackInstruction() {
  namespace D = ShaderRecompiler::Decoder;
  const std::array code{0xdb840008u, 0x01010a09u,
                        0xdb840010u, 0x02010a09u, EncodeSopp(0x01)};
  D::Program decoded;
  D::DecodeProgram(code, decoded);
  const auto& first = decoded.instructions.front();
  Check(first.opcode == D::Opcode::DS_BVH_STACK_PUSH8_POP1_RTN_B32 &&
            first.src_count == 3 && first.offset == 8 && first.dst.reg == 1 &&
            first.src0.reg == 9 && first.src1.reg == 10 && first.src2.reg == 1,
        "Trinity stack operands or capacity were not decoded");
  ShaderComputeInputInfo compute{};
  compute.lds_size_dwords = 1024;
  auto options = MakeCompileOptions(ShaderType::Compute);
  options.input_info.compute = &compute;
  const auto result = RecompileForTest(code, options);
  Check(CountSourceOccurrences(result.ir_dump, " = BvhStack ") == 2 &&
            result.program.info.uses_lds && result.program.info.bvh_stack &&
            !result.program.info.shared_int64_atomics,
        "unused stack results discarded LDS writes or selected 64-bit atomics");
  Check(SpirvInstructionOpcodeCount(result.spirv, 54u) == 2 &&
            SpirvInstructionOpcodeCount(result.spirv, 246u) == 1,
        "Trinity stack calls duplicated the function or unrolled its eight pushes");
  CheckSpirvBinaryValidates(result.spirv);
  compute.lds_storage = true;
  CheckSpirvBinaryValidates(RecompileForTest(code, options).spirv);
  options = MakeCompileOptions(ShaderType::Pixel);
  CheckSpirvBinaryValidates(RecompileForTest(code, options).spirv);
  for (const auto words : {std::array{0xdb860008u, 0x01010a09u},
                           std::array{0xdb840007u, 0x01010a09u},
                           std::array{0xdb840020u, 0x01010a09u},
                           std::array{0xdb840008u, 0x01f90a09u},
                           std::array{0xdb840008u, 0x09010a09u}}) {
    const std::array invalid{words[0], words[1], EncodeSopp(0x01)};
    D::DecodeProgram(invalid, decoded);
    Check(decoded.instructions[0].opcode == D::Opcode::UNSUPPORTED,
          "unsupported stack encoding was silently accepted");
  }
}

void TestTrinityBvhInstruction() {
  namespace D = ShaderRecompiler::Decoder;
  const std::array code{0xf1949f05u, 0x00010000u, 0x0a030201u, 0x0000040du,
      EncodeMubuf0(0x1e, 0, false), EncodeMubuf1(0, 0, 0),
      EncodeMubuf0(0x1e, 16, false), EncodeMubuf1(4, 0, 0),
      EncodeMubuf0(0x1d, 32, false), EncodeMubuf1(8, 0, 0),
      EncodeMubuf0(0x1e, 40, false), EncodeMubuf1(10, 0, 0),
      EncodeMubuf0(0x1d, 56, false), EncodeMubuf1(14, 0, 0), EncodeSopp(0x01)};
  D::Program decoded;
  D::DecodeProgram(code, decoded);
  const auto& inst = decoded.instructions.front();
  Check(inst.opcode == D::Opcode::IMAGE_BVH8_INTERSECT_RAY && inst.word_count == 4 &&
            inst.data_dwords == 10 && inst.image_address_components == 11 &&
            inst.image_nsa_addr[3] == 10 && inst.image_nsa_addr[5] == 12 &&
            inst.image_nsa_addr[6] == 13 && inst.image_nsa_addr[8] == 15 &&
            inst.image_nsa_addr[9] == 4 && inst.src1.reg == 4,
        "Trinity BVH8 grouped ray registers were not decoded");
  const auto result = RecompileForTest(code, MakeCompileOptions(ShaderType::Compute));
  Check(result.program.info.bvh8 && !result.program.info.bvh &&
            result.program.info.uses_dma && result.program.info.images.empty(),
        "Trinity BVH8 did not use its separate raw-memory helper");
  Check(CountSourceOccurrences(result.ir_dump, " = Bvh8Intersect ") == 1,
        "Trinity BVH8 result and ray writes duplicated the intersection");
  CheckSpirvBinaryValidates(result.spirv);
  Check(SpirvInstructionOpcodeCount(result.spirv, 54u) == 6 &&
            SpirvInstructionOpcodeCount(result.spirv, 246u) == 1,
        "Trinity BVH8 duplicated helpers or unrolled its four-child loop");
  CheckSpirvBinaryValidates(RecompileForTest(code, MakeCompileOptions(ShaderType::Pixel)).spirv);
  for (const auto words : {std::array{0xf1949f05u, 0x40010000u, 0x0a030201u, 0x0000040du},
                           std::array{0xf1949f05u, 0x0001f700u, 0x0a030201u, 0x0000040du},
                           std::array{0xf1949f05u, 0x00010000u, 0xfe030201u, 0x0000040du},
                           std::array{0xf1949f05u, 0x00010700u, 0x0a030201u, 0x0000040du},
                           std::array{0xf1949f05u, 0x00010000u, 0x0a030201u, 0x0000040bu}}) {
    const std::array invalid{words[0], words[1], words[2], words[3], EncodeSopp(0x01)};
    D::DecodeProgram(invalid, decoded);
    Check(decoded.instructions[0].opcode == D::Opcode::UNSUPPORTED,
          "Trinity BVH8 accepted truncated ray or destination registers");
  }
}

// Closest hit traversals of guest BVHs as the native ray tracing patch finds them, reduced from
// Astro Bot's (the TLAS header from user data, its T# (S_BFE_U64 0x280008) and root 37 before the
// loop; the loop header (copy of EXEC, node type test); the instance setup: record
// base + (node >> 3) << 7 or the inline record at header + 0x80 of type 7 leaves, mask test of
// dword 15, object space ray from the transform in dwords 0-11, BLAS T# from dwords 12-13; the
// wave-wide BLAS walk from root dword 14 with the leaf test that keeps the nearest hit; the box test
// of the TLAS and the exit), in waves of 64 lanes and of 32. Waves of 32 lanes keep lane masks in
// one SGPR (S_MOV_B32 <s>, exec_lo; S_AND/OR/ANDN2_B32; S_CSELECT_B32 <s>, exec_lo, 0; V_CMP into
// one SGPR), put S_AND_B32 vcc_hi, exec_lo, <s> between the node type test's S_AND_B32 and
// S_ADD_I32, and pop the node stack from two VGPRs (lanes 0-31 and 32-59) after the loop's last
// backward branch.
constexpr uint32_t kClosestWave64[] = {
    0xf4040480u, 0xfa000028u, 0xf4040a09u, 0xfa000030u, 0x94b8ff12u, 0x00280008u, 0x80341228u, 0xbeaa03a5u,
    0x82351329u, 0x803cff12u, 0x00000080u, 0x823d8013u, 0xbe94047eu, 0x876a872au, 0x816ac46au, 0xbf0b826au,
    0x8736872au, 0xbf073687u, 0xbeb2043cu, 0x906a832au, 0xbeeb0380u, 0x8fea876au, 0x80326a34u, 0x82336b35u,
    0xf4100519u, 0xfa000000u, 0xbeeb0321u, 0xbf08ff23u, 0x00ffffffu, 0xbf840040u, 0xbf800000u, 0x10022614u,
    0x10042618u, 0x1006261cu, 0xd541000fu, 0x005e7a14u, 0xd5410010u, 0x006e7a18u, 0xd5410000u, 0x007e7a1cu,
    0xd5410008u, 0x04062a15u, 0xd541000bu, 0x040a2a19u, 0xd541000cu, 0x040e2a1du, 0x8721ff21u, 0x003fffffu,
    0xd5410011u, 0x04027c1du, 0x3e1e7c15u, 0x8f948320u, 0x3e207c19u, 0x3e102416u, 0x3e16241au, 0x3e18241eu,
    0x3e1e7816u, 0x3e20781au, 0x3e22781eu, 0x94a8ff14u, 0x00280008u, 0x852022a5u, 0x8710ff6bu, 0x01000000u,
    0x93a3ff6bu, 0x00010019u, 0x876a8720u, 0x816ac46au, 0xbf0b826au, 0xbf071080u, 0x859c807eu, 0xf1989f07u,
    0x000a0000u, 0x11100f19u, 0x1a0c0b08u, 0x00001c1bu, 0x2c04009fu, 0x7d8a00f9u, 0x0686a06au, 0x7d840423u,
    0x10000300u, 0x879e1c20u, 0x87ea206au, 0x7c0832f9u, 0x0606a000u, 0x8dea1e6au, 0x8dea6a20u, 0xbeea246au,
    0xbf880004u, 0x7e1a021bu, 0x7e120232u, 0x7e1c021au, 0x7e320300u, 0xbf82ffe4u, 0xf1989f07u, 0x000e0000u,
    0x3c3e3d19u, 0x04121513u, 0x00000605u, 0xbf840001u, 0xbf82ffa7u, 0xbf810000u, 0xbf810000u,
};

constexpr uint32_t kClosestWave32[] = {
    0xf4040080u, 0xfa000000u, 0x9484ff02u, 0x00280008u, 0xf4041a81u, 0xfa000030u, 0x802c026au, 0xbe8b03a5u,
    0x822e036bu, 0x8032ff02u, 0x00000080u, 0x82338003u, 0xbe88037eu, 0x876a870bu, 0x876b087eu, 0x816ac46au,
    0xbf0b826au, 0x870a870bu, 0xbf070a87u, 0xbe880432u, 0x906a830bu, 0xbeeb0380u, 0x8fea876au, 0x80086a2cu,
    0x82096b2eu, 0xf4100404u, 0xfa000000u, 0xbeeb031du, 0xbf08ff1fu, 0x00ffffffu, 0xbf840039u, 0x10081c10u,
    0x100a1c14u, 0x100c1c18u, 0xd541000du, 0x004e9c10u, 0xd5410044u, 0x005e9c14u, 0xd5410003u, 0x006e9c18u,
    0xd5410040u, 0x04121e11u, 0xd5410041u, 0x04161e15u, 0xd5410042u, 0x041a1e19u, 0x871dff1du, 0x003fffffu,
    0xd5410043u, 0x040e9e19u, 0x8f88831cu, 0x3e1a9e11u, 0x3e889e15u, 0x3e802012u, 0x3e822016u, 0x3e84201au,
    0x3e1a1212u, 0x3e881216u, 0x3e86121au, 0x9490ff08u, 0x00280008u, 0x85151ea5u, 0x8716ff6bu, 0x01000000u,
    0x9397ff6bu, 0x00010019u, 0x876a8715u, 0x816ac46au, 0xbf0b826au, 0xbf071680u, 0x8514807eu, 0xf1989f07u,
    0x00040303u, 0x43440d3fu, 0x46424140u, 0x00004847u, 0x2c0a069fu, 0x7d8a06f9u, 0x06869b6au, 0x7d840a17u,
    0x10060903u, 0x876b141bu, 0x876a1b6au, 0x886a6b6au, 0x8a6a6a7eu, 0x027a7a15u, 0x027e7f03u, 0xbf82ffeau,
    0xf1989f07u, 0x00010303u, 0x094f4e3fu, 0x11100f0eu, 0x00001312u, 0xbf82ffafu, 0xbf072d80u, 0xbf840009u,
    0x812dc12du, 0xbf09a02du, 0xbf840003u, 0xd760000bu, 0x00005b3cu, 0xbf82ffa6u, 0xd760000bu, 0x00005b3bu,
    0xbf82ffa3u, 0xbf810000u,
};

// The same traversal as ASTRO's PLAYROOM's builds it (reduced from 2224c5845b4dfdb9, waves of 32
// lanes): instance records of 160 bytes (index S_MUL_HI_U32 / S_MULK_I32 by 0xa0) at header +
// [header + 0x28], the transform by columns in dwords 4-18, the BLAS address in dwords 2-3 as it
// is, root dword 38 (S_MOV_B32 <root>, 37 then a branch around its load for the inline records at
// header + 0x8c), a mask of 20 bits (S_AND_B32 <dword 0>, 0xfffff00), flags bits 2 and 3 of dword
// 0 read through S_AND_B32 <dword 0>, 12, and the T# size from the node count at header + 0x50.
constexpr uint32_t kClosestPlayroomWave32[] = {
    0xf4040080u, 0xfa000000u, 0xf4041a81u, 0xfa000050u, 0x8004c16au, 0x8205c16bu, 0xf4041a81u, 0xfa000028u,
    0x9492ff04u, 0x002a0000u, 0x9490ff02u, 0x00280008u, 0x80256a02u, 0xbe8903a5u, 0x82266b03u, 0x802aff02u,
    0x0000008cu, 0x822b8003u, 0xbe86037eu, 0x876b8709u, 0x816ac46bu, 0xbf0b816au, 0xbf076b87u, 0xbe9c042au,
    0xbea803a5u, 0xbf860008u, 0x906a8309u, 0x9aebff6au, 0x000000a0u, 0xb86a00a0u, 0x801c6a25u, 0x821d6b26u,
    0xf4000a0eu, 0xfa000098u, 0xf4000a4eu, 0xfa000000u, 0x876aff29u, 0x0fffff00u, 0xbf84003du, 0xf40c0c0eu,
    0xfa000008u, 0xf408050eu, 0xfa000040u, 0xd5410010u, 0x00526632u, 0xd5410011u, 0x005a6634u, 0xd5410012u,
    0x00566633u, 0x10266032u, 0x10786034u, 0x107a6033u, 0xf4080b0eu, 0xfa000028u, 0xd5410016u, 0x04426836u,
    0xd5410039u, 0x0446682cu, 0xd541003au, 0x044a6837u, 0xd541003bu, 0x044e6236u, 0x3e78622cu, 0x3e7a6237u,
    0xf4041a8eu, 0xfa000038u, 0x3e2c6a2eu, 0x3e726a6au, 0x3e746a2fu, 0x3e76642eu, 0x3e78646au, 0x3e7a642fu,
    0x9488ff30u, 0x00280008u, 0x87028c29u, 0x876a8728u, 0x816ac46au, 0xbf0b816au, 0x87198402u, 0x901a8302u,
    0xbf071980u, 0x8514807eu, 0xf1989f07u, 0x00021010u, 0x393a160cu, 0x3e3c3d3bu, 0x0000403fu, 0x2c22209fu,
    0x7d8a20f9u, 0x06869c6au, 0x10202510u, 0x7d84221au, 0x876a1c6au, 0x8a6b146au, 0x876a141cu, 0x886b6a6bu,
    0x8a6a6b7eu, 0x02181910u, 0x022a2a15u, 0xbf82ffe7u, 0xf1989f07u, 0x00041010u, 0x3534330cu, 0x36323130u,
    0x00003837u, 0xbf840001u, 0xbf82ffa7u, 0xbf810000u,
};

void TestRayTracingSiteWaveSizes() {
  namespace D = ShaderRecompiler::Decoder;
  const auto derive = [](std::span<const uint32_t> code, uint32_t wave_size, D::RayTracingPatch &patch) {
    D::Program decoded;
    D::DecodeProgram(code, decoded);
    return D::DeriveRayTracingPatch(decoded, patch, wave_size);
  };
  struct Case {
    std::span<const uint32_t> code;
    uint32_t wave_size;
    uint32_t header_offset;
    uint32_t result_word;
    uint32_t result_instance;
    bool playroom;
  };
  for (const auto &test : {Case{kClosestWave64, 64, 0x28, 13, 9, false}, Case{kClosestWave32, 32, 0, 61, D::NoRegister, false},
                           Case{kClosestPlayroomWave32, 32, 0, 21, D::NoRegister, true}}) {
    D::RayTracingPatch patch;
    Check(derive(test.code, test.wave_size, patch) && patch.sites.size() == 1,
          "closest hit traversal was not recognized in its wave size");
    D::RayTracingPatch other;
    Check(!derive(test.code, 96 - test.wave_size, other),
          "closest hit traversal was recognized with the lane masks of the other wave size");
    const auto &site = patch.sites[0];
    const auto &layout = patch.layout;
    Check(site.kind == D::RayTracingKind::ClosestOpaque && patch.user_sgpr == 0 &&
              patch.header_offset == test.header_offset && patch.root == 37 && site.result_t == site.t_max &&
              site.result_word == test.result_word && site.result_instance == test.result_instance,
          "closest hit traversal site has the wrong TLAS source or hit registers");
    if (test.playroom) {
      const std::array<uint8_t, 12> transform{4, 8, 12, 16, 5, 9, 13, 17, 6, 10, 14, 18};
      Check(layout.instance_offset == 0x28 && layout.record_stride == 160 && layout.transform == transform &&
                layout.blas_dword == 2 && !layout.blas_packed && layout.root_dword == 38 && layout.mask_dword == 0 &&
                layout.mask_shift == 8 && layout.mask_width == 20 && layout.flags_dword == 0 &&
                layout.flags_shift == 0 && !layout.flip_facing && layout.inline_type == 7 &&
                layout.inline_offset == 0x8c && layout.inline_root == 37 && layout.node_count_80 &&
                layout.Pack() == std::array{0xd951c840u, 0x4c25ea62u, 0x28482080u, 0x308f2511u},
            "closest hit traversal has the wrong instance record layout");
    } else {
      Check(layout.instance_offset == 0x30 && layout.record_stride == 128 && layout.blas_dword == 12 &&
                layout.blas_packed && layout.root_dword == 14 && layout.mask_dword == 15 &&
                layout.mask_shift == 24 && layout.mask_width == 0 && layout.flags_dword == 13 &&
                layout.flags_shift == 22 && !layout.flip_facing && layout.inline_type == 7 &&
                layout.inline_offset == 0x80 && layout.inline_root == 37 && !layout.node_count_80 &&
                layout.Pack() == std::array{0x76543210u, 0x1dc6ba98u, 0x000dbb8fu, 0x000f2510u},
            "closest hit traversal has the wrong instance record layout");
    }
    // The patched program traces with ray queries.
    auto options = MakeCompileOptions(ShaderType::Compute);
    options.wave_size = test.wave_size;
    options.native_ray_tracing = true;
    auto translated = ShaderRecompiler::TranslateProgram(test.code, options);
    const auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
    ShaderRecompiler::IR::ResourceSpecialization specialization;
    specialization.buffers.assign(plan.info.buffers.size(), {});
    specialization.images.assign(plan.info.images.size(), {});
    specialization.samplers.assign(plan.info.samplers.size(), {});
    const auto compiled = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
    Check(compiled.program.info.ray_query && compiled.program.info.ray_query_sites == 1 &&
              compiled.program.info.ray_query_layout == layout.Pack() &&
              SpirvInstructionOpcodeCount(compiled.spirv, 4473u) != 0,
          "patched closest hit traversal does not trace with ray queries");
    CheckSpirvBinaryValidates(compiled.spirv);
  }
}

void TestRayTracingInstructions() {
  TestTrinityBvhInstruction();
  TestTrinityStackInstruction();
  TestRayTracingSiteWaveSizes();
  namespace D = ShaderRecompiler::Decoder;
  const std::array code{
      0xf1989f01u, 0x00081014u,
      EncodeMubuf0(0x1e, 0, false), EncodeMubuf1(16, 0, 0),
      0xf1989f07u, 0x00071001u, 0x3d3e3c15u, 0x38095851u, 0x00003a39u,
      EncodeMubuf0(0x1e, 16, false), EncodeMubuf1(16, 0, 0),
      0xf19c9f01u, 0x00081014u,
      EncodeMubuf0(0x1e, 32, false), EncodeMubuf1(16, 0, 0),
      // Captured 016c1ab44ee6a526: pointer v2:v7 and final inverse component v5.
      0xf19c9f07u, 0x00090202u, 0x28263607u, 0x221f1e29u, 0x00050403u,
      EncodeMubuf0(0x1e, 48, false), EncodeMubuf1(2, 0, 0),
      EncodeSopp(0x01)};
  D::Program decoded;
  D::DecodeProgram(code, decoded);
  Check(decoded.instructions.size() == 9 &&
            decoded.instructions.back().opcode == D::Opcode::S_ENDPGM,
        "decoder stopped before instructions following BVH intersections");
  for (const auto index : {0u, 2u}) {
    const auto &inst = decoded.instructions[index];
    Check(inst.opcode == D::Opcode::IMAGE_BVH_INTERSECT_RAY &&
              inst.image_address_components == 11 && inst.data_dwords == 4 && inst.image_r128,
          "BVH32 instruction lost its ray or descriptor width");
  }
  for (const auto index : {4u, 6u}) {
    const auto &inst = decoded.instructions[index];
    Check(inst.opcode == D::Opcode::IMAGE_BVH_INTERSECT_RAY && inst.opcode_id == 0xe7 &&
              inst.image_address_components == 12 && inst.data_dwords == 4 && inst.image_r128,
          "BVH64 instruction lost its full node pointer or ray width");
  }
  Check(decoded.instructions[6].word_count == 5 &&
            decoded.instructions[6].image_nsa_addr[0] == 7 &&
            decoded.instructions[6].image_nsa_addr[10] == 5,
        "captured BVH64 NSA pointer or final ray component was not decoded");
  Check(decoded.instructions[2].image_nsa_addr[9] == 58,
        "BVH NSA ray components were not decoded");
  const auto result = RecompileForTest(code, MakeCompileOptions(ShaderType::Compute));
  Check(CountSourceOccurrences(result.ir_dump, "BvhIntersect") == 4 &&
            result.program.info.uses_dma && result.program.info.images.empty() &&
            result.program.info.samplers.empty(),
        "BVH intersections did not use GPU descriptors and raw guest memory");
  CheckSpirvBinaryValidates(result.spirv);

  const std::array literal{EncodeSMovB32(0, 255), 0xf1989f01u, EncodeSopp(0x01)};
  D::DecodeProgram(literal, decoded);
  Check(decoded.instructions.size() == 2 && decoded.instructions[0].opcode == D::Opcode::S_MOV_B32,
        "BVH-shaped scalar literal was decoded as an instruction");
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
  for (const auto words : {std::array{0xf19c9f01u, 0x40081014u}, // BVH64 A16
                           std::array{0xf1989f01u, 0x40081014u}, // BVH32 A16
                           std::array{0xf1981f01u, 0x00081014u}, // not R128
                           std::array{0xf1989f01u, 0x80081014u}}) { // D16
    const std::array unsupported{words[0], words[1], EncodeSopp(0x01)};
    ExpectFatal([&]() { (void)RecompileForTest(unsupported, MakeCompileOptions(ShaderType::Compute)); },
                "unsupported BVH encoding was silently accepted");
  }
  for (const auto encoding : {0xf1989f01u, 0xf19c9f01u}) {
    for (const auto nsa : {1u, 2u}) {
      std::array<uint32_t, 5> truncated{encoding | (nsa << 1), 0x00081014u};
      truncated[2 + nsa] = EncodeSopp(0x01);
      ExpectFatal([&]() {
        (void)RecompileForTest(std::span(truncated).first(3 + nsa), MakeCompileOptions(ShaderType::Compute));
      }, "BVH NSA instruction accepted an incomplete ray address");
    }
  }
#endif
}
