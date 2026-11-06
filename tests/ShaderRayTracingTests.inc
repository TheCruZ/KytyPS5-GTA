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

// Closest hit traversals whose rays start at one point for every lane (camera rays), reduced from
// ASTRO BOT's cs 9833d01b08e21d08 and ASTRO's PLAYROOM's cs 29289be6a0202b6d (waves of 64 lanes):
// the world origin stays in SGPRs (s_buffer_load_dwordx4 before the loop), the loop copies it into
// the TLAS box test's VGPRs with V_MOV_B32 right before the test, and the instance setup computes
// the object space origin from those SGPRs (V_MAD_F32 <v>, <record dword>, <origin SGPR>, <v>).
constexpr uint32_t kClosestUniformOrigin[] = {
    0xf4040080u, 0xfa000028u, 0xf4040201u, 0xfa000030u, 0x9498ff02u, 0x00280008u, 0x802a0208u, 0xbe8c03a5u,
    0x822b0309u, 0x802eff02u, 0x00000080u, 0x822f8003u, 0xbe84047eu, 0x876a870cu, 0x816ac46au, 0xbf0b826au,
    0x872c870cu, 0xbf072c87u, 0xbea8042eu, 0x906a830cu, 0xbeeb0380u, 0x8fea876au, 0x80286a2au, 0x82296b2bu,
    0xf4100114u, 0xfa000000u, 0xbeeb0311u, 0xbf08ff13u, 0x00ffffffu, 0xbf840044u, 0x10002604u, 0x10022608u,
    0x1004260cu, 0x7e160207u, 0x7e10020bu, 0x7e18020fu, 0xd5410022u, 0x04023605u, 0xd5410023u, 0x04063609u,
    0xd5410003u, 0x042c2804u, 0xd5410008u, 0x04202808u, 0xd541000cu, 0x0430280cu, 0xd5410024u, 0x040a360du,
    0x8711ff11u, 0x003fffffu, 0xd5410003u, 0x040c2a05u, 0x8f848310u, 0xd5410008u, 0x04202a09u, 0xd541000cu,
    0x04302a0du, 0x3e443806u, 0x3e46380au, 0x3e48380eu, 0xd541001fu, 0x040c2c06u, 0xd5410020u, 0x04202c0au,
    0xd5410021u, 0x04302c0eu, 0x94a0ff04u, 0x00280008u, 0x851012a5u, 0x8702ff6bu, 0x01000000u, 0x9393ff6bu,
    0x00010019u, 0x876a8710u, 0x816ac46au, 0xbf0b826au, 0xbf070280u, 0x858c807eu, 0xf1989f01u, 0x0008001du,
    0x2c04009fu, 0x7d8a00f9u, 0x0686906au, 0x7d840413u, 0x10000300u, 0x878e0c10u, 0x87ea106au, 0x7c083cf9u,
    0x06069000u, 0x8dea0e6au, 0x8dea6a10u, 0xbeea246au, 0xbf880004u, 0x7e1a020bu, 0x7e120228u, 0x7e1c020au,
    0x7e3c0300u, 0xbf82ffe7u, 0x7e020214u, 0x7e040215u, 0x7e060216u, 0xf1989f07u, 0x00060000u, 0x0302011eu,
    0x041c1b13u, 0x00000605u, 0xbf840001u, 0xbf82ffa0u, 0xbf810000u, 0xbf810000u,
};

constexpr uint32_t kClosestUniformOriginPlayroom[] = {
    0xf4040080u, 0xfa000028u, 0xf4041a81u, 0xfa000050u, 0x8004c16au, 0x8205c16bu, 0xf4041a81u, 0xfa000028u,
    0x9492ff04u, 0x002a0000u, 0xbe9403a5u, 0x9490ff02u, 0x00280008u, 0x80246a02u, 0x82256b03u, 0x802cff02u,
    0x0000008cu, 0x822d8003u, 0xbe84047eu, 0x87068714u, 0x816ac406u, 0xbf0b816au, 0xbf070687u, 0xbea0042cu,
    0xbea203a5u, 0xbf860008u, 0x906a8314u, 0x9aebff6au, 0x000000a0u, 0xb86a00a0u, 0x80206a24u, 0x82216b25u,
    0xf4000890u, 0xfa000098u, 0xf40008d0u, 0xfa000000u, 0xb72300ffu, 0xbf850049u, 0xf4080610u, 0xfa000040u,
    0xf40c0f10u, 0xfa000008u, 0xf4080510u, 0xfa000028u, 0xf4041a90u, 0xfa000038u, 0x7e020218u, 0x7e06021au,
    0x7e180219u, 0x1000263eu, 0x10202640u, 0xd5410001u, 0x04047c0cu, 0xd5410003u, 0x040c800cu, 0xd541000cu,
    0x04307e0cu, 0x1004263fu, 0xd5410017u, 0x040c280du, 0xd541000du, 0x0430860du, 0xd541000cu, 0x04023c42u,
    0xd541000bu, 0x04041a42u, 0xd541000eu, 0x040a3c43u, 0x3e203c14u, 0xd5410017u, 0x045cd40eu, 0xd541000du,
    0x04342e0eu, 0xd541000bu, 0x042c1c16u, 0x3e183e16u, 0x3e203e6au, 0x3e1c3e17u, 0x9488ff3cu, 0x00280008u,
    0x87278c23u, 0x876a8722u, 0x816ac46au, 0xbf0b816au, 0x87288427u, 0x90298327u, 0xbf072880u, 0x8598807eu,
    0xf1989f07u, 0x00020000u, 0x170d0b12u, 0x14100e0cu, 0x00001615u, 0x2c04009fu, 0x7d8a00f9u, 0x06869a6au,
    0x7d840429u, 0x87ea1a6au, 0x8aa2186au, 0x87ea181au, 0x88ea6a22u, 0xbeea246au, 0xbf880007u, 0x10000300u,
    0x7c262500u, 0xbf880004u, 0x7e240300u, 0x7e1e0217u, 0x7e080216u, 0x7e120220u, 0xbf82ffe2u, 0x7e06020cu,
    0x7e02020du, 0x7e04020eu, 0xf1989f07u, 0x00040000u, 0x02010312u, 0x051f1e13u, 0x00000806u, 0xbf840001u,
    0xbf82ff99u, 0xbf810000u, 0xbf810000u,
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

// Gran Turismo 7's closest hit traversal (CS, wave64; the code at 0x05e0cbbc of its eboot): its
// instance code walks the BLAS itself, and the leaf code calls the any-hit function of the
// triangle's material through a call table whose V# the shader loads once, before its loop.
constexpr uint32_t kGt7ClosestLeafWave64[] = {
    0xbfa00003u, 0xd746001au, 0x04010c02u, 0xf4080700u, 0xfa000060u, 0xf4100100u, 0xfa000090u, 0xf40c0500u,
    0xfa0000d0u, 0xf4001a80u, 0xfa0000f0u, 0x2c043486u, 0x36023487u, 0xd5480000u, 0x020d071au, 0xbe8203ffu,
    0x3dc8c8cau, 0x7e3002ffu, 0x21ffffffu, 0xbf8cc07fu, 0xe0302000u, 0x80070302u, 0xbf8c3f70u, 0x360406ffu,
    0x0000ffffu, 0x2c060690u, 0xd7460016u, 0x04050702u, 0xd7460017u, 0x04010703u, 0x7e160d16u, 0xf0000f08u,
    0x00030416u, 0xf0000f08u, 0x00010016u, 0x7e140d17u, 0xd5410009u, 0x005a1614u, 0xd5410008u, 0x00661617u,
    0xd541002eu, 0x01aa161au, 0x3e121415u, 0x3e101418u, 0x3e5c141bu, 0x10161309u, 0x3e161108u, 0x3e165d2eu,
    0x7e165d0bu, 0x101e130bu, 0x101c110bu, 0x101a5d0bu, 0xbf8c3f71u, 0x7e161704u, 0x7d940880u, 0x7e081705u,
    0x7e362507u, 0x7e0a1706u, 0xbf8c3f70u, 0x0216010bu, 0x020c0304u, 0x101836f9u, 0x06862602u, 0x020a0505u,
    0x10081f0bu, 0xd5410011u, 0x03ca18f0u, 0xd541000au, 0x04121d06u, 0x7e185511u, 0x3e140b0du, 0x1020190cu,
    0x10281d0cu, 0x10381b0cu, 0x1024150au, 0x10081f0cu, 0xd5410012u, 0x84422112u, 0x063224f2u, 0x7c0224f3u,
    0x7e2a6719u, 0x0225e4f9u, 0x86860680u, 0xd5410015u, 0xa456190au, 0x3e280d15u, 0x3e380b15u, 0x3e081715u,
    0x10262912u, 0x10283912u, 0x102a0912u, 0x7c1a26f9u, 0x06868280u, 0x7c1a2880u, 0x7c1a2af9u, 0x06868480u,
    0x88ea6a02u, 0x88fe6a04u, 0xbf88058bu, 0x7e242307u, 0xf4001a80u, 0xfa000168u, 0x100e0703u, 0xbf8cc07fu,
    0xd76d0019u, 0x045e2c6au, 0x7c023680u, 0x102424ffu, 0x3b808081u, 0x7e080280u, 0x7e380280u, 0x36323281u,
    0xbe8403ffu, 0x3a83126fu, 0x10242512u, 0x7e0602f2u, 0xbe82047eu, 0x02323280u, 0x7d8a3280u, 0x023a2507u,
    0x7c3c08f9u, 0x8606001du, 0xbf88001au, 0xd541001cu, 0x03ce3b1du, 0xf4041a80u, 0xfa000158u, 0xf40c0100u,
    0xfa000138u, 0xbf8cc07fu, 0x4a082c6au, 0x4a062e6bu, 0x363a08bfu, 0x363c06bfu, 0xf0000308u, 0x0001031du,
    0xbf8c3f70u, 0xd541001eu, 0x03ca3904u, 0x7e386d03u, 0x7e3a6b03u, 0x7e06551eu, 0xd51f8003u, 0x20020903u,
    0x080806f2u, 0x7e066703u, 0x7e3c6704u, 0x10083d1cu, 0x10383d1du, 0xbefe0402u, 0xbe860b81u, 0x7c03e0f9u,
    0x86260006u, 0x7c0de0f9u, 0x86268406u, 0x7d8a32f9u, 0x06868280u, 0x023c0a80u, 0x023a0cf9u, 0x86160605u,
    0xd501001fu, 0x20120d0bu, 0x10403d1eu, 0x3e403b1du, 0x3e403f1fu, 0x7e405d20u, 0x103c3d20u, 0x103a3b20u,
    0x103e3f20u, 0xbeea2402u, 0xbf88000fu, 0x10443b05u, 0x104a3f0bu, 0x10463d06u, 0x10483d04u, 0x104c3b04u,
    0x10083f04u, 0xd51f0022u, 0x20020d1fu, 0xd51f0025u, 0x20020b1eu, 0xd51f0023u, 0x2002171du, 0x3e483922u,
    0x3e4c3925u, 0x3e083923u, 0x8afe7e6au, 0xbf880020u, 0x10443b02u, 0x10463f00u, 0x10403d01u, 0xd51f0022u,
    0x2002031fu, 0xd51f0023u, 0x2002051eu, 0xd51f0020u, 0x2002011du, 0x103a4522u, 0x3e3a4723u, 0x3e3a4120u,
    0x7e425d1du, 0x103a4721u, 0x103c4121u, 0x10464521u, 0x10423b02u, 0x104a3d00u, 0x10444701u, 0x10484704u,
    0x104c3b04u, 0xd51f0021u, 0x2002031eu, 0x103c3d04u, 0xd51f0025u, 0x20020523u, 0xd51f0022u, 0x2002011du,
    0x3e483921u, 0x3e4c3925u, 0xd5410004u, 0x047a451cu, 0xbefe046au, 0x7d8a3280u, 0x103e2b15u, 0x02381700u,
    0x02400d01u, 0x02320b02u, 0x3e3e2713u, 0x3e48071cu, 0xd541001du, 0x049a4103u, 0xd541001eu, 0x04123303u,
    0x3e3e2914u, 0x10384924u, 0x7e3e5d1fu, 0x3e383b1du, 0x10322af9u, 0x1606061fu, 0x100826f9u, 0x1606061fu,
    0x3e383d1eu, 0x104228f9u, 0x1606061fu, 0x02441ef9u, 0x16060619u, 0x023e1cf9u, 0x16060604u, 0x7e465d1cu,
    0x024a1af9u, 0x16060621u, 0x10384923u, 0x103a3b23u, 0x103c3d23u, 0x10463922u, 0x3e463f1du, 0xd51f0023u,
    0x08024b1eu, 0xd5410022u, 0x848a471cu, 0xd541001fu, 0x847e471du, 0xd5410025u, 0x8496471eu, 0x103a4522u,
    0x3e3a3f1fu, 0x3e3a4b25u, 0x7e405d1du, 0x103a4520u, 0x10383f20u, 0x103c4b20u, 0xbe842402u, 0xbf880009u,
    0x7c1a2af9u, 0x06868680u, 0x7c1a2680u, 0x7c1a28f9u, 0x06868880u, 0x88ea6a06u, 0x88ea086au, 0x022302f9u,
    0x86860680u, 0x8afe7e04u, 0xbf880026u, 0x10403b00u, 0x103e22f9u, 0x06160611u, 0x102a3b11u, 0x10283911u,
    0x10443d11u, 0x3e40031cu, 0x3e40051eu, 0x103840f9u, 0x06160620u, 0x3e3e391fu, 0x063c3ef2u, 0x7c023ef3u,
    0x7e3a66f9u, 0x0026061eu, 0x0239e4f9u, 0x86860680u, 0xd541001du, 0x84764111u, 0xd51f0015u, 0x4002011du,
    0xd51f0014u, 0x4002031du, 0xd51f0022u, 0x4002051du, 0x10262b1cu, 0x1028291cu, 0x102a451cu, 0x10222713u,
    0x3e222914u, 0x3e222b15u, 0x7c1a2280u, 0x7e3c5d11u, 0x022302f9u, 0x86860680u, 0x103a271eu, 0x1038291eu,
    0x103c2b1eu, 0xbefe0404u, 0xbeea03ffu, 0x3c23d70au, 0x10283b00u, 0x1026171du, 0x100e0f07u, 0x10242512u,
    0x3e28031cu, 0x3e263906u, 0x200e0effu, 0x358637bdu, 0x202424ffu, 0x358637bdu, 0x3e28051eu, 0x3e263d05u,
    0x7c0228f9u, 0x0686866au, 0x7c0226f9u, 0x0686846au, 0xbe822402u, 0xbf880042u, 0x102a2713u, 0xd5410014u,
    0x840e2503u, 0x101e191du, 0x101c191cu, 0x101a191eu, 0xd5410010u, 0x84422115u, 0xd5410014u, 0x03ca2903u,
    0xd501001fu, 0x00110280u, 0x060620f2u, 0x7c0220f3u, 0x10202914u, 0x7e286703u, 0x0207e4f9u, 0x86860680u,
    0x7e205510u, 0xd5410014u, 0x8452270cu, 0x10202510u, 0xd51f000fu, 0x40021714u, 0xd51f000eu, 0x40020d14u,
    0xd51f000du, 0x40020b14u, 0x104020ffu, 0x3ea2f983u, 0x100a1f03u, 0x100c1d03u, 0x10061b03u, 0x10160b05u,
    0x3e160d06u, 0x3e160703u, 0x7e165d0bu, 0x3e321705u, 0x3e081706u, 0x3e421703u, 0x10063319u, 0x100a0119u,
    0x3e060904u, 0x3e0a0901u, 0x3e064321u, 0x3e0a4302u, 0x7e045d03u, 0x7c0406f9u, 0x06868880u, 0x10020b02u,
    0x200002f9u, 0x06862680u, 0x7c0c0280u, 0xd5410001u, 0x84020107u, 0x88ea6a08u, 0x8bea066au, 0xd5410000u,
    0x03ca0300u, 0x024300f9u, 0x86860681u, 0x10000100u, 0x7e005500u, 0x10000f00u, 0x104400ffu, 0x3ea2f983u,
    0x8afe7e02u, 0xbf880029u, 0x08041f1du, 0x08021d1cu, 0xd5410004u, 0x840e0f03u, 0x08001b1eu, 0xd5010021u,
    0x00190280u, 0x10180502u, 0x10041702u, 0xd5410003u, 0x03ca0903u, 0x3e180301u, 0xd5410004u, 0x040a0d01u,
    0x10060703u, 0x3e180100u, 0x3e080105u, 0x7e065503u, 0x7e045d0cu, 0x7c0418f9u, 0x06868680u, 0x10020f03u,
    0x10000902u, 0x104402ffu, 0x3ea2f983u, 0x200200f9u, 0x06862680u, 0x7c0c0080u, 0xd5410000u, 0x84060312u,
    0x88ea6a06u, 0x8bea046au, 0xd5410000u, 0x03ca0101u, 0x023f00f9u, 0x86860681u, 0x10000100u, 0x7e005500u,
    0x10002500u, 0x104000ffu, 0x3ea2f983u, 0xbefe0402u, 0x38003f21u, 0x36000111u, 0x7daa0080u, 0xbf880426u,
    0xf4040400u, 0xfa000118u, 0xbf8cc07fu, 0xf4001a88u, 0xfa000018u, 0xbf8cc07fu, 0xbf0d926au, 0xbf840001u,
    0xbf920001u, 0x7e50551du, 0xf40c0100u, 0xfa000070u, 0xf4040588u, 0xfa000058u, 0xbe820380u, 0xbf8cc07fu,
    0xf0000108u, 0x00010016u, 0x876a17ffu, 0x000003ffu, 0xbeeb03ffu, 0x461c4000u, 0xd7610027u, 0x00011802u,
    0xf4040300u, 0xfa000160u, 0xf4040380u, 0xfa0000f4u, 0xf40004c0u, 0xfa0000fcu, 0x7e52551cu, 0x7e54551eu,
    0xbf8cc07fu, 0x085618f9u, 0x8686066bu, 0x7e4a02c1u, 0x7e4c0280u, 0x9494ff10u, 0x00280008u, 0x8817ff6au,
    0x81000000u, 0xf4041a88u, 0xfa000030u, 0xf4081400u, 0xfa000120u, 0xf4041600u, 0xfa000130u, 0xbe9203a5u,
    0xbf8cc07fu, 0x8002106au, 0xbe951d9fu, 0xd7610027u, 0x00010002u, 0x8202116bu, 0x8003ff10u, 0x00000080u,
    0xd7610027u, 0x00010202u, 0x82028011u, 0xd7610027u, 0x00010c03u, 0xd7610027u, 0x00010e02u, 0xbf8c3f70u,
    0x0a00000du, 0xd541002du, 0x003a0109u, 0xd541002eu, 0x004e012eu, 0xd541002fu, 0x003e0108u, 0x3e5a3a0cu,
    0x3e5c3c0cu, 0x3e5e380cu, 0xbe84047eu, 0x876a8712u, 0x816ac46au, 0xbf0b826au, 0xbf850227u, 0xbf0712c1u,
    0xbf800000u, 0x85ea807eu, 0x87188712u, 0x87ea6a04u, 0xbeda246au, 0xbf88021eu, 0xd7600002u, 0x00010d27u,
    0xd7600003u, 0x00010f27u, 0xbf071887u, 0xbedc0402u, 0xbf840009u, 0xd7600002u, 0x00010127u, 0x906a8312u,
    0xbeeb0380u, 0x8fea876au, 0x805c6a02u, 0xd7600002u, 0x00010327u, 0x825d6b02u, 0xf410012eu, 0xfa000000u,
    0xbf8cc07fu, 0xbeeb0311u, 0xbf0d9813u, 0xbf840208u, 0xd7610027u, 0x00010414u, 0x10023a04u, 0x10043a08u,
    0x10063a0cu, 0x8711ff11u, 0x003fffffu, 0xd7610027u, 0x0001105au, 0xd5410004u, 0x001e5a04u, 0xd5410005u,
    0x002e5a08u, 0xd5410000u, 0x003e5a0cu, 0xd5410030u, 0x04063805u, 0xd5410031u, 0x040a3809u, 0xd5410032u,
    0x040e380du, 0x8fde8310u, 0xd5410033u, 0x04125e05u, 0xf408152fu, 0xfa000058u, 0xd5410034u, 0x04165e09u,
    0xbf8cc07fu, 0x816a8154u, 0xd5410035u, 0x04025e0du, 0x3e603c06u, 0x3e623c0au, 0x3e643c0eu, 0xd7600002u,
    0x00011927u, 0xd7610027u, 0x00010615u, 0x7e6c5530u, 0x7e6e5531u, 0x7e705532u, 0x3e665c06u, 0xd7610027u,
    0x00010816u, 0x80066ac1u, 0x3e685c0au, 0x820780c1u, 0x3e6a5c0eu, 0xd7610027u, 0x00010a17u, 0x876a07ffu,
    0x000003ffu, 0x9484ff5eu, 0x00280008u, 0x9062966bu, 0x8807ff6au, 0x81000000u, 0x8855ff5fu, 0x00080000u,
    0xbed4035eu, 0xbed703ffu, 0x00016204u, 0x875a8262u, 0xbf061887u, 0xbee30302u, 0x851012a5u, 0x8764ff6bu,
    0x01000000u, 0x93e5ff6bu, 0x00010019u, 0xbe851d9fu, 0xbe88047eu, 0x876a8710u, 0x816ac46au, 0xbf0b826au,
    0xbf850071u, 0xbf0710c1u, 0xbf84006eu, 0xbe822408u, 0xbf88006bu, 0x876ac510u, 0x906b8310u, 0xf420032au,
    0xd4000000u, 0xbf8cc07fu, 0x876a870cu, 0xbf076a87u, 0xbf840063u, 0xd7610027u, 0x00011402u, 0x876ac710u,
    0x8102816bu, 0xf420036au, 0xd4000004u, 0xbf8cc07fu, 0x7e00020cu, 0x926a9788u, 0xbf076480u, 0xbf800000u,
    0x8588807eu, 0xf1989f07u, 0x00010000u, 0x3534332bu, 0x36323130u, 0x00003837u, 0xbf8c3f70u, 0x7e085501u,
    0x2c02009fu, 0x7d8a00f9u, 0x06868a6au, 0x10000900u, 0x7d840265u, 0x878e080au, 0x7c0856f9u, 0x06069000u,
    0x87ea0a6au, 0x8aea086au, 0x8dea0e6au, 0x8dea6a10u, 0xbee0246au, 0xbf880037u, 0x10020902u, 0xbf8cc07fu,
    0x906a9e0du, 0x10040903u, 0x886a6a62u, 0xbf0d806au, 0xbf800000u, 0x856a8081u, 0xbf075a80u, 0xbf800000u,
    0x856a6a80u, 0x7e0e026au, 0xbf076a80u, 0xbf850020u, 0x7e080301u, 0x906b830cu, 0xf4001aafu, 0xfa00007cu,
    0xbe880bc4u, 0xbf8cc07fu, 0x80eb6a6bu, 0xf4001aafu, 0xfa000028u, 0x360c1af9u, 0x86860608u, 0xbf8cc07fu,
    0x97ea6a6bu, 0x7e0a0302u, 0xf4001aefu, 0xd4000000u, 0xf4001aaeu, 0xfa000078u, 0x7e0e0281u, 0xbf8cc07fu,
    0x816a6a6bu, 0x936a6a58u, 0x816a6a59u, 0x8f6a846au, 0xf4280228u, 0xd4000000u, 0xbf8cc07fu, 0xbe8e0308u,
    0xbe8f0309u, 0xbe90030au, 0xbe91030bu, 0xbe8e210eu, 0x7daa0e80u, 0xbf880007u, 0x7e4a020du, 0x7e30025cu,
    0x7e32025du, 0x7e4c020cu, 0x7e480301u, 0x7e460302u, 0x7e560300u, 0xbf8cc07fu, 0xbf04800du, 0xbefe0460u,
    0xbf850005u, 0x8f6a8302u, 0x81028102u, 0xf424032au, 0xd4000000u, 0xbf82ffa5u, 0xd7600002u, 0x00011527u,
    0xbefe0402u, 0xbf820134u, 0x7e0002c1u, 0x7e0202c1u, 0x7e0402c1u, 0x7e0602c1u, 0xbeea2408u, 0xbf88000au,
    0x7e000210u, 0xbf800000u, 0xf1989f07u, 0x00010000u, 0x3534332bu, 0x36323130u, 0x00003837u, 0xbf8c3f70u,
    0x7d8a00f9u, 0x068688c1u, 0xbf138008u, 0xbefe046au, 0xbf840121u, 0x7d8a02c1u, 0xbf13806au, 0xbf8400e7u,
    0x7d8a04c1u, 0xbf13806au, 0xbf84007cu, 0x7d8a06c1u, 0xbf13806au, 0xbf840019u, 0x7d8a06c1u, 0xd7600002u,
    0x00011927u, 0xbe83146au, 0xd760006au, 0x00000703u, 0xd760006bu, 0x00000701u, 0xd7600010u, 0x00000700u,
    0xd761002cu, 0x0000046au, 0xd760006au, 0x00000702u, 0x81028102u, 0xd761002cu, 0x0000046au, 0x81028102u,
    0xd761002cu, 0x0000046bu, 0x81028102u, 0x8382bc02u, 0xd7610027u, 0x00011802u, 0xbf82ff56u, 0x7d8a04f9u,
    0x06868ac1u, 0xd7600002u, 0x00011927u, 0xbeea140au, 0xd7600010u, 0x0000d500u, 0xd7600011u, 0x0000d501u,
    0xd7600012u, 0x0000d502u, 0x7d8a0410u, 0x7d8a04f9u, 0x06868c11u, 0x7d8a04f9u, 0x06868e12u, 0xd761002cu,
    0x00000412u, 0x81028102u, 0x87ea6a0au, 0xd7610027u, 0x00011802u, 0x87ea0c6au, 0x87ea0e6au, 0xbf13806au,
    0xbf84000eu, 0xd7600002u, 0x00011927u, 0xbeea146au, 0xd760006au, 0x0000d502u, 0xd761002cu, 0x0000046au,
    0x81028102u, 0xd761002cu, 0x00000411u, 0x81028102u, 0xd7610027u, 0x00011802u, 0xbf820032u, 0x7d8a02f9u,
    0x06868ac1u, 0x7d8a0210u, 0xd7600002u, 0x00011927u, 0x7d8a02f9u, 0x06868c11u, 0x7d8a02f9u, 0x06868e12u,
    0x87ea6a0au, 0xd761002cu, 0x00000411u, 0x81028102u, 0x87ea0c6au, 0xd7610027u, 0x00011802u, 0x87ea0e6au,
    0xbf84000bu, 0xd7600002u, 0x00011927u, 0xbeea146au, 0xd760006au, 0x0000d501u, 0xd761002cu, 0x0000046au,
    0x81028102u, 0xd7610027u, 0x00011802u, 0xbf820015u, 0x7d8a00f9u, 0x068682c1u, 0x7d8a0010u, 0x7d8a00f9u,
    0x06868a11u, 0x7d8a00f9u, 0x06868c12u, 0x87ea6a02u, 0x87ea0a6au, 0x87ea0c6au, 0xbf84000au, 0xd7600002u,
    0x00011927u, 0xbeea146au, 0xd760006au, 0x0000d500u, 0xd761002cu, 0x0000046au, 0x81028102u, 0xd7610027u,
    0x00011802u, 0xd7600002u, 0x00011927u, 0x8382bc02u, 0xd7610027u, 0x00011802u, 0xbf82fef6u, 0x7d8a02f9u,
    0x06868ac1u, 0xd7600002u, 0x00011927u, 0xbeea140au, 0xd7600010u, 0x0000d500u, 0xd760000eu, 0x0000d501u,
    0x7d8a0210u, 0x7d8a02f9u, 0x06868c0eu, 0xd761002cu, 0x0000040eu, 0x81028102u, 0x87ea6a0au, 0xd7610027u,
    0x00011802u, 0x878a0c6au, 0xbf84002eu, 0xd7600002u, 0x00011927u, 0xbeea140au, 0xd760000cu, 0x0000d501u,
    0x7d8a020cu, 0xd761002cu, 0x0000040cu, 0x81028102u, 0xd7610027u, 0x00011802u, 0x87ea0a6au, 0xbf84000bu,
    0xd7600002u, 0x00011927u, 0xbeea146au, 0xd760006au, 0x0000d501u, 0xd761002cu, 0x0000046au, 0x81028102u,
    0xd7610027u, 0x00011802u, 0xbf820036u, 0x7d8a00f9u, 0x068682c1u, 0x7d8a0010u, 0x7d8a00f9u, 0x06868a0eu,
    0x7d8a00f9u, 0x06868c0cu, 0x87ea6a02u, 0x87ea0a6au, 0x87ea0c6au, 0xbf84000au, 0xd7600002u, 0x00011927u,
    0xbeea146au, 0xd760006au, 0x0000d500u, 0xd761002cu, 0x0000046au, 0x81028102u, 0xd7610027u, 0x00011802u,
    0xbf820020u, 0x7d8a00f9u, 0x068682c1u, 0x7d8a0010u, 0x7d8a00f9u, 0x06868a0eu, 0x87ea6a02u, 0x878a0a6au,
    0xbf840018u, 0xd7600002u, 0x00011927u, 0xbeea140au, 0xd760006au, 0x0000d500u, 0x7d8a00f9u, 0x06868c6au,
    0xd761002cu, 0x0000046au, 0x81028102u, 0xd7610027u, 0x00011802u, 0x87ea0a0cu, 0xbf84000au, 0xd7600002u,
    0x00011927u, 0xbeea146au, 0xd760006au, 0x0000d500u, 0xd761002cu, 0x0000046au, 0x81028102u, 0xd7610027u,
    0x00011802u, 0xd7600002u, 0x00011927u, 0x8382bc02u, 0xd7610027u, 0x00011802u, 0xbf82fe8eu, 0x7d8a00f9u,
    0x068682c1u, 0xbe9003c1u, 0xbf138002u, 0xbf840031u, 0xbeea1402u, 0xd7600010u, 0x0000d500u, 0x7d8a0010u,
    0x878a6a02u, 0xbf840026u, 0xd7600002u, 0x00011927u, 0xbeea140au, 0xd760006au, 0x0000d500u, 0x7d8a00f9u,
    0x06868c6au, 0xd761002cu, 0x0000046au, 0x81028102u, 0xd7610027u, 0x00011802u, 0x878a0a0cu, 0xbf840018u,
    0xd7600002u, 0x00011927u, 0xbeea140au, 0xd760006au, 0x0000d500u, 0x7d8a00f9u, 0x06868c6au, 0xd761002cu,
    0x0000046au, 0x81028102u, 0xd7610027u, 0x00011802u, 0x87ea0a0cu, 0xbf84000au, 0xd7600002u, 0x00011927u,
    0xbeea146au, 0xd760006au, 0x0000d500u, 0xd761002cu, 0x0000046au, 0x81028102u, 0xd7610027u, 0x00011802u,
    0xd7600002u, 0x00011927u, 0x8382bc02u, 0xd7610027u, 0x00011802u, 0xbf82fe57u, 0xd7600002u, 0x00011927u,
    0xbf076302u, 0xbf840008u, 0xd7600002u, 0x00011927u, 0x8102c102u, 0xd7600010u, 0x0000052cu, 0xd7610027u,
    0x00011802u, 0xbf82fe4au, 0xd7600014u, 0x00010527u, 0xd7600015u, 0x00010727u, 0xd7600016u, 0x00010927u,
    0xd7600017u, 0x00010b27u, 0xd760005au, 0x00011127u, 0xbefe045au, 0xbf820133u, 0x7e0002c1u, 0x7e0202c1u,
    0x7e0402c1u, 0x7e0602c1u, 0xbeea2404u, 0xbf88000au, 0x7e000212u, 0xbf800000u, 0xf1989f07u, 0x00050000u,
    0x2e2f2d2bu, 0x281e1c1du, 0x00002a29u, 0xbf8c3f70u, 0x7d8a00f9u, 0x068684c1u, 0xbf138004u, 0xbefe046au,
    0xbf840120u, 0x7d8a02c1u, 0xbf13806au, 0xbf8400e6u, 0x7d8a04c1u, 0xbf13806au, 0xbf84007bu, 0x7d8a06c1u,
    0xbf13806au, 0xbf840019u, 0x7d8a06c1u, 0xd7600002u, 0x00011927u, 0xbeea146au, 0xd760006bu, 0x0000d503u,
    0xd7600006u, 0x0000d502u, 0xd7600003u, 0x0000d501u, 0xd7600012u, 0x0000d500u, 0xd761002cu, 0x0000046bu,
    0x81028102u, 0xd761002cu, 0x00000406u, 0x81028102u, 0xd761002cu, 0x00000403u, 0x81028102u, 0x8382bc02u,
    0xd7610027u, 0x00011802u, 0xbf82fda0u, 0x7d8a04f9u, 0x068686c1u, 0xd7600002u, 0x00011927u, 0xbeea1406u,
    0xd7600012u, 0x0000d500u, 0xd760000cu, 0x0000d501u, 0xd760000du, 0x0000d502u, 0x7d8a0412u, 0x7d8a04f9u,
    0x0686880cu, 0x7d8a04f9u, 0x06868a0du, 0xd761002cu, 0x0000040du, 0x81028102u, 0x87ea6a06u, 0xd7610027u,
    0x00011802u, 0x87ea086au, 0x87ea0a6au, 0xbf84000eu, 0xd7600002u, 0x00011927u, 0xbeea146au, 0xd760006au,
    0x0000d502u, 0xd761002cu, 0x0000046au, 0x81028102u, 0xd761002cu, 0x0000040cu, 0x81028102u, 0xd7610027u,
    0x00011802u, 0xbf820032u, 0x7d8a02f9u, 0x068686c1u, 0x7d8a0212u, 0xd7600002u, 0x00011927u, 0x7d8a02f9u,
    0x0686880cu, 0x7d8a02f9u, 0x06868a0du, 0x87ea6a06u, 0xd761002cu, 0x0000040cu, 0x81028102u, 0x87ea086au,
    0xd7610027u, 0x00011802u, 0x87ea0a6au, 0xbf84000bu, 0xd7600002u, 0x00011927u, 0xbeea146au, 0xd760006au,
    0x0000d501u, 0xd761002cu, 0x0000046au, 0x81028102u, 0xd7610027u, 0x00011802u, 0xbf820015u, 0x7d8a00f9u,
    0x068682c1u, 0x7d8a0012u, 0x7d8a00f9u, 0x0686860cu, 0x7d8a00f9u, 0x0686880du, 0x87ea6a02u, 0x87ea066au,
    0x87ea086au, 0xbf84000au, 0xd7600002u, 0x00011927u, 0xbeea146au, 0xd760006au, 0x0000d500u, 0xd761002cu,
    0x0000046au, 0x81028102u, 0xd7610027u, 0x00011802u, 0xd7600002u, 0x00011927u, 0x8382bc02u, 0xd7610027u,
    0x00011802u, 0xbf82fd41u, 0x7d8a02f9u, 0x068686c1u, 0xd7600002u, 0x00011927u, 0xbeea1406u, 0xd7600012u,
    0x0000d500u, 0xd760000au, 0x0000d501u, 0x7d8a0212u, 0x7d8a02f9u, 0x0686880au, 0xd761002cu, 0x0000040au,
    0x81028102u, 0x87ea6a06u, 0xd7610027u, 0x00011802u, 0x8786086au, 0xbf84002eu, 0xd7600002u, 0x00011927u,
    0xbeea1406u, 0xd7600008u, 0x0000d501u, 0x7d8a0208u, 0xd761002cu, 0x00000408u, 0x81028102u, 0xd7610027u,
    0x00011802u, 0x87ea066au, 0xbf84000bu, 0xd7600002u, 0x00011927u, 0xbeea146au, 0xd760006au, 0x0000d501u,
    0xd761002cu, 0x0000046au, 0x81028102u, 0xd7610027u, 0x00011802u, 0xbf820036u, 0x7d8a00f9u, 0x068682c1u,
    0x7d8a0012u, 0x7d8a00f9u, 0x0686860au, 0x7d8a00f9u, 0x06868808u, 0x87ea6a02u, 0x87ea066au, 0x87ea086au,
    0xbf84000au, 0xd7600002u, 0x00011927u, 0xbeea146au, 0xd760006au, 0x0000d500u, 0xd761002cu, 0x0000046au,
    0x81028102u, 0xd7610027u, 0x00011802u, 0xbf820020u, 0x7d8a00f9u, 0x068682c1u, 0x7d8a0012u, 0x7d8a00f9u,
    0x0686860au, 0x87ea6a02u, 0x8786066au, 0xbf840018u, 0xd7600002u, 0x00011927u, 0xbeea1406u, 0xd760006au,
    0x0000d500u, 0x7d8a00f9u, 0x0686886au, 0xd761002cu, 0x0000046au, 0x81028102u, 0xd7610027u, 0x00011802u,
    0x87ea0608u, 0xbf84000au, 0xd7600002u, 0x00011927u, 0xbeea146au, 0xd760006au, 0x0000d500u, 0xd761002cu,
    0x0000046au, 0x81028102u, 0xd7610027u, 0x00011802u, 0xd7600002u, 0x00011927u, 0x8382bc02u, 0xd7610027u,
    0x00011802u, 0xbf82fcd9u, 0x7d8a00f9u, 0x068682c1u, 0xbe9203c1u, 0xbf138002u, 0xbf840031u, 0xbeea1402u,
    0xd7600012u, 0x0000d500u, 0x7d8a0012u, 0x87866a02u, 0xbf840026u, 0xd7600002u, 0x00011927u, 0xbeea1406u,
    0xd760006au, 0x0000d500u, 0x7d8a00f9u, 0x0686886au, 0xd761002cu, 0x0000046au, 0x81028102u, 0xd7610027u,
    0x00011802u, 0x87860608u, 0xbf840018u, 0xd7600002u, 0x00011927u, 0xbeea1406u, 0xd760006au, 0x0000d500u,
    0x7d8a00f9u, 0x0686886au, 0xd761002cu, 0x0000046au, 0x81028102u, 0xd7610027u, 0x00011802u, 0x87ea0608u,
    0xbf84000au, 0xd7600002u, 0x00011927u, 0xbeea146au, 0xd760006au, 0x0000d500u, 0xd761002cu, 0x0000046au,
    0x81028102u, 0xd7610027u, 0x00011802u, 0xd7600002u, 0x00011927u, 0x8382bc02u, 0xd7610027u, 0x00011802u,
    0xbf82fca2u, 0xd7600002u, 0x00011927u, 0xbf070280u, 0xbf840008u, 0xd7600002u, 0x00011927u, 0x8102c102u,
    0xd7600012u, 0x0000052cu, 0xd7610027u, 0x00011802u, 0xbf82fc95u, 0xbe82047eu, 0x7daa4ac1u, 0xbf88001fu,
    0xdc348030u, 0x017d0018u, 0xdc308078u, 0x057d0018u, 0xd5480006u, 0x02710726u, 0x36084affu, 0x3fffffffu,
    0xbf8c3f71u, 0x360404ffu, 0x003fffffu, 0x34000283u, 0xd54e0001u, 0x02760302u, 0xdc30807cu, 0x077d0000u,
    0xdc348028u, 0x027d0000u, 0xbf8c3f71u, 0x4c0c0f06u, 0xbf8c3f70u, 0xd70f6a00u, 0x00020102u, 0x34040c82u,
    0x50020303u, 0x2c060c9eu, 0xd70f6a00u, 0x00020500u, 0x50020701u, 0xdc308000u, 0x007d0000u, 0x8afe7e02u,
    0xbf880001u, 0x7e0802c1u, 0xbefe0402u, 0x7daa08c1u, 0xbf880018u, 0xbf8c3f70u, 0x4a000105u, 0xf4080100u,
    0xfa000050u, 0x100248ffu, 0x47800000u, 0x100446ffu, 0x47800000u, 0x7c023680u, 0xbf8cc07fu, 0xe0302000u,
    0x80011800u, 0xbe880b81u, 0x1e0202ffu, 0x477fff00u, 0x1e0604ffu, 0x477fff00u, 0x020100f9u, 0x86860608u,
    0x7e040f01u, 0x7e060f03u, 0x38020104u, 0xd76f0000u, 0x04092103u, 0x8afe7e02u, 0xbf88000bu, 0x7c023680u,
    0x7e0214f9u, 0x0006061eu, 0xbf8c3f70u, 0x5e00391du, 0x7e3002ffu, 0x20fffffeu, 0xbe840b81u, 0x020500f9u,
    0x86860604u, 0x38020501u, 0xbefe0402u, 0xf4100100u, 0xfa000010u, 0x7c023680u, 0x7d8a42f9u, 0x06868280u,
    0x7d8a3ef9u, 0x06869480u, 0xbf8cc07fu, 0xf0200308u, 0x00010016u, 0x0203e4f9u, 0x86860680u, 0xd5010000u,
    0x0009e480u, 0xd5010001u, 0x00520280u, 0x10060122u, 0x06000300u, 0x3e064101u, 0x7e045503u, 0x7c020680u,
    0x10000102u, 0x02000080u, 0x10020122u, 0x10000120u, 0xd5010001u, 0x000a0280u, 0xd5010002u, 0x00520080u,
    0x1e0002ffu, 0x477de800u, 0x1e0204ffu, 0x477de800u, 0xf0200308u, 0x00030016u, 0xbefe04c1u, 0xf4080100u,
    0xfa000000u, 0xd76f0019u, 0x04592117u, 0xbf8c0070u, 0xe0742000u, 0x8001181au, 0xbf810000u, 0xbf810000u,
};

void TestRayTracingClosestLeafCalls() {
  namespace D = ShaderRecompiler::Decoder;
  // User data s[0:1] point at the shader's table of V#s; the call table's V# (at +0x120, loaded
  // before the traversal loop) has 4 records of 16 bytes: two any-hit functions (one accepts the
  // hit, v7 = 0, the other returns at once) and two null slots.
  constexpr uint64_t Tables = 0x10000000u, CallTable = 0x20000000u, Functions = 0x30000000u;
  const std::array<uint32_t, 4> vsharp{static_cast<uint32_t>(CallTable), 16u << 16u, 4u, 0u};
  const auto read = [&](uint64_t address, std::span<uint32_t> words) {
    for (size_t i = 0; i < words.size(); i++) {
      const uint64_t at = address + i * 4u;
      if (at >= Tables && at < Tables + 0x1000u) {
        words[i] = vsharp[(at / 4u) % 4u];
      } else if (at >= CallTable && at < CallTable + 64u) {
        const uint64_t slot = (at - CallTable) / 16u, word = ((at - CallTable) % 16u) / 4u;
        const uint64_t target = slot < 2u ? Functions + slot * 0x100u : 0u;
        words[i] = word == 0u ? static_cast<uint32_t>(target) : word == 1u ? static_cast<uint32_t>(target >> 32u) : 0u;
      } else if (at >= Functions && at < Functions + 0x200u) {
        const uint64_t word = (at - Functions) / 4u;
        words[i] = word == 0u ? 0x7e0e0280u   // v_mov_b32 v7, 0
                   : word == 1u || word == 0x40u ? 0xbe80200eu // s_setpc_b64 s[14:15]
                                                  : 0xbf800000u; // s_nop 0
      } else {
        return false;
      }
    }
    return true;
  };
  std::vector<uint32_t> user_data(16, 0u);
  user_data[0] = static_cast<uint32_t>(Tables);
  D::ShaderFunctionExpander expander;
  std::vector<uint32_t> expanded;
  std::string reason;
  Check(expander.Expand(kGt7ClosestLeafWave64, 0x400000000ull, user_data, read, expanded, reason, 64u) &&
            !expanded.empty() && expander.Bodies().size() == 2u,
        ("the call table loaded before the traversal loop did not expand: " + reason).c_str());
  D::Program program;
  D::DecodeProgram(expanded, program, expander.LongBranches());
  program.code = expanded;
  program.call_bodies = expander.Bodies();
  program.runtime_literals = expander.RuntimeLiterals();
  D::RayTracingPatch patch;
  Check(D::DeriveRayTracingPatch(program, patch, 64u) && patch.sites.size() == 1u,
        "the closest hit traversal with any-hit calls was not recognized");
  const auto &site = patch.sites[0];
  Check(site.kind == D::RayTracingKind::ClosestLeaf && patch.user_sgpr == 0 && patch.header_offset == 0x118 &&
            patch.root == 0x25 && site.t_max == 43 && site.result_t == 43 && site.result_word == 37 &&
            site.result_instance == 24 && site.result_u == 36 && site.result_v == 35 && site.result_node == 38,
        "the closest hit traversal with any-hit calls has the wrong TLAS source or hit registers");
  const auto &layout = patch.layout;
  Check(layout.instance_offset == 0x30 && layout.record_stride == 128 && layout.blas_dword == 12 && layout.blas_packed &&
            layout.root_dword == 14 && layout.mask_dword == 15 && layout.mask_shift == 24 && layout.flags_dword == 13 &&
            layout.flags_shift == 22 && layout.inline_type == 7 && layout.inline_offset == 0x80,
        "the closest hit traversal with any-hit calls has the wrong instance record layout");
  const auto bodies = program.call_bodies.size();
  const auto literals = program.runtime_literals.size();
  Check(D::ApplyRayTracingPatch(patch, program), "the closest hit traversal with any-hit calls was not patched");
  // The leaf code (with the inlined calls) runs in the native loop too: its bodies and literals
  // follow the copies.
  Check(program.call_bodies.size() >= bodies && program.runtime_literals.size() >= literals &&
            std::ranges::is_sorted(program.runtime_literals, {}, &D::RuntimeLiteral::pc),
        "the patch lost the inlined calls' bodies or runtime literals");
  for (const auto &literal : program.runtime_literals) {
    Check(std::ranges::find(program.instructions, literal.pc, &D::Instruction::pc) != program.instructions.end(),
          "a runtime literal of the patched program names no instruction");
  }
  auto options = MakeCompileOptions(ShaderType::Compute);
  options.wave_size = 64;
  options.native_ray_tracing = true;
  auto translated = ShaderRecompiler::TranslateProgram(program, options);
  const auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
  ShaderRecompiler::IR::ResourceSpecialization specialization;
  specialization.buffers.assign(plan.info.buffers.size(), {});
  specialization.images.assign(plan.info.images.size(), {});
  for (size_t i = 0; i < plan.info.images.size(); i++) {
    specialization.images[i].numeric_class = Prospero::TextureNumericClass::Float;
    specialization.images[i].dimension = plan.info.images[i].dimension != D::ImageDimension::Unknown
                                             ? plan.info.images[i].dimension
                                             : D::ImageDimension::Dim2D;
  }
  specialization.samplers.assign(plan.info.samplers.size(), {});
  const auto compiled = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
  Check(compiled.program.info.ray_query && compiled.program.info.ray_query_sites == 1 &&
            compiled.program.info.ray_query_layout == layout.Pack() &&
            SpirvInstructionOpcodeCount(compiled.spirv, 4473u) != 0,
        "the patched closest hit traversal with any-hit calls does not trace with ray queries");
  CheckSpirvBinaryValidates(compiled.spirv);
}

void TestRayTracingUniformOrigin() {
  namespace D = ShaderRecompiler::Decoder;
  const auto derive = [](std::span<const uint32_t> code, D::RayTracingPatch &patch) {
    D::Program decoded;
    D::DecodeProgram(code, decoded);
    return D::DeriveRayTracingPatch(decoded, patch, 64);
  };
  struct Case {
    std::span<const uint32_t> code;
    std::array<uint32_t, 3> origin;
    std::array<uint32_t, 4> layout;
  };
  for (const auto &test : {Case{kClosestUniformOrigin, {20, 21, 22}, {0x76543210u, 0x1dc6ba98u, 0x000dbb8fu, 0x000f2510u}},
                           Case{kClosestUniformOriginPlayroom, {12, 13, 14}, {0xd951c840u, 0x4c25ea62u, 0x30482080u, 0x308f2511u}}}) {
    D::RayTracingPatch patch;
    Check(derive(test.code, patch) && patch.sites.size() == 1, "closest hit traversal of uniform rays was not recognized");
    const auto &site = patch.sites[0];
    Check(std::ranges::equal(site.origin_sgpr, test.origin) &&
              std::ranges::all_of(site.origin, [](uint32_t reg) { return reg == D::NoRegister; }) &&
              patch.user_sgpr == 0 && patch.header_offset == 0x28 && patch.root == 37 && patch.layout.Pack() == test.layout,
          "closest hit traversal of uniform rays has the wrong origin or layout");
    auto options = MakeCompileOptions(ShaderType::Compute);
    options.native_ray_tracing = true;
    auto translated = ShaderRecompiler::TranslateProgram(test.code, options);
    const auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
    ShaderRecompiler::IR::ResourceSpecialization specialization;
    specialization.buffers.assign(plan.info.buffers.size(), {});
    specialization.images.assign(plan.info.images.size(), {});
    specialization.samplers.assign(plan.info.samplers.size(), {});
    const auto compiled = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
    Check(compiled.program.info.ray_query && compiled.program.info.ray_query_sites == 1 &&
              SpirvInstructionOpcodeCount(compiled.spirv, 4473u) != 0,
          "patched traversal of uniform rays does not trace with ray queries");
    CheckSpirvBinaryValidates(compiled.spirv);
  }
  // A copy into the box test from an SGPR the loop writes (the instance record's s4) is no uniform origin.
  std::vector<uint32_t> rewritten(std::begin(kClosestUniformOrigin), std::end(kClosestUniformOrigin));
  const auto copy = std::ranges::find(rewritten, 0x7e020214u); // v_mov_b32 v1, s20
  Check(copy != rewritten.end(), "uniform origin test program changed");
  *copy = 0x7e020204u; // v_mov_b32 v1, s4
  D::RayTracingPatch patch;
  Check(!derive(rewritten, patch), "an origin SGPR the traversal loop writes was taken as the ray origin");
  // The user data pair the TLAS header is loaded through may be written in another case of a switch
  // (a path that does not reach the load), but not on the path to it.
  for (const bool reaches : {false, true}) {
    std::vector<uint32_t> code {0xbf068005u,                              // s_cmp_eq_u32 s5, 0
                                EncodeSopp(0x05, reaches ? 1u : 2u),      // s_cbranch_scc1 <load>
                                0xbe800412u};                             // s_mov_b64 s[0:1], s[18:19]
    if (!reaches) {
      // s_branch to the program's end.
      code.push_back(EncodeSopp(0x02, static_cast<uint32_t>(std::size(kClosestUniformOrigin) - 2)));
    }
    code.insert(code.end(), std::begin(kClosestUniformOrigin), std::end(kClosestUniformOrigin));
    Check(derive(code, patch) == !reaches, reaches ? "a TLAS header loaded through a pair written before was recognized"
                                                    : "a write of the header's user data pair on another path hid the traversal");
  }
}

void TestRayTracingInstructions() {
  TestTrinityBvhInstruction();
  TestTrinityStackInstruction();
  TestRayTracingSiteWaveSizes();
  TestRayTracingClosestLeafCalls();
  TestRayTracingUniformOrigin();
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
