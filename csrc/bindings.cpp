// bindings.cpp: host 函数注册进 Python
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> quant_int4(torch::Tensor x);                                  // quant_int4.cu
torch::Tensor w4a4_gemm(torch::Tensor xq, torch::Tensor xs,
                        torch::Tensor wq, torch::Tensor ws);                             // gemm_int4.cu (v0)
torch::Tensor w4a4_gemm_v1(torch::Tensor xq, torch::Tensor xs,
                           torch::Tensor wq, torch::Tensor ws);                          // gemm_wmma.cu (v1)
torch::Tensor quant_int4_tensor_scale(torch::Tensor x, double scale);              // fast_path.cu
torch::Tensor quant_int4_tensor_scale_dev(torch::Tensor x, torch::Tensor inv_scale);     // fast_path.cu (graph 化)
torch::Tensor w4a4_gemm_cutlass(torch::Tensor xq, torch::Tensor wq, double alpha);       // gemm_cutlass_int4.cu
torch::Tensor w4a4_gemm_cutlass_dev(torch::Tensor xq, torch::Tensor wq, torch::Tensor alpha); // gemm_cutlass_int4.cu (graph 化)
torch::Tensor w4a4_gemm_v2(torch::Tensor xq, torch::Tensor xs,
                           torch::Tensor wq, torch::Tensor ws);                          // gemm_wmma2.cu (v2)
torch::Tensor w4a4_gemv(torch::Tensor xq, torch::Tensor xs,
                        torch::Tensor wq, torch::Tensor ws);                    // gemv_int4.cu (decode)
torch::Tensor w4a4_gemv_multi(torch::Tensor xq, torch::Tensor xs,
                              torch::Tensor wq, torch::Tensor ws);              // gemv_int4.cu (decode, M<=8)
torch::Tensor w4a4_gemm_s4(torch::Tensor xq, torch::Tensor xs,
                           torch::Tensor wq, torch::Tensor ws);             // gemm_s4.cu (主线)
torch::Tensor s4_atom_selftest(torch::Tensor a, torch::Tensor b);           // gemm_s4.cu
std::vector<torch::Tensor> fused_transform_quant_int4_smallm(torch::Tensor x, torch::Tensor t); // fused_tq_smallm.cu
std::vector<torch::Tensor> fused_transform_quant_int4_v3(torch::Tensor x, torch::Tensor t);  // fused_tq_v3.cu (纯寄存器)
std::vector<torch::Tensor> fused_transform_quant_int4(torch::Tensor x, torch::Tensor t);                               // fused_tq_int4.cu (v0)
std::vector<torch::Tensor> fused_transform_quant_int4_wmma(torch::Tensor x, torch::Tensor t);          // fused_tq_wmma.cu (v1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant_int4_tensor_scale", &quant_int4_tensor_scale,
          "Per-tensor absmax INT4 quantize (fast path)",
          py::arg("x"), py::arg("scale"));
    m.def("quant_int4_tensor_scale_dev", &quant_int4_tensor_scale_dev,
          "Per-tensor INT4 quantize, inv_scale from 1-elem fp32 device tensor (graph-safe)",
          py::arg("x"), py::arg("inv_scale"));
    m.def("quant_int4", &quant_int4, "INT4 symmetric group quantize (G=32, fp16 scale)",
          py::arg("x"));
    m.def("w4a4_gemm", &w4a4_gemm, "W4A4 INT4 matmul v0 (dequant + fp32 FMA)",
          py::arg("xq"), py::arg("xs"), py::arg("wq"), py::arg("ws"));
    m.def("w4a4_gemm_v1", &w4a4_gemm_v1, "W4A4 INT4 matmul v1 (WMMA fp16 tensor core)",
          py::arg("xq"), py::arg("xs"), py::arg("wq"), py::arg("ws"));
    m.def("w4a4_gemm_cutlass", &w4a4_gemm_cutlass,
          "W4A4 INT4 matmul via CUTLASS s4 tensor core (per-tensor scale)",
          py::arg("xq"), py::arg("wq"), py::arg("alpha"));
    m.def("w4a4_gemm_cutlass_dev", &w4a4_gemm_cutlass_dev,
          "CUTLASS s4 W4A4 matmul, alpha from 1-elem fp32 device tensor (graph-safe)",
          py::arg("xq"), py::arg("wq"), py::arg("alpha"));
    m.def("w4a4_gemm_s4", &w4a4_gemm_s4, "W4A4 INT4 matmul (native s4 tensor core + per-group scale)",
          py::arg("xq"), py::arg("xs"), py::arg("wq"), py::arg("ws"));
    m.def("s4_atom_selftest", &s4_atom_selftest, "verify m8n8k32 s4 fragment layout");
    m.def("w4a4_gemv", &w4a4_gemv, "W4A4 INT4 gemv (M=1 decode path)",
          py::arg("xq"), py::arg("xs"), py::arg("wq"), py::arg("ws"));
    m.def("w4a4_gemv_multi", &w4a4_gemv_multi, "W4A4 INT4 gemv, 1<=M<=8 (batched decode)",
          py::arg("xq"), py::arg("xs"), py::arg("wq"), py::arg("ws"));
    m.def("fused_transform_quant_int4_v3", &fused_transform_quant_int4_v3,
          "Fused WUSH transform + INT4 quantize (v3 pure-register)",
          py::arg("x"), py::arg("t"));
    m.def("fused_transform_quant_int4_smallm", &fused_transform_quant_int4_smallm,
          "Fused WUSH transform + INT4 quantize (M<=8 decode path)",
          py::arg("x"), py::arg("t"));
    m.def("fused_transform_quant_int4", &fused_transform_quant_int4,
          "Fused WUSH transform + INT4 quantize (per-block T)",
          py::arg("x"), py::arg("t"));
    m.def("w4a4_gemm_v2", &w4a4_gemm_v2, "W4A4 INT4 matmul v2 (WMMA, 2-group stage, vectorized)",
          py::arg("xq"), py::arg("xs"), py::arg("wq"), py::arg("ws"));
    m.def("fused_transform_quant_int4_wmma", &fused_transform_quant_int4_wmma,
          "Fused WUSH transform + INT4 quantize (WMMA tensor core)",
          py::arg("x"), py::arg("t"));
}
