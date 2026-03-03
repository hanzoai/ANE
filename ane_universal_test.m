// ane_universal_test.m — Test the universal ANE runtime on any Apple Silicon
// Demonstrates auto-detection and fallback between M4+ and M1-M3 paths
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <mach/mach_time.h>
#include <sys/sysctl.h>
#include "ane_universal.h"

static mach_timebase_info_data_t g_tb;
static double ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static NSString *get_chip_name(void) {
    char buf[256] = {0};
    size_t len = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, NULL, 0) == 0)
        return [NSString stringWithUTF8String:buf];
    return @"Unknown";
}

// Generate CoreML mlpackage via coremltools (Python) for M1-M3 path
static BOOL generate_coreml_model(int ch_in, int ch_out, int seq, const char *label) {
    NSString *script = [NSString stringWithFormat:
        @"import sys; sys.path.insert(0, '.venv/lib/python3.12/site-packages')\n"
        "import coremltools as ct\n"
        "import numpy as np\n"
        "from coremltools.converters.mil import Builder as mb\n"
        "@mb.program(input_specs=[mb.TensorSpec(shape=(1, %d, 1, %d))])\n"
        "def prog(x):\n"
        "    W = mb.const(val=np.full((%d, %d, 1, 1), 0.25, dtype=np.float16), name='W')\n"
        "    x_fp16 = mb.cast(x=x, dtype='fp16', name='x_fp16')\n"
        "    y = mb.conv(x=x_fp16, weight=W, strides=[1, 1], pad_type='valid', name='out')\n"
        "    return y\n"
        "model = ct.convert(prog, compute_units=ct.ComputeUnit.CPU_AND_NE,\n"
        "    minimum_deployment_target=ct.target.macOS15, convert_to='mlprogram')\n"
        "model.save('/tmp/ane_u_%s.mlpackage')\n"
        "print('OK')\n",
        ch_in, seq, ch_out, ch_in, label];

    NSString *tmpScript = [NSString stringWithFormat:@"/tmp/ane_gen_%s.py", label];
    [script writeToFile:tmpScript atomically:YES encoding:NSUTF8StringEncoding error:nil];

    NSTask *task = [[NSTask alloc] init];
    task.launchPath = @".venv/bin/python";
    task.arguments = @[tmpScript];
    task.currentDirectoryURL = [NSURL fileURLWithPath:@"/Users/z/work/hanzo/ANE"];
    NSPipe *pipe = [NSPipe pipe];
    task.standardOutput = pipe;
    task.standardError = pipe;
    [task launch];
    [task waitUntilExit];
    return task.terminationStatus == 0;
}

int main() {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        printf("╔═══════════════════════════════════════════════╗\n");
        printf("║  Universal ANE Runtime Test                    ║\n");
        printf("╚═══════════════════════════════════════════════╝\n\n");

        printf("Chip: %s\n", [get_chip_name() UTF8String]);
        printf("macOS: %s\n", [[[NSProcessInfo processInfo] operatingSystemVersionString] UTF8String]);

        // Initialize — auto-detects M1-M3 vs M4+
        printf("\nDetecting ANE backend...\n");
        ane_universal_init();
        printf("Backend: %s\n\n", ane_backend_name());

        if (g_ane_backend == ANE_BACKEND_INMEM) {
            // ============================================================
            // M4+ path: test with in-memory MIL compilation
            // ============================================================
            printf("=== M4+ In-Memory ANE Test ===\n");
            int C = 256, S = 64;
            size_t wSize = 64 + C * C * 2;
            NSMutableData *wd = [NSMutableData dataWithLength:wSize];
            uint16_t *wp = (uint16_t *)(wd.mutableBytes + 64);
            for (int i = 0; i < C * C; i++) wp[i] = 0x3400; // 0.25

            NSString *mil = [NSString stringWithFormat:
                @"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-version\", \"3520.5.1\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n"
                "{\n"
                "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
                "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
                "        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string(\"out\")];\n"
                "    } -> (out);\n"
                "}\n",
                C, S, C, C, C, C, C, S];

            NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
            size_t inBytes = C * S * 2;
            size_t outBytes = C * S * 2;

            uint64_t t0 = mach_absolute_time();
            UANEKernel *k = ane_compile_inmem(milData, wd, 1, &inBytes, 1, &outBytes);
            printf("  Compile: %.1f ms\n", ms(mach_absolute_time() - t0));

            if (k) {
                // Fill input
                size_t nElems = inBytes / 2;
                uint16_t *inp = malloc(inBytes);
                for (size_t i = 0; i < nElems; i++) inp[i] = 0x3C00; // 1.0
                ane_universal_write_input(k, 0, inp, inBytes);

                // Warmup + benchmark
                for (int i = 0; i < 10; i++) ane_universal_eval(k);
                int iters = 100;
                t0 = mach_absolute_time();
                for (int i = 0; i < iters; i++) ane_universal_eval(k);
                double avgMs = ms(mach_absolute_time() - t0) / iters;
                double gflops = 2.0 * C * C * S / 1e9;
                printf("  Eval: %.3f ms, %.2f TFLOPS\n", avgMs, gflops / avgMs);

                // Check output
                uint16_t *out = malloc(outBytes);
                ane_universal_read_output(k, 0, out, outBytes);
                printf("  Output[0..3]: 0x%04x 0x%04x 0x%04x 0x%04x\n",
                       out[0], out[1], out[2], out[3]);

                free(inp); free(out);
                ane_universal_free(k);
            }

        } else if (g_ane_backend == ANE_BACKEND_COREML) {
            // ============================================================
            // M1-M3 path: test with CoreML compiled models
            // ============================================================
            printf("=== M1-M3 CoreML ANE Test ===\n\n");

            struct { int ci, co, seq; const char *label; } configs[] = {
                {64, 64, 16, "tiny"},
                {256, 256, 64, "small"},
                {768, 768, 64, "medium"},
            };

            for (int c = 0; c < 3; c++) {
                int ci = configs[c].ci, co = configs[c].co, seq = configs[c].seq;
                const char *label = configs[c].label;

                printf("--- Config: %s (%dx%d, seq=%d) ---\n", label, ci, co, seq);

                // Generate model via Python/coremltools
                NSString *pkgPath = [NSString stringWithFormat:@"/tmp/ane_u_%s.mlpackage", label];
                if (![[NSFileManager defaultManager] fileExistsAtPath:pkgPath]) {
                    printf("  Generating CoreML model...\n");
                    if (!generate_coreml_model(ci, co, seq, label)) {
                        printf("  FAILED to generate model (need: .venv/bin/python + coremltools)\n");
                        continue;
                    }
                }
                printf("  Model: %s\n", [pkgPath UTF8String]);

                // Compile via CoreML
                NSError *e = nil;
                uint64_t t0 = mach_absolute_time();
                NSURL *compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:pkgPath] error:&e];
                double compileMs = ms(mach_absolute_time() - t0);
                if (!compiled) {
                    printf("  Compile FAILED: %s\n", [[e description] UTF8String]);
                    continue;
                }
                printf("  CoreML compile: %.1f ms\n", compileMs);

                // Load with ANE
                MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
                config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
                MLModel *model = [MLModel modelWithContentsOfURL:compiled configuration:config error:&e];
                if (!model) {
                    printf("  Load FAILED\n");
                    continue;
                }

                // Get I/O info
                MLModelDescription *desc = model.modelDescription;
                NSString *inputName = desc.inputDescriptionsByName.allKeys.firstObject;
                NSString *outputName = desc.outputDescriptionsByName.allKeys.firstObject;

                // Create input
                MLMultiArray *input = [[MLMultiArray alloc]
                    initWithShape:@[@1, @(ci), @1, @(seq)]
                    dataType:MLMultiArrayDataTypeFloat32 error:&e];
                float *fp = (float *)input.dataPointer;
                for (int i = 0; i < ci * seq; i++) fp[i] = 1.0f;

                MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc]
                    initWithDictionary:@{inputName: input} error:&e];

                // Warmup
                for (int i = 0; i < 10; i++)
                    [model predictionFromFeatures:provider error:&e];

                // Benchmark ANE
                int iters = 200;
                t0 = mach_absolute_time();
                for (int i = 0; i < iters; i++)
                    [model predictionFromFeatures:provider error:&e];
                double aneMs = ms(mach_absolute_time() - t0) / iters;

                // Benchmark CPU
                config.computeUnits = MLComputeUnitsCPUOnly;
                MLModel *cpuModel = [MLModel modelWithContentsOfURL:compiled configuration:config error:&e];
                for (int i = 0; i < 10; i++)
                    [cpuModel predictionFromFeatures:provider error:&e];
                t0 = mach_absolute_time();
                for (int i = 0; i < iters; i++)
                    [cpuModel predictionFromFeatures:provider error:&e];
                double cpuMs = ms(mach_absolute_time() - t0) / iters;

                double gflops = 2.0 * ci * co * seq / 1e9;
                printf("  ANE:  %.3f ms  (%.2f TFLOPS)\n", aneMs, gflops / aneMs);
                printf("  CPU:  %.3f ms  (%.2f TFLOPS)\n", cpuMs, gflops / cpuMs);
                printf("  Speedup: %.2fx\n\n", cpuMs / aneMs);

                // Read output to verify correctness
                id<MLFeatureProvider> result = [model predictionFromFeatures:provider error:&e];
                MLMultiArray *outArr = [result featureValueForName:outputName].multiArrayValue;
                if (outArr) {
                    float *op = (float *)outArr.dataPointer;
                    printf("  Output[0..3]: %.4f %.4f %.4f %.4f\n", op[0], op[1], op[2], op[3]);
                }

                [[NSFileManager defaultManager] removeItemAtPath:[compiled path] error:nil];
                printf("\n");
            }
        }

        // ============================================================
        // Summary
        // ============================================================
        printf("╔═══════════════════════════════════════════════╗\n");
        printf("║  Results                                       ║\n");
        printf("╚═══════════════════════════════════════════════╝\n");
        printf("  Chip: %s\n", [get_chip_name() UTF8String]);
        printf("  Backend: %s\n", ane_backend_name());
        if (g_ane_backend == ANE_BACKEND_INMEM)
            printf("  Status: Full ANE training support via private API\n");
        else
            printf("  Status: ANE inference via CoreML (training via MLX recommended)\n");
    }
    return 0;
}
