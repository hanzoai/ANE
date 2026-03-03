// ane_m1_multiweight.m — Test separate weight files (the approach that works on M1)
// Key insight from upstream issue #3: M1 ANE compiler rejects single-blob offset references
// but accepts separate weight files per matrix (wq.bin, wk.bin, etc.)
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double tms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static Class g_D, g_I, g_R, g_AIO;

static void init_ane(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D   = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I   = NSClassFromString(@"_ANEInMemoryModel");
    g_R   = NSClassFromString(@"_ANERequest");
    g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
}

// Build ANE weight blob: 128-byte header + fp16 data
static NSData *build_blob_const(float val, int rows, int cols) {
    int ws = rows * cols * 2;
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);
    for (int i = 0; i < rows * cols; i++) fp16[i] = (_Float16)val;
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

typedef struct {
    id model;
    IOSurfaceRef ioIn, ioOut;
    id request;
    NSString *tmpDir;
} Kern;

// Compile with separate weight files (the M1-compatible approach)
static Kern *compile_multi_weight(NSString *mil, NSDictionary *weights,
                                   int inBytes, int outBytes, const char *label) {
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_D, @selector(modelWithMILText:weights:optionsPlist:), md, weights, nil);
    if (!desc) { printf("  %-30s DESC=nil\n", label); return NULL; }

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_I, @selector(inMemoryModelWithDescriptor:), desc);
    if (!mdl) { printf("  %-30s MODEL=nil\n", label); return NULL; }

    // Setup temp dir with SEPARATE weight files (critical for M1!)
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

    // Write each weight as its own file
    for (NSString *path in weights) {
        NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        NSString *fullPath = [td stringByAppendingPathComponent:rel];
        NSString *dir = [fullPath stringByDeletingLastPathComponent];
        [fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
        [weights[path][@"data"] writeToFile:fullPath atomically:YES];
    }

    NSError *e = nil;
    uint64_t t0 = mach_absolute_time();
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    double compileMs = tms(mach_absolute_time() - t0);

    if (!ok) {
        NSString *errStr = e ? [e localizedDescription] : @"unknown";
        if ([e userInfo][@"NSUnderlyingError"])
            errStr = [[e userInfo][@"NSUnderlyingError"] localizedDescription];
        printf("  %-30s COMPILE FAIL (%.1fms): %s\n", label, compileMs,
               [errStr UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return NULL;
    }

    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        printf("  %-30s LOAD FAIL\n", label);
        [fm removeItemAtPath:td error:nil];
        return NULL;
    }

    Kern *k = (Kern *)calloc(1, sizeof(Kern));
    k->model = mdl;
    k->tmpDir = td;
    k->ioIn = make_surface(inBytes);
    k->ioOut = make_surface(outBytes);
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
        g_AIO, @selector(objectWithIOSurface:), k->ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
        g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_R,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

    printf("  %-30s OK! compile=%.1fms\n", label, compileMs);
    return k;
}

static BOOL ane_eval(Kern *k) {
    NSError *e = nil;
    return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        k->model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, k->request, &e);
}

static void free_kern(Kern *k) {
    if (!k) return;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
        k->model, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(k->ioIn);
    CFRelease(k->ioOut);
    [[NSFileManager defaultManager] removeItemAtPath:k->tmpDir error:nil];
    free(k);
}

int main() {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        init_ane();

        printf("=== ANE Multi-Weight Test (M1-Compatible Approach) ===\n\n");

        if (!g_D || !g_I) {
            printf("FATAL: ANE classes not found\n");
            return 1;
        }

        // ============================================================
        // Test 1: Single weight file (known to fail on M1)
        // ============================================================
        printf("--- Test 1: Single weight file (baseline, fails on M1) ---\n");
        {
            int C = 256, S = 64;
            NSData *wdata = build_blob_const(0.25f, C, C);
            NSDictionary *weights = @{
                @"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wdata}
            };
            NSString *mil = [NSString stringWithFormat:
                @"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-version\", \"3520.5.1\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n"
                "{\n"
                "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
                "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), "
                "offset=uint64(64)))];\n"
                "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
                "        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,"
                "weight=W,x=x)[name=string(\"out\")];\n"
                "    } -> (out);\n"
                "}\n", C, S, C, C, C, C, C, S];

            Kern *k = compile_multi_weight(mil, weights, C*S*2, C*S*2, "single-blob-256x256");
            if (k) free_kern(k);
        }

        // ============================================================
        // Test 2: Separate weight files (the stories_mil.h approach)
        // ============================================================
        printf("\n--- Test 2: Separate weight files (stories_mil.h approach) ---\n");
        {
            int C = 256, S = 64;
            NSData *wq_data = build_blob_const(0.1f, C, C);
            NSData *wk_data = build_blob_const(0.2f, C, C);

            NSDictionary *weights = @{
                @"@model_path/weights/wq.bin": @{@"offset": @0, @"data": wq_data},
                @"@model_path/weights/wk.bin": @{@"offset": @0, @"data": wk_data},
            };

            NSString *mil = [NSString stringWithFormat:
                @"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n"
                "{\n"
                "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [%d,%d,1,1]> Wq = const()[name=string(\"Wq\"), "
                "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), "
                "offset=uint64(64)))];\n"
                "        tensor<fp16, [%d,%d,1,1]> Wk = const()[name=string(\"Wk\"), "
                "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), "
                "offset=uint64(64)))];\n"
                "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
                "        tensor<fp16, [1,%d,1,%d]> q = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,"
                "weight=Wq,x=x)[name=string(\"cq\")];\n"
                "        tensor<fp16, [1,%d,1,%d]> k = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,"
                "weight=Wk,x=x)[name=string(\"ck\")];\n"
                "        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"
                "        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"
                "        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(q,k))"
                "[name=string(\"cat\")];\n"
                "    } -> (out);\n"
                "}\n", C, S, C, C, C, C, C, C, C, C, C, S, C, S, 2*C, S];

            Kern *k = compile_multi_weight(mil, weights, C*S*2, 2*C*S*2, "multi-weight-2conv");
            if (k) {
                // Fill input with 1.0
                IOSurfaceLock(k->ioIn, 0, NULL);
                _Float16 *inp = (_Float16 *)IOSurfaceGetBaseAddress(k->ioIn);
                for (int i = 0; i < C * S; i++) inp[i] = (_Float16)1.0f;
                IOSurfaceUnlock(k->ioIn, 0, NULL);

                // Warmup
                for (int i = 0; i < 5; i++) ane_eval(k);

                // Benchmark
                int iters = 100;
                uint64_t t0 = mach_absolute_time();
                for (int i = 0; i < iters; i++) ane_eval(k);
                double avg = tms(mach_absolute_time() - t0) / iters;
                double gflops = 2.0 * (2.0 * C * C * S) / 1e9; // 2 convs
                printf("  Eval: %.3f ms, %.2f TFLOPS\n", avg, gflops / avg);

                // Read output
                IOSurfaceLock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *out = (_Float16 *)IOSurfaceGetBaseAddress(k->ioOut);
                printf("  Output[0..3]: %.4f %.4f %.4f %.4f\n",
                       (float)out[0], (float)out[1], (float)out[2], (float)out[3]);
                IOSurfaceUnlock(k->ioOut, kIOSurfaceLockReadOnly, NULL);

                free_kern(k);
            }
        }

        // ============================================================
        // Test 3: Single conv, separate weight file, small (64x64)
        // ============================================================
        printf("\n--- Test 3: Single conv, separate weight, 64x64 ---\n");
        {
            int C = 64, S = 16;
            NSData *w_data = build_blob_const(0.25f, C, C);
            NSDictionary *weights = @{
                @"@model_path/weights/w.bin": @{@"offset": @0, @"data": w_data}
            };
            NSString *mil = [NSString stringWithFormat:
                @"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n"
                "{\n"
                "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
                "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), "
                "offset=uint64(64)))];\n"
                "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
                "        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,"
                "weight=W,x=x)[name=string(\"out\")];\n"
                "    } -> (out);\n"
                "}\n", C, S, C, C, C, C, C, S];

            Kern *k = compile_multi_weight(mil, weights, C*S*2, C*S*2, "sep-weight-64x64");
            if (k) {
                IOSurfaceLock(k->ioIn, 0, NULL);
                _Float16 *inp = (_Float16 *)IOSurfaceGetBaseAddress(k->ioIn);
                for (int i = 0; i < C * S; i++) inp[i] = (_Float16)1.0f;
                IOSurfaceUnlock(k->ioIn, 0, NULL);
                for (int i = 0; i < 5; i++) ane_eval(k);
                int iters = 100;
                uint64_t t0 = mach_absolute_time();
                for (int i = 0; i < iters; i++) ane_eval(k);
                double avg = tms(mach_absolute_time() - t0) / iters;
                printf("  Eval: %.3f ms, %.2f GFLOPS\n", avg, 2.0*C*C*S/1e6/avg);
                free_kern(k);
            }
        }

        // ============================================================
        // Test 4: 768x768 (Stories110M dim) with separate weights
        // ============================================================
        printf("\n--- Test 4: 768x768 conv (Stories110M dim), separate weight ---\n");
        {
            int C = 768, S = 256;
            NSData *w_data = build_blob_const(0.01f, C, C);
            NSDictionary *weights = @{
                @"@model_path/weights/w.bin": @{@"offset": @0, @"data": w_data}
            };
            NSString *mil = [NSString stringWithFormat:
                @"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n"
                "{\n"
                "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
                "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), "
                "offset=uint64(64)))];\n"
                "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
                "        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,"
                "weight=W,x=x)[name=string(\"out\")];\n"
                "    } -> (out);\n"
                "}\n", C, S, C, C, C, C, C, S];

            Kern *k = compile_multi_weight(mil, weights, C*S*2, C*S*2, "sep-weight-768x768");
            if (k) {
                IOSurfaceLock(k->ioIn, 0, NULL);
                _Float16 *inp = (_Float16 *)IOSurfaceGetBaseAddress(k->ioIn);
                for (int i = 0; i < C * S; i++) inp[i] = (_Float16)1.0f;
                IOSurfaceUnlock(k->ioIn, 0, NULL);
                for (int i = 0; i < 3; i++) ane_eval(k);
                int iters = 50;
                uint64_t t0 = mach_absolute_time();
                for (int i = 0; i < iters; i++) ane_eval(k);
                double avg = tms(mach_absolute_time() - t0) / iters;
                double tflops = 2.0*C*C*S / 1e9 / avg;
                printf("  Eval: %.3f ms, %.2f TFLOPS\n", avg, tflops);
                free_kern(k);
            }
        }

        // ============================================================
        // Test 5: buildInfo from stories_mil.h exactly
        // ============================================================
        printf("\n--- Test 5: Exact stories_mil.h buildInfo + format ---\n");
        {
            int C = 256, S = 64;
            NSData *w_data = build_blob_const(0.25f, C, C);
            // Use the exact same weight dict key format as stories_mil.h
            NSDictionary *weights = @{
                @"@model_path/weights/wq.bin": @{@"offset": @0, @"data": w_data}
            };
            // Exact buildInfo from stories_mil.h MIL_HDR
            NSString *mil = [NSString stringWithFormat:
                @"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
                "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
                "{\"coremltools-version\", \"9.0\"}})]\n"
                "{\n"
                "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [%d,%d,1,1]> Wq = const()[name=string(\"Wq\"), "
                "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), "
                "offset=uint64(64)))];\n"
                "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
                "        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,"
                "weight=Wq,x=x)[name=string(\"out\")];\n"
                "    } -> (out);\n"
                "}\n", C, S, C, C, C, C, C, S];

            Kern *k = compile_multi_weight(mil, weights, C*S*2, C*S*2, "stories-exact-256x256");
            if (k) {
                IOSurfaceLock(k->ioIn, 0, NULL);
                _Float16 *inp = (_Float16 *)IOSurfaceGetBaseAddress(k->ioIn);
                for (int i = 0; i < C * S; i++) inp[i] = (_Float16)1.0f;
                IOSurfaceUnlock(k->ioIn, 0, NULL);
                for (int i = 0; i < 5; i++) ane_eval(k);
                int iters = 100;
                uint64_t t0 = mach_absolute_time();
                for (int i = 0; i < iters; i++) ane_eval(k);
                double avg = tms(mach_absolute_time() - t0) / iters;
                double tflops = 2.0*C*C*S / 1e9 / avg;
                printf("  Eval: %.3f ms, %.2f TFLOPS\n", avg, tflops);

                IOSurfaceLock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *out = (_Float16 *)IOSurfaceGetBaseAddress(k->ioOut);
                // Expected: 256 * 0.25 * 1.0 = 64.0 per output element
                printf("  Output[0..3]: %.2f %.2f %.2f %.2f (expected 64.0)\n",
                       (float)out[0], (float)out[1], (float)out[2], (float)out[3]);
                IOSurfaceUnlock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
                free_kern(k);
            }
        }

        printf("\n=== Summary ===\n");
        printf("If tests 2-5 pass but test 1 fails, the M1 ANE compiler\n");
        printf("requires separate weight files, not single-blob offsets.\n");
        printf("This matches upstream issue #3 findings.\n");
    }
    return 0;
}
