// ane_probe_m1.m — Comprehensive M1 ANE diagnostic
// Tests every step of the private API pipeline to identify exactly where M1 diverges from M4
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// Dump all classes in the ANE framework
static void dump_ane_classes(void) {
    printf("\n=== ANE Framework Classes ===\n");
    unsigned int count = 0;
    Class *classes = objc_copyClassList(&count);
    int found = 0;
    for (unsigned int i = 0; i < count; i++) {
        const char *name = class_getName(classes[i]);
        if (strstr(name, "ANE") || strstr(name, "ane")) {
            printf("  %s\n", name);
            found++;
        }
    }
    free(classes);
    printf("  Total ANE classes: %d\n", found);
}

// Check if key classes and selectors exist
static void check_api_surface(void) {
    printf("\n=== API Surface Check ===\n");
    struct { const char *name; BOOL required; } classes[] = {
        {"_ANEInMemoryModelDescriptor", YES},
        {"_ANEInMemoryModel", YES},
        {"_ANERequest", YES},
        {"_ANEIOSurfaceObject", YES},
        {"_ANEClient", NO},
        {"_ANECompiler", NO},
        {"_ANEChainingRequest", NO},
        {"_ANEDeviceController", NO},
        {"_ANEBuffer", NO},
        {"_ANEPerformanceStats", NO},
        {"_ANEModel", NO},
        {NULL, NO}
    };
    for (int i = 0; classes[i].name; i++) {
        Class c = NSClassFromString([NSString stringWithUTF8String:classes[i].name]);
        printf("  %-40s %s %s\n", classes[i].name,
               c ? "FOUND" : "MISSING",
               (!c && classes[i].required) ? "<-- REQUIRED!" : "");
    }
}

// Generate a minimal MIL program: y = conv(x, W) — simplest possible ANE op
static NSString *gen_simple_mil(int ch_in, int ch_out, int seq) {
    return [NSString stringWithFormat:
        @"program(1.0)\n"
        @"[buildInfo]\n"
        @"coremltools-version = \"8.0\"\n"
        @"[main]\n"
        @"func main(tensor<fp16, [1, %d, 1, %d]> x) -> (tensor<fp16, [1, %d, 1, %d]> y) {\n"
        @"  y = conv(x, weight=@model_path/weights/weight.bin:0:uint8[%d], "
        @"strides=[1,1], pad_type=\"valid\", pad=[0,0,0,0], "
        @"dilations=[1,1], groups=1)\n"
        @"}\n",
        ch_in, seq, ch_out, seq, ch_out * ch_in * 2];
}

// Alternative MIL format — some versions want different syntax
static NSString *gen_mil_v2(int ch_in, int ch_out, int seq) {
    return [NSString stringWithFormat:
        @"program(1.0)\n"
        @"[buildInfo]\n"
        @"coremltools-version = \"8.1\"\n"
        @"[main]\n"
        @"func main(tensor<fp16, [1, %d, 1, %d]> x) -> (tensor<fp16, [1, %d, 1, %d]> out) {\n"
        @"  out = conv(x, weight=const<fp16, [%d, %d, 1, 1]>(@model_path/weights/weight.bin:0:uint8[%d]), "
        @"strides=[1,1], pad_type=\"valid\", pad=[0,0,0,0], "
        @"dilations=[1,1], groups=1)\n"
        @"}\n",
        ch_in, seq, ch_out, seq, ch_out, ch_in, ch_out * ch_in * 2];
}

// Another variation — explicit const declaration before use
static NSString *gen_mil_v3(int ch_in, int ch_out, int seq) {
    return [NSString stringWithFormat:
        @"program(1.0)\n"
        @"[buildInfo]\n"
        @"coremltools-version = \"8.0\"\n"
        @"[main]\n"
        @"func main(tensor<fp16, [1, %d, 1, %d]> x) -> (tensor<fp16, [1, %d, 1, %d]> out) {\n"
        @"  const tensor<fp16, [%d, %d, 1, 1]> W = @model_path/weights/weight.bin:0:uint8[%d]\n"
        @"  out = conv(x, weight=W, "
        @"strides=[1,1], pad_type=\"valid\", pad=[0,0,0,0], "
        @"dilations=[1,1], groups=1)\n"
        @"}\n",
        ch_in, seq, ch_out, seq, ch_out, ch_in, ch_out * ch_in * 2];
}

static BOOL try_compile_mil(NSString *milStr, NSData *weightData, const char *label,
                            int ch_in, int ch_out, int seq) {
    printf("\n--- Trying: %s ---\n", label);

    Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class IMM = NSClassFromString(@"_ANEInMemoryModel");
    Class AR = NSClassFromString(@"_ANERequest");
    Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");

    if (!Desc || !IMM || !AR || !AIO) {
        printf("  SKIP: required classes missing\n");
        return NO;
    }

    NSData *milData = [milStr dataUsingEncoding:NSUTF8StringEncoding];
    printf("  MIL text (%lu bytes):\n", (unsigned long)milData.length);
    // Print first few lines
    NSArray *lines = [milStr componentsSeparatedByString:@"\n"];
    for (int i = 0; i < (int)lines.count && i < 12; i++)
        printf("    %s\n", [lines[i] UTF8String]);

    NSDictionary *wdict = nil;
    if (weightData) {
        wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weightData}};
    }

    // Step 1: Create descriptor
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        Desc, @selector(modelWithMILText:weights:optionsPlist:),
        milData, wdict, nil);
    if (!desc) {
        printf("  FAIL: modelWithMILText returned nil\n");

        // Try without optionsPlist (2-arg variant?)
        SEL alt = @selector(modelWithMILText:weights:);
        if ([Desc respondsToSelector:alt]) {
            printf("  Trying 2-arg variant...\n");
            desc = ((id(*)(Class,SEL,id,id))objc_msgSend)(Desc, alt, milData, wdict);
            if (desc) printf("  2-arg variant worked!\n");
        }
        if (!desc) return NO;
    }
    printf("  Step 1 OK: descriptor created (%s)\n", [NSStringFromClass([desc class]) UTF8String]);

    // Step 2: Create model
    id model = ((id(*)(Class,SEL,id))objc_msgSend)(
        IMM, @selector(inMemoryModelWithDescriptor:), desc);
    if (!model) {
        printf("  FAIL: inMemoryModelWithDescriptor returned nil\n");
        return NO;
    }
    printf("  Step 2 OK: model created\n");

    // Step 3: Get hex ID and populate tmp dir
    id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    printf("  Hex ID: %s\n", hexId ? [[hexId description] UTF8String] : "(nil)");
    NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId ?: @"ane_probe"];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    if (weightData)
        [weightData writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    // Step 4: Compile
    NSError *e = nil;
    uint64_t t0 = mach_absolute_time();
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    double compileMs = ms(mach_absolute_time() - t0);
    printf("  Step 4 Compile: %s (%.1f ms)\n", ok ? "OK" : "FAIL", compileMs);
    if (e) {
        printf("    Error: %s\n", [[e localizedDescription] UTF8String]);
        // Print full error
        printf("    Detail: %s\n", [[e description] UTF8String]);
        e = nil;
    }
    if (!ok) {
        [fm removeItemAtPath:tmpDir error:nil];
        return NO;
    }

    // Step 5: Load
    t0 = mach_absolute_time();
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    double loadMs = ms(mach_absolute_time() - t0);
    printf("  Step 5 Load: %s (%.1f ms)\n", ok ? "OK" : "FAIL", loadMs);
    if (e) {
        printf("    Error: %s\n", [[e description] UTF8String]);
        e = nil;
    }
    if (!ok) {
        [fm removeItemAtPath:tmpDir error:nil];
        return NO;
    }

    // Step 6: Create I/O surfaces and evaluate
    size_t inBytes = ch_in * seq * 2;   // fp16
    size_t outBytes = ch_out * seq * 2;
    IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(inBytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(inBytes),
        (id)kIOSurfaceAllocSize: @(inBytes), (id)kIOSurfacePixelFormat: @0});
    IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(outBytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(outBytes),
        (id)kIOSurfaceAllocSize: @(outBytes), (id)kIOSurfacePixelFormat: @0});

    // Fill input with test pattern
    IOSurfaceLock(ioIn, 0, NULL);
    uint16_t *inp = (uint16_t *)IOSurfaceGetBaseAddress(ioIn);
    for (int i = 0; i < ch_in * seq; i++) inp[i] = 0x3C00; // 1.0 in fp16
    IOSurfaceUnlock(ioIn, 0, NULL);

    id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioIn);
    id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

    t0 = mach_absolute_time();
    ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        model, @selector(evaluateWithQoS:options:request:error:),
        21, @{}, req, &e);
    double evalMs = ms(mach_absolute_time() - t0);
    printf("  Step 6 Eval: %s (%.3f ms)\n", ok ? "OK" : "FAIL", evalMs);
    if (e) {
        printf("    Error: %s\n", [[e description] UTF8String]);
        e = nil;
    }

    if (ok) {
        // Benchmark: warmup + measure
        for (int i = 0; i < 10; i++)
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);

        int iters = 100;
        t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++)
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        double avgMs = ms(mach_absolute_time() - t0) / iters;
        double gflops = 2.0 * ch_in * ch_out * seq / 1e9;
        double tflops = gflops / avgMs;
        printf("  BENCHMARK: %.3f ms/eval, %.3f GFLOPS, %.4f TFLOPS\n", avgMs, gflops/avgMs*1000, tflops);

        // Read output sample
        IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
        uint16_t *out = (uint16_t *)IOSurfaceGetBaseAddress(ioOut);
        printf("  Output[0..3]: 0x%04x 0x%04x 0x%04x 0x%04x\n", out[0], out[1], out[2], out[3]);
        IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);
    }

    // Cleanup
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(ioIn); CFRelease(ioOut);
    [fm removeItemAtPath:tmpDir error:nil];

    return ok;
}

// Try CoreML-based approach as fallback
static void try_coreml_ane(int ch_in, int ch_out, int seq) {
    printf("\n=== CoreML ANE Fallback ===\n");
    printf("Generating mlpackage with coremltools would go here.\n");
    printf("CoreML schedules to ANE automatically for supported ops.\n");
    printf("Use: MLComputeUnits = .cpuAndNeuralEngine\n");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        printf("╔══════════════════════════════════════════╗\n");
        printf("║  ANE M1 Max Diagnostic Probe             ║\n");
        printf("╚══════════════════════════════════════════╝\n");

        // System info
        NSProcessInfo *pi = [NSProcessInfo processInfo];
        printf("\nmacOS: %s\n", [[pi operatingSystemVersionString] UTF8String]);
        printf("Cores: %lu physical\n", (unsigned long)[pi processorCount]);
        printf("RAM: %.0f GB\n", [pi physicalMemory] / 1073741824.0);

        // Load ANE framework
        void *h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        printf("ANE framework: %s\n", h ? "LOADED" : "FAILED");
        if (!h) {
            printf("FATAL: Cannot load ANE framework\n");
            return 1;
        }

        dump_ane_classes();
        check_api_surface();

        // Create weight data (ch_out * ch_in * 2 bytes for fp16)
        int ch_in = 64, ch_out = 64, seq = 16;
        size_t wSize = ch_out * ch_in * 2;
        NSMutableData *weightData = [NSMutableData dataWithLength:wSize];
        uint16_t *w = (uint16_t *)weightData.mutableBytes;
        // Identity-ish weights (small values)
        for (int i = 0; i < ch_out * ch_in; i++) {
            w[i] = (i % (ch_in + 1) == 0) ? 0x3C00 : 0x0000;  // 1.0 on diagonal, 0 elsewhere
        }

        // Try multiple MIL formats
        BOOL worked = NO;
        worked = try_compile_mil(gen_simple_mil(ch_in, ch_out, seq), weightData, "MIL v1 (basic)", ch_in, ch_out, seq);
        if (!worked)
            worked = try_compile_mil(gen_mil_v2(ch_in, ch_out, seq), weightData, "MIL v2 (typed const)", ch_in, ch_out, seq);
        if (!worked)
            worked = try_compile_mil(gen_mil_v3(ch_in, ch_out, seq), weightData, "MIL v3 (explicit const)", ch_in, ch_out, seq);

        // Try larger sizes if basic works
        if (worked) {
            printf("\n=== Scaling Tests ===\n");
            int configs[][3] = {
                {128, 128, 32},
                {256, 256, 64},
                {512, 512, 64},
                {768, 768, 64},   // Stories110M dim
                {768, 2048, 64},  // Stories110M FFN
            };
            for (int i = 0; i < 5; i++) {
                int ci = configs[i][0], co = configs[i][1], s = configs[i][2];
                size_t ws = co * ci * 2;
                NSMutableData *wd = [NSMutableData dataWithLength:ws];
                uint16_t *wp = (uint16_t *)wd.mutableBytes;
                for (size_t j = 0; j < (size_t)(co * ci); j++) wp[j] = 0x2000; // small fp16
                char label[64];
                snprintf(label, sizeof(label), "Scale %dx%d seq=%d", ci, co, s);
                // Use whichever MIL format worked
                try_compile_mil(gen_simple_mil(ci, co, s), wd, label, ci, co, s);
            }
        }

        if (!worked) {
            printf("\n=== DIAGNOSIS ===\n");
            printf("Private ANE in-memory API does not work on this M1 Max.\n");
            printf("Possible reasons:\n");
            printf("  1. MIL format differs between M1 and M4 ANE compilers\n");
            printf("  2. _ANEInMemoryModel may require M2+ or M4+ silicon\n");
            printf("  3. ANE compiler version mismatch (macOS version dependent)\n");
            printf("\nAlternatives:\n");
            printf("  - CoreML with .cpuAndNeuralEngine compute units\n");
            printf("  - MLX framework (uses Metal + ANE via CoreML internally)\n");
            printf("  - PyTorch MPS backend (Metal only, no ANE)\n");
        }

        printf("\n=== DONE ===\n");
    }
    return 0;
}
