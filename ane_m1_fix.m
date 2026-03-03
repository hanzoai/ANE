// ane_m1_fix.m — Systematic MIL format fuzzer for M1 ANE compiler
// Tests every combination of: program version, opset, buildInfo format, op syntax
// Goal: find the exact MIL format that M1's H14 ANE compiler accepts
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double tms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

static Class g_Desc, g_IMM, g_Req, g_AIO;

static void init_ane(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_IMM  = NSClassFromString(@"_ANEInMemoryModel");
    g_Req  = NSClassFromString(@"_ANERequest");
    g_AIO  = NSClassFromString(@"_ANEIOSurfaceObject");
}

// Try to compile+load+eval a MIL program. Returns eval time in ms, or -1/-2/-3 on failure.
static double try_mil(NSString *mil, NSData *weightData, const char *label,
                      int nInBytes, int nOutBytes) {
    NSError *e = nil;
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    NSDictionary *wdict = nil;
    if (weightData) {
        wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weightData}};
    }

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
        milData, wdict, nil);
    if (!desc) { printf("  %-40s DESC=nil\n", label); return -1; }

    id model = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_IMM, @selector(inMemoryModelWithDescriptor:), desc);
    if (!model) { printf("  %-40s MODEL=nil\n", label); return -1; }

    // Setup tmp dir
    id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    if (weightData)
        [weightData writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    // Compile
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        NSString *errStr = [e localizedDescription] ?: @"unknown";
        // Extract inner error
        NSError *inner = [e userInfo][NSUnderlyingErrorKey];
        if (inner) errStr = [inner localizedDescription];
        // Shorten
        if (errStr.length > 60) errStr = [errStr substringToIndex:60];
        printf("  %-40s COMPILE FAIL: %s\n", label, [errStr UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return -2;
    }

    // Load
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        printf("  %-40s LOAD FAIL\n", label);
        [fm removeItemAtPath:td error:nil];
        return -3;
    }

    // Eval
    IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(nInBytes),(id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(nInBytes),
        (id)kIOSurfaceAllocSize:@(nInBytes),(id)kIOSurfacePixelFormat:@0});
    IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(nOutBytes),(id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(nOutBytes),
        (id)kIOSurfaceAllocSize:@(nOutBytes),(id)kIOSurfacePixelFormat:@0});

    // Fill input with 1.0 fp16
    IOSurfaceLock(ioIn, 0, NULL);
    uint16_t *inp = IOSurfaceGetBaseAddress(ioIn);
    for (int i = 0; i < nInBytes/2; i++) inp[i] = 0x3C00;
    IOSurfaceUnlock(ioIn, 0, NULL);

    id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
    id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_Req,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

    ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    if (!ok) {
        printf("  %-40s EVAL FAIL\n", label);
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioIn); CFRelease(ioOut);
        [fm removeItemAtPath:td error:nil];
        return -4;
    }

    // Benchmark
    for (int i = 0; i < 10; i++)
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    int iters = 100;
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++)
        ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    double ms = tms(mach_absolute_time() - t0) / iters;

    // Read output
    IOSurfaceLock(ioOut, kIOSurfaceLockReadOnly, NULL);
    uint16_t *out = IOSurfaceGetBaseAddress(ioOut);
    printf("  %-40s OK! %.3f ms  out[0]=0x%04x\n", label, ms, out[0]);
    IOSurfaceUnlock(ioOut, kIOSurfaceLockReadOnly, NULL);

    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(ioIn); CFRelease(ioOut);
    [fm removeItemAtPath:td error:nil];
    return ms;
}

// Weight blob with 64-byte header (like CoreML compiled format)
static NSMutableData *make_weights_with_header(int count) {
    NSMutableData *d = [NSMutableData dataWithLength:64 + count * 2];
    uint16_t *w = (uint16_t *)(d.mutableBytes + 64);
    for (int i = 0; i < count; i++) w[i] = 0x3400; // 0.25 in fp16
    return d;
}

// Weight blob without header
static NSMutableData *make_weights_raw(int count) {
    NSMutableData *d = [NSMutableData dataWithLength:count * 2];
    uint16_t *w = (uint16_t *)d.mutableBytes;
    for (int i = 0; i < count; i++) w[i] = 0x3400;
    return d;
}

int main() {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        init_ane();

        printf("╔═════════════════════════════════════════════╗\n");
        printf("║  ANE M1 MIL Format Fuzzer                   ║\n");
        printf("╚═════════════════════════════════════════════╝\n\n");

        int C = 64, S = 16;
        int wCount = C * C;
        NSMutableData *wHdr = make_weights_with_header(wCount);  // 64-byte hdr + data
        NSMutableData *wRaw = make_weights_raw(wCount);          // raw data
        int inBytes = C * S * 2;
        int outBytes = C * S * 2;

        // ============================================================
        // Phase 1: Program version + opset combinations
        // ============================================================
        printf("=== Phase 1: Program version × opset ===\n");
        NSArray *versions = @[@"1.0", @"1.1", @"1.2", @"1.3"];
        NSArray *opsets = @[@"", @"<ios15>", @"<ios16>", @"<ios17>", @"<ios18>"];

        for (NSString *ver in versions) {
            for (NSString *opset in opsets) {
                // v1.3 style buildInfo
                NSString *mil = [NSString stringWithFormat:
                    @"program(%@)\n"
                    "[buildInfo = dict<string, string>({{\"coremlc-version\", \"3520.5.1\"}, "
                    "{\"coremltools-version\", \"9.0\"}})]\n"
                    "{\n"
                    "    func main%@(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                    "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
                    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
                    "        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string(\"out\")];\n"
                    "    } -> (out);\n"
                    "}\n",
                    ver, opset, C, S, C, C, C, C, C, S];

                char label[64];
                snprintf(label, sizeof(label), "v%s %s hdr",
                         [ver UTF8String], opset.length ? [opset UTF8String] : "(none)");
                try_mil(mil, wHdr, label, inBytes, outBytes);
            }
        }

        // ============================================================
        // Phase 2: Offset 0 vs 64 for weight blob
        // ============================================================
        printf("\n=== Phase 2: Weight offset variants ===\n");
        for (NSString *ver in @[@"1.0", @"1.3"]) {
            for (NSString *opset in @[@"", @"<ios16>", @"<ios18>"]) {
                // offset=0 with raw weights
                NSString *mil0 = [NSString stringWithFormat:
                    @"program(%@)\n"
                    "[buildInfo = dict<string, string>({{\"coremlc-version\", \"3520.5.1\"}, "
                    "{\"coremltools-version\", \"9.0\"}})]\n"
                    "{\n"
                    "    func main%@(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                    "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(0)))];\n"
                    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
                    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
                    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
                    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
                    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
                    "        tensor<fp16, [1,%d,1,%d]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string(\"out\")];\n"
                    "    } -> (out);\n"
                    "}\n",
                    ver, opset, C, S, C, C, C, C, C, S];

                char label[64];
                snprintf(label, sizeof(label), "v%s %s off=0",
                         [ver UTF8String], opset.length ? [opset UTF8String] : "(none)");
                try_mil(mil0, wRaw, label, inBytes, outBytes);
            }
        }

        // ============================================================
        // Phase 3: Old-style MIL (program 1.0, no buildInfo dict)
        // ============================================================
        printf("\n=== Phase 3: Old-style MIL formats ===\n");

        // Minimalist v1.0
        {
            NSString *mil = [NSString stringWithFormat:
                @"program(1.0)\n"
                "[buildInfo]\n"
                "coremltools-version = \"7.0\"\n"
                "[main]\n"
                "func main(tensor<fp16, [1, %d, 1, %d]> x) -> (tensor<fp16, [1, %d, 1, %d]> out) {\n"
                "  tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"), offset=uint64(64)))];\n"
                "  out = conv(x, weight=W, strides=[1,1], pad_type=\"valid\", pad=[0,0,0,0], dilations=[1,1], groups=1)\n"
                "}\n",
                C, S, C, S, C, C, C, C];
            try_mil(mil, wHdr, "old v1.0 mixed syntax", inBytes, outBytes);
        }

        // Pure old style with inline weight ref
        {
            NSString *mil = [NSString stringWithFormat:
                @"program(1.0)\n"
                "[buildInfo]\n"
                "coremltools-version = \"7.0\"\n"
                "[main]\n"
                "func main(tensor<fp16, [1, %d, 1, %d]> x) -> (tensor<fp16, [1, %d, 1, %d]> out) {\n"
                "  out = conv(x, weight=@model_path/weights/weight.bin:0:uint8[%d], "
                "strides=[1,1], pad_type=\"valid\", pad=[0,0,0,0], dilations=[1,1], groups=1)\n"
                "}\n",
                C, S, C, S, wCount * 2];
            try_mil(mil, wRaw, "old v1.0 inline weight ref", inBytes, outBytes);
        }

        // v1.0 with return arrow and typed const
        {
            NSString *mil = [NSString stringWithFormat:
                @"program(1.0)\n"
                "[buildInfo]\n"
                "coremltools-version = \"8.0\"\n"
                "[main]\n"
                "func main(tensor<fp16, [1, %d, 1, %d]> x) -> (tensor<fp16, [1, %d, 1, %d]> out) {\n"
                "  const tensor<fp16, [%d, %d, 1, 1]> W = @model_path/weights/weight.bin:64:uint8[%d]\n"
                "  out = conv(x, weight=W, strides=[1,1], pad_type=\"valid\", pad=[0,0,0,0], dilations=[1,1], groups=1)\n"
                "}\n",
                C, S, C, S, C, C, wCount * 2];
            try_mil(mil, wHdr, "old v1.0 typed const decl", inBytes, outBytes);
        }

        // ============================================================
        // Phase 4: Simple identity / relu / add (no weights needed)
        // ============================================================
        printf("\n=== Phase 4: Weight-free ops (isolate compiler issue) ===\n");

        // Identity
        for (NSString *ver in @[@"1.0", @"1.3"]) {
            for (NSString *opset in @[@"", @"<ios16>", @"<ios18>"]) {
                NSString *mil = [NSString stringWithFormat:
                    @"program(%@)\n"
                    "[buildInfo = dict<string, string>({{\"coremlc-version\", \"3520.5.1\"}})]\n"
                    "{\n"
                    "    func main%@(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                    "        tensor<fp16, [1, %d, 1, %d]> out = identity(x=x)[name=string(\"out\")];\n"
                    "    } -> (out);\n"
                    "}\n",
                    ver, opset, C, S, C, S];
                char label[64];
                snprintf(label, sizeof(label), "identity v%s %s",
                         [ver UTF8String], opset.length ? [opset UTF8String] : "(none)");
                try_mil(mil, nil, label, inBytes, outBytes);
            }
        }

        // relu
        for (NSString *ver in @[@"1.0", @"1.3"]) {
            NSString *mil = [NSString stringWithFormat:
                @"program(%@)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-version\", \"3520.5.1\"}})]\n"
                "{\n"
                "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1, %d, 1, %d]> out = relu(x=x)[name=string(\"out\")];\n"
                "    } -> (out);\n"
                "}\n",
                ver, C, S, C, S];
            char label[64];
            snprintf(label, sizeof(label), "relu v%s <ios18>", [ver UTF8String]);
            try_mil(mil, nil, label, inBytes, outBytes);
        }

        // add scalar
        {
            NSString *mil = [NSString stringWithFormat:
                @"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-version\", \"3520.5.1\"}})]\n"
                "{\n"
                "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];\n"
                "        tensor<fp16, [1, %d, 1, %d]> out = add(x=x,y=one)[name=string(\"out\")];\n"
                "    } -> (out);\n"
                "}\n",
                C, S, C, S];
            try_mil(mil, nil, "add scalar v1.3 <ios18>", inBytes, outBytes);
        }

        // mul element-wise
        {
            NSString *mil = [NSString stringWithFormat:
                @"program(1.3)\n"
                "[buildInfo = dict<string, string>({{\"coremlc-version\", \"3520.5.1\"}})]\n"
                "{\n"
                "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
                "        tensor<fp16, [1, %d, 1, %d]> out = mul(x=x,y=x)[name=string(\"out\")];\n"
                "    } -> (out);\n"
                "}\n",
                C, S, C, S];
            try_mil(mil, nil, "mul self v1.3 <ios18>", inBytes, outBytes);
        }

        // ============================================================
        // Phase 5: Use _ANEModel (not InMemory) with compiled mlmodelc
        // ============================================================
        printf("\n=== Phase 5: _ANEModel with pre-compiled mlmodelc ===\n");
        {
            Class ANEModel = NSClassFromString(@"_ANEModel");
            printf("  _ANEModel: %s\n", ANEModel ? "found" : "missing");
            if (ANEModel) {
                // List methods
                unsigned int count;
                Method *methods = class_copyMethodList(object_getClass(ANEModel), &count);
                printf("  Class methods (%d):\n", count);
                for (unsigned int i = 0; i < count && i < 20; i++) {
                    printf("    %s\n", sel_getName(method_getName(methods[i])));
                }
                free(methods);

                methods = class_copyMethodList(ANEModel, &count);
                printf("  Instance methods (%d):\n", count);
                for (unsigned int i = 0; i < count && i < 20; i++) {
                    printf("    %s\n", sel_getName(method_getName(methods[i])));
                }
                free(methods);
            }
        }

        // ============================================================
        // Phase 6: Try loading compiled mlmodelc via _ANEModel
        // ============================================================
        printf("\n=== Phase 6: CoreML compile → _ANEModel ===\n");
        {
            NSError *e = nil;
            NSString *pkg = @"/tmp/ane_m1_small.mlpackage";
            if ([[NSFileManager defaultManager] fileExistsAtPath:pkg]) {
                NSURL *compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:pkg] error:&e];
                if (compiled) {
                    printf("  Compiled: %s\n", [[compiled path] UTF8String]);

                    // Check for espresso.net / coreml.bin (the actual ANE program)
                    NSFileManager *fm = [NSFileManager defaultManager];
                    NSDirectoryEnumerator *de = [fm enumeratorAtPath:[compiled path]];
                    NSString *file;
                    while ((file = [de nextObject])) {
                        NSDictionary *attrs = [de fileAttributes];
                        printf("    %s (%llu bytes)\n", [file UTF8String],
                               [attrs[NSFileSize] unsignedLongLongValue]);
                    }

                    // Try _ANEModel with this path
                    Class ANEModel = NSClassFromString(@"_ANEModel");
                    if (ANEModel) {
                        SEL initSel = @selector(initWithModelAtPath:error:);
                        if ([ANEModel instancesRespondToSelector:initSel]) {
                            id aneModel = [[ANEModel alloc] init];
                            aneModel = ((id(*)(id,SEL,id,NSError**))objc_msgSend)(
                                aneModel, initSel, [compiled path], &e);
                            printf("  _ANEModel initWithModelAtPath: %s\n",
                                   aneModel ? "OK" : "nil");
                            if (e) printf("    Error: %s\n", [[e description] UTF8String]);
                        } else {
                            printf("  _ANEModel doesn't respond to initWithModelAtPath:\n");

                            // Try alloc variants
                            SEL alts[] = {
                                @selector(modelWithPath:),
                                @selector(modelWithURL:),
                                @selector(initWithPath:),
                                @selector(initWithURL:),
                                @selector(modelAtPath:error:),
                            };
                            for (int i = 0; i < 5; i++) {
                                if ([ANEModel instancesRespondToSelector:alts[i]] ||
                                    [ANEModel respondsToSelector:alts[i]]) {
                                    printf("    Found: %s\n", sel_getName(alts[i]));
                                }
                            }
                        }
                    }

                    [fm removeItemAtPath:[compiled path] error:nil];
                }
            } else {
                printf("  No mlpackage found. Run: .venv/bin/python gen_mil_m1.py\n");
            }
        }

        printf("\n=== Done ===\n");
    }
    return 0;
}
