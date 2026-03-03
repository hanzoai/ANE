// ane_coreml_bridge.m — Bridge CoreML to ANE on M1
// Uses CoreML's official API to compile models, then tests ANE execution
// This is the M1-compatible path: CoreML handles MIL compilation internally
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// Approach 1: CoreML official API with ANE compute units
static void test_coreml_ane(const char *pkg_path, int ch_in, int ch_out, int seq) {
    printf("\n=== CoreML ANE Test: %s ===\n", pkg_path);
    NSError *e = nil;

    // Compile the model
    uint64_t t0 = mach_absolute_time();
    NSURL *compiledURL = [MLModel compileModelAtURL:
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:pkg_path]] error:&e];
    if (e) {
        printf("  Compile FAILED: %s\n", [[e localizedDescription] UTF8String]);
        return;
    }
    printf("  CoreML compile: %.1f ms\n", ms(mach_absolute_time() - t0));
    printf("  Compiled to: %s\n", [[compiledURL path] UTF8String]);

    // List compiled contents
    NSFileManager *fm = [NSFileManager defaultManager];
    NSArray *contents = [fm contentsOfDirectoryAtPath:[compiledURL path] error:nil];
    printf("  Compiled contents: %s\n", [[contents description] UTF8String]);

    // Check for MIL in compiled output
    NSString *milPath = [[compiledURL path] stringByAppendingPathComponent:@"model.mil"];
    if ([fm fileExistsAtPath:milPath]) {
        NSString *mil = [NSString stringWithContentsOfFile:milPath
                          encoding:NSUTF8StringEncoding error:nil];
        printf("  Found compiled MIL! (%lu bytes)\n", (unsigned long)mil.length);
        // Print first 2000 chars
        if (mil.length > 2000)
            mil = [mil substringToIndex:2000];
        printf("%s\n", [mil UTF8String]);
    } else {
        printf("  No model.mil in compiled output\n");
        // List all files recursively
        NSDirectoryEnumerator *de = [fm enumeratorAtPath:[compiledURL path]];
        NSString *file;
        while ((file = [de nextObject])) {
            NSDictionary *attrs = [de fileAttributes];
            printf("    %s (%llu bytes)\n", [file UTF8String],
                   [attrs[NSFileSize] unsignedLongLongValue]);
        }
    }

    // Load with ANE compute units
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

    t0 = mach_absolute_time();
    MLModel *model = [MLModel modelWithContentsOfURL:compiledURL
                       configuration:config error:&e];
    if (e) {
        printf("  Load FAILED: %s\n", [[e localizedDescription] UTF8String]);
        return;
    }
    printf("  Model loaded: %.1f ms\n", ms(mach_absolute_time() - t0));

    // Get model description
    MLModelDescription *desc = model.modelDescription;
    printf("  Inputs: %s\n", [[desc.inputDescriptionsByName description] UTF8String]);
    printf("  Outputs: %s\n", [[desc.outputDescriptionsByName description] UTF8String]);

    // Create input - find the input name
    NSString *inputName = desc.inputDescriptionsByName.allKeys.firstObject;
    NSString *outputName = desc.outputDescriptionsByName.allKeys.firstObject;
    printf("  Input name: '%s', Output name: '%s'\n",
           [inputName UTF8String], [outputName UTF8String]);

    // Create MLMultiArray input
    MLMultiArray *inputArray = [[MLMultiArray alloc]
        initWithShape:@[@1, @(ch_in), @1, @(seq)]
        dataType:MLMultiArrayDataTypeFloat16
        error:&e];
    if (e) {
        // Try float32
        inputArray = [[MLMultiArray alloc]
            initWithShape:@[@1, @(ch_in), @1, @(seq)]
            dataType:MLMultiArrayDataTypeFloat32
            error:&e];
    }
    if (e) {
        printf("  Input creation FAILED: %s\n", [[e localizedDescription] UTF8String]);
        return;
    }

    // Fill with 1.0
    for (int i = 0; i < ch_in * seq; i++) {
        inputArray[i] = @1.0f;
    }

    // Create feature provider
    MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{inputName: inputArray} error:&e];
    if (e) {
        printf("  Provider FAILED: %s\n", [[e localizedDescription] UTF8String]);
        return;
    }

    // Predict (warmup)
    for (int i = 0; i < 5; i++) {
        [model predictionFromFeatures:provider error:&e];
    }

    // Benchmark
    int iters = 200;
    t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        id<MLFeatureProvider> result = [model predictionFromFeatures:provider error:&e];
        if (e && i == 0) {
            printf("  Predict FAILED: %s\n", [[e localizedDescription] UTF8String]);
            return;
        }
    }
    double avgMs = ms(mach_absolute_time() - t0) / iters;
    double gflops = 2.0 * ch_in * ch_out * seq / 1e9;
    double tflops = gflops / avgMs;

    printf("\n  BENCHMARK (CoreML → ANE):\n");
    printf("    %.3f ms/inference\n", avgMs);
    printf("    %.4f TFLOPS\n", tflops);
    printf("    %.1f GFLOPS\n", gflops / avgMs * 1000);

    // Now compare with CPU-only
    config.computeUnits = MLComputeUnitsCPUOnly;
    MLModel *cpuModel = [MLModel modelWithContentsOfURL:compiledURL
                          configuration:config error:&e];
    for (int i = 0; i < 5; i++)
        [cpuModel predictionFromFeatures:provider error:&e];
    t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++)
        [cpuModel predictionFromFeatures:provider error:&e];
    double cpuMs = ms(mach_absolute_time() - t0) / iters;
    double cpuTflops = gflops / cpuMs;

    printf("\n  BENCHMARK (CPU only):\n");
    printf("    %.3f ms/inference\n", cpuMs);
    printf("    %.4f TFLOPS\n", cpuTflops);

    printf("\n  ANE speedup: %.2fx\n", cpuMs / avgMs);

    // Cleanup
    [fm removeItemAtPath:[compiledURL path] error:nil];
}

// Approach 2: Try to extract compiled MIL and feed to private API
static void try_compiled_to_private_api(const char *pkg_path) {
    printf("\n=== Private API from compiled model ===\n");
    NSError *e = nil;

    NSURL *compiledURL = [MLModel compileModelAtURL:
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:pkg_path]] error:&e];
    if (e) { printf("  Compile failed\n"); return; }

    // Check what files the compiled model has
    NSFileManager *fm = [NSFileManager defaultManager];
    NSDirectoryEnumerator *de = [fm enumeratorAtPath:[compiledURL path]];
    NSString *file;
    NSString *milFile = nil;
    NSString *weightFile = nil;
    while ((file = [de nextObject])) {
        if ([file hasSuffix:@".mil"]) milFile = [[compiledURL path] stringByAppendingPathComponent:file];
        if ([file hasSuffix:@"weight.bin"]) weightFile = [[compiledURL path] stringByAppendingPathComponent:file];
        printf("  %s\n", [file UTF8String]);
    }

    if (milFile) {
        printf("  MIL file found: %s\n", [milFile UTF8String]);
        NSString *milContent = [NSString stringWithContentsOfFile:milFile
                                 encoding:NSUTF8StringEncoding error:nil];
        if (milContent) {
            printf("  MIL content (%lu bytes):\n", (unsigned long)milContent.length);
            printf("%s\n", milContent.length > 3000 ?
                   [[milContent substringToIndex:3000] UTF8String] :
                   [milContent UTF8String]);

            // Try feeding this to the private API
            dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
            Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
            Class IMM = NSClassFromString(@"_ANEInMemoryModel");
            if (!Desc || !IMM) { printf("  Classes not found\n"); return; }

            NSData *milData = [milContent dataUsingEncoding:NSUTF8StringEncoding];
            NSData *wData = weightFile ? [NSData dataWithContentsOfFile:weightFile] : nil;
            NSDictionary *wdict = wData ?
                @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": wData}} : nil;

            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                Desc, @selector(modelWithMILText:weights:optionsPlist:),
                milData, wdict, nil);
            printf("  Descriptor: %s\n", desc ? "OK" : "nil");

            if (desc) {
                id model = ((id(*)(Class,SEL,id))objc_msgSend)(
                    IMM, @selector(inMemoryModelWithDescriptor:), desc);
                printf("  Model: %s\n", model ? "OK" : "nil");

                if (model) {
                    id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
                    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
                    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                        withIntermediateDirectories:YES attributes:nil error:nil];
                    [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
                    if (wData)
                        [wData writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

                    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                        model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
                    printf("  Private compile: %s\n", ok ? "OK" : "FAIL");
                    if (e) printf("    Error: %s\n", [[e description] UTF8String]);
                    [fm removeItemAtPath:td error:nil];
                }
            }
        }
    } else {
        printf("  No .mil file in compiled model (uses protobuf format)\n");
    }

    [fm removeItemAtPath:[compiledURL path] error:nil];
}

int main() {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        printf("╔════════════════════════════════════════════╗\n");
        printf("║  CoreML → ANE Bridge for M1 Max           ║\n");
        printf("╚════════════════════════════════════════════╝\n");

        NSProcessInfo *pi = [NSProcessInfo processInfo];
        printf("\nmacOS: %s\n", [[pi operatingSystemVersionString] UTF8String]);
        printf("RAM: %.0f GB\n", [pi physicalMemory] / 1073741824.0);

        // Test with models generated by gen_mil_m1.py
        struct { const char *path; int ci, co, seq; } tests[] = {
            {"/tmp/ane_m1_small.mlpackage", 64, 64, 16},
            {"/tmp/ane_m1_medium.mlpackage", 256, 256, 64},
        };

        for (int i = 0; i < 2; i++) {
            NSString *path = [NSString stringWithUTF8String:tests[i].path];
            if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
                test_coreml_ane(tests[i].path, tests[i].ci, tests[i].co, tests[i].seq);
                try_compiled_to_private_api(tests[i].path);
            } else {
                printf("\nModel not found: %s\n", tests[i].path);
                printf("Run: .venv/bin/python gen_mil_m1.py first\n");
            }
        }

        printf("\n=== Summary ===\n");
        printf("On M1 Max, ANE is accessed via CoreML (official API).\n");
        printf("The private _ANEInMemoryModel API MIL format differs from M4.\n");
        printf("For training: use CoreML-compiled models + gradient computation on CPU/MPS.\n");
    }
    return 0;
}
