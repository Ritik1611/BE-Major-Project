fn main() {
    tonic_build::compile_protos("proto/orchestrator.proto")
        .expect("proto compile failed");
}
