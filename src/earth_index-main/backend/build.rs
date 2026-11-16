fn main() -> std::io::Result<()> {
    println!("cargo:rerun-if-changed=proto/earth_index.proto");
    tonic_prost_build::configure()
        .file_descriptor_set_path(
            std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("descriptor.bin"),
        )
        .compile_protos(&["proto/earth_index.proto"], &["proto"])
}
