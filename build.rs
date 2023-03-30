fn main() {
    let mut config = cc::Build::new();
    config.opt_level(3);
    config.cpp(true);
    config.file("bindings.cpp");
    config.include(".");
    config.compile("libflow.a");
    println!("cargo:rerun-if-changed=lemon");
}
