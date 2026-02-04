//! API server entry point for Burn Qwen3 inference

mod api;

// Use single-threaded runtime to avoid Send/Sync requirements for WGPU model
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    api::run_api_server().await
}
