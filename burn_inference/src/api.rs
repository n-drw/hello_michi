//! REST API server for Burn Qwen3 inference

use axum::{
    Router,
    routing::post,
    extract::State,
    Json,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tower_http::cors::{Any, CorsLayer};

use burn::prelude::*;
use burn::backend::Wgpu;
use burn::tensor::TensorData;
use tokenizers::Tokenizer;

// Include the generated model
#[path = "../../burn_qoperator_model/model.rs"]
mod model;

use model::Model;

type MyBackend = Wgpu;

const NUM_LAYERS: usize = 28;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const EOS_TOKEN_ID: i64 = 151645;

// Request sent to inference thread
struct InferenceRequest {
    input_ids: Vec<i64>,
    max_tokens: usize,
    response_tx: oneshot::Sender<Vec<i64>>,
}

// AppState with channel to inference thread (Send + Sync safe)
#[derive(Clone)]
pub struct AppState {
    inference_tx: mpsc::Sender<InferenceRequest>,
    tokenizer: Arc<Tokenizer>,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_max_tokens() -> usize { 256 }
fn default_temperature() -> f32 { 0.7 }

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub content: String,
    pub tokens_generated: usize,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
}

/// KV cache for transformer layers
pub struct KvCache {
    pub keys: Vec<Tensor<MyBackend, 4>>,
    pub values: Vec<Tensor<MyBackend, 4>>,
}

impl KvCache {
    fn new(batch_size: usize, device: &<MyBackend as burn::tensor::backend::Backend>::Device) -> Self {
        let keys: Vec<Tensor<MyBackend, 4>> = (0..NUM_LAYERS)
            .map(|_| Tensor::zeros([batch_size, NUM_KV_HEADS, 1, HEAD_DIM], device))
            .collect();
        let values: Vec<Tensor<MyBackend, 4>> = (0..NUM_LAYERS)
            .map(|_| Tensor::zeros([batch_size, NUM_KV_HEADS, 1, HEAD_DIM], device))
            .collect();
        Self { keys, values }
    }
}

fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str("<|im_start|>system\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<|im_end|>\n");
            }
            "user" => {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<|im_end|>\n");
            }
            "assistant" => {
                prompt.push_str("<|im_start|>assistant\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<|im_end|>\n");
            }
            _ => {}
        }
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: "qwen3-0.6b-q4".to_string(),
    })
}

async fn chat_handler(
    State(state): State<AppState>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    let prompt = format_chat_prompt(&request.messages);
    
    let encoding = state.tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    
    // Create oneshot channel for response
    let (response_tx, response_rx) = oneshot::channel();
    
    // Send inference request to dedicated thread
    state.inference_tx
        .send(InferenceRequest {
            input_ids,
            max_tokens: request.max_tokens,
            response_tx,
        })
        .await
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Inference thread unavailable".to_string()))?;
    
    // Wait for response
    let generated = response_rx
        .await
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "Inference failed".to_string()))?;
    
    let response_tokens: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
    let content = state.tokenizer
        .decode(&response_tokens, true)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(ChatResponse {
        content,
        tokens_generated: generated.len(),
    }))
}

fn sample_greedy(logits: &Tensor<MyBackend, 3>) -> i64 {
    let dims = logits.dims();
    let last_logits = logits.clone().slice([0..1, (dims[1] - 1)..dims[1], 0..dims[2]]);
    let last_logits: Tensor<MyBackend, 1> = last_logits.flatten(0, 2);
    let token_id = last_logits.argmax(0);
    let token_data: TensorData = token_id.into_data();
    token_data.as_slice::<i32>().unwrap()[0] as i64
}

fn forward_with_cache(
    model: &Model<MyBackend>,
    input_ids: Tensor<MyBackend, 2, Int>,
    attention_mask: Tensor<MyBackend, 2, Int>,
    position_ids: Tensor<MyBackend, 2, Int>,
    cache: KvCache,
) -> (Tensor<MyBackend, 3>, KvCache) {
    let result = model.forward(
        input_ids,
        attention_mask,
        position_ids,
        cache.keys[0].clone(), cache.values[0].clone(),
        cache.keys[1].clone(), cache.values[1].clone(),
        cache.keys[2].clone(), cache.values[2].clone(),
        cache.keys[3].clone(), cache.values[3].clone(),
        cache.keys[4].clone(), cache.values[4].clone(),
        cache.keys[5].clone(), cache.values[5].clone(),
        cache.keys[6].clone(), cache.values[6].clone(),
        cache.keys[7].clone(), cache.values[7].clone(),
        cache.keys[8].clone(), cache.values[8].clone(),
        cache.keys[9].clone(), cache.values[9].clone(),
        cache.keys[10].clone(), cache.values[10].clone(),
        cache.keys[11].clone(), cache.values[11].clone(),
        cache.keys[12].clone(), cache.values[12].clone(),
        cache.keys[13].clone(), cache.values[13].clone(),
        cache.keys[14].clone(), cache.values[14].clone(),
        cache.keys[15].clone(), cache.values[15].clone(),
        cache.keys[16].clone(), cache.values[16].clone(),
        cache.keys[17].clone(), cache.values[17].clone(),
        cache.keys[18].clone(), cache.values[18].clone(),
        cache.keys[19].clone(), cache.values[19].clone(),
        cache.keys[20].clone(), cache.values[20].clone(),
        cache.keys[21].clone(), cache.values[21].clone(),
        cache.keys[22].clone(), cache.values[22].clone(),
        cache.keys[23].clone(), cache.values[23].clone(),
        cache.keys[24].clone(), cache.values[24].clone(),
        cache.keys[25].clone(), cache.values[25].clone(),
        cache.keys[26].clone(), cache.values[26].clone(),
        cache.keys[27].clone(), cache.values[27].clone(),
    );

    let (logits, k0, v0, k1, v1, k2, v2, k3, v3, k4, v4, k5, v5, k6, v6, k7, v7,
         k8, v8, k9, v9, k10, v10, k11, v11, k12, v12, k13, v13, k14, v14, k15, v15,
         k16, v16, k17, v17, k18, v18, k19, v19, k20, v20, k21, v21, k22, v22, k23, v23,
         k24, v24, k25, v25, k26, v26, k27, v27) = result;

    let new_cache = KvCache {
        keys: vec![k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15,
                   k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27],
        values: vec![v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
                     v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27],
    };

    (logits, new_cache)
}

// Inference thread that owns the model (non-Send types stay on this thread)
fn run_inference_thread(mut rx: mpsc::Receiver<InferenceRequest>) {
    println!("1. Initializing WGPU backend...");
    let device = burn::backend::wgpu::WgpuDevice::default();
    
    println!("2. Loading model...");
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/Users/perro/work/hello_michi/burn_qoperator_model/model.bpk".to_string());
    let model: Model<MyBackend> = Model::from_file(&model_path, &device);
    println!("   Model loaded from: {}", model_path);
    
    // Process inference requests
    while let Some(request) = rx.blocking_recv() {
        let generated = generate_sync_inner(&model, &device, request.input_ids, request.max_tokens);
        let _ = request.response_tx.send(generated);
    }
}

// Inner generate function that takes model and device separately
fn generate_sync_inner(
    model: &Model<MyBackend>,
    device: &<MyBackend as burn::tensor::backend::Backend>::Device,
    input_ids: Vec<i64>,
    max_new_tokens: usize,
) -> Vec<i64> {
    let batch_size = 1;
    let mut cache = KvCache::new(batch_size, device);
    let mut generated_tokens = Vec::new();
    let seq_len = input_ids.len();

    // Prefill
    let input_tensor: Tensor<MyBackend, 2, Int> = Tensor::from_data_dtype(
        TensorData::new(input_ids.clone(), [batch_size, seq_len]),
        device,
        burn::tensor::DType::I64,
    );
    
    let total_seq_len = 1 + seq_len;
    let mut mask_data: Vec<i64> = Vec::with_capacity(total_seq_len);
    mask_data.push(0i64);
    mask_data.extend(vec![1i64; seq_len]);
    let attention_mask: Tensor<MyBackend, 2, Int> = Tensor::from_data_dtype(
        TensorData::new(mask_data, [batch_size, total_seq_len]),
        device,
        burn::tensor::DType::I64,
    );
    
    let position_ids: Tensor<MyBackend, 2, Int> = Tensor::from_data_dtype(
        TensorData::new((0..seq_len as i64).collect::<Vec<_>>(), [batch_size, seq_len]),
        device,
        burn::tensor::DType::I64,
    );

    let (logits, new_cache) = forward_with_cache(model, input_tensor, attention_mask, position_ids, cache);
    cache = new_cache;

    let mut next_token = sample_greedy(&logits);
    generated_tokens.push(next_token);

    // Autoregressive loop
    for i in 1..max_new_tokens {
        if next_token == EOS_TOKEN_ID {
            break;
        }

        let pos = seq_len + i;
        let input_tensor: Tensor<MyBackend, 2, Int> = Tensor::from_data_dtype(
            TensorData::new(vec![next_token], [1, 1]),
            device,
            burn::tensor::DType::I64,
        );
        
        let cache_len = cache.keys[0].dims()[2];
        let total_mask_len = cache_len + 1;
        let mut mask_data: Vec<i64> = Vec::with_capacity(total_mask_len);
        mask_data.push(0i64);
        mask_data.extend(vec![1i64; total_mask_len - 1]);
        let attention_mask: Tensor<MyBackend, 2, Int> = Tensor::from_data_dtype(
            TensorData::new(mask_data, [1, total_mask_len]),
            device,
            burn::tensor::DType::I64,
        );
        
        let position_ids: Tensor<MyBackend, 2, Int> = Tensor::from_data_dtype(
            TensorData::new(vec![pos as i64 - 1], [1, 1]),
            device,
            burn::tensor::DType::I64,
        );

        let (logits, new_cache) = forward_with_cache(model, input_tensor, attention_mask, position_ids, cache);
        cache = new_cache;

        next_token = sample_greedy(&logits);
        generated_tokens.push(next_token);
    }

    generated_tokens
}

pub async fn run_api_server() -> anyhow::Result<()> {
    println!("=== Burn Qwen3 API Server ===\n");
    
    // Create channel for inference requests
    let (inference_tx, inference_rx) = mpsc::channel::<InferenceRequest>(10);
    
    // Spawn inference thread with large stack (model has deep nesting)
    std::thread::Builder::new()
        .name("inference".to_string())
        .stack_size(64 * 1024 * 1024)  // 64MB stack
        .spawn(move || {
            run_inference_thread(inference_rx);
        })
        .expect("Failed to spawn inference thread");
    
    println!("3. Loading tokenizer...");
    let tokenizer_path = std::env::var("TOKENIZER_PATH")
        .unwrap_or_else(|_| "/Users/perro/work/hello_michi/onnx_models/qwen3-0.6b-qoperator/tokenizer.json".to_string());
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!("   Tokenizer loaded from: {}", tokenizer_path);
    
    let state = AppState {
        inference_tx,
        tokenizer: Arc::new(tokenizer),
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", axum::routing::get(health_handler))
        .route("/chat", post(chat_handler))
        .layer(cors)
        .with_state(state);

    let addr = "0.0.0.0:3001";
    println!("\n4. Starting server on http://{}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}
