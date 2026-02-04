//! Burn inference for Qwen3-0.6B Q4 model with KV cache and autoregressive generation

use anyhow::Result;
use burn::prelude::*;
use burn::backend::Wgpu;
use tokenizers::Tokenizer;
use std::time::Instant;

// Include the generated model
#[path = "../../burn_qoperator_model/model.rs"]
mod model;

use model::Model;

type MyBackend = Wgpu;

const NUM_LAYERS: usize = 28;
const NUM_KV_HEADS: usize = 8;  // Qwen3 uses GQA with 8 KV heads
const HEAD_DIM: usize = 128;  // Head dimension from ONNX model
const EOS_TOKEN_ID: i64 = 151645; // <|im_end|>
const MAX_NEW_TOKENS: usize = 128;

/// KV cache for all 28 transformer layers
pub struct KvCache<B: Backend> {
    pub keys: Vec<Tensor<B, 4>>,
    pub values: Vec<Tensor<B, 4>>,
}

impl<B: Backend> KvCache<B> {
    /// Create empty KV cache (uses seq_len=1 with zeros since WGPU can't handle size 0)
    pub fn new(batch_size: usize, device: &B::Device) -> Self {
        // WGPU doesn't support 0-sized tensors, so we start with seq_len=1
        // The attention mask will handle masking these out
        let keys: Vec<Tensor<B, 4>> = (0..NUM_LAYERS)
            .map(|_| Tensor::zeros([batch_size, NUM_KV_HEADS, 1, HEAD_DIM], device))
            .collect();
        let values: Vec<Tensor<B, 4>> = (0..NUM_LAYERS)
            .map(|_| Tensor::zeros([batch_size, NUM_KV_HEADS, 1, HEAD_DIM], device))
            .collect();
        Self { keys, values }
    }

    /// Get current sequence length from cache (subtract 1 for the dummy initial token)
    pub fn seq_len(&self) -> usize {
        if self.keys.is_empty() {
            0
        } else {
            self.keys[0].dims()[2].saturating_sub(1)
        }
    }
}

/// Wrapper for Qwen3 model with cleaner interface
pub struct Qwen3<B: Backend> {
    model: Model<B>,
    device: B::Device,
}

impl<B: Backend> Qwen3<B> {
    pub fn from_file(path: &str, device: &B::Device) -> Self {
        let model: Model<B> = Model::from_file(path, device);
        Self {
            model,
            device: device.clone(),
        }
    }

    /// Run forward pass with KV cache, returns (logits, updated_cache)
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2, Int>,
        position_ids: Tensor<B, 2, Int>,
        cache: KvCache<B>,
    ) -> (Tensor<B, 3>, KvCache<B>) {
        // Unpack cache into individual tensors
        let result = self.model.forward(
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

        // Repack the output cache
        let (logits, k0, v0, k1, v1, k2, v2, k3, v3, k4, v4, k5, v5, k6, v6, k7, v7,
             k8, v8, k9, v9, k10, v10, k11, v11, k12, v12, k13, v13, k14, v14, k15, v15,
             k16, v16, k17, v17, k18, v18, k19, v19, k20, v20, k21, v21, k22, v22, k23, v23,
             k24, v24, k25, v25, k26, v26, k27, v27) = result;

        let new_cache = KvCache {
            keys: vec![k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13,
                      k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27],
            values: vec![v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
                        v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27],
        };

        (logits, new_cache)
    }

    /// Sample next token using greedy decoding (argmax)
    pub fn sample_greedy(&self, logits: &Tensor<B, 3>) -> i64 {
        // logits shape: [batch, seq_len, vocab_size]
        // Get last token's logits
        let dims = logits.dims();
        let last_logits = logits.clone().slice([0..1, (dims[1] - 1)..dims[1], 0..dims[2]]);
        // Flatten to 1D: [1, 1, vocab_size] -> [vocab_size]
        let last_logits: Tensor<B, 1> = last_logits.flatten(0, 2);
        
        // Argmax returns I32, convert to i64
        let token_id = last_logits.argmax(0);
        let token_data: TensorData = token_id.into_data();
        token_data.as_slice::<i32>().unwrap()[0] as i64
    }

    /// Generate text autoregressively
    pub fn generate(
        &self,
        mut input_ids: Vec<i64>,
        tokenizer: &Tokenizer,
        max_new_tokens: usize,
    ) -> Result<String> {
        let batch_size = 1;
        let mut cache = KvCache::<B>::new(batch_size, &self.device);
        let mut generated_tokens = Vec::new();

        // Initial prefill with full prompt
        let seq_len = input_ids.len();
        // Use explicit I64 dtype to match model's expectations
        let input_tensor: Tensor<B, 2, Int> = Tensor::from_data_dtype(
            TensorData::new(input_ids.clone(), [batch_size, seq_len]),
            &self.device,
            burn::tensor::DType::I64,
        );
        // Attention mask should cover past_seq_len + current_seq_len
        // Initially cache has 1 dummy token that should be MASKED OUT (0)
        // Then seq_len real tokens that should be attended (1)
        let total_seq_len = 1 + seq_len;  // dummy cache token + input tokens
        let mut mask_data: Vec<i64> = Vec::with_capacity(total_seq_len);
        mask_data.push(0i64);  // Mask out the dummy cache token
        mask_data.extend(vec![1i64; seq_len]);  // Attend to real tokens
        let attention_mask: Tensor<B, 2, Int> = Tensor::from_data_dtype(
            TensorData::new(mask_data, [batch_size, total_seq_len]),
            &self.device,
            burn::tensor::DType::I64,
        );
        let position_ids: Tensor<B, 2, Int> = Tensor::from_data_dtype(
            TensorData::new((0..seq_len as i64).collect::<Vec<_>>(), [batch_size, seq_len]),
            &self.device,
            burn::tensor::DType::I64,
        );

        println!("   Prefill ({} tokens)...", seq_len);
        let start = Instant::now();
        let (logits, mut cache) = self.forward(input_tensor, attention_mask, position_ids, cache);
        let prefill_time = start.elapsed();
        println!("   Prefill completed in {:.2}s ({:.1} tokens/s)", 
                 prefill_time.as_secs_f32(), 
                 seq_len as f32 / prefill_time.as_secs_f32());

        // Sample first token
        let mut next_token = self.sample_greedy(&logits);
        generated_tokens.push(next_token);
        print!("{}", tokenizer.decode(&[next_token as u32], false).unwrap_or_default());

        // Autoregressive generation loop
        let gen_start = Instant::now();
        for i in 1..max_new_tokens {
            if next_token == EOS_TOKEN_ID {
                println!("\n   [EOS reached at token {}]", i);
                break;
            }

            let pos = seq_len + i;
            
            // Single token input with explicit I64 dtype
            let input_tensor: Tensor<B, 2, Int> = Tensor::from_data_dtype(
                TensorData::new(vec![next_token], [1, 1]),
                &self.device,
                burn::tensor::DType::I64,
            );
            
            // Attention mask covers past_seq_len + 1 (new token)
            // First position is the dummy token (mask=0), rest are real tokens (mask=1)
            let cache_len = cache.keys[0].dims()[2];  // actual cache dimension including dummy
            let total_mask_len = cache_len + 1;  // cache + new token
            let mut mask_data: Vec<i64> = Vec::with_capacity(total_mask_len);
            mask_data.push(0i64);  // Mask out the dummy cache token
            mask_data.extend(vec![1i64; total_mask_len - 1]);  // Attend to real tokens
            let attention_mask: Tensor<B, 2, Int> = Tensor::from_data_dtype(
                TensorData::new(mask_data, [1, total_mask_len]),
                &self.device,
                burn::tensor::DType::I64,
            );
            
            // Position ID for current token with explicit I64 dtype
            let position_ids: Tensor<B, 2, Int> = Tensor::from_data_dtype(
                TensorData::new(vec![pos as i64 - 1], [1, 1]),
                &self.device,
                burn::tensor::DType::I64,
            );

            let (logits, new_cache) = self.forward(input_tensor, attention_mask, position_ids, cache);
            cache = new_cache;

            next_token = self.sample_greedy(&logits);
            generated_tokens.push(next_token);
            
            // Stream output
            let token_str = tokenizer.decode(&[next_token as u32], false).unwrap_or_default();
            print!("{}", token_str);
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
        println!();

        let gen_time = gen_start.elapsed();
        let tokens_generated = generated_tokens.len() - 1; // exclude first token from prefill
        if tokens_generated > 0 {
            println!("   Generation: {} tokens in {:.2}s ({:.1} tokens/s)",
                     tokens_generated, gen_time.as_secs_f32(),
                     tokens_generated as f32 / gen_time.as_secs_f32());
        }

        // Decode full response
        let response = tokenizer.decode(
            &generated_tokens.iter().map(|&t| t as u32).collect::<Vec<_>>(),
            true
        ).unwrap_or_default();

        Ok(response)
    }
}

fn main() -> Result<()> {
    println!("=== Burn Qwen3-0.6B Q4 Inference ===\n");
    
    // Initialize WGPU backend
    println!("1. Initializing WGPU backend...");
    let device = burn::backend::wgpu::WgpuDevice::default();
    println!("   Device: {:?}", device);
    
    // Load tokenizer
    println!("\n2. Loading tokenizer...");
    let tokenizer_path = "/Users/perro/work/hello_michi/onnx_models/qwen3-0.6b-qoperator/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!("   Vocab size: {}", tokenizer.get_vocab_size(true));
    
    // Load the burn model
    println!("\n3. Loading Qwen3 model...");
    let model_path = "/Users/perro/work/hello_michi/burn_qoperator_model/model.bpk";
    let start = Instant::now();
    let model = Qwen3::<MyBackend>::from_file(model_path, &device);
    println!("   Model loaded in {:.2}s", start.elapsed().as_secs_f32());
    
    // Prepare prompt
    println!("\n4. Generating response...");
    let prompt = "<|im_start|>system\nYou are Michi, a helpful AI assistant.<|im_end|>\n<|im_start|>user\nHello! What can you help me with?<|im_end|>\n<|im_start|>assistant\n";
    
    let encoding = tokenizer.encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    println!("   Prompt tokens: {}", input_ids.len());
    
    // Generate
    println!("\n--- Response ---");
    let response = model.generate(input_ids, &tokenizer, MAX_NEW_TOKENS)?;
    println!("--- End Response ---\n");
    
    println!("=== Inference Complete ===");
    
    Ok(())
}
