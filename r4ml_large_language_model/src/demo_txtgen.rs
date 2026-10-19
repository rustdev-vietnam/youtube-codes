//! Chương trình minh họa tạo sinh văn bản dùng mô hình Qwen
//!
use rust_002_51::ModelIdx;

use candle_core::{DType, Device, Tensor, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{
    ModelForCausalLM as Qwen2Model, Config as Qwen2Config
};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
//use std::io::Write;

const MODEL_ID: &str = "Qwen/Qwen2.5-3B";
const MODEL_RV: &str = " ";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 01 - Tải mô hình về máy
    let model_idx = ModelIdx::new(MODEL_ID, Some(MODEL_RV)).await?;
    let (tok_file, cfg_file, sft_files) = model_idx.get_model_files().await?;
   
    // 02 - Khởi tạo các đối tượng mô hình dành cho nhiệm vụ suy luận
    // [02-01]
    let tokenizer: Tokenizer = Tokenizer::from_file(&tok_file)
        .map_err(|e| format!("{e}"))?;
    let eos_token_id = tokenizer.token_to_id("<|enoftext|>").unwrap_or(151643);
/*  // [02-02]
    let device = match Device::cuda_if_available(0) {
        Ok(cuda) => cuda,
        _ => { println!("No CUDA found!"); Device::Cpu }
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&sft_files, DType::BF16, &device)?
    };
    };
*/
    // [02-02]
    let device = Device::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&sft_files, DType::F16, &device)?
    };

    // [02-03]
    let model_config: Qwen2Config = serde_json::from_slice(
        &std::fs::read(cfg_file)?
    )?;
    let mut model: Qwen2Model = Qwen2Model::new(&model_config, vb)?; 

    // 03 - Chuẩn bị dữ liệu đầu vào cho mô hình
    // [03-01]
    let prompt: &str = "Rust programming language is";
    //let prompt: &str = "Ngôn ngữ lập trình Rust là";
    let prompt_tokens: Vec<u32> = tokenizer
        .encode(prompt, true)
        .map_err(|e| format!("{e}"))?
        .get_ids()
        .to_vec();
    //println!("{:?}", prompt_tokens);

    // [03-02]
    let mut logits_processor = LogitsProcessor::new(
        123456789, Some(0.7), Some(0.8)
    );

    // Lưu tất cả token (prompt + generated)
    let mut all_tokens = prompt_tokens.clone();

    // Xóa KV cache trước khi bắt đầu
    model.clear_kv_cache();

    // 04 - Thực hiện suy luận, tạo sinh văn bản
    // [04-01]
    let sample_length = 64;

    // [04-02]
    for i in 0..sample_length {
        let (input_ids, context_len) = if i == 0 {
            // Lần chạy đầu tiên: xử lý toàn bộ prompt
            let tensor = Tensor::new(vec![all_tokens.clone()], &device)?;
            (tensor, 0)
        } else {
            // Các lần chạy tiếp theo: chỉ xử lý token cuối cùng (nhờ KV cache)
            // [04-03]
            let last_token = *all_tokens.last().unwrap();
            let tensor = Tensor::new(vec![vec![last_token]], &device)?;
            (tensor, all_tokens.len() - 1)
        };
        //print!("{:?}", input_ids.shape());
        print!("{context_len} -> ");

        // [04-04]
        // logits sẽ có hình dạng: [batch_size, 1, vocab_size]
        let logits: Tensor = model.forward(&input_ids, context_len)?;
        //println!("{logits:?}");

        // logits sẽ có hình dạng: [vocab_size]
        let logits: Tensor = logits.squeeze(0)?.i(logits.dim(1)? - 1)?;
        //println!("{logits:?}");

        // Lấy mẫu (nucleus sampling) logits để sinh ra token ID đại diện cho 
        // từ tiếp theo.
        let next_token_id: u32 = logits_processor.sample(&logits)?;

        // Dừng nếu gặp token kết thúc (End Of Stream). Tình huống này khó xảy
        // ra nhưng để đảm bảo có kết quả đẹp mắt vẫn giữ lại.
        if next_token_id == eos_token_id {
            println!("\n[Đã gặp token kết thúc]");
            break;
        }

        // Nối thêm token mới sinh vào chuỗi token kết quả.
        all_tokens.push(next_token_id);
/*
        let dbg = tokenizer
            .decode(&[next_token_id], true)
            .unwrap_or("".to_string());
        println!("{}", dbg);
*/
    }

    println!("\n--- Hoàn thành ---");

    // 05 - Giải mã và in kết quả tạo sinh
    let final_text: String = tokenizer
        .decode(&all_tokens, true)
        .unwrap_or("".to_string());
    println!("{} {}", prompt, final_text);

    Ok(())
}
