use candle_core::{Result, Device, Tensor};

fn main() -> Result<()> {
    let cuda_device = Device::cuda_if_available(0)?; 

    let size = 8192;
    let a_gpu = Tensor::randn(0.0f32, 1.0, &[size, size], &cuda_device)?;
    let b_gpu = Tensor::randn(0.0f32, 1.0, &[size, size], &cuda_device)?;

    // Khởi động GPU
    let _c_warmup = a_gpu.matmul(&b_gpu)?;
    cuda_device.synchronize()?;
    
    let num_runs = 10;
    let mut total_time = std::time::Duration::ZERO;
    
    for _ in 0..num_runs {
        // Chờ GPU xong hết các nhiệm vụ trước đó nếu có
        cuda_device.synchronize()?;
        
        let start = std::time::Instant::now();
        let c_gpu = a_gpu.matmul(&b_gpu)?;

        // Lấy ra kết quả nhân chỉ để đảm bảo phép nhân phải thực hiện xong mới
        // tiếp tục
        let _anything = c_gpu.get(0)?;

        total_time += start.elapsed();
    }
    
    let avg_time = total_time / num_runs;
    println!("(v2) Dùng candle để nhân ten-xơ: {:?}", avg_time);

    Ok(())
}
