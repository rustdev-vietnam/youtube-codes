//! Chương trình minh họa sự khác nhau giữa năng lực tính toán của CPU và GPU 
//! khi cả hai đều thực hiện một nhiệm vụ.
use candle_core::{Device, Tensor, Result};
use std::time::Instant;

fn main() -> Result<()> {
    // [01]
    let cpu_device = Device::Cpu;
    println!("CPU: {:?}", cpu_device);
    
    let cuda_device = match Device::cuda_if_available(0) {
        Ok(device) => {
            println!("GPU: {:?}", device);
            device
        }
        Err(e) => {
            panic!("Không thấy GPU! ({})", e);
        }
    };
    
    // [02]
    let size = 4096;
    let cpu_tensor_a = Tensor::randn(0f32, 1.0, (size, size), &cpu_device)?;
    let cpu_tensor_b = Tensor::randn(0f32, 1.0, (size, size), &cpu_device)?;
    
    println!("\nThực hiện nhân ma trận trên CPU...");
    let start = Instant::now();
    let cpu_result: Tensor = cpu_tensor_a.matmul(&cpu_tensor_b)?;
    let cpu_time = start.elapsed();
    println!("Thời gian thực hiện: {:?}", cpu_time);
    println!("Ma trận kết quả    : {:?}", cpu_result.shape());

    // [03]
    let cuda_tensor_a = cpu_tensor_a.to_device(&cuda_device)?;
    let cuda_tensor_b = cpu_tensor_b.to_device(&cuda_device)?;
    
    println!("\nThực hiện nhân ma trận trên GPU...");
    let start = Instant::now();
    let cuda_result = cuda_tensor_a.matmul(&cuda_tensor_b)?;
    let cuda_time = start.elapsed();
    println!("Thời gian thực hiện: {:?}", cuda_time);
    println!("Ma trận kết quả    : {:?}", cuda_result.shape());
        
    // [04]
    println!("\nGPU / CPU = {:.2}x", 
        cpu_time.as_secs_f64() / cuda_time.as_secs_f64()
    );
   
    let _ = demonstrate_memory_constraints();

    Ok(())
}

fn demonstrate_memory_constraints() -> Result<()> {
    println!("\n=== RAM và VRAM ===");
    let cpu_device = Device::Cpu;
    let gpu_device = Device::cuda_if_available(0)?;

    //[05]
    let large_size = 65536;

    let _large_cpu_tensor = match Tensor::zeros((large_size, large_size), 
                                                candle_core::DType::F64, 
                                                &cpu_device)
    {
        Ok(t) => {
            println!("CPU ten-xơ trên RAM : {:?}", t.shape());
            Some(t)
        }
        Err(e) => {
            println!("Không đủ RAM: {}", e);
            None
        }
    };
 
    let _large_gpu_tensor = match Tensor::zeros((large_size, large_size), 
                                                candle_core::DType::F64, 
                                                &gpu_device)
    {
        Ok(t) => {
            println!("GPU ten-xơ trên VRAM : {:?}", t.shape());
            Some(t)
        }
        Err(e) => {
            println!("Không đủ VRAM: {}", e);
            None
        }
    };
 
    Ok(())
}
