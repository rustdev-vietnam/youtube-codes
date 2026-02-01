//! Chương trình minh họa dung sai giữa các phép tính được thực hiện trên CPU 
//! và GPU.
use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> 
{
    if !candle_core::utils::cuda_is_available() {
        println!("Không có CUDA !!!");
        return Ok(());
    }
    let cpu_device = Device::Cpu;
    let cuda_device = Device::new_cuda(0)?;

    let m = 1024;
    let k = 2048;
    let n = 1024;

    // [01]
    let a_cpu = Tensor::randn(0.0f32, 1.0f32, (m, k), &cpu_device)?;
    let b_cpu = Tensor::randn(0.0f32, 1.0f32, (k, n), &cpu_device)?;

    let c_cpu = a_cpu.matmul(&b_cpu)?;

    let a_gpu = a_cpu.to_device(&cuda_device)?;
    let b_gpu = b_cpu.to_device(&cuda_device)?;

    let c_gpu = a_gpu.matmul(&b_gpu)?;

    let c_gpu_on_cpu = c_gpu.to_device(&cpu_device)?;

    // [02] 
    let vec_cpu = c_cpu.flatten_all()?.to_vec1::<f32>()?;
    let vec_gpu = c_gpu_on_cpu.flatten_all()?.to_vec1::<f32>()?;

    let mut max_abs_diff = 0.0f32;
    let mut total_diff_elements = 0;

    for i in 0..vec_cpu.len() {
        let diff = (vec_cpu[i] - vec_gpu[i]).abs();
        if diff > 0.0 {
            total_diff_elements += 1;
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
        }
    }

    // [03]
    println!("\n--- BÁO CÁO DUNG SAI ---");
    println!("Tổng số phần tử (CPU | GPU): {}", vec_cpu.len());
    println!("Số phần tử sai lệch        : {}", total_diff_elements);
    println!("Dung sai cao nhất          : {:.10e}", max_abs_diff);
    println!("\n5 phần tử đầu tiên (CPU)   : {:?}", &vec_cpu[0..5]);
    println!("5 phần tử đầu tiên (GPU)   : {:?}", &vec_gpu[0..5]);

    Ok(())
}
