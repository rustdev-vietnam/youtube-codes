use tch::{Device, Kind, Tensor, Cuda};

fn main() {
    let cuda_device = Device::cuda_if_available();

    if !cuda_device.is_cuda() {
        println!("Không tìm thấy CUDA!");
        return;
    }

    let size = 8192;
    let a_gpu = Tensor::randn([size, size], (Kind::Float, cuda_device));
    let b_gpu = Tensor::randn([size, size], (Kind::Float, cuda_device));

    // Khởi động GPU
    let _c_warmup = a_gpu.matmul(&b_gpu);
    Cuda::synchronize(0);
    
    let num_runs = 10;
    let mut total_time = std::time::Duration::ZERO;
    for _ in 0..num_runs {
        // Chờ GPU xong hết các nhiệm vụ trước đó nếu có
        Cuda::synchronize(0);
        
        let start = std::time::Instant::now();
        let c_gpu = a_gpu.matmul(&b_gpu);

        // Lấy ra kết quả nhân chỉ để đảm bảo phép nhân phải thực hiện xong mới
        // tiếp tục
        let _anything = c_gpu.get(0);

        total_time += start.elapsed();
    }
    
    let avg_time = total_time / num_runs;
    println!("(v2) Dùng tch-rs & libTorch để nhân ten-xơ: {:?}", avg_time);
}
