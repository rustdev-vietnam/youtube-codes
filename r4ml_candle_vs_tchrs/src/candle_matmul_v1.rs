//! Đo tốc độ thi hành phép nhân ma trận dùng libtorch
//! 
//! Phiên bản này đo chưa thực sự chính xác do cơ chế tương tác với GPU là asyn
//! và bản chất của GPU là nhận lệnh từ CPU rồi đưa vào hàng đợi trước khi thi
//! hành nên đôi khi nếu GPU đang bận xử lý việc khác thì sẽ ảnh hưởng đến phép
//! đo.
use candle_core::{Result, Device, Tensor};

fn  main() -> Result<()> {
    let cuda_device = Device::cuda_if_available(0)?; 

    let size  = 8192;
    let a_gpu = Tensor::randn(0.0f32, 1.0, &[size, size], &cuda_device)?;
    let b_gpu = Tensor::randn(0.0f32, 1.0, &[size, size], &cuda_device)?;

    let start  = std::time::Instant::now();
    let _c_gpu = a_gpu.matmul(&b_gpu)?;

    println!("(v1) Dùng candle để nhân ten-xơ: {:?}", start.elapsed());

    Ok(())
}
