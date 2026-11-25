use std::f32::consts::PI;
use ndarray::{Array1, Array2, s};
use rustfft::{FftPlanner, num_complex::Complex};
use plotters::prelude::*;

use asr_app_02::load_audio_mono;

const SOUND_FILE: &str =
                    "/home/trinhtuan/tuyenngondoclap_30-00s_16000_mono.wav";
const IMAGE_FILE: &str =
                    "plot_melspectrogram.png";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- 1. Tải nạp dữ liệu âm thanh vào ndarray::Array1 ---
    // [01]
    let (waveform, sr)  = load_audio_mono(SOUND_FILE, None)?;
    let waveform        = Array1::from(waveform); 

    println!("Dữ liệu âm thanh - Hình dạng: {:?}, Tsố lấy mẫu: {}",
        waveform.shape(), sr
    );

    // --- 2. Thiết lập các tham số phân tích phổ ---
    // [02]
    let n_fft = 400;
    let hop_length = 160;
    let n_mels = 80;
    let fmax = 8000.0;

    // --- 3. Tính STFT (Short-Time Fourier Transform) ---
    // [03]
    let stft_power = compute_power_spectrogram(&waveform, n_fft, hop_length);
    
    // --- 4. Tạo bộ lọc thông giải - Mel Filterbank ---
    // [04]
    //let mel_basis = asr_app_02::mel_filter_matrix(sr, n_fft, n_mels, fmax);

    let mel_bytes = include_bytes!("melfilters.bytes").as_slice();
    let mut mf = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
        mel_bytes, &mut mf
    );
    let mel_basis = convert_filters_to_ndarray(mf, n_mels)?;

    // --- 5. Tính Mel Spectrogram (nhân ma trận) ---
    // Công thức:
    //    Mel_Spectrogram = Mel_Basis * Power_Spectrogram
    // Hình dạng ma trện kết quả:
    //    (n_mels, n_freqs) dot (n_freqs, n_time) -> (n_mels, n_time)
    let mel_spec = mel_basis.dot(&stft_power);

    // --- 6. Chuyển đổi Power -> dB (Log Amplitude) ---
    // Mô phỏng lệnh: librosa.power_to_db(S, ref=np.max)
    let mel_spec_db = power_to_db(&mel_spec);

    // --- 7. Plotting ---
    plot_melspectrogram(&mel_spec_db, sr, hop_length, fmax, IMAGE_FILE)?;
    println!("Done! Saved to {}", IMAGE_FILE);

    Ok(())
}

// -----------------------------------------------------------------------------
// 2. DSP Functions (STFT & Mel)
// -----------------------------------------------------------------------------

/// Computes |STFT|^2 using rustfft and ndarray
/// Returns Array2 with shape (Frequency Bins, Time Frames)
fn compute_power_spectrogram(
    waveform: &Array1<f32>, n_fft: usize, hop_length: usize
) -> Array2<f32>
{
    // Tính số lượng cửa sổ / khung sẽ dùng để phân tích tần số, bỏ cửa sổ
    // không đủ độ dài cuối cùng (phép chia lấy phần nguyên). 
    let n_frames = (waveform.len() - n_fft) / hop_length + 1;
    let n_freqs = n_fft / 2 + 1;
    //println!("len: {len} -> n_frames: {n_frames} -> n_freqs: {n_freqs}");

    // 1. Prepare FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    
    // 2. Create Hann Window
    // w[n] = 0.5 * (1 - cos(2*pi*n / n_fft))
    let window = Array1::from_shape_fn(n_fft, |i| {
        0.5 * (1.0 - (2.0 * PI * i as f32 / n_fft as f32).cos())
    });

    // 3. Initialize Output Matrix (Columns = Time, Rows = Freq)
    // We store it as (n_freqs, n_frames) to match standard Linear Algebra notation
    // for the subsequent Mel Matrix multiplication.
    let mut spectrogram = Array2::<f32>::zeros((n_freqs, n_frames));

    // 4. Processing Loop
    let mut buffer = vec![Complex { re: 0.0, im: 0.0 }; n_fft];

    for t in 0..n_frames {
        let start = t * hop_length;
        let end = start + n_fft;
        
        // Extract frame using ndarray slicing
        let frame = waveform.slice(s![start..end]);

        // Apply Window & Copy to FFT buffer
        for (i, &sample) in frame.iter().enumerate() {
            buffer[i] = Complex { re: sample * window[i], im: 0.0 };
        }

        // Compute FFT
        fft.process(&mut buffer);

        // Calculate Magnitude Squared for first n_freqs bins (discard symmetric negative freqs)
        for f in 0..n_freqs {
            spectrogram[[f, t]] = buffer[f].norm_sqr();
        }
    }

    spectrogram
}


/// Converts a flat vector of Mel filters into a 2D ndarray preserving Structure-1.
///
/// # Arguments
/// * `flat_filters` - The flat vector of coefficients (e.g., loaded from bytes).
/// * `n_mels` - The number of Mel bands (rows), usually 80 for Whisper.
///
/// # Returns
/// * `Result<Array2<f32>, ...>` - An Array2 with shape (n_mels, calculated_bins).
fn convert_filters_to_ndarray(
    flat_filters: Vec<f32>, 
    n_mels: usize
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let total_len = flat_filters.len();

    // 1. Safety check to ensure the vector can be reshaped perfectly
    //if total_len == 0 || total_len % n_mels != 0 {
    if total_len == 0 || !total_len.is_multiple_of(n_mels) {
        return Err(format!(
            "Shape mismatch: Vector length {} is not divisible by n_mels {}", 
            total_len, n_mels
        ).into());
    }

    // 2. Calculate the number of frequency bins (columns)
    // For standard Whisper: 16080 / 80 = 201
    let n_freq_bins = total_len / n_mels;

    // 3. Create the Array2
    // Array2::from_shape_vec reads the vector in Row-Major order (C-order).
    // This perfectly matches Structure-1: It fills Row 0 first, then Row 1, etc.
    let matrix = Array2::from_shape_vec((n_mels, n_freq_bins), flat_filters)?;

    Ok(matrix)
}


// -----------------------------------------------------------------------------
// 3. Post-Processing (dB conversion)
// -----------------------------------------------------------------------------

/// Convert Power Spectrogram to Decibels
/// 10 * log10(S / ref) where ref is max(S)
fn power_to_db(spectrogram: &Array2<f32>) -> Array2<f32> {
    // 1. Find Reference Value (Max of the whole array)
    // unwrap_or is safety for empty arrays
    let ref_value = spectrogram.iter().fold(0.0f32, |a, &b| a.max(b));
    let min_threshold = 1e-10; // Avoid log(0)

    // 2. Compute dB using ndarray mapv (Vectorized map)
    spectrogram.mapv(|x| {
        let val = x / ref_value;
        let val = val.max(min_threshold);
        10.0 * val.log10()
    })
}

// -----------------------------------------------------------------------------
// 4. Trực quan hóa dùng plotters
// -----------------------------------------------------------------------------

/// Approximate Magma Colormap (Black -> Purple -> Red -> Yellow)
fn get_magma_color(t: f32) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    // Key color stops (Approximate Magma)
    let stops = [
        (0.0, (1, 1, 5)),         // Very dark (almost black)
        (0.25, (80, 18, 123)),    // Purple
        (0.5, (182, 54, 121)),    // Magenta
        (0.75, (251, 135, 97)),   // Orange-Peach
        (1.0, (252, 253, 191)),   // Light Yellow
    ];

    // Find segment
    for i in 0..stops.len()-1 {
        if t >= stops[i].0 && t <= stops[i+1].0 {
            let local_t = (t - stops[i].0) / (stops[i+1].0 - stops[i].0);
            let (r1, g1, b1) = stops[i].1;
            let (r2, g2, b2) = stops[i+1].1;
            
            return RGBColor(
                (r1 as f32 + local_t * (r2 as f32 - r1 as f32)) as u8,
                (g1 as f32 + local_t * (g2 as f32 - g1 as f32)) as u8,
                (b1 as f32 + local_t * (b2 as f32 - b1 as f32)) as u8,
            );
        }
    }
    RGBColor(252, 253, 191)
}

fn plot_melspectrogram(
    spec_db: &Array2<f32>, 
    sr: u32, 
    hop_length: usize,
    fmax: f32,
    out_path: &str
) -> Result<(), Box<dyn std::error::Error>> {
    let margin_left  = 20;  // không gian để trống cạnh trái (từ lề đến vùng vẽ)
    let margin_right = 40;  // không gian để trộng cạnh phải
    let margin_bottom= 10;  // không gian để trống cạnh dưới
    let margin_top   = 10;  // không gian để trống cạnh trên
    let y_label_room = 80;  // không gian cho nhãn trục Y
    let x_label_room = 40;  // không gian cho nhãn trục X
    let chart_width  = 1200;// độ rộng của riêng phần nội dung biểu đồ

    // Tạo ảnh có tỉ lệ dài / rộng = 2.5; độ phân giải sẽ được tính toán sao
    // cho phần nội dung biểu đồ sẽ có chiều dài tính theo pixel là chart_width.
    // BitMapBackend của plotters mặc định sẽ tạo các ảnh PNG với độ sâu 32-bit
    // RGBA màu.
    // Tham số IMAGE_FILE sẽ quyết định ảnh sẽ được vẽ ra đâu, ở đây sẽ là ra
    // tệp tin.
    let root = BitMapBackend::new(
        out_path,
        (
            chart_width + margin_left + margin_right,
            (chart_width as f32 / 2.5) as u32
                + margin_top + margin_bottom + x_label_room
        )
    ).into_drawing_area();
    root.fill(&WHITE)?;

    // Chia toan vẽ thành hai phần: bên trái cho đồ thị Mel Spectrogram (1160)
    // và bên phải (chart_width-1160) cho chú giải nhiệt & cường độ.
    let (chart_area, colorbar_area) = root.split_horizontally(1160);

    let (n_mels, n_frames) = spec_db.dim();
    let duration = (n_frames * hop_length) as f32 / sr as f32;

    // Định nghĩa các hàm chuyển đổi Mel <-> Hz theo công thức HTK/Slaney giản
    // lược.
    let hz_to_mel = |f: f32| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f32| 700.0 * (10.0f32.powf(m / 2595.0) - 1.0);

    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(fmax);

    // ------------------------------------------------------------------------
    // Vẽ Phần Đồ Thị Mel Spectrogram
    // ------------------------------------------------------------------------
    let mut chart = ChartBuilder::on(&chart_area)
        .margin_left(margin_left)
        .margin_right(margin_right - 30)
        .margin_top(margin_top)
        .margin_bottom(margin_bottom)
        .x_label_area_size(x_label_room + margin_top)
        .y_label_area_size(y_label_room + margin_left)
        .caption("Mel Spectrogram", ("Arial", 28))
        // We use a coordinate system of [Time] x [Mel Index (0..128)]
        .build_cartesian_2d(0f32..duration, 0f32..(n_mels as f32))?;

    // Custom Axis Configuration to mimic Librosa's "Mel" axis
    chart.configure_mesh()
        .x_desc("Time")
        .y_desc("Hz")
        .label_style(("Arial", 18).into_font())
        // Custom Formatter: Convert Mel Index Y back to Frequency string
        .y_label_formatter(&|y_idx| {
            let pct = y_idx / n_mels as f32; // 0.0 to 1.0
            let m = min_mel + pct * (max_mel - min_mel); // Interpolate Mel
            let f = mel_to_hz(m); // Convert to Hz
            
            // Nếu muốn định dạng lại cách hiển thị giá trị trên trục Y
            //if f >= 1000.0 {
            //    format!("{:.0}k", f / 1000.0)
            //} else {
                format!("{:.0}", f)
            //}
        })
        // Có thể tự vẽ đơn vị trục Y bằng cách đổi thành .y_labels(0) và dùng
        // khối lệnh tự vẽ đơn vị trục Y ngay dưới.
        // Nếu không muốn tự vẽ, đặt thành .y_labels(10) hoặc bỏ hẳn để dùng
        // giá trị mặc định tương đương .y_labels(11) đồng thời vô hiệu khối
        // lệnh vẽ đơn vị trục Y ở dưới.
        .y_labels(0) 
        .draw()?;

    //--- Bắt Đầu - Khối lệnh (tùy chọn) tự vẽ đơn vị trục Y
    // Khối lệnh này sẽ tính toán chính xác vị trí các đơn vị tần số trên trục
    // Y theo thang tỉ lệ Mel. Do chỉ quan tâm đến dải tần số liên quan xử lý
    // tiếng nói với tần số lấy mẫu 16KHz nên dừng lại ở 8192Hz.
    let target_freqs = vec![0.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0];

    // Lấy hệ tọa độ của toan vẽ làm cơ sở tính vị trí vẽ & đặt kiểu chữ.
    let coord_spec = chart.as_coord_spec();
    let font = ("Arial", 18).into_font();

    for &f in &target_freqs {
        // Không vẽ tần số ngoài dải quan tâm
        if f > fmax { continue; }

        // Chuyển đổi từ thang đo Hz -> Mel
        let m = hz_to_mel(f);
        
        // Chuyển đổi từ đơn vị Mel -> tọa độ trên trục Y (0..n_mels=128)
        // Vì trục Y biểu diễn tuyến tính 0..n_mels nên sẽ cần ánh xạ các giá
        // trị Mel "m" -> vào dải 0..n_mels. (giống cách làm normalization)
        let y_chart_val = (m - min_mel) / (max_mel - min_mel) * n_mels as f32;

        // Lấy tọa độ tuyệt đối điểm vẽ trên toan vẽ. Về cơ bản là trả lời câu
        // hỏi "Điểm trên đồ thị có cặp giá trị (x=0, y=y_chart_val) nằm ở đâu
        // trên toan vẽ?"
        let (x_px, y_px) = coord_spec.translate(&(0.0, y_chart_val));

        // Vẽ đánh dấu tick trên trục Y (một đường kẻ nhỏ)
        root.draw(&PathElement::new(
            vec![(x_px - 5, y_px), (x_px, y_px)], 
            BLACK
        ))?;

        // Vẽ nhãn biểu diễn giá trị tần số tương ứng bên cạnh dấu tick
        let label_text = format!("{:.0}", f);
        let (w, h) = font.box_size(&label_text).unwrap_or((20, 10));
        root.draw(&Text::new(
            label_text,
            // Căn bên phải theo chiều ngang và căn giữa theo chiều dọc
            (x_px - 8 - w as i32, y_px - (h as i32 / 2)),
            font.clone()
        ))?;
    }
    //--- Kết thúc - Khối lệnh (tùy chọn) tự vẽ đơn vị trục Y

    let max_db = 0.0;
    let min_db = -80.0;

    // Vẽ nội dung nhiệt màu của Mel Spectrogram
    chart.draw_series(
        (0..n_mels).flat_map(|row_idx| {
            (0..n_frames).map(move |col_idx| {
                let db_val = spec_db[[row_idx, col_idx]];
                let norm_val = ((db_val - min_db) / (max_db - min_db)).clamp(0.0, 1.0);

                let x0 = col_idx as f32 * (duration / n_frames as f32);
                let x1 = (col_idx + 1) as f32 * (duration / n_frames as f32);
                // Y coordinates are simply the indices 0..128
                let y0 = row_idx as f32;
                let y1 = (row_idx + 1) as f32;

                Rectangle::new(
                    [(x0, y0), (x1, y1)],
                    get_magma_color(norm_val).filled(),
                )
            })
        })
    )?;

    // ------------------------------------------------------------------------
    // Vẽ Phần Diễn Giải Nhiệt Màu
    // ------------------------------------------------------------------------
    let mut cb_chart = ChartBuilder::on(&colorbar_area)
        .margin_top(10)
        .margin_bottom(10)
        .margin_right(20)
        .margin_top(margin_top + x_label_room)
        .margin_bottom(margin_bottom + x_label_room + 10)
        .margin_right(margin_right - 30)
        .margin_left(0)
        // Put labels on the RIGHT side of the bar (standard for colorbars)
        .set_label_area_size(LabelAreaPosition::Right, y_label_room - 20)
        .set_label_area_size(LabelAreaPosition::Left, 0) // No labels on left

//        .set_label_area_size(LabelAreaPosition::Right, 50) // Space for labels
        .build_cartesian_2d(0f32..1f32, min_db..max_db)?; // X is dummy, Y is dB

    // Draw the gradient
    // We draw 200 small rectangles stacked vertically to create a smooth gradient
    let steps = 200;
    let step_size = (max_db - min_db) / steps as f32;

    cb_chart.draw_series(
        (0..steps).map(|i| {
            let y0 = min_db + i as f32 * step_size;
            let y1 = min_db + (i + 1) as f32 * step_size;
            let norm = i as f32 / steps as f32; // Map 0..1 for color function

            Rectangle::new(
                [(0.0, y0), (1.0, y1)], // Fill full width of colorbar area
                get_magma_color(norm).filled(),
            )
        })
    )?;

    cb_chart.configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .disable_x_axis()
        .label_style(("Arial", 18).into_font())
        .y_desc("dB")
        .y_label_formatter(&|y| format!("{:.0}", y))
        .draw()?;

    Ok(())
}
