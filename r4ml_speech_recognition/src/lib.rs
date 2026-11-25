#![allow(dead_code)]

use std::fs::File;
use std::error::Error as StdErr;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Tự động dò tìm định dạng tệp âm thanh bất kỳ rồi đọc dữ liệu âm thanh và
/// trả về cùng thông số tần số lấy mẫu. Hiện tại phiên bản này chỉ đọc và trả
/// về dữ liệu âm thanh mono.
///
/// # Tham số đầu vào
/// * path: chuỗi ký tự biễu diên đường dẫn tuyệt đối đến tệp âm thanh.
/// * limit_samples: nếu là None sẽ đọc toàn bộ dữ liệu tệp âm thanh, nếu là
///   Some(x) thì chỉ đọc x giá trị âm thanh tính từ vị trí đầu tiên của dữ
///   liệu âm thanh.
///
/// # Kết quả trả về
/// * `Ok((sound_data, sample_rate))` - `sound_data` là dữ liệu âm thanh dạng 
///   Vec<f32> và `sample_rate` là tần số lấy mẫu tương ứng.
/// * `Err` - Nếu đọc tệp dữ liệu không thành công vì bất kỳ lý do gì.
///
/// # Ví dụ
/// ```
/// let (sound_data, rate) = load_audio_stereo("audio.ogg", None)?;
/// println!("Loaded {} samples at {} Hz", sound_data.len(), rate);
/// ```
pub fn load_audio_mono(
    path: &str,
    limit_samples: Option<usize>
) -> Result<(Vec<f32>, u32), Box<dyn StdErr>>
{
    let src = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let mut hinted = Hint::new();
    hinted.with_extension("ogg");

    let format_opts   = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let decoder_opts  = DecoderOptions::default();

    let probed = symphonia::default::get_probe().format(
        &hinted, mss, &format_opts, &metadata_opts
    )?;
    let mut format = probed.format;

    // [mono-01]
    let track = format.default_track().ok_or(
        "Không xác định được track mặc định!"
    )?;

    let sample_rate = track.codec_params.sample_rate.ok_or(
        "Không xác định được tần số lấy mẫu!"
    )?;

    let mut decoder = symphonia::default::get_codecs().make(
        &track.codec_params, &decoder_opts
    )?;

    let track_id = track.id;
    let mut audio_data = Vec::new();

    // Vòng lặp giải mã dữ liệu âm thanh.
    loop {
        // [mono-02]
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(_)) => break,        // EOF
            Err(e) => return Err(Box::new(e)),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet)?;
        let mut sample_buf = SampleBuffer::<f32>::new(
            decoded.capacity() as u64, *decoded.spec()
        );
        sample_buf.copy_interleaved_ref(decoded.clone());

        // Mix to mono (average all channels)
        let channels = decoded.spec().channels.count();
        let samples = sample_buf.samples();
        
        for frame in samples.chunks(channels) {
            let mut sum = 0.0;
            for &sample in frame {
                sum += sample;
            }
            audio_data.push(sum / channels as f32);
            
            if let Some(limit) = limit_samples && audio_data.len() >= limit {
                break;
            }
        }

        if let Some(limit) = limit_samples && audio_data.len() >= limit {
            break;
        }
    }

    Ok((audio_data, sample_rate))
}


pub fn mel_filter_matrix(
    sr: u32, n_fft: usize, n_mels: usize, fmax: f32
) -> ndarray::Array2<f32>
{
    let n_freqs = n_fft / 2 + 1;
    let fmin = 0.0;

    // Helper closures for Hz <-> Mel conversion
    let hz_to_mel = |f: f32| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f32| 700.0 * (10.0f32.powf(m / 2595.0) - 1.0);

    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);

    // Create mel bin points
    let mel_points = ndarray::Array1::linspace(min_mel, max_mel, n_mels + 2);
    let hz_points = mel_points.mapv(mel_to_hz);

    // Create FFT bin frequencies
    // linspace from 0 to sr/2
    let fft_freqs = ndarray::Array1::linspace(0.0, sr as f32 / 2.0, n_freqs);

    let mut weights = ndarray::Array2::<f32>::zeros((n_mels, n_freqs));

    for i in 0..n_mels {
        // Get the triangle limits
        let f_prev = hz_points[i];
        let f_curr = hz_points[i+1];
        let f_next = hz_points[i+2];

        // Fill the row for this mel band
        for j in 0..n_freqs {
            let f = fft_freqs[j];
            if f >= f_prev && f < f_curr {
                weights[[i, j]] = (f - f_prev) / (f_curr - f_prev);
            } else if f >= f_curr && f < f_next {
                weights[[i, j]] = (f_next - f) / (f_next - f_curr);
            }
        }

        // Slaney Normalization (Area normalization)
        // To match Librosa default: divide by the width of the mel band
        let enorm = 2.0 / (f_next - f_prev);
        let mut row = weights.row_mut(i);
        row.mapv_inplace(|x| x * enorm);
    }

    weights
}
