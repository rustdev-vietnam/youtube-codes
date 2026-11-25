use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::whisper::{self as m, audio, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::SeedableRng;
use tokenizers::Tokenizer;

use std::error::Error as StdErr;

pub enum Model
{
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

impl Model
{
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    pub fn encoder_forward(
        &mut self, x: &Tensor, flush: bool
    ) -> candle_core::Result<Tensor>
    {
        match self {
            Self::Normal(m)    => m.encoder.forward(x, flush),
            Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    // [Model::decoder_forward - 00]
    pub fn decoder_forward(
        &mut self,   x: &Tensor,
        xa: &Tensor, flush: bool,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m)    => m.decoder.forward(x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    pub fn decoder_final_linear(
        &self, x: &Tensor
    ) -> candle_core::Result<Tensor>
    {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Segment { start: f64, duration: f64, dr: DecodingResult, }

#[allow(dead_code)]
struct Decoder {
    model: Model,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    timestamps: bool,
    max_initial_timestamp_index: Option<u32>,
    verbose: bool,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

impl Decoder
{
    // [Decoder::new - 00]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model  : Model,       tokenizer     : Tokenizer, seed     : u64,
        device : &Device,    language_token : Option<u32>,
        task   : Option<Task>, timestamps   : bool,
        max_initial_timestamp_index         : Option<u32>, verbose: bool,
    ) -> Result<Self, Box<dyn StdErr>>
    {
        // [Decoder::new - 01]
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i)
                    || timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens  = Tensor::new(suppress_tokens.as_slice(), device)?;
        
        // [Decoder::new - 02]
        let sot_token        = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token  = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token        = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token  = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token  = match no_speech_token {
            None => panic!("Không tìm thấy token nào đại diện cho non-speech"),
            Some(n) => n,
        };

        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            task,
            timestamps,
            max_initial_timestamp_index,
            verbose,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
        })
    }

    // [Decoder::decode - 00]
    fn decode(
        &mut self, mel: &Tensor, t: f64
    ) -> Result<DecodingResult, Box<dyn StdErr>>
    {
        // [Decoder::decode - 01]
        let audio_features: Tensor = self.model.encoder_forward(mel, true)?;
        if self.verbose {
            println!("Các đặc trưng của đoạn ghi âm:\n{:?}",
                audio_features.dims()
            );
        }

        // [Decoder::decode - 02]
        let sample_len         = self.model.config().max_target_positions / 2;
        let mut sum_logprob    = 0f64;
        let mut no_speech_prob = f64::NAN;

        // [Decoder::decode - 03]
        let mut tokens              = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task 
        {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps
        {
            tokens.push(self.no_timestamps_token);
        }

        // [Decoder::decode - 04]
        for i in 0..sample_len
        {
            //println!("tokens: {:?}", tokens);
            // [Decoder::decode - 05]
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;
            let tokens_t = tokens_t.unsqueeze(0)?;

            // [Decoder::decode - 06]
            let ys = self.model
                         .decoder_forward(&tokens_t, &audio_features, i == 0)?;
            //println!("ys: {:?}", ys.shape());

            // [Decoder::decode - 07]
            if i == 0
            {
                let logits = self.model
                    .decoder_final_linear(&ys.i(..1)?)?
                    .i(0)?.i(0)?;

                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            // [Decoder::decode - 08]
            let (_, seq_len, _) = ys.dims3()?;
            let logits = self.model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?.i(0)?;

            // [Decoder::decode - 09]
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs: Tensor = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr: WeightedIndex<f32> = WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };

            tokens.push(next_token);

            // [Decoder::decode - 10]
            if  next_token == self.eot_token ||
                tokens.len() > self.model.config().max_target_positions
            {
                break;
            }

            let prob = softmax(&logits, candle_core::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;

            sum_logprob += prob.ln();
        }

        // [Decoder::decode - 11]
        let text = self.tokenizer.decode(&tokens, true)
            .map_err(|e| format!("{e}"))?;

        // [Decoder::decode - 12]
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    // [Decoder::decode_with_fallback - 00]
    fn decode_with_fallback(
        &mut self, segment: &Tensor
    ) -> Result<DecodingResult, Box<dyn StdErr>>
    {
        for (i, &t) in m::TEMPERATURES.iter().enumerate()
        {
            let dr: Result<DecodingResult, Box<dyn StdErr>> =
                                                       self.decode(segment, t);

            // [Decoder::decode_with_fallback - 01]
            if i == m::TEMPERATURES.len() - 1
            {
                return dr;
            }

            // [Decoder::decode_with_fallback - 02]
            match dr
            {
                Ok(dr) => {
                    let needs_fallback: bool =
                        dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;

                    if  !needs_fallback ||
                        dr.no_speech_prob > m::NO_SPEECH_THRESHOLD
                    {
                        return Ok(dr);
                    }
                },
                Err(err) => {
                    println!("Gặp lỗi tại {t}: {err}")
                }
            }
        }

        unreachable!()
    }

    // [Decoder::run - 00]
    fn run(&mut self, mel: &Tensor) -> Result<Vec<Segment>, Box<dyn StdErr>>
    {
        // [Decoder::run - 01]
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments: Vec<Segment> = vec![];

        // Vòng lặp chính sẽ duyệt qua toàn bộ các Audio Frame có trong ten-xơ
        // hay chính xác hơn là trong ảnh phổ biểu diễn nội dung lời nói trong
        // bản ghi âm.
        while seek < content_frames
        {
            // Lưu mốc thời gian để nếu dùng verbose thì cuối cúng sẽ in ra
            // tổng thời gian suy luận.
            let start = std::time::Instant::now();

            // Tính thời điểm bắt đầu (so với mốc 0, theo đơn vị giây) của
            // khung âm thanh đang được xử lý. Con trỏ seek sẽ giúp trượt
            // chuyển sang Audio Frame kế tiếp theo đúng bước trượt tiêu chuẩn
            // của Whisper.
            //   seek=0 -> time_offset=0, ..., seel=4500 -> time_offset=45
            // time_offset sẽ được lưu vào Segment tương ứng.
            let time_offset =
                        (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;

            // [Decoder::run - 02]
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment  = mel.narrow(2, seek, segment_size)?;
            let segment_duration =
                (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            println!("Decoder::run - segment_duration: {:?}", segment_duration);
            println!("Decoder::run - mel_segment: {:?}", mel_segment.shape());

            // [Decoder::run - 03]
            let dr: DecodingResult = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;

            // [Decoder::run - 04]
            if  dr.no_speech_prob > m::NO_SPEECH_THRESHOLD &&
                dr.avg_logprob    < m::LOGPROB_THRESHOLD
            {
                if self.verbose {
                    println!("Không dò thấy lời nói, bỏ qua {seek} {dr:?}");
                }
                continue;
            }

            // [Decoder::run - 05]
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            println!("{:.1}s -- {:.1}s: {}",
                segment.start,
                segment.start + segment.duration,
                segment.dr.text,
            );

            if self.verbose {
                println!("{seek}: {segment:?}, in {:?}", start.elapsed());
            }
            segments.push(segment)
        }

        Ok(segments)
    }
}


/// Hàm trợ giúp chuyển đổi từ token sang token-ID tương ứng.
///
/// # Tham số đầu vào:
/// * tokenizer: đối tượng thực hiện token-hóa đã được khởi tạo.
/// * toke: token muốn tìm token-ID.
///
/// # Kết quả:
/// Giá trị u32 là token-ID tương ứng hoặc Error nếu không tìm thấy.
pub fn token_id(
    tokenizer: &Tokenizer, token: &str
) -> candle_core::Result<u32>
{
    match tokenizer.token_to_id(token) {
        None => candle_core::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
enum Task { Transcribe /* văn bản */, Translate /* dịch ra tiếng Anh */}

fn main() -> Result<(), Box<dyn StdErr>>
{
    //let device = candle_core::Device::Cpu;
    let device = candle_core::Device::cuda_if_available(0)?;

    //let default_model       = String::from("openai/whisper-base");
    let default_model       = String::from("openai/whisper-small");
    let default_revision    = String::from("main");
    let (model_id, revision)= (default_model, default_revision);

    let (config_filename, tokenizer_filename, weights_filename, input) = {
        let api = Api::new()?;
        let repo = api.repo(
            Repo::with_revision(model_id, RepoType::Model, revision)
        );
        //let sample = "/home/trinhtuan/tuyenngondoclap_30-05s_16000_mono.wav";
        let sample = "/home/trinhtuan/tuyenngondoclap_30-00s_16000_mono.wav";
        
        let config    = repo.get("config.json")?;
        let tokenizer = repo.get("tokenizer.json")?;
        let model     = repo.get("model.safetensors")?;
        
        (config, tokenizer, model, sample)
    };

    let config: Config = serde_json::from_str(
        &std::fs::read_to_string(config_filename)?
    )?;

    // [main - 01]
    let tokenizer = Tokenizer::from_file(tokenizer_filename)
        .map_err(|e| format!("{e}"))?;

    // [main - 02]
    let mel_bytes = include_bytes!("melfilters.bytes").as_slice();
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
        mel_bytes, &mut mel_filters
    );
    //println!("{} - {} - {}", mel_filters.len(), config.num_mel_bins, 1);

    // [main - 03]
    let (pcm_data, sample_rate) = asr_app_02::load_audio_mono(input, None)?;
    if sample_rate != m::SAMPLE_RATE as u32 {
        panic!("Tệp ghi âm phải có tần số lấy mẫu là {}", m::SAMPLE_RATE)
    }
    println!("Dữ liệu PCM (véc-tơ): {}", pcm_data.len());
   
    // [main -04]
    let mel: Vec<f32> = audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
    println!("Ảnh phổ Mel (véc-tơ): {:?}", mel.len());
   
    // [main - 05]
    let mel_len = mel.len();
    let mel = Tensor::from_vec(
        mel,
        (1, config.num_mel_bins, mel_len / config.num_mel_bins),
        &device,
    )?;
    println!("Ảnh phổ Mel (ten-xơ): {:?}", mel.shape());

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[weights_filename], m::DTYPE, &device
        )?
    };
    let model = Model::Normal(m::model::Whisper::load(&vb, config)?);

/*  // Hiển thị các token sẽ bị phớt lờ
    let stoken_ids: &Vec<u32> = &model.config().suppress_tokens;
    println!("{stoken_ids:?}");
    let stoken_wrd = tokenizer
        .decode(stoken_ids, false)
        .map_err(|e| format!("{e}"))?;
    println!("{stoken_wrd:?}");
*/

    // [main - 06]
    let mut dc = Decoder::new(
        model,
        tokenizer,
        299792458,                          // seed
        &device,
        Some(50278),                        // language_token vi=50278,en=50259
        Some(Task::Transcribe),             // task: Transcribe / Translate
        false,                              // timestamps
        None,                               // max_initial_timestamp_index
        false,                              // verbose
    )?;

    // [main - 07]
    dc.run(&mel)?;

    Ok(())
}
