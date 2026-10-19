#![allow(clippy::all)]
//! Phiên bản này bổ sung khả năng tự chuyển tải về "model.safetensors" mếu mô
//! hình nhỏ và không có tệp chỉ số "model.safetensors.index.json".
//!
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use hf_hub::{Repo, RepoType, api::RepoInfo, api::tokio::{ApiBuilder, ApiRepo}};
use std::path::PathBuf;
use tokio::task::JoinSet;

use std::error::Error as StdErr;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ModelIdxMetadata {
    total_size: Option<u64>,
    format: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ModelIdxFile {
    metadata: ModelIdxMetadata,
    weight_map: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct ModelIdx {
    metadata: ModelIdxMetadata,
    weight_map: HashMap<String, String>,
    #[serde(skip)]
    model_files_list: Vec<String>,
    #[serde(skip)]
    repo: std::sync::Arc<ApiRepo>
}

impl ModelIdx {
    pub async fn new(
        name: &str, rev: Option<&str>
    ) -> Result<Self, Box<dyn StdErr>>
    {
        // [01]
        let model_name = if name.trim().is_empty() {
            Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "No model name provided!"
            ))
        } else {
            Ok(name)
        };

        let access_token = std::env::var("HF_TOKEN")
            .map_err(|e| format!("HF_TOKEN: {e}"))?;
        
        // [02]
        let api: hf_hub::api::tokio::Api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(access_token))
            .build()?;
        // [03]
        let mut revision = rev.unwrap_or("main").to_string();
        if revision.trim().is_empty() { revision = "main".to_string() }
        let repo: ApiRepo = api.repo(Repo::with_revision(
            model_name?.to_string(),
            RepoType::Model,
            revision
        ));

        // [04]
        let files = ModelIdx::get_repo_info(&repo).await?;
        let mut model_idx_file = ModelIdxFile::default();

        // [06]
        if files.contains("model.safetensors.index.json") {
            let idx_file = repo.get("model.safetensors.index.json").await
                                                .map_err(|e| format!("{e}"))?;

            let idx_json: String = std::fs::read_to_string(idx_file)?;
            //[07]
            model_idx_file = serde_json::from_str(&idx_json)?;

            let mut file_list: Vec<String> = model_idx_file.weight_map
                                                .values()
                                                .map(|s| s.clone()).collect();
            file_list.sort(); file_list.dedup();

            Ok(Self {
                metadata: model_idx_file.metadata, 
                weight_map: model_idx_file.weight_map,
                model_files_list: file_list, repo: std::sync::Arc::new(repo)
            })
        } else {
            let pattern = regex::Regex::new(r"\.safetensors$").unwrap();
            let file_list = files.into_iter()
                .filter(|f| pattern.is_match(f))
                .collect();

            Ok(Self {
                metadata: model_idx_file.metadata,
                weight_map: model_idx_file.weight_map,
                model_files_list: file_list, repo: std::sync::Arc::new(repo)
            })
        }
    }

    pub async fn get_repo_info(api: &ApiRepo)
        -> Result<std::collections::HashSet<String>, Box<dyn StdErr>>
    {
        let rinfo: RepoInfo = api.info().await.map_err(|e| format!("{e}"))?;

        // [05]
        let files: std::collections::HashSet<String> = rinfo.siblings
            .into_iter()
            .map(|rf| rf.rfilename).collect();

        Ok(files)
    }

    pub fn get_model_file_list(&self) -> Vec<String> {
        self.model_files_list.clone()
    }

    pub fn get_size(&self) -> Option<u64> {
        self.metadata.total_size
    }

    pub fn get_format(&self) -> Option<String> {
        self.metadata.format.clone()
    }

    pub async fn get_model_files(
        &self
    ) -> Result<(PathBuf, PathBuf, Vec<PathBuf>), Box<dyn StdErr>>
    {
        // [08]
        let tokenizer = self.repo
            .get("tokenizer.json").await
            .map_err(|e| format!("tokenizer.json: {e}"))?;
        let config = self.repo
            .get("config.json").await
            .map_err(|e| format!("config.json: {e}"))?;

        // [09]
        let sf_file_list = self.get_model_file_list();
        // [10]
        let mut set: JoinSet<Result<PathBuf, _>> = JoinSet::new();
        
        for f in sf_file_list {
            // [11]
            let repo_in_task = self.repo.clone();
            let file_to_get = f.to_string();
            
            set.spawn(async move {
                repo_in_task.get(&file_to_get).await
                .map_err(|e| format!("{file_to_get}: {e}"))
            });
        }

        let mut safetensors = Vec::new();
        while let Some(resource) = set.join_next().await {
            let task_result: Result<PathBuf, String> = resource?;
            let file_result: PathBuf = task_result?;
            safetensors.push(file_result);
        }

        Ok((tokenizer, config, safetensors))
    }
}
