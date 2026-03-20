use anyhow::{Context, Result, anyhow};

#[derive(serde::Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
enum SecretFromFileOrEnv {
    Env(String),
    FileName(String),
    EnvFileName(String),
}

impl SecretFromFileOrEnv {
    fn get(self) -> Result<Vec<u8>> {
        fn trim_last_newline(mut v: Vec<u8>) -> Vec<u8> {
            v.pop_if(|b| *b == '\n' as u8);
            v
        }

        match self {
            SecretFromFileOrEnv::Env(var_name) => std::env::var_os(&var_name)
                .with_context(|| format!("environment variable {var_name:?} is not set"))
                .map(|r| r.into_encoded_bytes()),
            SecretFromFileOrEnv::FileName(file_name) => std::fs::read(&file_name)
                .with_context(|| format!("couldn't read file {file_name:?}"))
                .map(trim_last_newline),
            SecretFromFileOrEnv::EnvFileName(var_name) => std::env::var_os(&var_name)
                .with_context(|| format!("environment variable {var_name:?} is not set"))
                .and_then(|file_name| {
                    std::fs::read(&file_name)
                        .with_context(|| format!("couldn't read file {file_name:?}"))
                        .map(trim_last_newline)
                }),
        }
    }
}

#[derive(serde::Deserialize, Debug)]
struct GitHubAppConfig {
    pub webhook_secret: SecretFromFileOrEnv,
    pub app_private_key: SecretFromFileOrEnv,
    pub app_id: u64,
    pub client_id: String,
}

#[derive(serde::Deserialize, Debug)]
struct HydraConfig {
    pub url: String,
    pub user: String,
    pub password: SecretFromFileOrEnv,
    pub project: String,
}

#[derive(serde::Deserialize, Debug)]
struct RepoConfig {
    pub check_run_name: String,
    pub hydra_jobset_template: hydra::Jobset,
    #[serde(default)]
    pub check_per_job: bool,
}

#[derive(serde::Deserialize, Debug)]
struct Config {
    pub listen: Option<std::net::SocketAddr>,
    pub user_agent: String,
    pub github_app: Option<GitHubAppConfig>,
    pub hydra: HydraConfig,
    pub repositories: std::collections::HashMap<String, RepoConfig>,
}

fn load_config(filename: &std::path::Path) -> Result<Config> {
    let contents = std::fs::read_to_string(filename)
        .with_context(|| format!("couldn't read config file {}", filename.to_string_lossy()))?;
    let deserialized: Config = serde_json::from_str(&contents)
        .with_context(|| format!("couldn't parse config file {}", filename.to_string_lossy()))?;
    Ok(deserialized)
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let mut args = std::env::args_os();
    let program_name = args
        .next()
        .map(|os_str| os_str.to_string_lossy().to_string())
        .unwrap_or("<hydra-github-app>".to_string());
    let usage = || anyhow!("usage: {program_name} <config> [{{webhook|one_pr <pr_url>}}]");
    let config_file_name = args.next().ok_or_else(usage)?;
    let action = args.next().map_or_else(
        || "webhook".to_string(),
        |s| s.to_string_lossy().to_string(),
    );

    // ugly way to get a reference to different async functions
    let run_fn: Box<dyn FnOnce(Config) -> std::pin::Pin<Box<dyn std::future::Future<Output = _>>>> =
        match action.as_ref() {
            "webhook" => Box::new(|cfg| Box::pin(webhook::run(cfg))),
            "one_pr" => {
                let pull_request_url = args.next().ok_or_else(usage)?.to_string_lossy().to_string();
                Box::new(|cfg| Box::pin(do_one_pr(cfg, pull_request_url)))
            }
            _ => return Err(usage()),
        };
    if args.next().is_some() {
        return Err(usage());
    }
    let cfg = load_config(config_file_name.as_ref())?;
    run_fn(cfg).await
}

fn parse_pr_url(pull_request_url: &str) -> Option<(String, u64)> {
    let url = url::Url::parse(pull_request_url).ok()?;
    if url.scheme() != "https" || url.host_str() != Some("github.com") {
        return None;
    }
    let [empty, org, repo, pull, pr_number_str] = url.path().split('/').collect::<Vec<_>>()[..]
    else {
        return None;
    };
    if !empty.is_empty() || pull != "pull" {
        return None;
    }
    Some((format!("{org}/{repo}"), pr_number_str.parse().ok()?))
}

async fn do_one_pr(cfg: Config, pull_request_url: String) -> Result<()> {
    let Some((repo_full_name, pr_number)) = parse_pr_url(&pull_request_url) else {
        return Err(anyhow!(
            "URL must point to a PR as in https://github.com/org/repo/pull/123"
        ));
    };
    let Some(repo_config) = cfg.repositories.get(&repo_full_name) else {
        return Err(anyhow!("repository not found in config: {repo_full_name}"));
    };

    let hydra_client = hydra::Client::new(
        &cfg.hydra.url,
        &cfg.user_agent,
        &cfg.hydra.user,
        cfg.hydra
            .password
            .get()
            .context("couldn't get Hydra password")?,
    )
    .await?;

    let github_client = github::Client::new(
        &cfg.user_agent,
        std::env::var("GITHUB_TOKEN")
            .context("couldn't get GitHub token from GITHUB_TOKEN env var")?,
    );

    let pull_request = github_client
        .get_pull_request(&repo_full_name, pr_number)
        .await?;

    let _jobset_id = create_jobset_for_pr(
        &github_client,
        &hydra_client,
        &cfg.hydra.project,
        repo_config,
        &repo_full_name,
        &pull_request,
    )
    .await?;

    Ok(())
}

mod github {
    use anyhow::{Context, Result, anyhow};

    // Signature verification

    pub async fn check_signature(
        secret: &[u8],
        sig: Vec<u8>,
        body: bytes::Bytes,
    ) -> Result<bytes::Bytes> {
        use hmac::{Hmac, Mac};
        let mut mac = Hmac::<sha2::Sha256>::new_from_slice(secret)
            .expect("Hmac can derives keys from slices of any length");
        mac.update(body.as_ref());
        if mac.verify_slice(sig.as_slice()).is_err() {
            return Err(anyhow!("Signature verification failed"));
        }
        Ok(body)
    }

    pub async fn parse_signature_header(s: String) -> Result<Vec<u8>> {
        let mut components = s.splitn(2, '=');
        let Some(algo) = components.next() else {
            return Err(anyhow!("Signature hash method missing"));
        };
        let Some(hash) = components.next() else {
            return Err(anyhow!("Signature hash missing"));
        };
        let Ok(hash) = hex::decode(hash) else {
            return Err(anyhow!("Invalid signature hash hex"));
        };
        if algo != "sha256" {
            return Err(anyhow!("Invalid signature hash method"));
        }
        Ok(hash)
    }

    // GitHub API

    #[derive(serde::Serialize, serde::Deserialize, Debug)]
    pub struct Repository {
        pub full_name: String,
        pub clone_url: String,
    }

    #[derive(serde::Deserialize, Debug)]
    #[serde(rename_all = "snake_case")]
    pub enum InstallationAction {
        Created,
        Deleted,
        NewPermissionsAccepted,
        Suspend,
        Unsuspend,
    }

    #[derive(serde::Deserialize, Debug)]
    #[serde(rename_all = "snake_case")]
    pub enum InstallationRepositoriesAction {
        Added,
        Removed,
    }

    #[derive(serde::Deserialize, Debug)]
    pub struct SimpleInstallation {
        pub id: u64,
    }

    #[derive(serde::Deserialize, Debug)]
    pub struct Installation {
        pub id: u64,
        pub app_id: u64,
        pub client_id: String,
    }

    #[derive(serde::Deserialize, Debug)]
    pub struct InstallationEvent {
        pub action: InstallationAction,
        pub installation: Installation,
        // Actions like suspend and uninstall don't provide repositories
        #[serde(default)]
        pub repositories: Vec<Repository>,
    }

    #[derive(serde::Deserialize, Debug)]
    pub struct RepositoryChanged {
        pub full_name: String,
    }

    #[derive(serde::Deserialize, Debug)]
    pub struct InstallationRepositoriesEvent {
        pub action: InstallationRepositoriesAction,
        pub installation: Installation,
        pub repositories_added: Vec<RepositoryChanged>,
        pub repositories_removed: Vec<RepositoryChanged>,
    }

    #[derive(serde::Deserialize, Debug)]
    #[serde(rename_all = "snake_case")]
    pub enum PullRequestAction {
        Assigned,
        AutoMergeDisabled,
        AutoMergeEnabled,
        Closed,
        ConvertedToDraft,
        Demilestoned,
        Dequeued,
        Edited,
        Enqueued,
        Labeled,
        Locked,
        Milestoned,
        Opened,
        ReadyForReview,
        Reopened,
        ReviewRequestRemoved,
        ReviewRequested,
        Synchronize,
        Unassigned,
        Unlabeled,
        Unlocked,
    }

    #[derive(serde::Deserialize, Debug)]
    pub struct PullRequest {
        pub head: PullRequestBase,
        pub number: u64,
        pub html_url: String,
        pub merge_commit_sha: Option<String>,
        pub mergeable: Option<bool>,
    }

    #[derive(serde::Deserialize, Debug)]
    pub struct PullRequestBase {
        pub sha: String,
    }

    #[derive(serde::Deserialize, Debug)]
    pub struct PullRequestEvent {
        pub action: PullRequestAction,
        pub installation: SimpleInstallation,
        pub pull_request: PullRequest,
        pub repository: Repository,
    }

    #[derive(serde::Deserialize, Debug)]
    pub struct App {
        id: u64,
    }

    #[derive(serde::Deserialize, Debug)]
    pub struct CheckRun {
        id: u64,
        #[serde(flatten)]
        status: CheckRunStatus,
        name: String,
        app: Option<App>,
        details_url: String,
    }

    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq, Clone)]
    #[serde(tag = "status", content = "conclusion")]
    #[serde(rename_all = "snake_case")]
    pub enum CheckRunStatus {
        Queued,
        InProgress,
        Completed(CheckRunConclusion),
        Waiting,
        Requested,
        Pending,
    }

    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq, Clone)]
    #[serde(rename_all = "snake_case")]
    pub enum CheckRunConclusion {
        ActionRequired,
        Cancelled,
        Failure,
        Neutral,
        Success,
        Skipped,
        Stale,
        TimedOut,
    }

    #[derive(serde::Serialize, serde::Deserialize, Debug)]
    pub struct UpsertCheckRunData {
        #[serde(flatten)]
        pub status: CheckRunStatus,
        pub details_url: String,
    }

    #[derive(serde::Deserialize)]
    pub struct GitCommitParent {
        pub sha: String,
    }

    #[derive(serde::Deserialize)]
    pub struct GitCommit {
        pub sha: String,
        pub parents: Vec<GitCommitParent>,
    }

    #[derive(Debug)]
    pub enum Payload {
        Ping,
        Installation(InstallationEvent),
        InstallationRepositories(InstallationRepositoriesEvent),
        PullRequest(PullRequestEvent),
    }

    use tokio::sync::{mpsc, oneshot};

    /// A lower-level client to manage GitHub application specifics like:
    /// - managing application tokens
    /// - getting installation tokens
    /// - getting installation ID for repo
    #[derive(Clone)]
    pub struct LowerApplicationClient {
        http_client: reqwest::Client,
        token_channel: mpsc::Sender<oneshot::Sender<String>>,
    }

    impl LowerApplicationClient {
        fn new(
            http_client: reqwest::Client,
            client_id: &str,
            key: rsa::RsaPrivateKey,
        ) -> LowerApplicationClient {
            let (tx, mut rx) = mpsc::channel::<oneshot::Sender<String>>(32);
            let client_id = client_id.to_owned();
            tokio::spawn(async move {
                let mut token_store: Option<(String, chrono::DateTime<_>)> = None;
                use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
                let jwt_header =
                    URL_SAFE_NO_PAD.encode("{\"typ\":\"JWT\",\"alg\":\"RS256\"}") + ".";
                let signer = rsa::pkcs1v15::SigningKey::<sha2::Sha256>::new(key);
                const TOKEN_VALIDITY_BEFORE: chrono::TimeDelta = chrono::TimeDelta::minutes(1);
                const TOKEN_VALIDITY: chrono::TimeDelta = chrono::TimeDelta::minutes(10);
                const TOKEN_REISSUE_MARGIN: chrono::TimeDelta = TOKEN_VALIDITY
                    .checked_div(5)
                    .unwrap()
                    .checked_mul(4)
                    .unwrap();
                while let Some(chan) = rx.recv().await {
                    let now = chrono::Utc::now();
                    let token = match token_store {
                        Some((ref token, ref expiration))
                            if *expiration > now + TOKEN_REISSUE_MARGIN =>
                        {
                            token.clone()
                        }
                        _ => {
                            let expiration = now + TOKEN_VALIDITY;
                            let payload = serde_json::json!({
                                "iat": (now - TOKEN_VALIDITY_BEFORE).timestamp(),
                                "exp": expiration.timestamp(),
                                "iss": client_id,
                            });
                            let to_sign =
                                jwt_header.clone() + &URL_SAFE_NO_PAD.encode(payload.to_string());
                            use rsa::signature::{SignatureEncoding, Signer};
                            let signature =
                                URL_SAFE_NO_PAD.encode(signer.sign(to_sign.as_bytes()).to_bytes());
                            let token = to_sign + "." + &signature;
                            token_store = Some((token.clone(), expiration));
                            token
                        }
                    };
                    chan.send(token)
                        .expect("failed to send a response over a channel");
                }
            });
            LowerApplicationClient {
                http_client,
                token_channel: tx,
            }
        }

        async fn get_token(&self) -> String {
            let (tx, rx) = oneshot::channel();
            self.token_channel
                .send(tx)
                .await
                .expect("failed to send a request to a channel");
            rx.await.expect("failed to receive from a channel")
        }

        async fn get_installation_token(
            &self,
            installation_id: u64,
        ) -> Result<(String, chrono::DateTime<chrono::Utc>)> {
            let app_token = self.get_token().await;
            let resp = self
                .http_client
                .post(format!(
                    "https://api.github.com/app/installations/{installation_id}/access_tokens",
                ))
                .bearer_auth(&app_token)
                .send()
                .await
                .with_context(|| {
                    format!("failed to send an installation token request for {installation_id}")
                })?
                .error_for_status()
                .with_context(|| {
                    format!("installation token request for {installation_id} failed")
                })?
                .json::<InstallationTokenResponse>()
                .await
                .with_context(|| {
                    format!("failed to parse installation token response for {installation_id}")
                })?;
            eprintln!(
                "got a new installation token for {installation_id} valid until {}",
                resp.expires_at,
            );
            Ok((resp.token, resp.expires_at))
        }

        async fn get_installation_for_repo(&self, repo_full_name: &str) -> Result<u64> {
            #[derive(serde::Deserialize)]
            struct Response {
                id: u64,
            }
            Ok(self
                .http_client
                .get(format!(
                    "https://api.github.com/repos/{repo_full_name}/installation",
                ))
                .bearer_auth(&self.get_token().await)
                .send()
                .await
                .with_context(|| {
                    format!("failed to send an installation request for {repo_full_name}")
                })?
                .error_for_status()
                .with_context(|| format!("installation request for {repo_full_name} failed"))?
                .json::<Response>()
                .await
                .with_context(|| {
                    format!("failed to parse installation response for {repo_full_name}")
                })?
                .id)
        }
    }

    /// This client contains methods for all actions that can be done with most GitHub tokens
    #[derive(Clone)]
    pub struct Client {
        http_client: reqwest::Client,
        token: String,
    }

    impl Client {
        pub fn new(user_agent: &str, token: String) -> Self {
            Self {
                http_client: reqwest::ClientBuilder::new()
                    //.connection_verbose(true)
                    .user_agent(user_agent)
                    .build()
                    .expect("couldn't build client"),
                token,
            }
        }

        pub async fn get_pull_request(
            &self,
            repo_full_name: &str,
            pr_number: u64,
        ) -> Result<PullRequest> {
            self.http_client
                .get(format!(
                    "https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"
                ))
                .bearer_auth(&self.token)
                .send()
                .await
                .with_context(|| format!("failed to send request for pull request {pr_number}"))?
                .error_for_status()
                .with_context(|| format!("request for pull request {pr_number} failed"))?
                .json::<PullRequest>()
                .await
                .context("couldn't parse JSON response")
        }

        pub async fn get_commit(
            &self,
            repo_full_name: &str,
            commit_sha: &str,
        ) -> Result<GitCommit> {
            self.http_client
                .get(format!(
                    "https://api.github.com/repos/{repo_full_name}/git/commits/{commit_sha}"
                ))
                .bearer_auth(&self.token)
                .send()
                .await
                .with_context(|| format!("failed to send request for commit {commit_sha}"))?
                .error_for_status()
                .with_context(|| format!("request for commit {commit_sha} failed"))?
                .json::<GitCommit>()
                .await
                .context("couldn't parse JSON response")
        }
    }

    /// This client covers actions that can only be done via a GitHub App installation.
    /// It can be created via `ApplicationClient::for_installation`
    #[derive(Clone)]
    pub struct InstallationClient {
        http_client: reqwest::Client,
        #[allow(dead_code)]
        installation_id: u64,
        token: String,
    }

    impl InstallationClient {
        pub fn get_client(&self) -> Client {
            Client {
                http_client: self.http_client.clone(),
                token: self.token.clone(),
            }
        }

        pub async fn patch_check_suites_preferences(
            &self,
            repo_full_name: String,
            app_id: u64,
            setting: bool,
        ) -> Result<()> {
            #[derive(serde::Serialize)]
            struct AutoTriggerChecks {
                app_id: u64,
                setting: bool,
            }
            #[derive(serde::Serialize)]
            struct Request<'a> {
                auto_trigger_checks: &'a [AutoTriggerChecks],
            }
            let _ = self.http_client
                .patch(format!(
                    "https://api.github.com/repos/{repo_full_name}/check-suites/preferences",
                ))
                .bearer_auth(&self.token)
                .json(&Request {
                    auto_trigger_checks: &[AutoTriggerChecks { app_id, setting }],
                })
                .send()
                .await
                .with_context(|| format!("failed to send request to update check suite preferences for {repo_full_name}"))?
                .error_for_status()
                .with_context(|| format!("request to update check suite preferences for {repo_full_name} failed"))?;
            eprintln!("updated check suite preferences for {repo_full_name}");
            Ok(())
        }

        pub async fn get_check_runs_for_commit(
            &self,
            repo_full_name: &str,
            app_id: u64,
            commit_sha: &str,
        ) -> Result<Vec<CheckRun>> {
            #[derive(serde::Deserialize)]
            struct Response {
                #[allow(unused)]
                total_count: u64,
                check_runs: Vec<CheckRun>,
            }
            Ok(self
                .http_client
                .get(format!(
                    "https://api.github.com/repos/{repo_full_name}/commits/{commit_sha}/check-runs"
                ))
                .query(&[("app_id", &app_id)])
                .bearer_auth(&self.token)
                .send()
                .await
                .with_context(|| {
                    format!("failed to send request for check runs for commit {commit_sha}")
                })?
                .error_for_status()
                .with_context(|| format!("request for check runs for commit {commit_sha} failed"))?
                .json::<Response>()
                .await
                .context("couldn't parse JSON response")?
                .check_runs)
        }

        pub async fn post_check_run(
            &self,
            repo_full_name: &str,
            check_run_name: &str,
            head_sha: &str,
            data: &UpsertCheckRunData,
        ) -> Result<()> {
            #[derive(serde::Serialize)]
            struct Request<'a> {
                name: &'a str,
                head_sha: &'a str,
                #[serde(flatten)]
                data: &'a UpsertCheckRunData,
            }
            let _ = self
                .http_client
                .post(format!(
                    "https://api.github.com/repos/{repo_full_name}/check-runs"
                ))
                .bearer_auth(&self.token)
                .json(&Request {
                    name: check_run_name,
                    head_sha,
                    data,
                })
                .send()
                .await
                .with_context(|| {
                    format!("failed to send request to create a check run for commit {head_sha}")
                })?
                .error_for_status()
                .with_context(|| {
                    format!("request to create a check run for commit {head_sha} failed")
                })?;
            Ok(())
        }

        pub async fn patch_check_run(
            &self,
            repo_full_name: &str,
            check_run_id: u64,
            check_run_name: &str,
            head_sha: &str,
            data: &UpsertCheckRunData,
        ) -> Result<()> {
            #[derive(serde::Serialize)]
            struct Request<'a> {
                name: &'a str,
                head_sha: &'a str,
                #[serde(flatten)]
                data: &'a UpsertCheckRunData,
            }
            let _ = self
                .http_client
                .patch(format!(
                    "https://api.github.com/repos/{repo_full_name}/check-runs/{check_run_id}"
                ))
                .bearer_auth(&self.token)
                .json(&Request {
                    name: check_run_name,
                    head_sha,
                    data,
                })
                .send()
                .await
                .with_context(|| {
                    format!("failed to send request to patch a check run for commit {head_sha}")
                })?
                .error_for_status()
                .with_context(|| {
                    format!("request to patch a check run for commit {head_sha} failed")
                })?;
            Ok(())
        }
    }

    #[derive(Clone)]
    pub struct ApplicationClient {
        http_client: reqwest::Client,
        token_channel: mpsc::Sender<(u64, oneshot::Sender<Result<String>>)>,
        repo_mapping_channel: mpsc::Sender<(String, oneshot::Sender<Result<u64>>)>,
    }

    impl ApplicationClient {
        pub fn new(
            user_agent: &str,
            client_id: &str,
            key: rsa::RsaPrivateKey,
        ) -> ApplicationClient {
            // TODO: wrap in a tower service with a buffer and rate limits to keep GitHub happy
            let http_client = reqwest::ClientBuilder::new()
                //.connection_verbose(true)
                .user_agent(user_agent)
                .build()
                .expect("couldn't build client");

            let application_client =
                LowerApplicationClient::new(http_client.clone(), client_id, key);

            ApplicationClient {
                http_client,
                token_channel: {
                    let (tx, rx) = mpsc::channel(32);
                    tokio::spawn(Self::installation_token_manager(
                        application_client.clone(),
                        rx,
                    ));
                    tx
                },
                repo_mapping_channel: {
                    let (tx, rx) = mpsc::channel(32);
                    tokio::spawn(Self::repo_mapping_manager(application_client.clone(), rx));
                    tx
                },
            }
        }

        async fn installation_token_manager(
            application_client: LowerApplicationClient,
            mut rx: mpsc::Receiver<(u64, oneshot::Sender<Result<String>>)>,
        ) -> ! {
            let mut installation_tokens = std::collections::HashMap::<u64, String>::new();
            let mut expiration_queue = tokio_util::time::DelayQueue::new();
            use futures_util::StreamExt;
            loop {
                tokio::select! {
                    Some((i, resp)) = rx.recv() => {
                        use std::collections::hash_map::Entry;
                        resp.send(match installation_tokens.entry(i) {
                            Entry::Occupied(e) => Ok(e.get().clone()),
                            Entry::Vacant(e) => async {
                                let (token, expires_at) = application_client
                                    .get_installation_token(i)
                                    .await?;
                                let diff = (expires_at - chrono::Utc::now())
                                    .to_std()
                                    .map_err(|_| anyhow!("new installation token for {i} expires in the past"))?;
                                expiration_queue.insert(i, diff);
                                e.insert(token.clone());
                                Ok(token)
                            }.await
                        })
                        .expect("failed to send a response over a channel");
                    }
                    Some(expiration) = expiration_queue.next() => {
                        let i = expiration.into_inner();
                        eprintln!("installation token {i} is about to expire, invalidating");
                        installation_tokens.remove(&i);
                    }
                }
            }
        }

        async fn get_installation_token(&self, installation_id: u64) -> Result<String> {
            let (tx, rx) = oneshot::channel();
            self.token_channel
                .send((installation_id, tx))
                .await
                .expect("failed to send a request to a channel");
            rx.await.expect("failed to receive response")
        }

        pub async fn for_installation(&self, installation_id: u64) -> Result<InstallationClient> {
            Ok(InstallationClient {
                http_client: self.http_client.clone(),
                installation_id,
                token: self.get_installation_token(installation_id).await?,
            })
        }

        async fn repo_mapping_manager(
            application_client: LowerApplicationClient,
            mut rx: mpsc::Receiver<(String, oneshot::Sender<Result<u64>>)>,
        ) {
            // TODO: expiration
            let mut repo_map = std::collections::HashMap::<String, u64>::new();
            while let Some((repo_full_name, chan)) = rx.recv().await {
                use std::collections::hash_map::Entry;
                chan.send(match repo_map.entry(repo_full_name.clone()) {
                    Entry::Occupied(e) => Ok(*e.get()),
                    Entry::Vacant(e) => {
                        async {
                            let installation_id = application_client
                                .get_installation_for_repo(&repo_full_name)
                                .await?;
                            e.insert(installation_id);
                            Ok(installation_id)
                        }
                        .await
                    }
                })
                .expect("failed to send a response over a channel");
            }
        }

        async fn get_installation_for_repo(&self, repo_full_name: &str) -> Result<u64> {
            let (tx, rx) = oneshot::channel();
            self.repo_mapping_channel
                .send((repo_full_name.to_string(), tx))
                .await
                .expect("failed to send a request to a channel");
            rx.await.expect("failed to receive response")
        }

        pub async fn for_repo(&self, repo_full_name: &str) -> Result<InstallationClient> {
            let installation_id = self.get_installation_for_repo(repo_full_name).await?;
            self.for_installation(installation_id).await
        }
    }

    #[derive(serde::Deserialize)]
    struct InstallationTokenResponse {
        token: String,
        expires_at: chrono::DateTime<chrono::Utc>,
    }

    pub async fn upsert_check(
        installation_client: &InstallationClient,
        app_id: u64,
        check_run_name: &str,
        repo: &str,
        sha: &str,
        data: &UpsertCheckRunData,
    ) -> Result<()> {
        // TODO: do multiple checks at once to avoid looping for each one
        let check_runs = installation_client
            .get_check_runs_for_commit(repo, app_id, sha)
            .await?;
        //eprintln!("got check runs: {check_runs:?}");
        match check_runs.into_iter().find(|cr| {
            cr.app.as_ref().is_some_and(|app| app.id == app_id) && cr.name == check_run_name
        }) {
            Some(check_run) => {
                eprintln!(
                    "found check run \"{check_run_name}\" for commit {sha} with id {}",
                    check_run.id
                );
                if check_run.status == data.status && check_run.details_url == data.details_url {
                    eprintln!(
                        "check run \"{check_run_name}\" for commit {sha} is already in the expected state"
                    );
                    return Ok(());
                }
                installation_client
                    .patch_check_run(repo, check_run.id, check_run_name, sha, data)
                    .await?;
                eprintln!(
                    "check run \"{check_run_name}\" for commit {sha} has been updated successfully"
                );
            }
            None => {
                eprintln!(
                    "check run \"{check_run_name}\" for commit {sha} not found, will create a new one"
                );
                installation_client
                    .post_check_run(repo, check_run_name, sha, data)
                    .await?;
                eprintln!(
                    "check run \"{check_run_name}\" for commit {sha} has been created successfully"
                );
            }
        };
        Ok(())
    }

    // See https://docs.github.com/en/rest/guides/using-the-rest-api-to-interact-with-your-git-database?apiVersion=2022-11-28#checking-mergeability-of-pull-requests
    // for details. We have to request PR data for the merge process to start, then poll it until it's ready.
    pub async fn ensure_pr_merge_commit(
        github_client: &Client,
        repo: &str,
        pr_number: u64,
        pr_head_sha: &str,
    ) -> Result<(String, String)> {
        eprintln!("waiting for merge commit for PR {pr_number} with head {pr_head_sha}");
        use std::cmp::min;
        use std::iter::{once, successors};
        let backoff = once(0).chain(successors(Some(1), |n| Some(min(2 * n, 32))));
        for timeout in backoff {
            if timeout > 0 {
                tokio::time::sleep(std::time::Duration::from_secs(timeout)).await;
            }
            let pr = github_client.get_pull_request(repo, pr_number).await?;
            //eprintln!("got PR: {:#?}", pr);
            if pr.head.sha != *pr_head_sha {
                return Err(anyhow!(
                    "PR {pr_number} head SHA has changed from {pr_head_sha} to {}",
                    pr.head.sha,
                ));
            }
            match pr.mergeable {
                None => {
                    eprintln!("PR {pr_number} mergeability is still being calculated");
                    continue;
                }
                Some(false) => return Err(anyhow!("PR {pr_number} is not mergeable")),
                Some(true) => (),
            };
            let Some(merge_commit_sha) = pr.merge_commit_sha else {
                continue;
            };
            let merge_commit = github_client.get_commit(repo, &merge_commit_sha).await?;
            let Ok([target, head]): Result<[_; 2], _> = merge_commit.parents.try_into() else {
                eprintln!("number of parents for merge commit for PR {pr_number} is not 2");
                continue;
            };
            if head.sha == *pr_head_sha {
                eprintln!(
                    "got merge commit for PR {pr_number} with head {pr_head_sha}: {}",
                    merge_commit.sha
                );
                return Ok((merge_commit.sha, target.sha));
            }
        }
        unreachable!()
    }
}

mod hydra {
    use anyhow::{Context, Result};

    #[derive(serde_repr::Serialize_repr, serde_repr::Deserialize_repr, Debug, Default, Clone)]
    #[repr(u8)]
    #[allow(unused)]
    pub enum JobsetEnabled {
        Disabled = 0,
        #[default]
        Enabled = 1,
        OneShot = 2,
        OneAtATime = 3,
    }

    #[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
    pub struct JobsetInput {
        pub r#type: String,
        #[serde(default)]
        pub value: String,
    }

    fn return_false() -> bool {
        false
    }

    #[derive(serde_repr::Serialize_repr, serde_repr::Deserialize_repr)]
    #[repr(u8)]
    enum JobsetDefinitionType {
        Legacy = 0,
        Flake = 1,
    }

    #[derive(serde::Serialize, serde::Deserialize, Default)]
    #[serde(default)]
    struct JobsetDefinitionFields {
        r#type: Option<JobsetDefinitionType>,
        nixexprinput: String,
        nixexprpath: String,
        inputs: std::collections::HashMap<String, JobsetInput>,
        flake: String,
    }

    #[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
    #[serde(try_from = "JobsetDefinitionFields")]
    #[serde(into = "JobsetDefinitionFields")]
    pub enum JobsetDefinition {
        Legacy {
            nixexprinput: String,
            nixexprpath: String,
            inputs: std::collections::HashMap<String, JobsetInput>,
        },
        Flake {
            flake: String,
        },
    }

    impl TryFrom<JobsetDefinitionFields> for JobsetDefinition {
        type Error = anyhow::Error;

        fn try_from(value: JobsetDefinitionFields) -> Result<Self> {
            let r#type = value.r#type.unwrap_or({
                if value.nixexprinput.is_empty() {
                    JobsetDefinitionType::Flake
                } else {
                    JobsetDefinitionType::Legacy
                }
            });
            Ok(match r#type {
                JobsetDefinitionType::Legacy => JobsetDefinition::Legacy {
                    nixexprinput: value.nixexprinput,
                    nixexprpath: value.nixexprpath,
                    inputs: value.inputs,
                },
                JobsetDefinitionType::Flake => JobsetDefinition::Flake { flake: value.flake },
            })
        }
    }

    impl From<JobsetDefinition> for JobsetDefinitionFields {
        fn from(val: JobsetDefinition) -> Self {
            match val {
                JobsetDefinition::Legacy {
                    nixexprinput,
                    nixexprpath,
                    inputs,
                } => JobsetDefinitionFields {
                    r#type: Some(JobsetDefinitionType::Legacy),
                    nixexprinput,
                    nixexprpath,
                    inputs,
                    flake: Default::default(),
                },
                JobsetDefinition::Flake { flake } => JobsetDefinitionFields {
                    r#type: Some(JobsetDefinitionType::Flake),
                    nixexprinput: Default::default(),
                    nixexprpath: Default::default(),
                    inputs: Default::default(),
                    flake,
                },
            }
        }
    }

    #[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
    pub struct Jobset {
        #[serde(default)]
        pub enabled: JobsetEnabled,
        #[serde(default = "return_false")]
        pub visible: bool,
        pub description: String,
        #[serde(flatten)]
        pub definition: JobsetDefinition,
        pub triggertime: Option<u64>,
        pub starttime: Option<u64>,
        pub errortime: Option<u64>,
        pub errormsg: Option<String>,
    }

    #[derive(serde::Deserialize)]
    pub struct JobsetEval {
        pub id: u64,
    }

    #[derive(serde_repr::Deserialize_repr, Debug)]
    #[repr(u8)]
    #[allow(unused)]
    pub enum BuildStatus {
        Succeeded = 0,
        #[serde(other)]
        Failed = 1,
    }

    #[test]
    fn test_build_status() {
        assert!(matches!(
            serde_json::from_str::<BuildStatus>("0"),
            Ok(BuildStatus::Succeeded)
        ));
        assert!(matches!(
            serde_json::from_str::<BuildStatus>("3"),
            Ok(BuildStatus::Failed)
        ));
    }

    #[derive(serde::Deserialize)]
    pub struct Build {
        pub id: u64,
        #[serde(deserialize_with = "bool_from_int")]
        pub finished: bool,
        pub buildstatus: Option<BuildStatus>,
        pub job: String,
    }

    fn bool_from_int<'de, D>(deserializer: D) -> Result<bool, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct BoolFromInt;

        impl<'de> serde::de::Visitor<'de> for BoolFromInt {
            type Value = bool;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("integer")
            }

            fn visit_u64<E>(self, value: u64) -> Result<bool, E>
            where
                E: serde::de::Error,
            {
                Ok(value == 1)
            }
        }
        deserializer.deserialize_u64(BoolFromInt)
    }

    #[derive(serde::Deserialize)]
    pub struct Project {
        pub jobsets: Vec<String>,
    }

    #[derive(Clone)]
    pub struct Client {
        http_client: reqwest::Client,
        base_url: String,
    }

    impl Client {
        pub async fn new(
            base_url: &str,
            user_agent: &str,
            username: &str,
            password: Vec<u8>,
        ) -> Result<Client> {
            let http_client = reqwest::ClientBuilder::new()
                .user_agent(user_agent)
                .cookie_store(true)
                //.connection_verbose(true)
                //.http1_only()
                .redirect(reqwest::redirect::Policy::custom(|attempt| {
                    if attempt.url().path() == "/current-user" {
                        attempt.stop()
                    } else {
                        reqwest::redirect::Policy::default().redirect(attempt)
                    }
                }))
                .build()
                .expect("couldn't build client");
            eprintln!("authorizing with {base_url} with username {username}");
            // TODO: Authorize on demand to keep the session going for longer
            http_client
                .post(format!("{base_url}/login"))
                .header(reqwest::header::ACCEPT, "application/json")
                .header(reqwest::header::REFERER, base_url)
                .json(&serde_json::json!({
                    "username": username,
                    "password": String::from_utf8_lossy(&password),
                }))
                .send()
                .await?
                .error_for_status()
                .context("couldn't login to Hydra")?;
            Ok(Client {
                http_client,
                base_url: base_url.to_string(),
            })
        }

        pub async fn put_jobset(
            &self,
            project: &str,
            jobset_id: &str,
            jobset: Jobset,
        ) -> Result<()> {
            let _ = self
                .http_client
                .put(format!(
                    "{}/jobset/{}/{}",
                    self.base_url, project, jobset_id,
                ))
                .header(reqwest::header::ACCEPT, "application/json")
                .json(&jobset)
                .send()
                .await
                .with_context(|| {
                    format!(
                        "failed to send put request for jobset {jobset_id} in project {project}"
                    )
                })?
                .error_for_status()
                .with_context(|| {
                    format!("put request for jobset {jobset_id} in project {project} failed")
                })?;
            Ok(())
        }

        pub async fn trigger_jobset(&self, project_jobset: &str) -> Result<Vec<String>> {
            #[derive(serde::Deserialize)]
            #[serde(rename_all = "camelCase")]
            pub struct Response {
                pub jobsets_triggered: Vec<String>,
            }
            let response = self
                .http_client
                .post(format!("{}/api/push", self.base_url))
                .query(&[("jobsets", project_jobset), ("force", "true")])
                .header(reqwest::header::ACCEPT, "application/json")
                .header(reqwest::header::REFERER, &self.base_url)
                .send()
                .await
                .with_context(|| {
                    format!("failed to send request to trigger jobset {project_jobset}")
                })?
                .error_for_status()
                .with_context(|| format!("request to trigger jobset {project_jobset} failed"))?
                .json::<Response>()
                .await
                .context("failed to parse JSON")?;
            Ok(response.jobsets_triggered)
        }

        pub async fn get_project(&self, project: &str) -> Result<Project> {
            self.http_client
                .get(format!("{}/project/{project}", self.base_url))
                .header(reqwest::header::ACCEPT, "application/json")
                .send()
                .await
                .with_context(|| format!("failed to send request to get project {project}"))?
                .error_for_status()
                .with_context(|| format!("request to get project {project} failed"))?
                .json::<Project>()
                .await
                .context("failed to parse JSON")
        }

        pub fn jobset_url(&self, project: &str, jobset: &str) -> String {
            format!("{}/jobset/{project}/{jobset}", self.base_url)
        }

        pub async fn get_jobset(&self, project: &str, jobset: &str) -> Result<Jobset> {
            self.http_client
                // Use undocumented /jobset/<project>/<jobset>/errors API that actually returns errormsg
                .get(self.jobset_url(project, jobset) + "/errors")
                .header(reqwest::header::ACCEPT, "application/json")
                .send()
                .await
                .with_context(|| {
                    format!("failed to send request to get jobset {project}:{jobset}")
                })?
                .error_for_status()
                .with_context(|| format!("request to get jobset {project}:{jobset} failed"))?
                .json::<Jobset>()
                .await
                .context("failed to parse JSON")
        }

        pub fn eval_url(&self, eval_id: u64) -> String {
            format!("{}/eval/{}", self.base_url, eval_id)
        }

        pub async fn get_jobset_evals(
            &self,
            project: &str,
            jobset: &str,
        ) -> Result<Vec<JobsetEval>> {
            // Ignore pagination because we don't expect many evals (more than 1, really)
            #[derive(serde::Deserialize)]
            struct Response {
                evals: Vec<JobsetEval>,
            }
            self.http_client
                .get(format!("{}/jobset/{project}/{jobset}/evals", self.base_url))
                .header(reqwest::header::ACCEPT, "application/json")
                .send()
                .await
                .with_context(|| {
                    format!("failed to send request to get evals for jobset {project}:{jobset}")
                })?
                .error_for_status()
                .with_context(|| {
                    format!("request to get evals for jobset {project}:{jobset} failed")
                })?
                .json::<Response>()
                .await
                .context("failed to parse JSON")
                .map(|res| res.evals)
        }

        pub async fn get_eval_builds(&self, eval_id: u64) -> Result<Vec<Build>> {
            self.http_client
                .get(format!("{}/eval/{eval_id}/builds", self.base_url))
                .header(reqwest::header::ACCEPT, "application/json")
                .send()
                .await
                .with_context(|| {
                    format!("failed to send request to get builds for eval {eval_id}")
                })?
                .error_for_status()
                .with_context(|| format!("request to get builds for eval {eval_id} failed"))?
                .json()
                .await
                .context("failed to parse JSON")
        }

        pub(crate) fn build_url(&self, build_id: u64) -> String {
            format!("{}/build/{build_id}", self.base_url)
        }
    }
}

fn parse_jobset_name(jobset_name: &str) -> Option<(u64, String)> {
    let mut parts = jobset_name.split("-");
    if parts.next()? != "pr" {
        return None;
    }
    let pr_number = parts.next()?.parse().ok()?;
    let commit_sha = parts.next()?;
    if commit_sha.len() != 40 || commit_sha.find(|c: char| !c.is_ascii_hexdigit()).is_some() {
        return None;
    }
    // check commit_sha
    if parts.next().is_some() {
        return None;
    }
    Some((pr_number, commit_sha.to_string()))
}

#[test]
fn test_parse_jobset_name() {
    assert_eq!(
        parse_jobset_name("pr-1-90640e3953b7ed1d1ccf9b888d60009ff22fdf5a"),
        Some((1, "90640e3953b7ed1d1ccf9b888d60009ff22fdf5a".to_string())),
    );
    assert_eq!(
        parse_jobset_name("pr-1-90640e3953b7ed1d1ccf9b888d60009ff22fdf5"),
        None,
    );
}

async fn create_jobset_for_pr(
    github_client: &github::Client,
    hydra_client: &hydra::Client,
    hydra_project: &str,
    repo_config: &RepoConfig,
    repo_full_name: &str,
    pull_request: &github::PullRequest,
) -> Result<String> {
    let (merge_commit_sha, target_commit_sha) = github::ensure_pr_merge_commit(
        github_client,
        repo_full_name,
        pull_request.number,
        &pull_request.head.sha,
    )
    .await
    .context("failed to wait for the merge commit")?;
    let jobset_id = format!("pr-{}-{}", pull_request.number, pull_request.head.sha);
    let mut jobset = repo_config.hydra_jobset_template.clone();
    // String::replace_first is only in nightly
    if let Some((start, match_str)) = jobset.description.match_indices("{pr_url}").next() {
        jobset
            .description
            .replace_range(start..start + match_str.len(), &pull_request.html_url);
    }
    match jobset.definition {
        hydra::JobsetDefinition::Legacy { ref mut inputs, .. } => {
            let clone_url = format!("https://github.com/{repo_full_name}.git");
            for input in inputs.values_mut() {
                match input.r#type.as_str() {
                    "pr merge" => {
                        input.r#type = "git".to_string();
                        input.value = format!("{clone_url} {merge_commit_sha}");
                    }
                    "pr base" => {
                        input.r#type = "git".to_string();
                        input.value = format!("{clone_url} {target_commit_sha}");
                    }
                    "pr head" => {
                        input.r#type = "git".to_string();
                        input.value = format!("{clone_url} {}", pull_request.head.sha);
                    }
                    _ => {}
                };
            }
            inputs
                .entry("repository_name".to_string())
                .insert_entry(hydra::JobsetInput {
                    r#type: "string".to_string(),
                    value: repo_full_name.to_string(),
                });
            inputs
                .entry("pr_number".to_string())
                .insert_entry(hydra::JobsetInput {
                    r#type: "string".to_string(),
                    value: pull_request.number.to_string(),
                });
        }
        hydra::JobsetDefinition::Flake { ref mut flake } => match flake.as_str() {
            "{pr merge}" => *flake = format!("github:{repo_full_name}/{merge_commit_sha}"),
            "{pr base}" => *flake = format!("github:{repo_full_name}/{target_commit_sha}"),
            "{pr head}" => *flake = format!("github:{repo_full_name}/{}", pull_request.head.sha),
            _ => {}
        },
    }
    hydra_client
        .put_jobset(hydra_project, &jobset_id, jobset)
        .await?;
    let project_jobset = format!("{}:{}", hydra_project, jobset_id);
    let jobsets_triggered = hydra_client.trigger_jobset(&project_jobset).await?;

    if !jobsets_triggered.contains(&project_jobset) {
        return Err(anyhow!("jobset {project_jobset} was not triggered"));
    }
    eprintln!("jobset {project_jobset} was triggered successfully");
    Ok(jobset_id)
}

mod webhook {
    use crate::github::{
        Installation, InstallationAction, InstallationRepositoriesAction, Payload,
    };
    use crate::{github, hydra};
    use anyhow::{Context, Result, anyhow};

    #[derive(Debug)]
    enum Error {
        Internal(anyhow::Error),
        Bad(String),
    }

    impl warp::reject::Reject for Error {}

    impl Error {
        fn reject_internal(err: anyhow::Error) -> warp::Rejection {
            warp::reject::custom(Error::Internal(err))
        }
    }

    fn check_signature_filter(
        secret: Vec<u8>,
    ) -> impl warp::Filter<Extract = (bytes::Bytes,), Error = warp::Rejection>
    + Clone
    + Send
    + Sync
    + 'static {
        use futures_util::TryFutureExt;
        use warp::Filter;
        warp::header::<String>("X-Hub-Signature-256")
            .and_then(async |header| {
                github::parse_signature_header(header)
                    .await
                    .map_err(Error::reject_internal)
            })
            .and(warp::body::content_length_limit(1024 * 1024))
            .and(warp::body::bytes())
            .and_then({
                let secret = std::sync::Arc::new(secret);
                move |sig, body| {
                    (async |secret: std::sync::Arc<Vec<u8>>, sig, body| {
                        github::check_signature(&secret.clone(), sig, body)
                            .map_err(Error::reject_internal)
                            .await
                    })(secret.clone(), sig, body)
                }
            })
    }

    #[tokio::test]
    async fn test_signature_verification() {
        use warp::Filter;
        let filter = warp::post()
            .and(check_signature_filter("It's a Secret to Everybody".into()))
            .map(|_| warp::reply());

        let response = warp::test::request()
            .method("POST")
            .path("/")
            .header(
                "X-Hub-Signature-256",
                "sha256=757107ea0eb2509fc211221cce984b8a37570b6d7586c22c46f4379c8b043e17",
            )
            .body("Hello, World!")
            .reply(&filter)
            .await;
        eprintln!("{:?}", response);
        assert_eq!(response.status(), 200);
    }

    async fn sync_hydra_jobsets(
        hydra_client: hydra::Client,
        github_client: github::ApplicationClient,
        hydra_project: String,
        app_id: u64,
        repositories_cfg: std::sync::Arc<std::collections::HashMap<String, crate::RepoConfig>>,
    ) {
        // TODO: keep a task per jobset in memory instead of looping
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            let _ =
                async {
                    let project = hydra_client
                        .get_project(&hydra_project)
                        .await
                        .with_context(|| {
                            format!("failed to get information about project {}", hydra_project)
                        })?;
                    //eprintln!("got project {}: {project:?}", hydra_project);
                    for jobset_name in project.jobsets {
                        let _ = async {
                        let (_pr_number, commit_sha) = match crate::parse_jobset_name(&jobset_name)
                        {
                            Some(x) => x,
                            None => return Ok::<(), anyhow::Error>(()),
                        };
                        let jobset = hydra_client.get_jobset(&hydra_project, &jobset_name).await?;
                        if matches!(jobset.enabled, hydra::JobsetEnabled::Disabled)
                            || !jobset.visible
                        {
                            return Ok(());
                        }
                        eprintln!("got jobset: {jobset_name}");

                        let repository_name = match jobset.definition {
                            hydra::JobsetDefinition::Legacy { ref inputs, .. } => {
                                &inputs
                                    .get("repository_name")
                                    .ok_or_else(|| anyhow!("repository_name input not found"))?
                                    .value
                            }
                            hydra::JobsetDefinition::Flake { ref flake } => {
                                // flake jobsets don't have inputs, so we have to parse repo name from the flake URI
                                let mut schema_split = flake.split(':');
                                if schema_split.next().is_none_or(|s| s != "github") {
                                    return Err(anyhow!("unexpected schema in flake URI {flake}"));
                                }
                                let path = schema_split.next().ok_or_else(|| {
                                    anyhow!("failed to get flake path from URI {flake}")
                                })?;
                                let mut path_split = path.split('/');
                                let org = path_split.next().ok_or_else(|| {
                                    anyhow!("failed to get GitHub org from flake URL {flake}")
                                })?;
                                let repo = path_split.next().ok_or_else(|| {
                                    anyhow!("failed to get GitHub repo from flake URL {flake}")
                                })?;
                                &format!("{org}/{repo}")
                            }
                        };

                        let repo_config = match repositories_cfg.get(repository_name) {
                            Some(c) => c,
                            None => {
                                return Err(anyhow!(
                                    "repository not found in config: {repository_name}"
                                ));
                            }
                        };

                        let installation_client = github_client.for_repo(repository_name).await?;

                        let (status, details_url, builds) = 'data: {
                            use github::CheckRunConclusion::*;
                            use github::CheckRunStatus::*;

                            let details_url = hydra_client.jobset_url(&hydra_project, &jobset_name);

                            if jobset.errortime.is_some()
                                && jobset.errormsg.as_ref().is_some_and(|m| !m.is_empty())
                            {
                                eprintln!("jobset {jobset_name} failed to evaluate");
                                break 'data (Completed(Failure), details_url, None);
                            }
                            if jobset.triggertime.is_some() || jobset.starttime.is_some() {
                                eprintln!("jobset {jobset_name} is being evaluated");
                                break 'data (Queued, details_url, None);
                            }
                            let evals = hydra_client
                                .get_jobset_evals(&hydra_project, &jobset_name)
                                .await
                                .context("failed to get evals")?;

                            let last_eval = match evals.first() {
                                Some(eval) => eval,
                                None => {
                                    eprintln!("jobset {jobset_name} has no evals");
                                    break 'data (Completed(Skipped), details_url, None);
                                }
                            };
                            eprintln!(
                                "got {} evals for jobset {jobset_name}, picking the first one, {}",
                                evals.len(),
                                last_eval.id
                            );
                            let details_url = hydra_client.eval_url(last_eval.id);

                            let builds = hydra_client
                                .get_eval_builds(last_eval.id)
                                .await
                                .context("failed to get builds")?;

                            let (all_finished, has_failures) = builds.iter().fold(
                                (true, false),
                                |(all_finished, has_failures), build| match (
                                    build.finished,
                                    &build.buildstatus,
                                ) {
                                    (false, _) => (false, has_failures),
                                    (true, Some(hydra::BuildStatus::Succeeded)) => {
                                        (all_finished, has_failures)
                                    }
                                    (true, _) => (all_finished, true),
                                },
                            );

                            (
                                if !all_finished {
                                    InProgress
                                } else {
                                    Completed(if has_failures { Failure } else { Success })
                                },
                                details_url,
                                Some(builds),
                            )
                        };

                        github::upsert_check(
                            &installation_client,
                            app_id,
                            &repo_config.check_run_name,
                            repository_name,
                            &commit_sha,
                            &github::UpsertCheckRunData {
                                status: status.clone(),
                                details_url,
                            },
                        )
                        .await?;

                        if let Some(builds) = builds
                            && repo_config.check_per_job
                        {
                            use github::CheckRunConclusion::*;
                            use github::CheckRunStatus::*;

                            for build in builds {
                                github::upsert_check(
                                    &installation_client,
                                    app_id,
                                    &build.job,
                                    repository_name,
                                    &commit_sha,
                                    &github::UpsertCheckRunData {
                                        status: match (build.finished, &build.buildstatus) {
                                            (false, _) => InProgress,
                                            (true, Some(hydra::BuildStatus::Succeeded)) => {
                                                Completed(Success)
                                            }
                                            // TODO: more granular conclusions?
                                            (true, _) => Completed(Failure),
                                        },
                                        details_url: hydra_client.build_url(build.id),
                                    },
                                )
                                .await?;
                            }
                        }

                        if matches!(status, github::CheckRunStatus::Completed(_)) {
                            eprintln!("disabling jobset {jobset_name}");
                            hydra_client
                                .put_jobset(
                                    &hydra_project,
                                    &jobset_name,
                                    hydra::Jobset {
                                        enabled: hydra::JobsetEnabled::Disabled,
                                        visible: false,
                                        ..jobset
                                    },
                                )
                                .await?;
                        }
                        Ok(())
                    }
                    .await
                    .with_context(|| format!("failed to process jobset {jobset_name}"))
                    .inspect_err(|err| eprintln!("{err:?}"));
                    }
                    Ok::<(), anyhow::Error>(())
                }
                .await
                .inspect_err(|err| eprintln!("{err:?}"));
        }
    }

    fn check_client_id(
        client_id: &str,
        installation: &Installation,
    ) -> Result<(), warp::Rejection> {
        if installation.client_id != client_id {
            return Err(Error::Bad(format!(
                "Unexpected app client ID: {}",
                installation.client_id
            ))
            .into());
        }
        Ok(())
    }

    fn warn_about_not_allowed_repositories<T: Iterator<Item = String>>(
        repositories_cfg: &std::collections::HashMap<String, crate::RepoConfig>,
        provided: T,
    ) {
        let notallowed: Vec<_> = provided
            .filter(|r| !repositories_cfg.contains_key(r))
            .collect();
        if !notallowed.is_empty() {
            eprintln!(
                "app was installed on repositories that are not allowed, events for these repositories will be ignored: {}",
                notallowed
                    .iter()
                    .map(|r| r.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }

    async fn handle_payload(
        hydra_client: hydra::Client,
        github_client: github::ApplicationClient,
        app_id: u64,
        app_client_id: String,
        hydra_project: String,
        repositories_cfg: std::sync::Arc<std::collections::HashMap<String, crate::RepoConfig>>,
        payload: Payload,
    ) -> Result<impl warp::Reply, warp::Rejection> {
        //eprintln!("handle_payload: {:?}", payload);
        match payload {
            Payload::Ping => (),
            Payload::Installation(event) => {
                check_client_id(&app_client_id, &event.installation)?;
                if matches!(event.action, InstallationAction::Created) {
                    warn_about_not_allowed_repositories(
                        &repositories_cfg,
                        event.repositories.iter().map(|r| r.full_name.clone()),
                    )
                }
                eprintln!(
                    "installation: {:?} for repositories: {}",
                    event.action,
                    event
                        .repositories
                        .iter()
                        .map(|r| r.full_name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                if let Ok(installation_client) =
                    github_client.for_installation(event.installation.id).await
                {
                    for r in event.repositories {
                        eprintln!("will configure check suite preferences for {}", r.full_name);
                        let installation_client = installation_client.clone();
                        tokio::spawn(async move {
                            eprintln!("configuring check suite preferences for {}", r.full_name);
                            let _ = installation_client
                                .patch_check_suites_preferences(
                                    r.full_name,
                                    event.installation.app_id,
                                    false,
                                )
                                .await
                                .inspect_err(|err| eprintln!("{err}"));
                        });
                    }
                }
            }
            Payload::InstallationRepositories(event) => {
                check_client_id(&app_client_id, &event.installation)?;
                if matches!(event.action, InstallationRepositoriesAction::Added) {
                    warn_about_not_allowed_repositories(
                        &repositories_cfg,
                        event.repositories_added.iter().map(|r| r.full_name.clone()),
                    )
                }
                eprintln!(
                    "installation repositories: {:?} for repositories: {}",
                    event.action,
                    match event.action {
                        InstallationRepositoriesAction::Added => event.repositories_added,
                        InstallationRepositoriesAction::Removed => event.repositories_removed,
                    }
                    .iter()
                    .map(|r| r.full_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
                );
            }
            Payload::PullRequest(event) => 'pr: {
                eprintln!("got pull_request event: {:#?}", event);
                if !matches!(
                    event.action,
                    github::PullRequestAction::Opened
                        | github::PullRequestAction::Reopened
                        | github::PullRequestAction::Synchronize
                ) {
                    break 'pr;
                }
                let repo_full_name = &event.repository.full_name;
                let repo_config = match repositories_cfg.get(repo_full_name) {
                    Some(c) => c,
                    None => {
                        eprintln!("repository not found in config: {repo_full_name}");
                        break 'pr;
                    }
                };
                let installation_client = github_client
                    .for_installation(event.installation.id)
                    .await
                    .context("couldn't get installation client")
                    .map_err(Error::reject_internal)?;
                let jobset_id = crate::create_jobset_for_pr(
                    &installation_client.get_client(),
                    &hydra_client,
                    &hydra_project,
                    repo_config,
                    repo_full_name,
                    &event.pull_request,
                )
                .await
                .map_err(Error::reject_internal)?;
                let _ = github::upsert_check(
                    &installation_client,
                    app_id,
                    &repo_config.check_run_name,
                    repo_full_name,
                    &event.pull_request.head.sha,
                    &github::UpsertCheckRunData {
                        status: github::CheckRunStatus::Queued,
                        details_url: hydra_client.jobset_url(&hydra_project, &jobset_id),
                    },
                )
                .await
                .inspect_err(|err| eprintln!("{err}"));
            }
        };
        Ok(warp::reply::with_status(
            "",
            warp::http::StatusCode::NO_CONTENT,
        ))
    }

    // Parse webhook payload

    #[allow(dead_code)]
    fn log_body(body: &bytes::Bytes) {
        eprintln!(
            "{}",
            serde_json::from_slice::<serde_json::Value>(body)
                .and_then(|v| serde_json::to_string_pretty(&v))
                .unwrap_or("<couldn't parse body into Value>".into())
        );
    }

    async fn parse_webhook(body: bytes::Bytes, event: String) -> Result<Payload, warp::Rejection> {
        eprintln!("Got event: {}", event);
        match event.as_str() {
            "ping" => Ok(Payload::Ping),
            "installation" => serde_json::from_slice(&body).map(Payload::Installation),
            "installation_repositories" => {
                serde_json::from_slice(&body).map(Payload::InstallationRepositories)
            }
            "pull_request" => serde_json::from_slice(&body).map(Payload::PullRequest),
            // TODO: also handle check_run event so that user can click "re-run"
            _ => {
                //log_body(&body);
                return Err(Error::Bad(format!("Unexpected event type: {}", event)).into());
            }
        }
        .with_context(|| format!("failed to parse {event}"))
        .map_err(Error::reject_internal)
    }

    pub async fn run(cfg: crate::Config) -> Result<()> {
        // it's too hard to properly deal with non-static lifetimes in async code, so leak it
        // let cfg: &crate::Config = Box::leak(Box::new(cfg));
        let crate::Config {
            listen: Some(listen_address),
            user_agent,
            github_app: Some(app_cfg),
            hydra: hydra_cfg,
            repositories: repositories_cfg,
        } = cfg
        else {
            return Err(anyhow!(
                "config must contain \"listen\" and \"github_app\" fields"
            ));
        };
        let webhook_secret = app_cfg
            .webhook_secret
            .get()
            .context("couldn't get webhook secret")?;

        let hydra_client = hydra::Client::new(
            &hydra_cfg.url,
            &user_agent,
            &hydra_cfg.user,
            hydra_cfg
                .password
                .get()
                .context("couldn't get Hydra password")?,
        )
        .await?;
        let hydra_project = hydra_cfg.project;

        use rsa::pkcs1::DecodeRsaPrivateKey;
        let github_client = github::ApplicationClient::new(
            &user_agent,
            &app_cfg.client_id,
            rsa::RsaPrivateKey::from_pkcs1_pem(&String::from_utf8_lossy(
                &app_cfg
                    .app_private_key
                    .get()
                    .context("couldn't get app private key")?,
            ))
            .context("couldn't parse app private key")?,
        );
        let repositories_cfg = std::sync::Arc::new(repositories_cfg);
        let crate::GitHubAppConfig {
            app_id, client_id, ..
        } = app_cfg;

        tokio::spawn(sync_hydra_jobsets(
            hydra_client.clone(),
            github_client.clone(),
            hydra_project.clone(),
            app_cfg.app_id,
            repositories_cfg.clone(),
        ));
        eprintln!("Will listen on {}", listen_address);
        use warp::Filter;
        let route = warp::path::end()
            .and(warp::post())
            .and(check_signature_filter(webhook_secret))
            .and(warp::header::exact_ignore_case(
                "content-type",
                "application/json",
            ))
            .and(warp::header::<String>("X-GitHub-Event"))
            .and_then(parse_webhook)
            .and_then(move |payload| {
                handle_payload(
                    hydra_client.clone(),
                    github_client.clone(),
                    app_id,
                    client_id.clone(),
                    hydra_project.clone(),
                    repositories_cfg.clone(),
                    payload,
                )
            })
            .recover(async |err: warp::Rejection| {
                if let Some(e) = err.find::<Error>() {
                    match e {
                        Error::Internal(e) => {
                            eprintln!("got internal error: {:?}", e);
                            Ok(warp::reply::with_status(
                                "Internal error".to_string(),
                                warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                            ))
                        }
                        Error::Bad(s) => {
                            eprintln!("bad request: {}", s);
                            Ok(warp::reply::with_status(
                                s.clone(),
                                warp::http::StatusCode::BAD_REQUEST,
                            ))
                        }
                    }
                } else {
                    Err(err)
                }
            })
            .with(warp::log::custom(|info| {
                eprintln!(
                    "{} {} {} {:?}",
                    info.method(),
                    info.path(),
                    info.status(),
                    info.request_headers(),
                );
            }));
        warp::serve(route).run(listen_address).await;
        Ok(())
    }
}
