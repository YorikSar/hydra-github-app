# Hydra GitHub app

This project aims to provide a GitHub app interface that works in front of [Hydra](https://nixos.org/hydra) instance.

For each new PR event received by the webhook, a new jobset is created in Hydra. We always create a new jobset to make sure Hydra creates a new eval, so that we can reflect its status in GitHub accordingly. Jobsets are created by template provided in the config, replacing special inputs of type `pr base` and `pr merge` with `git` inputs pointing to the appropriate commits: the tip of the PR's target branch and a merge commit that was created by merging the PR into it.

When a new jobset is created, a new check is created on the target commit in *queued* status. After jobset is evaluated, if no evals were created, the check status is set to *completed* with the conclusion *skipped*, otherwise it is set to *in progress*. When all jobs in the jobset finish, the check status is set to *completed* with the conclusion of *success* or *failure* depending on the result.

## Usage

You can run this project with either:

```
nix run . config.json
```

or

```
cargo run config.json
```

passing the `config.json` with all the configuration as the argument.

### Set up

#### GitHub

To create and install GitHub app:
1. go to your org's Settings, Developer settings, GitHub apps (https://github.com/organizations/ORG/settings/apps)
1. click "New GitHub app", then potentially authenticate, then fill out at least:
    * *GitHub App name* - whatever you like, but must be unique for the all of GitHub
    * *Homepage URL* - any URL, doesn't have to be working
    * in *Webhook*:
        * keep *Active* checked 
        * fill the URL that will point to the app's server
        * write any text in *Secret* field and save it securely to put in the config
    * in *Permissions* under *Repository permissions* for *Checks* select *Read and write*
        * it means that the key for this app can create any checks for any commit in the repo
    * under *Where can this GitHub App be installed?* select the appropriate option
1. press *Create GitHub App*
1. on the configuration screen for the new app:
    * take note of *App ID* and *Client ID* to put into the config later
    * in *Private keys* section click *Generate a private key*, it will download a key to your machine, save it securely to pass to the app as well
1. in the left menu click *Install App* and install it in the target organisation(s) for target repo(s) or ask their administrators to do so

#### Hydra

1. Create a user that will be used by the app, by running this on the machine with Hydra and specifying a password for it (replace `github-app` with the desired user name):

    ```console
    $ hydra-create-user github-app --password-prompt
    Password:
    Password Confirmation:
    ```

2. Create a project with the new user set as an owner. Some actions performed by the app can only be done by the project owner or a global Hydra admin.

### Configuration

Configuration file is a JSON with following fields:

* `listen` is a string containing IPv4 or IPv6 address and port to that the webhook server will listen on (see Rust stdlib documentation for the supported textual representations for [IPv4](https://doc.rust-lang.org/std/net/struct.SocketAddrV4.html#textual-representation) and [IPv6](https://doc.rust-lang.org/std/net/struct.SocketAddrV6.html#textual-representation)).
* `user_agent` is a string that will be used as `User-Agent` header for HTTP requests. GitHub [requires](https://docs.github.com/en/rest/using-the-rest-api/troubleshooting-the-rest-api?apiVersion=2022-11-28#user-agent-required) it to be valid.
* `github_app` is an object with GitHub app config with following properties:
    * `webhook_secret_file` is a path to the file containing GitHub webhook secret. It can be any text specified during webhook creation in GitHub.
    * `app_private_key_file` is a path to the file containing GitHub app's private key generated in the GitHub app's settings.
    * `app_id` is the ID of the GitHub app, can be obtained on the GitHub app's settings page.
    * `client_id` is the client ID of the GitHub app, can be obtained on the GitHub app's settings page.
* `hydra` is an object with Hydra config with following properties:
    * `url` is the base URL for Hydra instance where jobsets will be created.
    * `user` and `password_env` are user name and the name of the environment variable containing the password that will be used to authenticate in Hydra.
    * `project` is the name of the project in Hydra where jobsets will be created and watched.
* `repositories` is an object with full names (`"ORG/REPO"`) of the GitHub repositories that are allowed to trigger Hydra builds as keys and following properties:
    * `check_run_name` is the name of the check representing the whole jobset that will be created on each commit.
    * `check_per_job` is a boolean flag, if it is enabled, every job in the jobset will be represented as a separate check named after its attribute path.
    * `hydra_jobset_template` is a representation of Hydra jobset configuration with following allowances:
        * `description` field can contain `{pr_url}` string that will be replaced with the URL of the PR that triggered this jobset;
        * `inputs` can contain inputs with no values with special types:
            * `pr base` will be replaced with `git` input pointing to the tip of the PR's target branch;
            * `pr merge` will be replaced with `git` input pointing to the PR merge commit generated by GitHub;
        * special string inputs will be added:
            * `repository_name` containing the name of the repository from which the jobset was triggered;
            * `pr_number` containing the number of the PR that triggered the jobset.

<details>
<summary>Full example</summary>

```json
{
    "listen": "127.0.0.1:3000",
    "user_agent": "Hi I am a GitHub app",
    "github_app": {
        "webhook_secret_file": "secret",
        "app_private_key_file": "private-key.pem",
        "app_id": 1234567,
        "client_id": "WFzzq5HUHbp8T484TFRT"
    },
    "hydra": {
        "url": "https://hydra.example.org",
        "user": "admin",
        "password_env": "HYDRA_PASSWORD",
        "project": "pr-tests"
    },
    "repositories": {
        "NixOS/nixpkgs": {
            "check_run_name": "Hydra check",
            "hydra_jobset_template": {
                "description": "triggered by PR {pr_url}",
                "nixexprinput": "jobsets",
                "nixexprpath": "pr.nix",
                "inputs": {
                    "jobsets": {
                        "type": "git",
                        "value": "https://github.com/example/jobsets"
                    },
                    "nixpkgsMerge": {
                        "type": "pr merge"
                    },
                    "nixpkgs": {
                        "type": "pr base"
                    }
                }
            }
        }
    }
}
```
</details>
