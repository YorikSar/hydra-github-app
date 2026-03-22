# Hydra GitHub app

This project aims to provide a GitHub app interface that works in front of [Hydra](https://nixos.org/hydra) instance.

For each new PR event received by the webhook, a new jobset is created in Hydra. We always create a new jobset to make sure Hydra creates a new eval, so that we can reflect its status in GitHub accordingly. Jobsets are created by template provided in the config, replacing special inputs of type `pr base` and `pr merge` with `git` inputs pointing to the appropriate commits: the tip of the PR's target branch and a merge commit that was created by merging the PR into it.

When a new jobset is created, a new check is created on the target commit in *queued* status. After jobset is evaluated, if no evals were created, the check status is set to *completed* with the conclusion *skipped*, otherwise it is set to *in progress*. When all jobs in the jobset finish, the check status is set to *completed* with the conclusion of *success* or *failure* depending on the result.

## Usage

You can run this project with either `nix run github:YorikSar/hydra-github-app config.json` or `cargo run config.json` passing the `config.json` with all the configuration as the argument.
You can also install the binary and call it directly, for example:

```
$ nix shell github:YorikSar/hydra-github-app
$ hydra-github-app config.json
```

### CLI mode

You can also just create one jobset from one PR by running:

```
hydra-github-app config.json one_pr <PR URL>
```

Where `<PR URL>` is a direct URL to a PR on GitHub in a repository mentioned in the config, for example, `https://github.com/YorikSar/hydra-github-app/pull/1`.
It will expect `GITHUB_TOKEN` environment variable to be set to a valid user or installation GitHub token that has access to this pull request.
In this mode only `user_agent`, `hydra` and `repositories` sections of the config are required.

This will trigger almost the same reaction as if webhook received a payload for this pull request, except it won't create any checks on the PR and will exit after the jobset is triggered.

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

Configuration file is a plain JSON. Secrets are represented as objects with one of following fields:

* `env` will read the secret from the environment variable with the provided name
* `file_name` will read the secret from the file with the provided name
* `env_file_name` will read the secret from the file with the name read from the environment variable with the provided name

When read from a file, if a secret ends with a newline, it will be removed.

Configuration contains following fields:

* `listen` is a string containing IPv4 or IPv6 address and port that the webhook server will listen on (see Rust stdlib documentation for the supported textual representations for [IPv4](https://doc.rust-lang.org/std/net/struct.SocketAddrV4.html#textual-representation) and [IPv6](https://doc.rust-lang.org/std/net/struct.SocketAddrV6.html#textual-representation)).
* `user_agent` is a string that will be used as `User-Agent` header for HTTP requests. GitHub [requires](https://docs.github.com/en/rest/using-the-rest-api/troubleshooting-the-rest-api?apiVersion=2022-11-28#user-agent-required) it to be valid.
* `github_app` is an object with GitHub app config with following properties:
    * `webhook_secret` is a secret containing GitHub webhook secret. It can be any text specified during webhook creation in GitHub.
    * `app_private_key` is a secret containing GitHub app's private key generated in the GitHub app's settings.
    * `app_id` is the ID of the GitHub app, can be obtained on the GitHub app's settings page.
    * `client_id` is the client ID of the GitHub app, can be obtained on the GitHub app's settings page.
* `hydra` is an object with Hydra config with following properties:
    * `url` is the base URL for Hydra instance where jobsets will be created.
    * `user` and `password` are user name and the secret containing the password that will be used to authenticate in Hydra.
    * `project` is the name of the project in Hydra where jobsets will be created and watched.
* `repositories` is an object with full names (`"ORG/REPO"`) of the GitHub repositories that are allowed to trigger Hydra builds as keys and following properties:
    * `check_run_name` is the name of the check representing the whole jobset that will be created on each commit.
    * `check_per_job` is a boolean flag, if it is enabled, every job in the jobset will be represented as a separate check named after its attribute path.
    * `hydra_jobset_template` is a representation of Hydra jobset configuration with following allowances:
        * `description` field can contain `{pr_url}` string that will be replaced with the URL of the PR that triggered this jobset;
        * `inputs` for legacy jobsets can contain inputs with no values with special types:
            * `pr head` will be replaced with `git` input pointing to the tip of the PR's branch;
            * `pr base` will be replaced with `git` input pointing to the tip of the PR's target branch;
            * `pr merge` will be replaced with `git` input pointing to the PR merge commit generated by GitHub;
        * special string inputs will be added:
            * `repository_name` containing the name of the repository from which the jobset was triggered;
            * `pr_number` containing the number of the PR that triggered the jobset.
        * `flake` field if present can match `{pr head}`, `{pr base}` or `{pr merge}` to be replaced with the appropriate flake URI (see above)

<details>
<summary>Full example</summary>

```json
{
    "listen": "127.0.0.1:3000",
    "user_agent": "Hi I am a GitHub app",
    "github_app": {
        "webhook_secret": {
            "file_name": "secret"
        },
        "app_private_key": {
            "file_name": "private-key.pem"
        },
        "app_id": 1234567,
        "client_id": "WFzzq5HUHbp8T484TFRT"
    },
    "hydra": {
        "url": "https://hydra.example.org",
        "user": "admin",
        "password": {
            "env": "HYDRA_PASSWORD"
        },
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
        },
        "NixOS/nix": {
            "check_run_name": "Hydra eval",
            "check_per_job": true,
            "hydra_jobset_template": {
                "description": "triggered by PR {pr_url}",
                "flake": "{pr head}"
            }
        }
    }
}
```
</details>

### NixOS module

You can also use the provided NixOS module to run this app as a systemd daemon on your NixOS system.

<details>
<summary>Generated NixOS module documentation</summary>

<!--begin generated NixOS module documentation-->
## services\.hydra-github-app\.enable

Whether to enable hydra-github-app\.



*Type:*
boolean



*Default:*

```nix
false
```



*Example:*

```nix
true
```



## services\.hydra-github-app\.package



The hydra-github-app package to use\.



*Type:*
package



*Default:*

```nix
pkgs.callPacakges ./package.nix
```



## services\.hydra-github-app\.settings



Configuration options that will be passed to the daemon



*Type:*
submodule



*Default:*

```nix
{ }
```



## services\.hydra-github-app\.settings\.github_app\.app_id



the ID of the GitHub app, can be obtained on the GitHub app’s settings page



*Type:*
positive integer, meaning >0



## services\.hydra-github-app\.settings\.github_app\.app_private_key_file



a path to the file containing GitHub app’s private key generated in the GitHub app’s settings



*Type:*
absolute path not in the Nix store



## services\.hydra-github-app\.settings\.github_app\.client_id



the client ID of the GitHub app, can be obtained on the GitHub app’s settings page



*Type:*
non-empty string



## services\.hydra-github-app\.settings\.github_app\.webhook_secret_file



a path to the file containing GitHub webhook secret\. It can be any text specified during webhook creation in GitHub



*Type:*
absolute path not in the Nix store



## services\.hydra-github-app\.settings\.hydra\.password_file



a path to the file containing password used to authenticate in Hydra



*Type:*
absolute path not in the Nix store



## services\.hydra-github-app\.settings\.hydra\.project



name of the project in Hydra where jobsets will be created and watched



*Type:*
non-empty string



## services\.hydra-github-app\.settings\.hydra\.url



the base URL for Hydra instance where jobsets will be created



*Type:*
non-empty string



## services\.hydra-github-app\.settings\.hydra\.user



user name used to authenticate in Hydra



*Type:*
non-empty string



## services\.hydra-github-app\.settings\.listen



IPv4 or IPv6 address and port that the webhook server will listen on (see Rust stdlib documentation for the supported textual representations for [IPv4](https://doc\.rust-lang\.org/std/net/struct\.SocketAddrV4\.html\#textual-representation) and [IPv6](https://doc\.rust-lang\.org/std/net/struct\.SocketAddrV6\.html\#textual-representation))



*Type:*
non-empty string



*Example:*

```nix
"127.0.0.1:3000"
```



## services\.hydra-github-app\.settings\.repositories



an object with full names (` "ORG/REPO" `) of the GitHub repositories that are allowed to trigger Hydra builds as keys



*Type:*
attribute set of (submodule)



## services\.hydra-github-app\.settings\.repositories\.\<name>\.check_per_job



if enabled, every job in the jobset will be represented as a separate check named after its attribute path



*Type:*
boolean



*Default:*

```nix
false
```



## services\.hydra-github-app\.settings\.repositories\.\<name>\.check_run_name



the name of the check representing the whole jobset that will be created on each commit



*Type:*
non-empty string



## services\.hydra-github-app\.settings\.repositories\.\<name>\.hydra_jobset_template



a representation of Hydra jobset configuration with following allowances:

 - ` description ` field can contain ` {pr_url} ` string that will be replaced with the URL of the PR that triggered this jobset;
 - ` inputs ` for legacy jobsets can contain inputs with no values with special types:
   
    - ` pr head ` will be replaced with ` git ` input pointing to the tip of the PR’s branch;
    - ` pr base ` will be replaced with ` git ` input pointing to the tip of the PR’s target branch;
    - ` pr merge ` will be replaced with ` git ` input pointing to the PR merge commit generated by GitHub;
 - special string inputs will be added:
   
    - ` repository_name ` containing the name of the repository from which the jobset was triggered;
    - ` pr_number ` containing the number of the PR that triggered the jobset\.
 - ` flake ` field if present can match ` {pr head} `, ` {pr base} ` or ` {pr merge} ` to be replaced with the appropriate flake URI (see above)



*Type:*
JSON value



## services\.hydra-github-app\.settings\.user_agent



a string that will be used as ` User-Agent ` header for HTTP requests\. GitHub [requires](https://docs\.github\.com/en/rest/using-the-rest-api/troubleshooting-the-rest-api?apiVersion=2022-11-28\#user-agent-required) it to be valid



*Type:*
non-empty string



*Example:*

```nix
"my-hydra-app"
```


<!--end generated NixOS module documentation-->
</details>
