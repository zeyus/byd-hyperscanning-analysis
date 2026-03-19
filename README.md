# Installation

First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/zeyus/byd-hyperscanning-analysis.git
cd byd-hyperscanning-analysis
```

## UV

- [Install UV](https://docs.astral.sh/uv/#installation)
  - Macos/Linux:  
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
  - Windows:
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
  - Alternatively, UV can be installed via pip, or possibly from your package manager (e.g. `brew`, `pacman`, `apt`, `WinGet`, `choco`).

## Optional: Direnv

If you want to use [direnv](https://direnv.net/) it can automatically load environment variables for you, which can be nice for setting the path to the data etc. It's optional, but it means you don't have to change anything in the script.

- [Install direnv](https://direnv.net/docs/installation.html), the easiest way is probable via your package manager. 
- Copy the default configuration file, and make changes as appropriate:
  ```bash
  cp .envrc.dist .envrc
  # Open .envrc in your editor and update the values as needed
  ```
- Once you are happy with the contents of the .envrc file, allow it to be loaded for this project:
  ```bash
  direnv allow
  ```

## Create a virtual environment and install dependencies

```bash
uv sync
```

# Usage



