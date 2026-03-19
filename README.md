# Installation

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

[Direnv](https://direnv.net/) can automatically load environment variables.