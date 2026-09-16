# TerraGuard AI - reproducible Phase 1 environment setup (Windows / PowerShell).
# Creates backend/.venv with Python 3.12 and installs the locked dependency stack.

$ErrorActionPreference = "Stop"

# 1. Virtual environment on the spec-locked Python (3.12)
py -3.12 -m venv backend\.venv

$py = "backend\.venv\Scripts\python.exe"

# 2. Modern packaging tools
& $py -m pip install --upgrade pip wheel setuptools

# 3. PyTorch first (CPU build on Windows), so the resolver in requirements.txt sees it satisfied
& $py -m pip install torch torchvision

# 3b. Windows: torch 2.14 needs the MSVC 17.10+ "threads" runtime family.
#     Preferred fix (elevated): winget install --id Microsoft.VCRedist.2015+.x64 -e
#     App-local fallback (no elevation): copy the DLLs from any app that bundles them.
$torchLib = Resolve-Path "backend\.venv\Lib\site-packages\torch\lib"
$threadsDll = Join-Path $torchLib "vcruntime140_threads.dll"
if (-not (Test-Path $threadsDll) -and -not (Test-Path "$env:SystemRoot\System32\vcruntime140_threads.dll")) {
    $resolveSource = Get-ChildItem 'C:\Program Files\*','C:\Program Files (x86)\*' -Filter 'vcruntime140_threads.dll' -ErrorAction SilentlyContinue |
        Get-ChildItem -Filter 'vcruntime140_threads.dll' -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
    if ($resolveSource) {
        Write-Host "Deploying MSVC threads runtime app-locally from: $resolveSource"
        $srcDir = Split-Path $resolveSource -Parent
        Copy-Item (Join-Path $srcDir '*threads*.dll') $torchLib -Force
    } else {
        Write-Warning "vcruntime140_threads.dll not found. Install the VC++ 2015-2022 x64 redistributable, then re-run this script."
    }
}

# 4. Everything else from the locked requirements file
& $py -m pip install -r backend/requirements.txt

& $py -m pip list

Write-Host "`nTerraGuard backend environment ready. Activate with: backend\.venv\Scripts\Activate.ps1"
Write-Host "Verify with: backend\.venv\Scripts\python.exe scripts/verify_environment.py"

