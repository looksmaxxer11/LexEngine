param(
  [switch]$DryRun = $true,
  [switch]$NoPrompt = $false,
  [switch]$KeepModels = $true,
  [switch]$KeepOutput = $false
)

# Cleanup script for removing legacy files/folders not used by the new pipeline.
# Run from the repository root. By default performs a dry run.

function Remove-PathSafe {
  param(
    [string]$Path
  )
  if (-not (Test-Path -LiteralPath $Path)) { return }
  if ($DryRun) {
    Write-Host "[DRY-RUN] Would remove: $Path"
  } else {
    Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Removed: $Path"
  }
}

# Define targets to delete. Full list from migration plan.
$FilesToDelete = @(
  'API_REFERENCE.md',
  'CHANGELOG.md',
  'cli.py',
  'config.ini',
  'config_v2.ini',
  'demo.py',
  'document_processor.py',
  'ENCODING_FIX.md',
  'EXAMPLE_OUTPUT.md',
  'install.bat',
  'install.sh',
  'INSTALLATION.md',
  'ocr_engine.py',
  'pipeline_v2.py',
  'preprocessing.py',
  'PROJECT_SUMMARY.md',
  'quick_process.bat',
  'QUICKSTART.md',
  'QUICKSTART_NEW.md',
  'QUICKSTART_V2.md',
  'requirements.txt',
  'requirements_new.txt',
  'requirements_v2.txt',
  'run_demo.bat',
  'setup.py',
  'setup_v2.py',
  'START_HERE.md',
  'START_HERE.txt',
  'start_web.bat',
  'test_config.py',
  'test_pipeline.py',
  'test_webapp.py',
  'train_corrector.py',
  'utils.py',
  'verify_setup.bat',
  'verify_setup.py',
  'web_app.py',
  'WEB_INTERFACE_GUIDE.md'
)

$DirsToDelete = @(
  'static',
  'templates',
  'training_data',
  'temp',
  '__pycache__',
  'input',
  'logs',
  'output'
)

# Keep list: explicitly avoid touching src/, data/, tests/, requirements_pipeline.txt, pipeline.py wrapper
$KeepSet = @(
  'src',
  'data',
  'tests',
  'requirements_pipeline.txt',
  'pipeline.py',
  'README.md'
)

# Confirm deletion when not DryRun and not NoPrompt
if (-not $DryRun -and -not $NoPrompt) {
  $answer = Read-Host "Proceed with deletion? Type 'yes' to confirm"
  if ($answer -ne 'yes') {
    Write-Host "Aborted by user."
    exit 1
  }
}

Write-Host "Starting cleanup (`$(if ($DryRun) { 'dry-run' } else { 'execute' })`)..."

foreach ($f in $FilesToDelete) {
  if ($KeepSet -contains $f) { continue }
  Remove-PathSafe -Path $f
}

foreach ($d in $DirsToDelete) {
  if ($KeepSet -contains $d) { continue }
  if ($d -eq 'models' -and $KeepModels) { continue }
  if ($d -eq 'output' -and $KeepOutput) { continue }
  Remove-PathSafe -Path $d
}

Write-Host "Cleanup complete."
