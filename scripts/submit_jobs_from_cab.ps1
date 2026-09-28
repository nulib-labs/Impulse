<#
.SYNOPSIS
    Queue document_extraction + image_processing jobs via impulse_cli.py.

.DESCRIPTION
    Each directory argument must be named <project_id>_<barcode> and contain
    the images/PDFs to process:

        data\
          projA_39015012345678\  page1.png page2.png ...
          projA_39015087654321\  scan1.tif scan2.tif ...

    The project ID and barcode are parsed from the directory name (split at
    the LAST underscore, so project IDs may contain underscores but barcodes
    may not). impulse_cli.py then uploads the files and inserts one workflow
    per directory into the FireWorks database. Nothing is run here -- workers
    pick the jobs up later.

    Files whose names match *.db (e.g. Thumbs.db) are left out: they are not
    uploaded, and a directory containing only such files counts as empty. Edit
    the $Exclude list near the top to change the patterns.

    Directories whose <project_id>_<barcode> identifier already exists in the
    FireWorks database are skipped (nothing is uploaded for them) unless
    -Force is given.

    Environment:
        S3_BUCKET, AWS_PROFILE, AWS_REGION   required (used by impulse_cli.py)
        MONGO_URI                            optional (default in impulse_cli.py)
        IMPULSE_CLI                          path to impulse_cli.py
                                             (default: next to this script)
        PYTHON                               interpreter (default: auto-detect)

    Exit status: 0 if no directory failed (submitted or skipped), 1 if any
    failed.

.PARAMETER Directory
    One or more job directories. Wildcards are expanded, so data\*\ works.

.PARAMETER DryRun
    Parse and print only; upload/submit nothing. Does not contact the
    database, so it can't report which directories would be skipped.

.PARAMETER Force
    Submit even if the identifier already exists in the FireWorks database.

.EXAMPLE
    .\submit_jobs.ps1 data\*\

.EXAMPLE
    .\submit_jobs.ps1 -DryRun data\*\
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0, ValueFromRemainingArguments = $true)]
    [string[]] $Directory,

    [Alias('n')]
    [switch] $DryRun,

    [Alias('f')]
    [switch] $Force
)

$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false

$Jobs = @('document_extraction', 'image_processing')
$Exclude = @('*.db')   # file-name globs to leave out (e.g. Thumbs.db)
$ExitSkipped = 3   # impulse_cli.py: everything already existed in FireWorks

function Write-Err([string] $Message) {
    [Console]::Error.WriteLine($Message)
}

# --- locate the CLI and a Python interpreter --------------------------------

$cli = if ($env:IMPULSE_CLI) { $env:IMPULSE_CLI } else { Join-Path $PSScriptRoot 'impulse_cli.py' }
if (-not (Test-Path -LiteralPath $cli -PathType Leaf)) {
    Write-Err "error: impulse_cli.py not found at $cli (set IMPULSE_CLI)"
    exit 1
}

$python = $env:PYTHON
if (-not $python) {
    $candidates = if ($env:OS -eq 'Windows_NT') { @('python', 'py', 'python3') } else { @('python3', 'python') }
    foreach ($name in $candidates) {
        if (Get-Command $name -ErrorAction SilentlyContinue) { $python = $name; break }
    }
}
if (-not $python) {
    Write-Err 'error: no Python interpreter found (set PYTHON)'
    exit 1
}

if (-not $DryRun) {
    foreach ($var in 'S3_BUCKET', 'AWS_PROFILE', 'AWS_REGION') {
        if ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($var))) {
            Write-Err "error: required environment variable $var is not set"
            exit 1
        }
    }
}

# --- build argument pieces --------------------------------------------------

$jobArgs = @()
foreach ($job in $Jobs) { $jobArgs += @('-j', $job) }
$forceArgs = if ($Force) { @('--force') } else { @() }
$excludeArgs = @()
foreach ($pattern in $Exclude) { $excludeArgs += @('--exclude', $pattern) }

# --- expand directory arguments (wildcards allowed) -------------------------

$targets = @()
foreach ($arg in $Directory) {
    if (Test-Path -LiteralPath $arg -PathType Container) {
        $items = @(Get-Item -LiteralPath $arg)
    } else {
        $items = @(Get-Item -Path $arg -ErrorAction SilentlyContinue)
    }

    if ($items.Count -eq 0) {
        Write-Err "error: not a directory: $arg"
        $targets += [pscustomobject]@{ Arg = $arg; Item = $null }
        continue
    }
    foreach ($item in $items) {
        $targets += [pscustomobject]@{ Arg = $item.FullName; Item = $item }
    }
}

# --- process ----------------------------------------------------------------

$submitted = 0
$skipped = 0
$failed = 0

foreach ($t in $targets) {
    if ($null -eq $t.Item) { $failed++; continue }

    if (-not $t.Item.PSIsContainer) {
        Write-Err "error: not a directory: $($t.Arg)"
        $failed++
        continue
    }

    $dir = $t.Item.FullName
    $name = $t.Item.Name
    $idx = $name.LastIndexOf('_')

    if ($idx -lt 0) {
        Write-Err "error: ${dir}: directory name must be <project_id>_<barcode>"
        $failed++
        continue
    }

    $projectId = $name.Substring(0, $idx)
    $barcode = $name.Substring($idx + 1)

    if ([string]::IsNullOrEmpty($projectId) -or [string]::IsNullOrEmpty($barcode)) {
        Write-Err "error: ${dir}: empty project ID or barcode in '$name'"
        $failed++
        continue
    }

    $hasFiles = Get-ChildItem -LiteralPath $dir -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $n = $_.Name
            -not $n.StartsWith('.') -and -not ($Exclude | Where-Object { $n -like $_ })
        } |
        Select-Object -First 1
    if (-not $hasFiles) {
        Write-Err "error: ${dir}: no files found"
        $failed++
        continue
    }

    Write-Output "[$name] project_id=$projectId barcode=$barcode jobs=$($Jobs -join ' ')"

    if ($DryRun) {
        $submitted++
        continue
    }

    $cliArgs = @('submit', '-p', $projectId, '-b', $barcode) + $jobArgs + $forceArgs + $excludeArgs + @('--files', $dir)
    & uv run $cli @cliArgs
    $rc = $LASTEXITCODE

    switch ($rc) {
        0 { $submitted++ }
        $ExitSkipped { $skipped++ }   # already in the FireWorks database
        default {
            Write-Err "error: ${dir}: submission failed (exit $rc)"
            $failed++
        }
    }
}

$verb = if ($DryRun) { 'would be submitted' } else { 'submitted' }
Write-Err "Done: $submitted $verb, $skipped skipped (already exist), $failed failed."

if ($failed -gt 0) { exit 1 } else { exit 0 }
