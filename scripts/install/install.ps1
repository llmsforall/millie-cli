# Millie installer for Windows: downloads the platform bundle from GitHub
# releases, verifies its checksum, unpacks it under $env:USERPROFILE\.millie\app,
# and adds the bin directory to the user PATH.
#
# Usage: powershell -ExecutionPolicy Bypass -File install.ps1 [-Release <version>]

param(
    [string]$Release = "latest"
)

$ErrorActionPreference = "Stop"
$Repo = "llmsforall/millie-cli"
$MillieHome = if ($env:MILLIE_HOME) { $env:MILLIE_HOME } else { Join-Path $env:USERPROFILE ".millie" }
$AppRoot = Join-Path $MillieHome "app"

function Step($msg) { Write-Host "==> $msg" }

if ($Release -eq "latest") {
    Step "Resolving latest release"
    $meta = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest"
    $Release = $meta.tag_name -replace '^v', ''
    if (-not $Release) { throw "could not resolve the latest release" }
}

$Name = "millie-$Release-windows-x86_64"
$Archive = "$Name.zip"
$Url = "https://github.com/$Repo/releases/download/v$Release/$Archive"
$SumsUrl = "https://github.com/$Repo/releases/download/v$Release/SHA256SUMS"

$TmpDir = Join-Path $env:TEMP ("millie-install-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TmpDir | Out-Null
try {
    Step "Downloading $Archive"
    $ArchivePath = Join-Path $TmpDir $Archive
    Invoke-WebRequest -Uri $Url -OutFile $ArchivePath

    Step "Verifying checksum"
    $Sums = (Invoke-WebRequest -Uri $SumsUrl).Content
    $Expected = ($Sums -split "`n" | Where-Object { $_ -match [regex]::Escape($Archive) } | Select-Object -First 1) -split '\s+' | Select-Object -First 1
    if (-not $Expected) { throw "no checksum listed for $Archive" }
    $Actual = (Get-FileHash -Algorithm SHA256 -Path $ArchivePath).Hash.ToLower()
    if ($Actual -ne $Expected.ToLower()) { throw "checksum mismatch for $Archive" }

    Step "Installing to $AppRoot\$Release"
    New-Item -ItemType Directory -Force -Path $AppRoot | Out-Null
    $Dest = Join-Path $AppRoot $Release
    if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
    Expand-Archive -Path $ArchivePath -DestinationPath $AppRoot
    Move-Item -Path (Join-Path $AppRoot $Name) -Destination $Dest

    $BinDir = Join-Path $Dest "bin"
    Step "Adding $BinDir to the user PATH"
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($UserPath -notlike "*$BinDir*") {
        [Environment]::SetEnvironmentVariable("Path", "$BinDir;$UserPath", "User")
        Write-Host "NOTE: open a new terminal for the PATH change to take effect."
    }

    Step "Done. Run 'millie' in a project directory to get started."
    Write-Host "The first run offers a model download (choose interactively; sizes shown)."
}
finally {
    Remove-Item -Recurse -Force $TmpDir -ErrorAction SilentlyContinue
}
