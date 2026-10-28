<#
.SYNOPSIS
  Export Jupyter notebooks to py:percent text (code + markdown only, outputs stripped).

.DESCRIPTION
  Used once to build legacy/*/notebooks_text/ in GN_Next_Gen from the original
  ADFSSN, GRPSSN and SN_Daisy notebooks (Windows PowerShell 5.1, no Python needed).
  Each export starts with a provenance header. Notebooks whose .ipynb bytes are
  identical to an already-exported one are not exported again; they are listed in
  the returned index with DuplicateOf set.

  The output can be read directly or converted back to .ipynb with
  `jupytext --to ipynb <file>.py` (outputs are not recoverable).

.PARAMETER Notebooks
  Full paths of the .ipynb files to export.
.PARAMETER RepoRoot
  Root of the source repository (used for git tracked/modified status). Optional.
.PARAMETER OutDir
  Destination folder for the .py exports.
#>
param(
    [Parameter(Mandatory = $true)][string[]]$Notebooks,
    [string]$RepoRoot = "",
    [Parameter(Mandatory = $true)][string]$OutDir
)

$utf8NoBom = New-Object System.Text.UTF8Encoding $false
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$seen = @{}
$index = @()

foreach ($nbPath in $Notebooks) {
    $hash = (Get-FileHash -LiteralPath $nbPath -Algorithm SHA256).Hash.ToLower()
    $rel = $nbPath
    $status = "not under git"
    if ($RepoRoot -ne "") {
        $rel = $nbPath.Substring($RepoRoot.Length).TrimStart('\', '/') -replace '\\', '/'
        $tracked = git -C $RepoRoot ls-files -- $rel
        if ($tracked) {
            $porc = git -C $RepoRoot status --porcelain -- $rel
            $status = if ($porc) { "tracked, modified (uncommitted)" } else { "tracked, clean" }
        } else {
            $status = "untracked (exists only in the original working copy)"
        }
    }
    $outName = ([System.IO.Path]::GetFileNameWithoutExtension($nbPath) -replace ' ', '_') + ".py"

    if ($seen.ContainsKey($hash)) {
        $index += [pscustomobject]@{ Source = $rel; Export = ""; Sha256 = $hash; Cells = ""; Status = $status; DuplicateOf = $seen[$hash] }
        continue
    }

    $nb = [System.IO.File]::ReadAllText($nbPath, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
    $nCode = 0; $nMd = 0; $nRaw = 0
    $sb = New-Object System.Text.StringBuilder
    foreach ($c in $nb.cells) {
        $src = if ($c.source -is [array]) { $c.source -join '' } else { [string]$c.source }
        $src = $src -replace "`r`n", "`n"
        switch ($c.cell_type) {
            'code' {
                $nCode++
                [void]$sb.Append("# %%`n")
                [void]$sb.Append($src.TrimEnd("`n"))
                [void]$sb.Append("`n`n")
            }
            default {
                if ($c.cell_type -eq 'markdown') { $nMd++; $tag = "markdown" } else { $nRaw++; $tag = "raw" }
                [void]$sb.Append("# %% [$tag]`n")
                foreach ($line in ($src.TrimEnd("`n") -split "`n")) {
                    if ($line -eq '') { [void]$sb.Append("#`n") } else { [void]$sb.Append("# $line`n") }
                }
                [void]$sb.Append("`n")
            }
        }
    }
    $nCells = $nCode + $nMd + $nRaw
    $header = @(
        "# ---------------------------------------------------------------------------",
        "# LEGACY NOTEBOOK EXPORT (code + markdown only; outputs/figures stripped)",
        "# Source:   $rel",
        "# Status:   $status",
        "# SHA-256 of original .ipynb: $hash",
        "# Cells:    $nCells (code $nCode, markdown $nMd, raw $nRaw)",
        "# Exported: $(Get-Date -Format 'yyyy-MM-dd') by legacy/tools/export_notebook_text.ps1",
        "# Format:   py:percent ('# %%' = code cell, '# %% [markdown]' = markdown cell)",
        "# Data paths inside refer to the ORIGINAL repo layout; see legacy/README.md.",
        "# ---------------------------------------------------------------------------",
        "", ""
    ) -join "`n"
    [System.IO.File]::WriteAllText((Join-Path $OutDir $outName), $header + $sb.ToString(), $utf8NoBom)
    $seen[$hash] = $rel
    $index += [pscustomobject]@{ Source = $rel; Export = $outName; Sha256 = $hash; Cells = $nCells; Status = $status; DuplicateOf = "" }
}

$index
