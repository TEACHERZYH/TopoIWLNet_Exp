$ErrorActionPreference = "Stop"

$Project = "F:\2026\Remote Sensing_codex\TopoIWLNet_Exp"
$Datasets = "F:\2026\Remote Sensing_codex\datasets"
$Python = "C:\Users\zyh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$Log = Join-Path $Project "results\sensitivity_local_run.log"

New-Item -ItemType Directory -Force -Path (Join-Path $Project "results") | Out-Null
"[$(Get-Date -Format s)] Starting sensitivity analysis" | Set-Content -LiteralPath $Log -Encoding UTF8

function Run-Sensitivity {
    param(
        [string]$DatasetName,
        [string]$Config,
        [string]$Checkpoint,
        [string]$DatasetRoot,
        [string]$OutDir,
        [string]$ExistingGrid
    )

    "[$(Get-Date -Format s)] Running $DatasetName" | Add-Content -LiteralPath $Log -Encoding UTF8
    & $Python `
        (Join-Path $Project "scripts\sensitivity_analysis.py") `
        --config $Config `
        --checkpoint $Checkpoint `
        --dataset-name $DatasetName `
        --dataset-root $DatasetRoot `
        --out-dir $OutDir `
        --existing-threshold-grid $ExistingGrid `
        --num-workers 0 `
        >> $Log 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Sensitivity analysis failed for $DatasetName with exit code $LASTEXITCODE"
    }
    "[$(Get-Date -Format s)] Finished $DatasetName" | Add-Content -LiteralPath $Log -Encoding UTF8
}

Run-Sensitivity `
    -DatasetName "GF6_TCUNet" `
    -Config (Join-Path $Project "configs\topoiwl_remote_gf6_mobilenetv3_ablate_full80.yaml") `
    -Checkpoint (Join-Path $Project "experiments\remote_gf6_mobilenetv3_ablate_full80\best.pt") `
    -DatasetRoot (Join-Path $Datasets "GF6_TCUNet\processed\topoiwl_format") `
    -OutDir (Join-Path $Project "experiments\remote_gf6_mobilenetv3_ablate_full80\sensitivity") `
    -ExistingGrid (Join-Path $Project "experiments\remote_gf6_mobilenetv3_ablate_full80\threshold_sweep_val_fast.csv")

Run-Sensitivity `
    -DatasetName "SeaLand_Coastline_2025" `
    -Config (Join-Path $Project "configs\topoiwl_remote_sealand_mobilenetv3_full80.yaml") `
    -Checkpoint (Join-Path $Project "experiments\remote_sealand_mobilenetv3_full80\best.pt") `
    -DatasetRoot (Join-Path $Datasets "SeaLand_Coastline_2025\processed\topoiwl_format") `
    -OutDir (Join-Path $Project "experiments\remote_sealand_mobilenetv3_full80\sensitivity") `
    -ExistingGrid (Join-Path $Project "experiments\remote_sealand_mobilenetv3_full80\threshold_sweep_val_fast.csv")

"[$(Get-Date -Format s)] Completed all sensitivity analyses" | Add-Content -LiteralPath $Log -Encoding UTF8
