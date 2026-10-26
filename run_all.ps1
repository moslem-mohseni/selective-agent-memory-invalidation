param([string]$Python='python', [switch]$SkipPdf)
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
function Run-Python([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python step failed: $Arguments" }
}
Run-Python @('-m','pytest','tests','-q')
Run-Python @('src/run_experiments.py','--seeds','300')
Run-Python @('src/analyze.py')
Run-Python @('src/llm_study/run.py','--backend','mock','--episodes','100')
Run-Python @('paper/build_manuscript.py')
Run-Python @('paper/audit_numbers.py')
Run-Python @('paper/audit_manuscript.py')
Run-Python @('paper/update_readme.py')
if (-not $SkipPdf) {
    & "$PSScriptRoot/paper/docx2pdf.ps1" -In "$PSScriptRoot/paper/P1_Selective_Invalidation_of_Procedural_Agent_Memory.docx" -Out "$PSScriptRoot/paper/P1_Selective_Invalidation_of_Procedural_Agent_Memory.pdf"
    if ($LASTEXITCODE -ne 0) { throw 'Word export failed' }
}
Run-Python @('paper/package_supplement.py')
