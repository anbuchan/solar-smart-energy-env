$app = New-Object -ComObject shell.application
$path = "C:\Users\anbuc\solar_smart_env"
$app.Namespace('shell:::{679f85cb-0220-4080-b29b-5540cc05aab6}').Items() | Where-Object { $_.Path -eq $path -or $_.Name -match "solar_smart_env" } | ForEach-Object { 
    $_.InvokeVerb("unpinfromhome") 
    Write-Host "Unpinned: $($_.Path)"
}
