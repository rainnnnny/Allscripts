$a=Get-ChildItem . | ForEach-Object -Process{if($_ -is [System.IO.FileInfo]){echo $_.name;}}

# echo $a

foreach($each in $a)
{
	(get-content $each) | Foreach-Object {$_ -replace "ScrollbackSize=.*", "ScrollbackSize=99999"} | Set-Content $each
	echo $each
}


